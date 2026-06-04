"""
LangGraph agentic pipeline for traceable test-case generation.

Graph topology:
    START -> generate_hyde -> retrieve -> draft -> ground_check -> END
                                  ^_______ (loop back if not grounded, max 2 retries)

Nodes:
    generate_hyde  - HyDE: write a hypothetical spec paragraph and retrieve on
                     that instead of the raw requirement (reduces hallucination).
    retrieve       - query selection + FAISS + re-ranking (see retrieval.py).
    draft          - produce a structured TestSuite. Structured-output models use
                     native parsing; text-only models get a strict-JSON prompt and
                     a tolerant parser, so downstream output is always structured.
    ground_check   - LLM-as-judge: verify every test case traces to the context.
                     If not, loop back to retrieve with a wider net.

The conditional edge is what makes this a proper cyclic graph rather than a
plain chain, and is a first step toward an LLM-as-a-judge self-optimising loop.
"""

import json
import re
from typing import List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from .llm import supports_structured_output
from .retrieval import retrieve as run_retrieval, format_context
from .schemas import TestCase, TestSuite

MAX_ATTEMPTS = 2


class AgentState(TypedDict, total=False):
    requirement: str
    hyde_doc: str
    context: str
    retrieved: List[Document]
    test_cases: list
    grounded: bool
    attempts: int


# ─── LLM response helpers ─────────────────────────────────────────────────────

def _text(resp) -> str:
    """Normalise an LLM response to a plain string (handles str or AIMessage)."""
    if isinstance(resp, str):
        content = resp
    else:
        content = getattr(resp, "content", str(resp))
    # DeepSeek-R1 and similar reasoning models emit <think>...</think> blocks.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip()


def _parse_testsuite(raw: str) -> List[dict]:
    """
    Tolerant parser: pull a TestSuite-shaped JSON object out of model text.
    Validates against the Pydantic schema and returns serialised test cases.
    Raises ValueError if nothing usable is found.
    """
    text = raw.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Grab the outermost JSON object.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    data = json.loads(text[start : end + 1])
    suite = TestSuite(**data)
    return [tc.model_dump() for tc in suite.test_cases]


_JSON_SHAPE = (
    '{"test_cases": [{"title": str, "priority": "High|Medium|Low", '
    '"preconditions": str, "steps": [{"action": str, "expected_result": str}], '
    '"source_citations": ["[source: filename p.N]"]}]}'
)


def build_agent(vectorstore, llm):
    """Build and compile the agent graph for the given vector store and LLM."""

    use_structured = supports_structured_output()
    structured_llm = llm.with_structured_output(TestSuite) if use_structured else None

    # ─── Node 1: HyDE ─────────────────────────────────────────────────────────
    def generate_hyde(state: AgentState) -> dict:
        resp = llm.invoke(
            "You are a senior QA engineer for safety-critical, regulated software. "
            "Given the requirement below, write a single concise paragraph (3-5 "
            "sentences) describing exactly how a fully compliant system should "
            "behave. Write it as a specification statement, not a question; it will "
            "be used to search technical documentation.\n\n"
            f"Requirement: {state['requirement']}"
        )
        return {"hyde_doc": _text(resp)}

    # ─── Node 2: Retrieve (query selection + re-ranking) ──────────────────────
    def retrieve(state: AgentState) -> dict:
        k = 4 + 2 * state.get("attempts", 0)
        docs = run_retrieval(
            vectorstore, state["requirement"], state["hyde_doc"], k=k
        )
        return {"retrieved": docs, "context": format_context(docs)}

    # ─── Node 3: Draft ────────────────────────────────────────────────────────
    def draft(state: AgentState) -> dict:
        rules = (
            "You are a QA engineer drafting formal test cases for regulated, "
            "safety-critical software. Rules:\n"
            "1. Use ONLY information in the CONTEXT below; do not invent behaviour.\n"
            "2. Every test case MUST list source_citations with the exact "
            "[source: filename p.N] tags it relies on.\n"
            "3. Expected results must be specific and verifiable.\n"
            "4. Produce 2-5 test cases covering the requirement.\n\n"
            f"REQUIREMENT:\n{state['requirement']}\n\n"
            f"CONTEXT:\n{state['context']}"
        )

        if use_structured:
            suite = structured_llm.invoke(rules)
            cases = [tc.model_dump() for tc in suite.test_cases]
        else:
            resp = llm.invoke(
                rules
                + "\n\nReturn ONLY a JSON object in EXACTLY this shape, no prose, "
                "no markdown:\n" + _JSON_SHAPE
            )
            try:
                cases = _parse_testsuite(_text(resp))
            except (ValueError, json.JSONDecodeError, Exception):
                # Last-resort fallback: keep the raw draft as one reviewable case
                # so the pipeline still returns something usable.
                cases = [
                    TestCase(
                        title="Generated test case (unparsed draft)",
                        priority="Medium",
                        preconditions="Document indexed; see steps for raw output.",
                        steps=[{"action": "Review the drafted text below",
                                "expected_result": _text(resp)[:1500]}],
                        source_citations=_citations_from_context(state["context"]),
                    ).model_dump()
                ]

        return {"test_cases": cases, "attempts": state.get("attempts", 0) + 1}

    # ─── Node 4: Ground check (LLM-as-judge) ──────────────────────────────────
    def ground_check(state: AgentState) -> dict:
        verdict = _text(llm.invoke(
            "You are a compliance reviewer for safety-critical software. Reply with "
            "ONLY 'YES' if EVERY expected result in EVERY test case can be traced to "
            "a specific statement in the CONTEXT, otherwise reply ONLY 'NO'.\n\n"
            f"CONTEXT:\n{state['context']}\n\n"
            f"TEST CASES:\n{json.dumps(state['test_cases'], ensure_ascii=False)}"
        )).upper()
        return {"grounded": verdict.startswith("YES")}

    def route_after_check(state: AgentState) -> str:
        if state.get("grounded") or state.get("attempts", 0) >= MAX_ATTEMPTS:
            return END
        return "retrieve"

    g = StateGraph(AgentState)
    g.add_node("generate_hyde", generate_hyde)
    g.add_node("retrieve", retrieve)
    g.add_node("draft", draft)
    g.add_node("ground_check", ground_check)

    g.add_edge(START, "generate_hyde")
    g.add_edge("generate_hyde", "retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "ground_check")
    g.add_conditional_edges("ground_check", route_after_check, ["retrieve", END])

    return g.compile()


def _citations_from_context(context: str) -> List[str]:
    cites = re.findall(r"\[source:[^\]]+\]", context)
    # De-duplicate, keep order, cap to 5.
    seen, out = set(), []
    for c in cites:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:5] or ["(see context)"]
