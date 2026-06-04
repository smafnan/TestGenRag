"""
Document analysis / execution with a relevance gate.

Flow:
    retrieve -> relevance gate -> (if relevant) analyse -> ground-check

The relevance gate is the important guard: it stops the assistant from
fabricating an answer when the uploaded document does not actually contain the
kind of information the request needs (for example, a restaurant menu uploaded
for a health assessment). Only when the document genuinely matches does the
assistant produce a grounded, citation-backed analysis, which a human still
approves before it counts as output.
"""

import json
import re
import time
from typing import List

from .retrieval import retrieve as run_retrieval, format_context, build_queries
from .schemas import AnalysisReport, Finding, RelevanceVerdict


# ─── helpers ──────────────────────────────────────────────────────────────────

def _text(resp) -> str:
    content = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip()


def _parse_json_obj(raw: str) -> dict:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


# ─── prompts ──────────────────────────────────────────────────────────────────

_RELEVANCE_PROMPT = (
    "You are a relevance gate for a document-analysis assistant. A user asked a "
    "question to be answered using one or more uploaded documents. Below are "
    "excerpts retrieved from those documents. The excerpts may come from SEVERAL "
    "different documents and may cover MULTIPLE unrelated topics at once. That is "
    "normal and is NOT by itself a reason to reject.\n\n"
    "Your job is permissive. Decide whether the excerpts contain ANY information "
    "that could help address the request.\n\n"
    "Mark relevant = true if at least some of the retrieved material is usable to "
    "address the request, EVEN IF other parts are unrelated, EVEN IF the document "
    "does not state a conclusion in words, and EVEN IF the request only names a "
    "person, topic, or document. Broad requests such as 'what do you know about "
    "X', 'summarise this', 'tell me about these documents', or 'what is in here' "
    "can be addressed by almost any document, so mark them relevant. When in "
    "doubt, mark relevant = true and let the analysis step explain any gaps.\n\n"
    "Mark relevant = false ONLY when the excerpts are entirely about a different "
    "subject and contain nothing at all that could address the request, for "
    "example a restaurant menu when asked to interpret clinical blood results. "
    "This should be rare.\n\n"
    "Examples:\n"
    "- A CV or resume, asked 'what do you know about <person>?' -> relevant: true "
    "(the CV describes that person).\n"
    "- A mix of a CV and a lab report, asked 'what do you know about <person>?' -> "
    "relevant: true (the CV gives the profile; other documents add context).\n"
    "- A blood test report, asked 'is the person healthy or do they need "
    "treatment?' -> relevant: true.\n"
    "- Any document, asked 'summarise this' -> relevant: true.\n"
    "- A restaurant menu, asked to 'interpret these blood test results' -> relevant: false.\n\n"
    "QUESTION:\n{question}\n\n"
    "RETRIEVED DOCUMENT EXCERPTS:\n{context}\n\n"
    "Return ONLY a JSON object, no prose:\n"
    '{{"relevant": true or false, "document_topic": "short label for what the '
    'documents are", "reason": "one or two sentences explaining the decision"}}'
)

_ANALYSIS_PROMPT = (
    "You are a careful analyst. Using ONLY the document context below, analyse "
    "and answer the user's request directly and usefully.\n"
    "Rules:\n"
    "1. Ground every observation in the context and cite it with the exact "
    "[source: filename p.N] tags shown in the context.\n"
    "2. Do not invent data, numbers, or facts that are not present in the context.\n"
    "3. If the content is clinical or lab data, interpret it: for each relevant "
    "value, state whether it falls inside or outside any reference range present "
    "in the document, and give a clear overall read of what the results indicate "
    "(for example, which markers are normal and which are flagged high or low, and "
    "what that generally suggests). Make clear this is an interpretation of the "
    "document, NOT a diagnosis, and that a qualified clinician should confirm and "
    "decide on any treatment. Do not prescribe specific treatment.\n"
    "4. Give a genuine answer to the question in the assessment; do not refuse to "
    "interpret data that is present. Always include honest caveats and limitations.\n\n"
    "REQUEST:\n{question}\n\n"
    "CONTEXT:\n{context}\n\n"
    "Return ONLY a JSON object in EXACTLY this shape, no prose, no markdown:\n"
    '{{"summary": str, "findings": [{{"observation": str, "evidence": str, '
    '"citation": "[source: filename p.N]"}}], "assessment": str, '
    '"recommendation": str, "caveats": str}}'
)


# ─── steps ────────────────────────────────────────────────────────────────────

def check_relevance(llm, question: str, context: str) -> RelevanceVerdict:
    """Relevance gate. Fails open (relevant=True) only if the verdict is unparseable."""
    try:
        data = _parse_json_obj(_text(llm.invoke(
            _RELEVANCE_PROMPT.format(question=question, context=context)
        )))
        return RelevanceVerdict(**data)
    except Exception:
        return RelevanceVerdict(
            relevant=True,
            document_topic="unknown",
            reason="The relevance check could not be parsed; proceeding with caution.",
        )


def produce_analysis(llm, question: str, context: str) -> AnalysisReport:
    """Produce a structured, grounded analysis. Falls back to wrapping raw text."""
    raw = _text(llm.invoke(_ANALYSIS_PROMPT.format(question=question, context=context)))
    try:
        return AnalysisReport(**_parse_json_obj(raw))
    except Exception:
        return AnalysisReport(
            summary="Automated analysis (unstructured fallback).",
            findings=[],
            assessment=raw[:1500],
            recommendation="Review the assessment text; the model did not return structured output.",
            caveats="This is an automated reading of the document, not professional advice.",
        )


def ground_check(llm, context: str, report: AnalysisReport) -> bool:
    verdict = _text(llm.invoke(
        "You are a strict reviewer. Reply with ONLY 'YES' if every finding and the "
        "assessment below can be traced to a specific statement in the CONTEXT, "
        "otherwise reply ONLY 'NO'.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANALYSIS:\n{json.dumps(report.model_dump(), ensure_ascii=False)}"
    )).upper()
    return verdict.startswith("YES")


# ─── orchestration ────────────────────────────────────────────────────────────

def analyze(vectorstore, llm, question: str, model_name: str = "") -> dict:
    """
    Run the full analysis flow and return a JSON-serialisable result:
        relevant, document_topic, reason, report (or None), grounded, attempts, trace
    """
    t0 = time.time()
    docs = run_retrieval(vectorstore, question, question, k=8)
    context = format_context(docs)

    trace = {
        "queries": build_queries(question, question),
        "chunks": [
            {
                "source": d.metadata.get("source", "?"),
                "page": d.metadata.get("page", "?"),
                "page_type": d.metadata.get("page_type", "text"),
                "snippet": (d.page_content or "")[:240].strip(),
            }
            for d in docs
        ],
        "model": model_name or "llm",
    }

    verdict = check_relevance(llm, question, context)
    result = {
        "relevant": verdict.relevant,
        "document_topic": verdict.document_topic,
        "reason": verdict.reason,
        "report": None,
        "grounded": False,
        "attempts": 0,
        "trace": trace,
    }

    if not verdict.relevant:
        trace["elapsed_ms"] = int((time.time() - t0) * 1000)
        return result

    report = produce_analysis(llm, question, context)
    result["report"] = report.model_dump()
    result["grounded"] = ground_check(llm, context, report)
    result["attempts"] = 1
    trace["elapsed_ms"] = int((time.time() - t0) * 1000)
    return result
