"""
Pydantic schemas for structured test-case output.

When the LLM provider supports it, with_structured_output() guarantees the
model returns exactly these shapes. For text-only providers (Ollama, NVIDIA
NIM) the agent prompts for this same JSON shape and parses it, so the rest of
the application sees one consistent contract.
"""

from typing import List
from pydantic import BaseModel, Field


class TestStep(BaseModel):
    action: str = Field(description="The concrete action the tester performs")
    expected_result: str = Field(
        description="The specific, verifiable expected outcome"
    )


class TestCase(BaseModel):
    title: str = Field(description="Short, descriptive title of the test case")
    priority: str = Field(description="Must be exactly: High, Medium, or Low")
    preconditions: str = Field(
        description="Required system state before the test can be executed"
    )
    steps: List[TestStep] = Field(
        description="Ordered list of test steps with expected results"
    )
    source_citations: List[str] = Field(
        description=(
            "List of [source: filename p.N] tags this test case is grounded in. "
            "Every claim in the test case must trace back to a cited source."
        )
    )


class TestSuite(BaseModel):
    test_cases: List[TestCase] = Field(
        description="2-5 test cases that together cover the stated requirement"
    )


class ApproveRequest(BaseModel):
    """Payload to persist a human-approved, e-signed test case."""
    requirement: str
    test_case: TestCase
    signed_by: str = Field(default="reviewer", description="Reviewer identity")


# ─── Analysis / execution ─────────────────────────────────────────────────────

class RelevanceVerdict(BaseModel):
    """Output of the relevance gate that guards against off-topic documents."""
    relevant: bool = Field(description="True only if the document can answer the request")
    document_topic: str = Field(default="unknown", description="What the document actually is")
    reason: str = Field(default="", description="One or two sentences explaining the decision")


class Finding(BaseModel):
    observation: str = Field(description="A single observation grounded in the document")
    evidence: str = Field(default="", description="The supporting detail quoted/paraphrased from the document")
    citation: str = Field(default="", description="The [source: filename p.N] tag this finding relies on")


class AnalysisReport(BaseModel):
    summary: str = Field(default="", description="One or two sentence overview")
    findings: List[Finding] = Field(default_factory=list)
    assessment: str = Field(default="", description="The analytical judgement, grounded in the findings")
    recommendation: str = Field(default="", description="Suggested next action, framed cautiously")
    caveats: str = Field(default="", description="Limitations and when to consult a professional")
