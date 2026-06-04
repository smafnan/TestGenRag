"""
Test doubles so the full pipeline runs without any network (no Hugging Face
model download, no Ollama, no hosted API).
"""

import hashlib
import json

from langchain_core.embeddings import Embeddings

from app.schemas import TestSuite, TestCase, TestStep


class FakeEmbeddings(Embeddings):
    """Deterministic hash-based vectors of fixed dimension."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self.dim)]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


_SAMPLE_SUITE = {
    "test_cases": [
        {
            "title": "Audible alarm triggers below threshold",
            "priority": "High",
            "preconditions": "Device powered on and monitoring an active channel.",
            "steps": [
                {
                    "action": "Drive the monitored value below the configured threshold.",
                    "expected_result": "An audible alarm sounds within 2 seconds.",
                }
            ],
            "source_citations": ["[source: sample.pdf p.0]"],
        },
        {
            "title": "Alarm clears when value recovers",
            "priority": "Medium",
            "preconditions": "Alarm currently active.",
            "steps": [
                {
                    "action": "Raise the monitored value back above the threshold.",
                    "expected_result": "The audible alarm stops automatically.",
                }
            ],
            "source_citations": ["[source: sample.pdf p.0]"],
        },
    ]
}


class FakeLLM:
    """Mimics a text-only provider (e.g. Ollama): invoke() returns a string."""

    def invoke(self, prompt, **kwargs):
        text = prompt if isinstance(prompt, str) else str(prompt)
        if "Reply with ONLY 'YES'" in text:
            return "YES"
        if "Return ONLY a JSON object" in text:
            # Wrap in a code fence + a stray reasoning block to exercise the parser.
            return "<think>drafting…</think>\n```json\n" + json.dumps(_SAMPLE_SUITE) + "\n```"
        # HyDE / anything else
        return (
            "The system shall emit an audible alarm within two seconds when the "
            "monitored value drops below the configured threshold, and shall clear "
            "the alarm automatically once the value recovers."
        )


class _StructuredRunner:
    def invoke(self, prompt, **kwargs):
        return TestSuite(
            test_cases=[
                TestCase(
                    title=tc["title"],
                    priority=tc["priority"],
                    preconditions=tc["preconditions"],
                    steps=[TestStep(**s) for s in tc["steps"]],
                    source_citations=tc["source_citations"],
                )
                for tc in _SAMPLE_SUITE["test_cases"]
            ]
        )


class FakeStructuredLLM(FakeLLM):
    """Mimics a structured-output provider (e.g. Claude/OpenAI)."""

    def with_structured_output(self, schema):
        return _StructuredRunner()
