"""
End-to-end pipeline tests (model layer stubbed) plus focused unit tests.

Run with:  pytest -q
"""

import os

import pytest
from fastapi.testclient import TestClient

from app import ingestion, main
from app.agent import _parse_testsuite, _citations_from_context
from app.extractors import _classify_page
from app import cache

from tests.conftest import SAMPLE_PDF
from tests.fakes import FakeEmbeddings, FakeLLM, FakeStructuredLLM


@pytest.fixture
def client(monkeypatch):
    # Swap the model layer for deterministic fakes.
    monkeypatch.setattr(ingestion, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(main, "get_llm", lambda *a, **k: FakeLLM())
    # Fresh in-memory cache + initialise the (temp sqlite) DB.
    cache._memory.clear()
    main.database.init_db()
    with TestClient(main.app) as c:
        yield c


def _ingest_sample(client) -> dict:
    with open(SAMPLE_PDF, "rb") as fh:
        res = client.post(
            "/ingest", files={"file": ("sample.pdf", fh, "application/pdf")}
        )
    assert res.status_code == 200, res.text
    return res.json()


# ─── End-to-end (text-mode provider, the default local Ollama path) ───────────

def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "ollama"
    assert body["db_backend"] == "sqlite"


def test_ingest_indexes_chunks(client):
    body = _ingest_sample(client)
    assert body["name"] == "sample.pdf"
    assert body["chunks"] > 0
    assert ingestion.index_exists()
    # Document registry reflects the upload.
    docs = client.get("/documents").json()
    assert docs[-1]["name"] == "sample.pdf"


def test_generate_returns_structured_cases(client):
    _ingest_sample(client)
    res = client.post("/generate", json={"requirement": "audible alarm on low value"})
    assert res.status_code == 200, res.text
    data = res.json()

    assert isinstance(data["test_cases"], list) and len(data["test_cases"]) >= 1
    tc = data["test_cases"][0]
    # The contract the frontend relies on.
    for key in ("title", "priority", "preconditions", "steps", "source_citations"):
        assert key in tc
    assert isinstance(tc["steps"], list) and tc["steps"]
    assert "action" in tc["steps"][0] and "expected_result" in tc["steps"][0]
    assert tc["source_citations"]                    # traceability present
    assert data["grounded"] is True                  # judge said YES
    assert data["cached"] is False


def test_generate_is_cached_second_time(client):
    _ingest_sample(client)
    payload = {"requirement": "alarm clears when value recovers"}
    first = client.post("/generate", json=payload).json()
    second = client.post("/generate", json=payload).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["test_cases"] == first["test_cases"]


def test_generate_requires_index(client):
    res = client.post("/generate", json={"requirement": "no docs yet"})
    assert res.status_code == 400


def test_approve_persists_to_db(client):
    _ingest_sample(client)
    gen = client.post("/generate", json={"requirement": "audible alarm"}).json()
    tc = gen["test_cases"][0]

    res = client.post(
        "/approve",
        json={"requirement": "audible alarm", "test_case": tc, "signed_by": "afnan"},
    )
    assert res.status_code == 200, res.text
    record = res.json()["record"]
    assert record["title"] == tc["title"]
    assert record["signed_by"] == "afnan"

    approved = client.get("/approved").json()
    assert any(r["title"] == tc["title"] for r in approved)


# ─── Structured-output provider path (e.g. Claude / OpenAI) ───────────────────

def test_structured_output_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")     # read at call time
    monkeypatch.setattr(ingestion, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(main, "get_llm", lambda *a, **k: FakeStructuredLLM())
    cache._memory.clear()
    main.database.init_db()
    with TestClient(main.app) as c:
        _ingest_sample(c)
        data = c.post("/generate", json={"requirement": "alarm"}).json()
    assert len(data["test_cases"]) == 2
    assert data["test_cases"][0]["priority"] == "High"


# ─── Unit tests ───────────────────────────────────────────────────────────────

def test_parser_handles_fences_and_think_blocks():
    raw = '<think>reasoning</think>\n```json\n{"test_cases": [{"title":"T",' \
          '"priority":"Low","preconditions":"p","steps":[{"action":"a",' \
          '"expected_result":"e"}],"source_citations":["[source: x p.1]"]}]}\n```'
    cases = _parse_testsuite(raw)
    assert cases[0]["title"] == "T"
    assert cases[0]["steps"][0]["expected_result"] == "e"


def test_parser_rejects_garbage():
    with pytest.raises(Exception):
        _parse_testsuite("this is not json at all")


def test_citations_extracted_from_context():
    ctx = "[source: a.pdf p.1]\nfoo\n\n[source: a.pdf p.1]\n[source: b.pdf p.2]"
    cites = _citations_from_context(ctx)
    assert cites == ["[source: a.pdf p.1]", "[source: b.pdf p.2]"]   # de-duped, ordered


def test_page_classification():
    assert _classify_page("") == "empty"
    prose = "The system shall emit an alarm when the value drops below threshold. " * 5
    assert _classify_page(prose) == "text"
    table = "1 | 22 | 333\n4 | 55 | 666\n7 | 88 | 999\n10 | 11 | 12\n13 | 14 | 15"
    assert _classify_page(table) in {"table", "mixed"}


def test_cache_roundtrip():
    cache._memory.clear()
    assert cache.get_cached("ollama", "req") is None
    cache.set_cached("ollama", "req", {"ok": True})
    assert cache.get_cached("ollama", "req") == {"ok": True}
    assert cache.backend_name() == "memory"
