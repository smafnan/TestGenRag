"""
Set environment BEFORE importing the app so module-level config (FAISS_DIR,
DATABASE_URL, provider) points at throwaway locations.
"""

import os
import shutil
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="testgenrag_")
os.environ.setdefault("LLM_PROVIDER", "ollama")          # text-mode by default
os.environ["EMBEDDINGS_PROVIDER"] = "huggingface"        # patched in tests anyway
os.environ["FAISS_DIR"] = os.path.join(_TMP, "faiss_index")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_TMP, 'test.db')}"
os.environ["AUTH_ENABLED"] = "false"
os.environ["RERANK_METHOD"] = "mmr"

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "sample.pdf")


@pytest.fixture(autouse=True)
def _clean_index():
    """Start every test with no FAISS index."""
    idx = os.environ["FAISS_DIR"]
    if os.path.isdir(idx):
        shutil.rmtree(idx)
    yield
    if os.path.isdir(idx):
        shutil.rmtree(idx)
