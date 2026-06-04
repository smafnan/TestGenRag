"""
RAG ingestion pipeline: PDF -> page-aware chunks -> embeddings -> FAISS index.

The extractor (PyPDF / PDFPlumber / Docling) yields one Document per page with
page + source metadata, which flows all the way through to the source_citations
in the generated test cases.
"""

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .extractors import extract_pdf
from .llm import get_embeddings


def _index_dir() -> str:
    """Resolve the FAISS directory at call time so config/tests can change it."""
    return os.getenv("FAISS_DIR", "faiss_index")


_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def ingest_pdf(file_path: str, source_name: str) -> int:
    """
    Parse a PDF, chunk it, embed the chunks, and upsert into the FAISS index.
    Creates the index if it does not exist yet; otherwise appends to it.

    Returns the number of chunks indexed.
    """
    docs = extract_pdf(file_path, source_name)

    chunks = _SPLITTER.split_documents(docs)
    # Propagate page_type onto chunks (splitter copies metadata already).
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        return 0

    embeddings = get_embeddings()

    if index_exists():
        vs = FAISS.load_local(
            _index_dir(), embeddings, allow_dangerous_deserialization=True
        )
        vs.add_documents(chunks)
    else:
        vs = FAISS.from_documents(chunks, embeddings)

    vs.save_local(_index_dir())
    return len(chunks)


def load_vectorstore() -> FAISS:
    """Load and return the persisted FAISS index."""
    embeddings = get_embeddings()
    return FAISS.load_local(
        _index_dir(), embeddings, allow_dangerous_deserialization=True
    )


def index_exists() -> bool:
    """Return True if the FAISS index has been created."""
    return os.path.isdir(_index_dir())
