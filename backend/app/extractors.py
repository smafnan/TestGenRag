"""
Advanced PDF extraction with pluggable strategies and dynamic page classification.

Set PDF_EXTRACTOR in .env to choose how documents are parsed:
    pypdf      - fast, lightweight, plain text per page (default)
    pdfplumber - better tables and layout-aware text extraction
    docling    - richest structure (headings, tables, reading order); heavier

Every extractor returns a list of LangChain Documents, one per page, each
carrying metadata: source, page, and a coarse `page_type` classification
(text | table | mixed | empty) used downstream for retrieval weighting.
"""

import os
from typing import List

from langchain_core.documents import Document


def _classify_page(text: str, table_ratio: float = 0.0) -> str:
    """
    Dynamic page classification.

    A lightweight heuristic that labels each page so retrieval can treat a
    dense specification table differently from prose. Real systems can swap
    this for a trained classifier; the interface stays the same.
    """
    stripped = text.strip()
    if not stripped:
        return "empty"
    if table_ratio >= 0.5:
        return "table"
    # Many short lines / lots of digits and separators => likely tabular.
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if lines:
        digit_heavy = sum(c.isdigit() or c in "|;\t" for c in stripped) / max(len(stripped), 1)
        short_lines = sum(len(ln) < 40 for ln in lines) / len(lines)
        if digit_heavy > 0.18 and short_lines > 0.5:
            return "table"
        if digit_heavy > 0.10 and short_lines > 0.4:
            return "mixed"
    return "text"


def _extract_pypdf(file_path: str, source_name: str) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    docs = PyPDFLoader(file_path).load()  # one Document per page, sets 'page'
    for d in docs:
        d.metadata["source"] = source_name
        d.metadata["page_type"] = _classify_page(d.page_content)
        d.metadata["extractor"] = "pypdf"
    return docs


def _extract_pdfplumber(file_path: str, source_name: str) -> List[Document]:
    import pdfplumber

    docs: List[Document] = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            # Append a flattened representation of any tables found.
            for tbl in tables:
                rows = [
                    " | ".join((cell or "").strip() for cell in row)
                    for row in tbl
                    if any(cell for cell in row)
                ]
                if rows:
                    text += "\n\n[TABLE]\n" + "\n".join(rows)
            table_ratio = len(tables) / max(len(page.lines) + len(tables), 1)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": i,
                        "page_type": _classify_page(text, table_ratio),
                        "extractor": "pdfplumber",
                    },
                )
            )
    return docs


def _extract_docling(file_path: str, source_name: str) -> List[Document]:
    # Docling produces structured Markdown with headings/tables in reading order.
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document

    docs: List[Document] = []
    # Docling exposes pages; export each to Markdown when available, else whole-doc.
    try:
        n_pages = doc.num_pages()
    except Exception:
        n_pages = 0

    if n_pages:
        for i in range(n_pages):
            try:
                md = doc.export_to_markdown(page_no=i + 1)
            except Exception:
                md = ""
            docs.append(
                Document(
                    page_content=md,
                    metadata={
                        "source": source_name,
                        "page": i,
                        "page_type": _classify_page(md),
                        "extractor": "docling",
                    },
                )
            )
    else:
        md = doc.export_to_markdown()
        docs.append(
            Document(
                page_content=md,
                metadata={
                    "source": source_name,
                    "page": 0,
                    "page_type": _classify_page(md),
                    "extractor": "docling",
                },
            )
        )
    return docs


_EXTRACTORS = {
    "pypdf": _extract_pypdf,
    "pdfplumber": _extract_pdfplumber,
    "docling": _extract_docling,
}


def extract_pdf(file_path: str, source_name: str) -> List[Document]:
    """Extract a PDF into page-level Documents using the configured strategy."""
    name = os.getenv("PDF_EXTRACTOR", "pypdf").lower()
    extractor = _EXTRACTORS.get(name)
    if extractor is None:
        raise ValueError(
            f"Unknown PDF_EXTRACTOR: '{name}'. "
            f"Valid options: {', '.join(_EXTRACTORS)}."
        )
    try:
        return extractor(file_path, source_name)
    except ImportError as exc:
        # Heavy optional deps (pdfplumber, docling) may not be installed.
        # Fall back to pypdf so ingestion never hard-fails.
        if name != "pypdf":
            print(f"[WARN] '{name}' unavailable ({exc}); falling back to pypdf.")
            return _extract_pypdf(file_path, source_name)
        raise
