"""
Retrieval utilities: query selection and re-ranking on top of FAISS.

This sits between the raw vector store and the agent. Two techniques:

1. Query selection - given a HyDE document and the original requirement, build
   several candidate queries (the hypothetical spec, the bare requirement, and
   keyword distillations) and union their hits. This widens recall beyond a
   single embedding lookup.

2. Re-ranking - re-order the unioned candidates with one of:
       mmr        - Maximal Marginal Relevance (diversity vs relevance)
       page_type  - boost spec-like pages (table/mixed) via metadata
       none       - keep FAISS similarity order

Configure with RERANK_METHOD in .env. Everything degrades gracefully.
"""

import os
import re
from typing import List

from langchain_core.documents import Document

_STOPWORDS = {
    "the", "a", "an", "must", "should", "shall", "when", "if", "is", "are",
    "to", "of", "and", "or", "that", "this", "it", "be", "system", "device",
}


def build_queries(requirement: str, hyde_doc: str) -> List[str]:
    """Query selection: produce a small, de-duplicated set of search queries."""
    queries = [hyde_doc.strip(), requirement.strip()]

    # Keyword distillation from the requirement.
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", requirement.lower())
    keywords = [w for w in words if w not in _STOPWORDS]
    if keywords:
        queries.append(" ".join(keywords[:8]))

    seen, unique = set(), []
    for q in queries:
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


def _dedupe(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:80])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _rerank_page_type(docs: List[Document]) -> List[Document]:
    """Boost spec-like pages (tables / mixed) which usually hold testable detail."""
    weight = {"table": 0, "mixed": 1, "text": 2, "empty": 3}
    return sorted(docs, key=lambda d: weight.get(d.metadata.get("page_type", "text"), 2))


def retrieve(vectorstore, requirement: str, hyde_doc: str, k: int = 4) -> List[Document]:
    """
    Run query selection + retrieval + re-ranking and return the top-k Documents.
    """
    method = os.getenv("RERANK_METHOD", "mmr").lower()
    queries = build_queries(requirement, hyde_doc)

    # Gather a candidate pool across all selected queries.
    pool: List[Document] = []
    per_query = max(k, 4)
    for q in queries:
        if method == "mmr" and hasattr(vectorstore, "max_marginal_relevance_search"):
            try:
                hits = vectorstore.max_marginal_relevance_search(
                    q, k=per_query, fetch_k=per_query * 3
                )
            except Exception:
                hits = vectorstore.similarity_search(q, k=per_query)
        else:
            hits = vectorstore.similarity_search(q, k=per_query)
        pool.extend(hits)

    pool = _dedupe(pool)

    if method == "page_type":
        pool = _rerank_page_type(pool)
    # For 'mmr' the per-query MMR already diversified; for 'none' keep order.

    return pool[:k]


def format_context(docs: List[Document]) -> str:
    """Render retrieved chunks with source + page tags for traceability."""
    return "\n\n---\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')} "
        f"p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )
