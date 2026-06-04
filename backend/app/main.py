"""
TestGenRAG - FastAPI backend.

An AI assistant that drafts traceable, citation-backed test cases from your
technical documentation using a RAG + LangGraph agentic pipeline, with a
human-in-the-loop approve / e-sign step before anything is persisted.

Endpoints:
    GET  /health    - liveness + active providers/backends
    GET  /documents - list ingested documents
    POST /ingest    - upload a PDF, index it in FAISS, optionally store in S3
    POST /generate  - run the agent and return structured, traceable drafts
    POST /approve   - persist a human-approved, e-signed test case
    GET  /approved  - list persisted approved test cases
"""

import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import analysis, cache, database
from .agent import build_agent
from .auth import auth_enabled, require_user
from .aws import upload_to_s3
from .ingestion import ingest_pdf, index_exists, load_vectorstore
from .llm import current_provider, get_llm
from .retrieval import build_queries
from .schemas import ApproveRequest

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()          # create tables on startup
    yield


app = FastAPI(
    title="TestGenRAG",
    description=(
        "AI-powered, citation-backed test-case drafting for regulated and "
        "safety-critical software via RAG + a LangGraph agentic pipeline, with "
        "human review and e-sign."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory registry of what was ingested this session (lightweight; the
# durable store is the approved-test-case database).
_INGESTED_DOCS: List[dict] = []


# ─── System ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health():
    return {
        "status": "ok",
        "index_ready": index_exists(),
        "llm_provider": current_provider(),
        "embeddings_provider": os.getenv("EMBEDDINGS_PROVIDER", "huggingface"),
        "pdf_extractor": os.getenv("PDF_EXTRACTOR", "pypdf"),
        "rerank_method": os.getenv("RERANK_METHOD", "mmr"),
        "cache_backend": cache.backend_name(),
        "db_backend": database.backend_name(),
        "auth_enabled": auth_enabled(),
    }


# ─── Ingestion ──────────────────────────────────────────────────────────────

@app.get("/documents", tags=["ingestion"])
def list_documents():
    return _INGESTED_DOCS


@app.post("/ingest", tags=["ingestion"])
async def ingest(file: UploadFile = File(...), user: dict = Depends(require_user)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        s3_uri = upload_to_s3(tmp_path, f"uploads/{file.filename}")
        n_chunks = ingest_pdf(tmp_path, file.filename)
    finally:
        os.remove(tmp_path)

    record = {"name": file.filename, "chunks": n_chunks, "s3": s3_uri}
    _INGESTED_DOCS.append(record)
    return record


# ─── Generation ─────────────────────────────────────────────────────────────

def _llm_error_to_http(exc: Exception, shown_model: str) -> HTTPException:
    """Map provider/SDK exceptions to clean, user-facing HTTP errors."""
    msg = str(exc)
    low = msg.lower()
    if "404" in msg or "not found" in low or "does not exist" in low or "unknown model" in low:
        return HTTPException(
            status_code=400,
            detail=(
                f"Model '{shown_model}' is not available on your NVIDIA account or "
                "the free endpoint. Pick a Llama or Mistral model from the dropdown."
            ),
        )
    if any(t in msg for t in ("401", "403")) or "unauthorized" in low or "invalid api key" in low:
        return HTTPException(status_code=401, detail="Your NVIDIA API key was rejected. Check the key and try again.")
    if "429" in msg or "rate limit" in low or "too many requests" in low:
        return HTTPException(status_code=429, detail="The model is rate limited right now. Wait a moment and try again.")
    return HTTPException(status_code=502, detail=f"Request failed: {msg[:300]}")


@app.post("/generate", tags=["generation"])
def generate(
    req: dict,
    x_api_key: Optional[str] = Header(default=None),
    user: dict = Depends(require_user),
):
    """
    Generate test cases for a requirement.

    x_api_key (sent as the HTTP header X-Api-Key) is the user's own LLM API key.
    It overrides any server-side key for this request only and is never stored.
    """
    requirement = (req or {}).get("requirement", "")
    user_model = (req or {}).get("model", "").strip() or None
    if not requirement.strip():
        raise HTTPException(status_code=400, detail="Requirement cannot be empty.")

    if not index_exists():
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a PDF via POST /ingest first.",
        )

    provider = current_provider()
    cached = cache.get_cached(provider, requirement)
    if cached:
        return {**cached, "cached": True}

    vectorstore = load_vectorstore()
    try:
        llm = get_llm(api_key=x_api_key, model=user_model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    agent = build_agent(vectorstore, llm)
    t0 = time.time()
    try:
        result = agent.invoke({"requirement": requirement, "attempts": 0})
    except Exception as exc:  # noqa: BLE001 - map provider errors to clean messages
        raise _llm_error_to_http(exc, user_model or os.getenv("NVIDIA_MODEL", "the selected model"))
    elapsed_ms = int((time.time() - t0) * 1000)

    # Pipeline trace: makes the agent's work visible in the UI.
    retrieved = result.get("retrieved", []) or []
    hyde_doc = result.get("hyde_doc", "")
    trace = {
        "hyde": hyde_doc,
        "queries": build_queries(requirement, hyde_doc) if hyde_doc else [requirement],
        "chunks": [
            {
                "source": d.metadata.get("source", "?"),
                "page": d.metadata.get("page", "?"),
                "page_type": d.metadata.get("page_type", "text"),
                "snippet": (d.page_content or "")[:240].strip(),
            }
            for d in retrieved
        ],
        "rerank_method": os.getenv("RERANK_METHOD", "mmr"),
        "extractor": os.getenv("PDF_EXTRACTOR", "pypdf"),
        "model": user_model or os.getenv("NVIDIA_MODEL", provider),
        "elapsed_ms": elapsed_ms,
    }

    payload = {
        "requirement": requirement,
        "test_cases": result.get("test_cases", []),
        "grounded": result.get("grounded", False),
        "attempts": result.get("attempts", 0),
        "trace": trace,
    }
    cache.set_cached(provider, requirement, payload)
    return {**payload, "cached": False}


# ─── Analysis / execution ─────────────────────────────────────────────────────

@app.post("/analyze", tags=["analysis"])
def analyze(
    req: dict,
    x_api_key: Optional[str] = Header(default=None),
    user: dict = Depends(require_user),
):
    """
    Analyse the indexed document against the user's request.

    A relevance gate runs first: if the document does not actually contain the
    kind of information the request needs (for example a menu uploaded for a
    health assessment), the endpoint returns relevant=False with the detected
    document topic and a reason, and produces no analysis. Only when the document
    matches does it return a grounded, citation-backed report for human approval.
    """
    question = (req or {}).get("requirement", "")
    user_model = (req or {}).get("model", "").strip() or None
    if not question.strip():
        raise HTTPException(status_code=400, detail="Request cannot be empty.")

    if not index_exists():
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a PDF via POST /ingest first.",
        )

    try:
        llm = get_llm(api_key=x_api_key, model=user_model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    vectorstore = load_vectorstore()
    try:
        return analysis.analyze(
            vectorstore, llm, question,
            model_name=user_model or os.getenv("NVIDIA_MODEL", current_provider()),
        )
    except Exception as exc:  # noqa: BLE001 - map provider errors to clean messages
        raise _llm_error_to_http(exc, user_model or os.getenv("NVIDIA_MODEL", "the selected model"))


# ─── Approval / persistence ───────────────────────────────────────────────────

@app.post("/approve", tags=["approval"])
def approve(req: ApproveRequest, user: dict = Depends(require_user)):
    signed_by = req.signed_by
    if isinstance(user, dict) and user.get("auth") != "disabled":
        signed_by = user.get("email") or user.get("sub") or signed_by
    saved = database.save_approved(
        requirement=req.requirement,
        test_case=req.test_case.model_dump(),
        signed_by=signed_by,
    )
    return {"status": "approved", "record": saved}


@app.get("/approved", tags=["approval"])
def approved(user: dict = Depends(require_user)):
    return database.list_approved()


# ─── Serve the built frontend (single-URL deploys) ────────────────────────────
# When the SvelteKit static build exists at frontend/build, mount it so the API
# and UI share one origin. Harmless during local dev (folder simply absent).

_FRONTEND_BUILD = Path(__file__).resolve().parents[2] / "frontend" / "build"
if _FRONTEND_BUILD.is_dir():
    app.mount(
        "/app",
        StaticFiles(directory=str(_FRONTEND_BUILD), html=True),
        name="frontend",
    )

    @app.get("/", include_in_schema=False)
    def _root_index():
        return FileResponse(str(_FRONTEND_BUILD / "index.html"))
