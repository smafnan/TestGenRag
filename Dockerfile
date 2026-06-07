# ── Single stage: Python backend that serves the HTML frontend ─────────────────
FROM python:3.12-slim
WORKDIR /app/backend

# System deps occasionally needed by faiss / psycopg wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app

# Place the HTML design as the frontend index, served by FastAPI at /app.
RUN mkdir -p /app/frontend/build
# JSON (exec) form is required: the source path contains spaces, which the
# shell form mis-parses on some BuildKit versions (e.g. Hugging Face Spaces).
COPY ["TestGenRag Design/TestGen RAG Redesign.html", "/app/frontend/build/index.html"]

EXPOSE 7860
# $PORT is provided by hosts like Render; default 7860 suits Hugging Face Spaces.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
