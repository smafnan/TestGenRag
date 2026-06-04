# ── Stage 1: build the SvelteKit frontend as a static SPA ─────────────────────
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
# Mount the UI under /app so it shares one origin with the API.
ENV BASE_PATH=/app
RUN npm run build

# ── Stage 2: Python backend that also serves the built frontend ───────────────
FROM python:3.12-slim
WORKDIR /app/backend

# System deps occasionally needed by faiss / psycopg wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
# Bring in the compiled frontend so FastAPI can serve it at /app.
COPY --from=frontend /fe/build /app/frontend/build

EXPOSE 7860
# $PORT is provided by hosts like Render; default 7860 suits Hugging Face Spaces.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
