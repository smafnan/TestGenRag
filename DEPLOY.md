# Deploying TestGenRAG online

The goal: a single public URL someone else can open and try.

**The catch:** the default local model (Ollama) needs a machine with the Ollama
server running — it can't run on free hosting. So online deploys use a **free
hosted model via NVIDIA NIM** instead. Everything else (FAISS, embeddings,
SQLite, the UI) ships inside one Docker image.

---

## 0. Get a free NVIDIA NIM API key

1. Go to <https://build.nvidia.com> and sign in.
2. Open any model (e.g. **DeepSeek-V3** or **Llama 3.3**).
3. Click **Get API Key** → copy the key (starts with `nvapi-`).

This key works with the OpenAI-compatible endpoint the app already uses
(`LLM_PROVIDER=nvidia_deepseek`).

---

## Option A — Hugging Face Spaces (recommended)

Free CPU Spaces give **16 GB RAM**, which comfortably runs the local
sentence-transformers embeddings, and you get one public URL. The repo's root
`Dockerfile` already builds the UI and serves it with the API.

1. **Create the Space**
   - <https://huggingface.co/new-space> → name it → **SDK: Docker** → **Blank** → Create

2. **Push the code** (from the project root):
   ```bash
   git init
   git add .
   git commit -m "TestGenRAG"
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
   (Or use the Space's web uploader / "Files" tab.)

3. **Tell the Space which port to expose.** Add this header to the top of
   `README.md` *in the Space* (Hugging Face reads YAML front-matter):
   ```yaml
   ---
   title: TestGenRAG
   sdk: docker
   app_port: 7860
   ---
   ```
   (The container already listens on 7860 by default.)

4. **Set secrets:** Space → **Settings → Variables and secrets**
   - Variable `LLM_PROVIDER` = `nvidia_deepseek`
   - Variable `NVIDIA_MODEL` = `deepseek-ai/deepseek-v3`
   - Variable `PDF_EXTRACTOR` = `pdfplumber`
   - **Secret** `NVIDIA_API_KEY` = `nvapi-…`

5. The Space builds and goes live at
   `https://<your-username>-<space-name>.hf.space`. The UI is at **`/app`**
   (the root redirects there). Share that link.

> Note: a Space's disk resets on rebuild, so the FAISS index and SQLite file
> are not permanent — fine for a demo (testers just upload a PDF first). To make
> approvals durable, point `DATABASE_URL` at a hosted Postgres (e.g. Neon free tier).

---

## Option B — Render (Blueprint)

The repo includes `render.yaml`.

1. Push the repo to GitHub.
2. <https://dashboard.render.com/blueprints> → **New Blueprint Instance** →
   pick the repo. Render reads `render.yaml`.
3. When prompted, set the `NVIDIA_API_KEY` secret.
4. Deploy. Your URL is `https://testgenrag.onrender.com` (UI at `/app`).

> Render's **free** web service has **512 MB RAM**, which is tight for local
> sentence-transformers. Options: use a paid instance, or set
> `EMBEDDINGS_PROVIDER=openai` with an `OPENAI_API_KEY`. The free instance also
> sleeps after ~15 min idle (first request then takes ~50 s to wake).

---

## Option C — Split deploy (most scalable)

Host the API and UI separately:

- **API** → Render / Fly.io / a small VM. Note its URL.
- **UI** → build the static frontend and host on Vercel or Netlify, with the
  env var `VITE_API_BASE=https://your-api-url` set at build time. Because CORS
  is open by default, the browser can call the API cross-origin.

```bash
cd frontend
VITE_API_BASE=https://your-api-url npm run build   # outputs to frontend/build
```

---

## Local Docker (smoke-test the image before deploying)

```bash
# from the project root
docker build -t testgenrag .
docker run -p 7860:7860 \
  -e LLM_PROVIDER=nvidia_deepseek \
  -e NVIDIA_API_KEY=nvapi-... \
  testgenrag
# open http://localhost:7860/app
```

Or with compose: `docker compose up --build` (then open `/app`). Uncomment the
`redis` / `postgres` services in `docker-compose.yml` to exercise those too.

---

## Sanity checklist

- `GET /health` returns `"status": "ok"` and shows your provider/backends.
- The `/app` page loads, a PDF uploads (chunk count appears), and **Generate
  drafts** returns structured, cited cases.
- **Approve & e-sign** succeeds (persisted; visible at `GET /approved`).
