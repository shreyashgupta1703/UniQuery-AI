# UniQuery AI

## Live Demo

GitHub Pages demo is deployed automatically from the main branch.

**Fully offline, citation-backed document Q&A with local LLMs — streamed live, runs even with nothing but Python installed.**

[![CI](https://github.com/xj16/localrag/actions/workflows/ci.yml/badge.svg)](https://github.com/xj16/localrag/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-in--browser-5b9dff)](https://xj16.github.io/localrag/)
[![coverage](https://img.shields.io/badge/coverage-89%25-4ade80)](#testing--evaluation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)](backend/pyproject.toml)

UniQuery AI is a self-hostable retrieval-augmented generation (RAG) assistant for university academic and student-service documents.
Point it at your PDFs and Markdown notes and ask questions in plain language.
It retrieves the most relevant passages, answers with **inline citations**
streamed token-by-token, and runs **100% locally** — no API keys, no cloud, no
data leaving your machine.

It is built to **degrade gracefully at every layer**, so it *always works*:

| Layer | Preferred (if available) | Zero-dependency fallback |
| --- | --- | --- |
| Embeddings | `sentence-transformers` (Hugging Face) | Deterministic **hashing vectorizer** (pure stdlib) |
| Generation | Local **Ollama** LLM (`llama3.2`, `qwen2.5`, …) | **Extractive** answer built from the retrieved passages |
| Search | **numpy** matmul over a cached matrix | Pure-Python cosine scan |
| Documents | PDF via `pypdf` | Markdown / plain text (always) |
| Vector store | — | **SQLite** (always) |

You get a genuinely useful, citation-backed answer even on a locked-down machine
with nothing but Python installed — and a much richer experience once you add a
local embedding model, numpy, and an Ollama server.

> **▶ Try it in your browser — [xj16.github.io/localrag](https://xj16.github.io/localrag/).**
> The full retrieval pipeline (embed → retrieve → rerank → cite) runs entirely
> client-side, with no backend and no network.

---

## Why

Most "chat with your docs" tools require an OpenAI key, ship your documents to a
third party, or collapse into an unusable error the moment a dependency or model
is missing. localrag is the opposite: **offline-first, dependency-optional, and
always answerable.** The hashing-vectorizer + extractive-answer path makes the
whole pipeline hermetic and testable with zero network — which is exactly how the
test suite (and the in-browser demo) run.

---

## Features

- **Streamed, cited answers.** Answers arrive token-by-token over SSE with a live
  cursor; every response carries numbered `[1]`, `[2]` markers mapping to the
  exact source passages, with similarity scores and snippets.
- **Honest citations.** When a local LLM answers, only the passages it actually
  cited are shown (renumbered to match) — the Sources panel never lists a passage
  the answer didn't use.
- **Grounding signal.** Answers are flagged *low-confidence* when no retrieved
  passage clears a similarity threshold, so you know when the docs don't cover the
  question.
- **Hybrid retrieval + reranking.** Vector search over-fetches candidates and
  blends in lexical term overlap to sharpen precision on keyword-heavy questions.
- **Fast at scale.** With numpy, search is a single cached matmul — see the
  [benchmark](#performance) (hundreds-to-thousands× faster than the pure-Python
  scan). Without numpy, it still works.
- **Measured quality.** A retrieval eval harness reports Recall@k / MRR over a
  labeled QA set, and CI gates on a minimum score.
- **Ingest** PDFs, Markdown, and plain text (upload, paste, CLI, or API) with
  paragraph-aware chunking and overlap.
- **Rich chat UI:** Markdown-rendered answers, clickable citations that scroll to
  and flash their source, a per-source filter, a light/dark toggle, and
  export-answer-as-Markdown.
- **Self-hostable:** one-command `docker compose up` (optionally with Ollama),
  configurable CORS, size-capped uploads, optional bearer auth, and a rate limit.
- **CLI** for ingest / ask / stats / serve.
- **Hermetic tests** — the full pipeline is covered by pytest with no network.

---

## Architecture

```
                ┌──────────────────────────────────────────────┐
  documents ──▶ │  chunking  ──▶  embeddings  ──▶  SQLite store │
  (md/pdf/txt)  └──────────────────────────────────────────────┘
                                                    │
  question ─────────────▶  embed query  ──▶  cosine search (top-k·3)
                                                    │
                                          hybrid lexical rerank ─▶ top-k
                                                    │
                                     ┌──────────────┴───────────────┐
                                     ▼                              ▼
                            Ollama (streamed)            extractive answer
                                     └──────────────┬───────────────┘
                                                    ▼
                              answer + honest citations + grounding flag
                                          (streamed over SSE)
```

- `backend/app/chunking.py` — document loading + paragraph-aware chunking
- `backend/app/embeddings.py` — sentence-transformers with hashing fallback
- `backend/app/store.py` — SQLite vector store, numpy-vectorized cosine search
- `backend/app/generation.py` — streaming Ollama client, honest citations,
  reranking, extractive answerer
- `backend/app/engine.py` — orchestrates the full RAG pipeline
- `backend/app/main.py` — FastAPI app (SSE streaming, hardened serving, static UI)
- `backend/app/security.py` — optional auth + rate-limit middleware
- `backend/app/cli.py` — command-line interface
- `backend/eval/` — labeled retrieval eval harness (Recall@k / MRR)
- `frontend/` — React + Vite + TypeScript chat UI
- `frontend/src/demo/` — self-contained in-browser port of the pipeline (the demo)

---

## Quick start

### Option A — Docker (one command)

```bash
docker compose up --build          # http://localhost:8000, sample docs pre-loaded
docker compose --profile ollama up # also start a local Ollama for LLM answers
# then, once: docker compose exec ollama ollama pull llama3.2
```

### Option B — local Python + Node

**1. Backend**

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate   •   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# (optional) real semantic embeddings, PDF support, and fast numpy search
pip install -r requirements-optional.txt

uvicorn app.main:app --reload      # serves the API (and built UI) on :8000
```

Ingest and ask from the CLI:

```bash
python -m app.cli ingest ../sample_docs/*.md
python -m app.cli ask "How efficient are commercial solar panels?"
python -m app.cli stats
```

**2. Frontend (dev mode with hot reload)**

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api to :8000
```

For production, `npm run build` emits `frontend/dist`, which the FastAPI app
serves automatically at `/`. The same build also produces `dist/demo.html`, the
standalone in-browser demo.

**3. (Optional) Local LLM via Ollama**

```bash
ollama pull llama3.2
ollama serve            # localrag auto-detects it at http://localhost:11434
```

With Ollama running, answers are generated by the model and streamed live;
without it, localrag returns a grounded extractive answer. Either way you get
citations.

---

## Configuration

All settings are environment variables (sensible defaults, zero config needed):

| Variable | Default | Description |
| --- | --- | --- |
| `LOCALRAG_DB` | `localrag.db` | SQLite path (`:memory:` for ephemeral) |
| `LOCALRAG_EMBED_MODEL` | `…all-MiniLM-L6-v2` | HF embedding model |
| `LOCALRAG_HASH_DIM` | `512` | Fallback hashing-vectorizer dimension |
| `LOCALRAG_CHUNK_SIZE` | `900` | Chunk size (characters) |
| `LOCALRAG_CHUNK_OVERLAP` | `150` | Overlap between chunks (characters) |
| `LOCALRAG_TOP_K` | `4` | Passages retrieved per question |
| `LOCALRAG_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `LOCALRAG_OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `LOCALRAG_DISABLE_OLLAMA` | `false` | Force the extractive fallback |
| `LOCALRAG_CORS_ORIGINS` | `localhost:5173` | Comma-separated CORS allowlist (`*` to open) |
| `LOCALRAG_MAX_UPLOAD_BYTES` | `10485760` | Max upload size (10 MiB) |
| `LOCALRAG_ALLOWED_EXTENSIONS` | `.md,.markdown,.txt,.pdf` | Accepted upload types |
| `LOCALRAG_API_TOKEN` | *(unset)* | If set, `/api/*` (except health) needs `Bearer <token>` |
| `LOCALRAG_RATE_LIMIT` | `120` | Requests/IP per window (`0` disables) |
| `LOCALRAG_RATE_WINDOW` | `60` | Rate-limit window (seconds) |

---

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness check (always public) |
| `GET` | `/api/stats` | Embedder, chunk count, sources, Ollama status |
| `POST` | `/api/ingest/text` | Ingest raw text `{text, source}` |
| `POST` | `/api/ingest/file` | Ingest an uploaded file (md/txt/pdf) |
| `POST` | `/api/ask` | Ask a question `{question, top_k?, source?}` → cited answer |
| `POST` | `/api/ask/stream` | Same, streamed as Server-Sent Events |
| `DELETE` | `/api/sources/{name}` | Remove a source and its chunks |

```bash
curl -N -X POST localhost:8000/api/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "How efficient are solar panels?"}'
```

```
data: {"type":"meta","mode":"extractive","model":"","grounded":true}
data: {"type":"token","text":"Commercial "}
data: {"type":"token","text":"silicon "}
…
data: {"type":"citations","citations":[{"marker":1,"source":"solar_power.md", … }]}
data: {"type":"done"}
```

---

## Performance

`scripts/benchmark_search.py` times search vs. corpus size for both paths.
Representative numbers (hashing vectorizer, dim 512, top-k 4):

| Corpus (chunks) | numpy | pure-Python | speedup |
| --- | --- | --- | --- |
| 500 | 0.07 ms | 56 ms | ~860× |
| 2 000 | 0.12 ms | 207 ms | ~1 700× |
| 8 000 | 0.22 ms | 826 ms | ~3 800× |

```bash
cd backend && python -m scripts.benchmark_search
```

---

## Testing & evaluation

```bash
cd backend
pip install -r requirements-dev.txt python-multipart
coverage run -m pytest && coverage report      # ~89% with numpy installed
```

Tests use the deterministic hashing embedder and an in-memory SQLite DB, so they
never touch the network or download a model. The flagship subsystems —
streaming, honest citation parsing, grounding, the numpy/pure-Python search
equivalence, and the serving-hardening layer — are all covered. The Ollama path
is exercised with a mocked server.

**Retrieval quality is measured, not asserted.** The eval harness runs a labeled
QA set and reports Recall@k / MRR; CI gates on a minimum score:

```bash
cd backend && python -m eval.harness
```

CI runs the suite on Python 3.11 and 3.12 (pure-Python and numpy paths), builds
the frontend, and enforces a coverage floor.

---

## Tech stack

**Backend:** Python, FastAPI, Pydantic, SQLite, numpy (optional),
sentence-transformers / Hugging Face (optional), Ollama (optional), pypdf
(optional).
**Frontend:** React, TypeScript, Vite (zero runtime dependencies beyond React).
**CI/CD:** GitHub Actions (tests + coverage gate + eval + Pages demo deploy).

> The concepts here (chunk → embed → retrieve → generate with citations) are the
> same building blocks frameworks like **LangChain** expose; localrag implements
> a focused, dependency-light version that runs fully offline. LangChain's
> `RecursiveCharacterTextSplitter` is used automatically when installed.

---

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 xj16.
