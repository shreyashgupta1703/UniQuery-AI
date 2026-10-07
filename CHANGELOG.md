# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-07

A substantial quality pass turning localrag from a polished demo into a
showcase-grade, self-hostable RAG app. No breaking changes to existing
endpoints; all additions are backward compatible and the zero-config local
experience is unchanged.

### Added

- **End-to-end token streaming.** New `POST /api/ask/stream` Server-Sent Events
  endpoint emits `meta` → `token…` → `citations` → `done` frames. The chat UI
  renders answers live with a blinking cursor for both the Ollama path (real
  streamed tokens) and the extractive path (word-streamed), replacing the static
  "Thinking…" state.
- **Honest citations.** For the Ollama path, only the passages the model
  actually referenced via `[n]` markers are returned, renumbered to match; markers
  pointing outside the retrieved set are dropped. No more Sources listing
  passages the answer never used.
- **Grounding signal.** Every answer reports a `grounded` flag (true when a
  retrieved chunk clears a confidence threshold). The UI shows a
  "low confidence" badge when the corpus doesn't really cover the question.
- **Hybrid lexical reranking.** Retrieval over-fetches `top_k * 3` candidates and
  blends vector score with query-term overlap before selecting the final
  `top_k`, improving precision on keyword-heavy questions.
- **Vectorized search.** With `numpy` installed, all embeddings load into one
  cached `float32` matrix and every query is a single matmul —
  hundreds-to-thousands of times faster than the per-row Python scan on large
  corpora (`scripts/benchmark_search.py`). Falls back to pure Python without
  numpy, so nothing is required.
- **Retrieval eval harness** (`backend/eval/`): a hand-labeled QA set with
  Recall@k / MRR / hit-rate, a printed comparison table, and a CI test gating a
  minimum score.
- **Serving hardening:** configurable CORS allowlist, size-capped streamed file
  uploads, server-side extension validation, an optional bearer-token auth hook,
  and a fixed-window per-IP rate limit — all off/permissive by default.
- **In-browser live demo** (`frontend/demo.html`): a faithful TypeScript port of
  the retrieval pipeline (hashing embedder, cosine, chunking, reranking, honest
  citations, extractive answers) that runs entirely client-side with no backend,
  no network, and no WASM runtime. Deployed to GitHub Pages via a workflow.
- **UI upgrades:** dependency-free Markdown rendering of answers, clickable `[n]`
  markers that scroll to and flash their source card, a per-source filter
  dropdown (wiring the already-supported `source=` param), a light/dark theme
  toggle, and an "export answer + citations as Markdown" button.
- **Docker:** a multi-stage `Dockerfile` (builds the frontend, serves it from
  FastAPI) and `docker-compose.yml` with an optional `ollama` profile and a seed
  step, so `docker compose up` lands on a working, pre-populated UI.
- **Coverage:** CI now runs the suite under `coverage` on the pure-Python path
  and again with numpy, gating on `--fail-under=85` (currently ~89%). Coverage
  and CI badges added to the README.

### Changed

- `AskResponse` gained a `grounded` boolean (additive; existing clients ignore it).
- CORS default tightened from `*` to the Vite dev origin (override with
  `LOCALRAG_CORS_ORIGINS`).
- Embeddings are stored and read the same way, but search now goes through the
  numpy matrix cache when available.

### Fixed

- Removed the dead `reload=bool(...) is False and False` expression in
  `main.main()` (it always evaluated to `False`).
- File uploads no longer read the entire payload into memory; they stream to a
  temp file under a configurable size cap.

## [0.1.0] - 2026

Initial release: offline-first RAG engine (FastAPI + React/TS + CLI) with
graceful degradation at every layer — sentence-transformers with a pure-stdlib
hashing-vectorizer fallback, Ollama generation with an extractive fallback, PDF
via pypdf with Markdown/text always, and a SQLite vector store. Hermetic,
network-free test suite in CI.

[0.2.0]: https://github.com/xj16/localrag/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/xj16/localrag/releases/tag/v0.1.0
