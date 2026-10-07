# syntax=docker/dockerfile:1

# --------------------------------------------------------------------- frontend
# Build the React/Vite SPA (and the standalone in-browser demo) into static
# assets that FastAPI serves at "/".
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --------------------------------------------------------------------- backend
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LOCALRAG_DB=/data/localrag.db

WORKDIR /app

# Install the core runtime plus numpy (vectorized search) and pypdf (PDF
# ingest). sentence-transformers is intentionally left out to keep the image
# small and fully offline; the hashing embedder is the default in-container.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt numpy>=1.24 pypdf>=4.0

# App source.
COPY backend/ backend/
COPY sample_docs/ sample_docs/

# Built frontend assets where main.py looks for them (frontend/dist).
COPY --from=frontend /app/frontend/dist/ frontend/dist/

# Persisted SQLite store lives on a volume.
RUN mkdir -p /data
VOLUME ["/data"]

WORKDIR /app/backend
EXPOSE 8000

# Seed the sample corpus on first boot (idempotent), then serve. Override with
# your own command to skip seeding.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
