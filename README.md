# RAG Backend API

FastAPI service for document ingestion, Pinecone-backed retrieval, and chat over a curated knowledge base.

## Repository Layout

```text
.
+-- backend/              # FastAPI application, provider adapters, services, and tests
|   +-- app/api/          # Health, auth, rate limiting, and v1 routers
|   +-- app/adapters/     # OpenAI, Pinecone, and Mistral integrations
|   +-- app/core/         # Retrieval, generation, chunking, indexing, and routing logic
|   +-- app/services/     # Higher-level indexing and benchmark helpers
|   +-- app/storage/      # Local metadata and artifact storage
|   +-- tests/            # Pytest suite
+-- scripts/              # Repo-level operational helpers
+-- data/                 # Local development storage, ignored by git
+-- .env.example          # Non-secret configuration template
+-- AGENTS.md             # Contributor and agent operating guidelines
```

The backend intentionally stays under `backend/` so imports, tests, and deployment commands remain stable.

## Quick Start

From the repo root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
Copy-Item .env.example .env
```

Fill in the required provider values in `.env`:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX`
- `PINECONE_HOST` if your Pinecone setup requires it
- `MISTRAL_API_KEY` for PDF ingestion
- `KB_APP_TOKEN` and `KB_ADMIN_TOKEN` for non-development environments

Run the API:

```powershell
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

Run tests:

```powershell
cd backend
python -m pytest
```

## Configuration And Security

Configuration is loaded from environment variables. Copy `.env.example` to `.env` for local development, but never commit `.env` or secrets.

Auth behavior:

- `APP_ENV=development`, `dev`, `local`, or `test`: auth defaults off unless `KB_REQUIRE_AUTH=true`.
- Any other `APP_ENV`: auth defaults on.
- When auth is on, startup fails unless both `KB_APP_TOKEN` and `KB_ADMIN_TOKEN` are set.

Token usage:

- `Authorization: Bearer <KB_APP_TOKEN>` for `/api/v1/chat` and `/api/v1/chat/stream`
- `Authorization: Bearer <KB_ADMIN_TOKEN>` for document admin endpoints

Operational notes for audit readiness:

- Secrets stay outside git in environment variables.
- `data/` is local runtime storage and is ignored by git.
- Uploaded PDFs are processed transiently; the app persists metadata, tree artifacts, caches, and vectors, not the original upload for download.
- Pinecone is treated as a clean corpus for this repo. If chunking, routing, tree, or vector schema changes, prefer a clean reingest over compatibility migrations unless a task explicitly requires migration support.
- Keep production auth enabled and use separate app/admin tokens.

## Retrieval Strategy

Ingestion and retrieval use a two-stage flow:

1. Each document is indexed into its own Pinecone namespace: `{doc_id}::{index_version}`.
2. Lightweight document-routing records are stored in a global summary namespace, `__doc_summaries` by default.
3. At query time, the service searches document summaries first to select likely documents.
4. Final chunk retrieval runs only against the selected document namespaces.

This avoids scanning one large mixed namespace for every query and keeps index-version rollouts cleaner.

## API Surface

- `GET /api/health`
- `GET /api/health/deps`
- `POST /api/v1/documents` with multipart form fields `file` and `source_url`
- `GET /api/v1/documents`
- `DELETE /api/v1/documents/{doc_id}`
- `POST /api/v1/chat` with JSON `{ "query": "...", "debug": false }`
- `POST /api/v1/chat/stream`

Set `RAG_GENERATE_ANSWERS_ENABLED=false` to use retrieval-only mode. In that mode, chat responses include selected chunks and skip final answer generation.

## Bulk Ingestion

Preview ordered PDF/source matches from the repo root:

```powershell
python scripts/ingest_folder.py --docs-dir "C:\path\to\pdfs" --sources-json "C:\path\to\sources.json"
```

After reviewing the preview, add `--apply` to upload. Add `--yes` to skip the interactive prompt.
