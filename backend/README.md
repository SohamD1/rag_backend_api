# RAG Backend API

## Setup
1. Create a virtual environment in `backend/`.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy repo-root `.env.example` to `.env` and fill in:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX` (and `PINECONE_HOST` if needed)
   - `MISTRAL_API_KEY` for PDF ingestion
   - Auth for non-development environments:
     - `APP_ENV=production` (or another non-development value)
     - `KB_APP_TOKEN`
     - `KB_ADMIN_TOKEN`
     - Optional override: `KB_REQUIRE_AUTH=false` only if you intentionally want auth disabled
4. Optional: set `RAG_GENERATE_ANSWERS_ENABLED=false` to run retrieval-only mode
   (`/api/v1/chat` returns selected chunks without LLM-generated answer text).

Auth behavior:
- In `APP_ENV=development`/`dev`/`local`/`test`, auth defaults to off unless `KB_REQUIRE_AUTH=true`.
- In any other `APP_ENV`, auth defaults to on.
- When auth is on, the backend now fails startup unless both `KB_APP_TOKEN` and `KB_ADMIN_TOKEN` are set.

Token usage:
- `Authorization: Bearer <KB_APP_TOKEN>` for `/api/v1/chat` and `/api/v1/chat/stream`
- `Authorization: Bearer <KB_ADMIN_TOKEN>` for document admin endpoints

OCR behavior:
- `POST /api/v1/documents` always sends uploaded PDFs through Mistral OCR.
- `MISTRAL_API_KEY` is therefore required for document ingestion.
- The default `INDEX_SCHEMA_VERSION` is now `v5` so reingests pick up the multi-vector-only
  document summary layout.

## Retrieval Strategy

This backend is designed around a two-stage retrieval flow:

1. Ingest each document into its own Pinecone namespace: `{doc_id}::{index_version}`.
2. Also store lightweight document-routing records for that doc in the global doc-summary namespace
   (`__doc_summaries` by default): multi-vector summaries such as `profile`, `headings`,
   and `keywords`.
3. At query time, search the doc-summary namespace first, aggregate hits by `doc_id`, blend vector
   and lexical metadata signals, and pick the most likely document or top few candidate documents.
4. Then search only those selected document namespaces for the final chunk/paragraph retrieval.

The goal is to avoid searching one huge mixed namespace for every query. It keeps document selection
cheap, keeps per-document search tighter, and makes index-version rollouts safer because each doc's
live vectors stay isolated.

A deeper review of the current ingestion pipeline, its weak points, and recommended improvements lives
in [docs/ingestion-pipeline-analysis.md](/C:/Users/sdave/projects/rag_backend_api/docs/ingestion-pipeline-analysis.md).

## Run
From `backend/`:

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

## API
- `GET /api/health` (liveness; always returns `{"status":"ok"}` if the app is running)
- `GET /api/health/deps` (readiness/config; checks whether required env vars are present)
- `POST /api/v1/documents` (multipart/form-data: `file`, `source_url`)
- `GET /api/v1/documents`
- `DELETE /api/v1/documents/{doc_id}`
- `POST /api/v1/chat` (JSON: `{ "query": "...", "debug": false }`)
- `POST /api/v1/chat/stream` (SSE)

When retrieval-only mode is enabled (`RAG_GENERATE_ANSWERS_ENABLED=false`), chat responses
include a `chunks` array with selected chunk text/metadata and skip final answer generation.

Uploaded PDFs are processed transiently during ingest. The backend persists document metadata,
tree artifacts, and vectors, but does not retain the original PDF for later download.
