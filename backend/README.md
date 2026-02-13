# RAG Backend API

## Setup
1. Create a virtual environment in `backend/`.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy repo-root `.env.example` to `.env` and fill in:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX` (and `PINECONE_HOST` if needed)
4. Optional: set `RAG_GENERATE_ANSWERS_ENABLED=false` to run retrieval-only mode
   (`/api/v1/chat` returns selected chunks without LLM-generated answer text).

## Run
From `backend/`:

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

## API
- `GET /api/health` (liveness; always returns `{"status":"ok"}` if the app is running)
- `GET /api/health/deps` (readiness/config; checks whether required env vars are present)
- `GET /api/health/deps?deep=true` (readiness/deep; actively calls OpenAI + Pinecone)
- `POST /api/v1/documents` (multipart/form-data: `file`)
- `GET /api/v1/documents`
- `GET /api/v1/documents/{doc_id}/file`
- `DELETE /api/v1/documents/{doc_id}`
- `POST /api/v1/chat` (JSON: `{ "query": "...", "debug": false }`)
- `POST /api/v1/chat/stream` (SSE)

When retrieval-only mode is enabled (`RAG_GENERATE_ANSWERS_ENABLED=false`), chat responses
include a `chunks` array with selected chunk text/metadata and skip final answer generation.
