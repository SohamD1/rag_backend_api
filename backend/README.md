# RAG Backend API

## Setup
1. Create a virtual environment in `backend/`.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy repo-root `.env.example` to `.env` and fill in:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX` (and `PINECONE_HOST` if needed)

## Run
From `backend/`:

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

## API
- `GET /api/health`
- `GET /api/health/deps` and `GET /api/health/deps?deep=true`
- `POST /api/v1/documents` (multipart/form-data: `file`)
- `GET /api/v1/documents`
- `GET /api/v1/documents/{doc_id}/file`
- `DELETE /api/v1/documents/{doc_id}`
- `POST /api/v1/chat` (JSON: `{ "query": "...", "debug": false }`)
- `POST /api/v1/chat/stream` (SSE)

