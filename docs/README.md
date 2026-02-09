Local PDF Drop Folder

Put PDFs in this folder (or set `LOCAL_INGEST_DIR` in `.env`), then call:

POST http://localhost:8000/api/v1/documents/ingest-local

This is meant for local development. PDFs are ignored by git via `.gitignore`.

