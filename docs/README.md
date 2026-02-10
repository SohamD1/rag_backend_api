PDFs (Optional)

This folder is a convenient place to keep PDFs during development.

To ingest a PDF, upload it via the API (admin token required when `KB_REQUIRE_AUTH=true`):

- `POST /api/v1/documents` (multipart/form-data: `file`, `source_url`)
