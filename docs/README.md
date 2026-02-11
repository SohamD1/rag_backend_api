PDFs (Optional)

This folder is a convenient place to keep PDFs during development.

To ingest a PDF, upload it via the API (admin token required when `KB_REQUIRE_AUTH=true`):

- `POST /api/v1/documents` (multipart/form-data: `file`, `source_url`)

Bulk ingest (ordered `sources.json` -> folder of PDFs):

- Preview matches (writes `ingest_matches.preview.json`): `python scripts/ingest_folder.py --docs-dir "C:\\path\\to\\pdfs" --sources-json "C:\\path\\to\\sources.json"`
- Upload after verifying: add `--apply` (and `--yes` to skip the prompt)
