# Repository Guidelines

## Project Structure & Module Organization
The application lives in `backend/`. Core API entrypoints are in `backend/app/api`, request orchestration and retrieval logic live in `backend/app/core`, provider integrations are in `backend/app/adapters`, persistence helpers are in `backend/app/storage`, and higher-level indexing services are in `backend/app/services`. Tests live in `backend/tests`. Operational notes and design writeups are in `docs/`. Use `scripts/ingest_folder.py` for ordered bulk PDF ingestion, and treat `data/` as local development storage.

## Build, Test, and Development Commands
Run from `backend/` unless noted otherwise:

- `python -m venv .venv` then `pip install -r requirements.txt`: create a local environment and install dependencies.
- `python -m uvicorn app.main:app --reload --port 8000`: start the FastAPI server with reload.
- `python -m pytest`: run the full backend test suite.
- `python -m pytest tests/test_retrieval_pipeline.py`: run a focused test module while iterating.
- From the repo root: `python scripts/ingest_folder.py --docs-dir "C:\\path\\to\\pdfs" --sources-json "C:\\path\\to\\sources.json"`: preview bulk ingest matches before using `--apply`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, explicit type hints where practical, and small focused modules. Use `snake_case` for functions, variables, and filenames; `PascalCase` for classes; and keep FastAPI routers grouped by version under `backend/app/api/v1`. There is no repo-wide formatter config checked in, so match the surrounding style and keep imports tidy.

## Testing Guidelines
Pytest is the test framework. Add tests beside the affected behavior in `backend/tests`, using `test_*.py` filenames and `test_*` function names. Prefer focused unit tests for retrieval, indexing, and API edge cases. When changing ingest or vector/index behavior, include coverage for the new branch and note any gaps in the PR.

## Commit & Pull Request Guidelines
Recent commits use scoped, imperative subjects such as `backend: improve first-pass doc routing` and `docs: add setup notes`. Keep that pattern. PRs should include a short summary, test evidence (`python -m pytest` or targeted modules), and call out any `.env` changes, index schema bumps, or Pinecone layout impacts. For API changes, include example request/response snippets instead of screenshots.

## Security, Configuration & Indexing Notes
Copy `.env.example` to `.env` and never commit secrets. `MISTRAL_API_KEY` is required for PDF ingestion, and non-development environments should use `KB_APP_TOKEN` and `KB_ADMIN_TOKEN`. Pinecone is treated as empty for this repo: prefer forward-only ingestion and retrieval changes, and do not add migration, backfill, or dual-write logic unless explicitly requested.

## Agent-Specific Notes
Assume there is no published Pinecone corpus, no live backfill requirement, and no legacy vector data that needs compatibility handling. If retrieval or indexing changes alter stored vectors or doc-summary records, prefer a clean reingest before launch instead of preserving older Pinecone layouts. Only add migration or compatibility paths when the task explicitly asks for them.
