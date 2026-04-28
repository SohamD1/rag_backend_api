# Agent Guidelines

Behavioral rules to reduce common LLM coding mistakes. Bias toward caution over speed; use judgment for trivial tasks.

## 1. Think Before Coding
**Do not assume. Do not hide confusion. Surface tradeoffs.**

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of silently choosing.
- If a simpler approach exists, say so. Push back when warranted.
- If unclear, stop, name what is confusing, and ask.

## 2. Simplicity First
**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes
**Touch only what you must. Clean up only your own mess.**

When editing:
- Do not improve adjacent code, comments, or formatting unless required.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it instead of deleting it.

When your changes create orphans, remove them. Do not remove pre-existing dead code unless asked. Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution
**Define success criteria. Loop until verified.**

Turn tasks into verifiable goals:
- "Add validation" -> write invalid-input tests, then make them pass.
- "Fix the bug" -> write a reproducing test, then make it pass.
- "Refactor X" -> ensure tests pass before and after.

For multi-step tasks, state a brief plan:
```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

These rules are working if diffs are smaller, rewrites are rarer, and clarifying questions happen before implementation.

# Repository Guidelines

## Structure
The app lives in `backend/`: routers in `app/api`, orchestration/retrieval in `app/core`, provider integrations in `app/adapters`, storage in `app/storage`, services in `app/services`, and tests in `tests`. Keep `README.md` canonical, `scripts/ingest_folder.py` for bulk PDF ingest, and `data/` as ignored local storage.

## Commands
Run from `backend/` unless noted:
- `python -m venv .venv` then `pip install -r requirements.txt`
- `python -m uvicorn app.main:app --reload --port 8000`
- `python -m pytest`
- `python -m pytest tests/test_retrieval_pipeline.py`
- From repo root: `python scripts/ingest_folder.py --docs-dir "C:\path\to\pdfs" --sources-json "C:\path\to\sources.json"`

## Style And Tests
Match existing Python style: 4-space indentation, practical type hints, small modules, tidy imports, `snake_case` for functions/files, and `PascalCase` for classes. Keep FastAPI v1 routers under `backend/app/api/v1`.

Use pytest. Add focused `test_*` coverage in `backend/tests` for changed retrieval, indexing, ingest, or API behavior. Note any test gaps in the PR.

## PRs, Docs, And Security
Use scoped, imperative commits like `backend: improve first-pass doc routing`. PRs need a summary, test evidence, and callouts for `.env`, index/schema, Pinecone layout, or API request/response changes.

Keep durable setup, operation, and security guidance in `README.md`. Do not commit one-off reports or stale planning docs.

Never commit secrets. Copy `.env.example` to `.env`; PDF ingest requires `MISTRAL_API_KEY`; non-dev envs should use `KB_APP_TOKEN` and `KB_ADMIN_TOKEN`.

Treat Pinecone as a clean corpus: assume no published corpus, live backfill, or legacy vector compatibility requirement. Prefer clean reingest over migrations unless explicitly asked.
