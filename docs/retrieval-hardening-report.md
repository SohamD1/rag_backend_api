# Retrieval Hardening Report

## Goal

Harden the retrieval pipeline so it is safer, more correct, and easier to resume work on in a fresh session.

Primary priorities:

1. stop caching degraded retrieval/results
2. fix tree structural correctness
3. improve context packing behavior
4. add a real second-stage rerank signal
5. capture findings/progress in this file for future sessions

## Baseline Findings

### High

- Partial retrieval failures were silently swallowed and cached as if complete.
- Low-confidence degraded responses could be response-cached without any degraded marker.
- Tree paragraph assignment was page-level, which can mis-attach paragraphs when headings change mid-page.
- The reranker mostly repeated cosine similarity rather than adding a materially different signal.

### Medium

- Context selection stopped at the first oversized item instead of skipping it and trying later evidence.
- Context dedupe was global by normalized text, which could remove corroborating evidence across docs.
- Tree retrieval pruned to a small set of sections/parents and could miss dispersed evidence.
- Response cache invalidation did not include all retrieval-policy knobs.
- File cache implementation was basic and not atomic.

## Planned Changes

### Orchestration / Cache Safety

- Track degraded retrieval states explicitly in `pipeline.py`.
- Do not write retrieval cache entries when any per-doc retrieval/fetch stage partially fails.
- Do not write response cache entries when retrieval was degraded.
- Expand response/retrieval cache keys to include more retrieval-policy settings.

### Tree Structural Correctness

- Reassign paragraph leaves by ordered blocks within a page, not by one page-level heading.
- Update active heading as the scan moves through a page.
- Preserve existing node schema as much as possible.

### Ranking / Context

- Make context packing skip too-large items rather than stopping entirely.
- Reduce over-aggressive cross-doc dedupe.
- Replace cosine-only rerank with a stronger local second-stage heuristic.

## In-Progress Notes

- Retrieval hardening requested after ingestion hardening landed.
- Two parallel subagents launched:
  - tree assignment worker
  - ranking/context worker
- Main rollout owns pipeline/cache safety and final integration.

## Verification Plan

- Add focused source tests for:
  - degraded retrieval not cached
  - degraded response not cached
  - context selection skipping oversized evidence
  - rerank behavior on ambiguous candidates
  - tree mid-page heading transitions

## Resume Guide

If a fresh session needs to resume:

1. Read this file.
2. Check `git status`.
3. Review pending changes in:
   - `backend/app/core/pipeline.py`
   - `backend/app/core/cache.py`
   - `backend/app/core/context.py`
   - `backend/app/core/rerank.py`
   - `backend/app/services/tree_index.py`
   - retrieval-focused tests under `backend/tests/`
4. Run targeted tests first before broader validation.

## Current Session Status

Implemented in this session:

- `backend/app/core/cache.py`
- `backend/app/core/context.py`
- `backend/app/core/pipeline.py`
- `backend/app/core/rerank.py`
- `backend/app/services/tree_index.py`
- `backend/tests/test_context_rerank.py`
- `backend/tests/test_retrieval_pipeline.py`
- `backend/tests/test_tree_index_assignment.py`

Behavior changes shipped:

- Retrieval and response caches no longer write results when retrieval degraded due to partial per-doc retrieval failures or rerank-fetch failures.
- Response and retrieval cache keys now include more retrieval-policy knobs so cache invalidation tracks behavior more honestly.
- The file-backed JSON cache now writes atomically with a temp file plus replace.
- Rerank fetch now degrades safely even if the vector store cannot provide `fetch`, instead of crashing before degradation can be recorded.
- Context packing now skips oversized items and keeps scanning for later evidence instead of stopping at the first overflow.
- Context dedupe is now scoped within a document, so identical passages from different docs can still survive as corroborating evidence.
- Rerank is no longer cosine-only. It now blends query similarity with a structure bonus and a redundancy penalty so the second stage can prefer focused, non-duplicate passages.
- Tree paragraph assignment is now page-local and ordered, so the active heading can switch mid-page when a new heading appears.

Verification completed:

- `pytest backend/tests/test_retrieval_pipeline.py backend/tests/test_context_rerank.py backend/tests/test_tree_index_assignment.py -q --basetemp=backend\.tmp_pytest`
- `pytest backend/tests/test_documents_ingest.py backend/tests/test_index_version.py -q --basetemp=backend\.tmp_pytest`
- `python -m compileall backend/app backend/tests`

Remaining follow-up candidates:

- tree retrieval still prunes fairly aggressively and may miss broad dispersed evidence
- doc-summary fallback to most recent docs is still a weak safety net in larger corpora
- local rerank is stronger now, but it is still heuristic rather than a learned reranker or cross-encoder
- cache/pytest temp-directory permissions in this environment still emit warnings during test runs

Fresh-session resume note:

1. Read this file.
2. Inspect `git status`.
3. If more retrieval work is requested, start with the remaining follow-up candidates above.
