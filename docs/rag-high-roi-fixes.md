# High-ROI RAG Fixes

## Goal

This document prioritizes the changes that are most likely to improve this RAG backend quickly across four dimensions:

- security,
- correctness,
- retrieval quality,
- operational reliability.

The emphasis is on highest return for the least implementation risk.

## Top Priorities

## 1. Make Auth Fail-Closed

### Why this is high ROI

Right now the biggest repo-level risk is accidental exposure. If auth is not explicitly enabled in deployment, both chat and admin document endpoints can become public.

This is the fastest security improvement with the largest downside reduction.

### What to change

- Make auth required by default in non-development environments.
- Fail startup if `KB_REQUIRE_AUTH=true` but tokens are missing.
- Consider failing startup in production unless auth is explicitly configured.
- Document the required env vars clearly in the backend README.

### Expected impact

- Prevents accidental public exposure of upload/delete/chat endpoints.
- Reduces the single highest-severity security risk in the repo.

## 2. Protect And Rate-Limit Deep Dependency Health Checks

### Why this is high ROI

`/api/health/deps?deep=true` currently performs real OpenAI and Pinecone calls. That means an unauthenticated caller can spend money and create load.

This is a small change with immediate security and cost benefits.

### What to change

- Require admin auth for deep dependency checks.
- Add rate limiting to deep health checks.
- Keep lightweight readiness checks public if needed, but gate expensive checks.

### Expected impact

- Stops abuse of paid dependency calls.
- Makes health endpoints safer in production.

## 3. Add Real Cleanup And Rollback For Failed Ingests

### Why this is high ROI

Right now failed uploads can leave:

- orphaned PDFs,
- partial vectors,
- partial tree artifacts,
- registry inconsistencies.

This creates long-term data hygiene problems and makes retrieval/debugging harder.

### What to change

- Track ingest state explicitly.
- On any failure after file save, perform compensating cleanup:
  - delete stored PDF,
  - delete vectors for that doc,
  - delete tree artifacts,
  - remove partial registry entry if created.
- Log cleanup failures explicitly.

### Expected impact

- Prevents “ghost documents.”
- Makes ingest behavior safer and easier to reason about.
- Reduces operational cleanup work later.

## 4. Make Delete Strict Or Auditable

### Why this is high ROI

Delete currently removes local state even if vector deletion fails. That can leave stale vectors retrievable after the document appears deleted.

This is a trust and correctness problem.

### What to change

- Either make deletion transactional enough to abort when vector deletion fails,
- or mark delete as incomplete and persist a cleanup task/retry record.
- Return degradation details instead of silently swallowing failures.

### Expected impact

- Prevents deleted docs from remaining searchable.
- Makes deletion behavior match user expectations.

## 5. Stop Caching Degraded Retrieval Results

### Why this is high ROI

A transient retrieval failure can currently produce a partial result set that gets cached as if it were correct.

That is one of the worst correctness multipliers in the repo because a single temporary error becomes a repeated one.

### What to change

- If any retrieval stage partially fails, do not write that result to cache.
- Record degraded retrieval state in debug/log output.
- Decide which failures should abort the request versus degrade gracefully.

### Expected impact

- Prevents transient backend issues from poisoning later answers.
- Improves trust in retrieval output.

## 6. Fix Tree Paragraph-To-Heading Assignment

### Why this is high ROI

This is likely the highest-value retrieval-quality fix for long structured documents.

Today tree paragraphs inherit headings at page level, which is wrong when sections change mid-page.

### What to change

- Detect heading transitions within the page.
- Split page content into ordered blocks.
- Reassign active heading as blocks progress through the page.
- Keep page-level fallback only when no confident intra-page transition is found.

### Expected impact

- Better retrieval on long PDFs.
- Better section breadcrumbs and evidence fidelity.
- Fewer structurally wrong answers in large docs.

## 7. Improve Chunking For OCR, Lists, Tables, And Forms

### Why this is high ROI

Current chunking flattens too much structure. That hurts retrieval quality even when embeddings are good.

This is especially important for messy PDFs, scanned documents, brochures, and forms.

### What to change

- Preserve more line structure when layout signals are strong.
- Add layout-aware normalization modes.
- Split by heading/list/table boundaries before falling back to token windows.
- Add metadata like `chunk_type`, `ocr_used`, and `structure_confidence`.

### Expected impact

- Better chunk coherence.
- Better retrieval for OCR-heavy and layout-heavy documents.
- Better future reranking and citation quality.

## 8. Make Index Versioning Include OCR And Chunking Semantics

### Why this is high ROI

Right now changing OCR behavior or chunking semantics can still reuse an old index.

That makes quality work harder because improvements may not actually take effect until a manual cleanup happens.

### What to change

- Include OCR mode and thresholds in `index_version`.
- Include structure-aware chunking flags and key chunking constants.
- Include tree paragraph assignment logic version.

### Expected impact

- Makes reindex behavior predictable.
- Prevents stale embeddings/text boundaries from surviving config changes.

## 9. Replace Cosine “Rerank” With A Distinct Signal

### Why this is high ROI

The current reranker mostly recomputes the same signal Pinecone already used. That adds complexity with limited benefit.

This is a quality improvement, but lower urgency than the security and cleanup issues above.

### What to change

- Add a true reranker:
  - cross-encoder,
  - API reranker,
  - or LLM reranker for small candidate sets.
- Keep cosine rerank only as fallback.

### Expected impact

- Better precision on top retrieved passages.
- More meaningful second-stage ranking.

## 10. Add A Minimal Source-Test Safety Net

### Why this is high ROI

The repo currently has almost no committed source test coverage. That makes every retrieval and ingest change riskier.

This does not need to become a huge harness to be valuable.

### What to change

- Add focused tests for:
  - failed upload cleanup,
  - failed delete cleanup,
  - doc selection fallback behavior,
  - mid-page heading transitions,
  - chunking on OCR/table/list fixtures,
  - degraded retrieval not being cached.

### Expected impact

- Catches regressions in the most fragile paths.
- Makes future retrieval changes much safer.

## Recommended Order

If the goal is best ROI in sequence, I would do this:

1. fail-closed auth,
2. protect deep health checks,
3. ingest rollback and delete reliability,
4. stop caching degraded retrieval,
5. fix tree paragraph assignment,
6. improve chunking/OCR preservation,
7. expand index versioning,
8. add a real reranker,
9. add focused regression tests alongside each fix.

## Fastest “Good Enough” Security Package

If you want the fastest possible security-focused pass, do these first:

1. require auth by default,
2. gate and rate-limit deep health checks,
3. document auth env vars in README,
4. make delete fail loudly when vector cleanup fails.

## Fastest “Better Answers” Package

If you want the fastest answer-quality pass, do these first:

1. fix tree paragraph assignment,
2. improve chunking for OCR/layout-heavy docs,
3. replace cosine rerank with a distinct signal,
4. stop document-level citation consolidation.

## My Recommendation

The best combined package is:

1. secure the service,
2. make ingest/delete failure-safe,
3. stop silent degraded caching,
4. then improve structural retrieval quality.

That sequence gives the backend a safer foundation before investing in more advanced retrieval improvements.
