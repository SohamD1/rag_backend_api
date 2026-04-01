# Doc Selection Improvements

## Goal

Improve stage-1 document routing before per-document retrieval begins.

The two changes to prioritize are:

1. replace the single doc centroid with 2-4 doc-level vectors per document
2. make first-pass doc selection smarter than raw-query-to-centroid plus "most recent docs" fallback

## Why This Matters

The current architecture is good:

- search a cheap global doc-selection namespace first
- then search only the chosen document namespace(s)

The weak point is the quality of the first routing step.

Right now the doc summary is one centroid over all chunk/node vectors. That can blur multi-topic docs,
overweight boilerplate, and make document routing less reliable.

## Priority 1: 2-4 Doc-Level Vectors Per Document

## Current State

Today each document gets exactly one doc-selection vector:

- build chunk/node embeddings during ingest
- average them into one centroid
- store that single vector in the doc-summary namespace

## Proposed Change

Store multiple lightweight routing vectors per document instead of one centroid.

Recommended vector types:

1. `profile`
   A short document profile built from filename, slug, source URL, route, and a concise abstract.

2. `headings`
   Title plus TOC/headings summary. Best for structured PDFs and section-oriented queries.

3. `keywords`
   Dense topic/entity/product/taxonomy keywords extracted from the document.

4. `faq` or `use_cases`
   Optional. A short synthetic "this document answers questions about..." representation.

If only two are implemented first, start with:

1. `profile`
2. `headings`

## Storage Shape

Keep using the global doc-summary namespace, but store multiple records per doc.

Suggested ids:

- `{doc_id}:profile`
- `{doc_id}:headings`
- `{doc_id}:keywords`
- `{doc_id}:faq`

Suggested metadata:

- `doc_id`
- `source_url`
- `filename`
- `slug`
- `route`
- `index_version`
- `summary_kind`
- `summary_text`

That preserves cheap global search while allowing multiple routing views of the same doc.

## Selection Logic

At query time:

1. search the global doc-summary namespace
2. aggregate matches by `doc_id`
3. combine scores across summary kinds
4. rank docs by aggregated score, not by one vector hit

Simple first scoring rule:

- take max score per summary kind
- sum the best 2 summary-kind scores per doc
- add a small bonus when more than one summary kind matches well

That gives better routing without much complexity.

## Ingest Implementation Notes

During ingest:

1. build the standard or tree index as usual
2. generate summary texts for the doc
3. embed those summary texts directly
4. upsert multiple doc-summary records instead of one centroid-only record

Do not remove the per-document namespace design. This change only upgrades stage-1 routing.

## Migration Strategy

Safe rollout:

1. keep centroid support temporarily
2. add multi-vector doc summaries behind a small flag or schema version bump
3. reingest documents
4. switch query-time selection to the new aggregated scoring
5. remove centroid-only fallback later if it is no longer needed

## Priority 2: Smarter First-Pass Doc Selection

## Current Weaknesses

Current first-pass routing is weaker than it should be because:

- it usually uses the raw user query directly
- it relies on one vector per doc
- if selection fails badly, fallback is most recent docs

## Proposed Change

Make first-pass doc selection use multiple signals.

Recommended order:

1. normalize and lightly rewrite the query for doc selection
2. search multi-vector doc summaries
3. aggregate by doc
4. if confidence is low, use a lexical fallback over doc metadata
5. only then fall back to broader retrieval behavior

## Better First-Pass Signals

### 1. Query Rewriting

Use a concise doc-selection rewrite on the first pass, not only after retrieval confidence drops.

The rewritten query should emphasize:

- product names
- entities
- acronyms
- topic keywords
- date or jurisdiction terms
- unique phrases

### 2. Lexical Metadata Match

Add a lightweight lexical score over:

- filename
- slug
- source URL
- headings text
- summary text

This helps when users mention exact names or phrases that vector search underweights.

### 3. Confidence-Aware Fallback

Replace "most recent docs" as the main empty-summary fallback with something safer:

- lexical top docs
- broader doc-summary top_k
- or all-doc search only when the corpus is still small

Most-recent can stay as a last-resort operational fallback, but not the main relevance fallback.

## Suggested Selection Algorithm

1. Build `raw_query_embedding`.
2. Build `rewritten_query` and `rewritten_query_embedding`.
3. Search doc-summary namespace with:
   - raw query embedding
   - rewritten query embedding
4. Aggregate all doc-summary hits by `doc_id`.
5. Blend vector score with lexical metadata score.
6. Mark a doc as strong only if:
   - top score clears a minimum threshold
   - and separation from the next doc is large enough
   - or multiple summary kinds agree on the same doc
7. If not strong, pass top 2-4 docs into per-doc retrieval.

That still keeps the design cheap while making stage-1 selection much more reliable.

## Concrete Implementation Tasks

## A. Ingest-Side

- Add a helper to generate doc summary texts from metadata plus tree/heading artifacts.
- Replace `upsert_doc_centroid` with a multi-record upsert flow.
- Store `summary_kind` in metadata for each doc-summary record.
- Update delete and cleanup logic so all doc-summary ids for a doc are removed.

## B. Query-Side

- Update doc-summary query logic to aggregate matches by `doc_id`.
- Add first-pass rewrite support for doc selection.
- Add lexical metadata scoring as a fallback or blended signal.
- Replace "most recent docs" as the primary empty-result fallback.
- Expose debug info showing which summary kinds matched and why a doc was selected.

## C. Versioning And Reindex

- Include doc-summary strategy in index/schema versioning or doc-summary version metadata.
- Reingest docs after rollout so all documents get the new summary records.

## Suggested Order

1. Implement multi-vector doc-summary storage.
2. Aggregate doc-summary hits by `doc_id`.
3. Enable first-pass rewrite for doc selection.
4. Add lexical metadata scoring.
5. Replace most-recent fallback.
6. Reingest and evaluate.

## Success Criteria

This work is successful if:

- the correct document is chosen more often on ambiguous queries
- exact doc-name queries route correctly
- multi-topic docs no longer feel blurred at stage 1
- fallback behavior is relevance-based, not recency-based
- per-document namespace retrieval remains unchanged and cheap
