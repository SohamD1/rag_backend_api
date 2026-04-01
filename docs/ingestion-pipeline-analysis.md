# Ingestion Pipeline Analysis

## Intended Design

The intended retrieval model is:

1. Create one cheap document-level vector per document.
2. Store those document-level vectors in one global namespace.
3. Query that global namespace first to decide which document or small set of documents to inspect.
4. Then query only the namespace for the chosen document(s).

That is the right high-level shape for this codebase. It is cheaper than querying one giant mixed
namespace, and it keeps document-level routing separate from passage retrieval.

## What The Current Code Does

Current ingest flow:

1. Save the uploaded PDF temporarily.
2. OCR every PDF with Mistral.
3. Route the document into:
   - `standard` indexing when page count is at or below `PAGE_TREE_THRESHOLD`
   - `tree` indexing when page count is above `PAGE_TREE_THRESHOLD`
4. Write all chunk/node vectors into a per-document namespace:
   - `{doc_id}::{index_version}`
5. Compute a centroid from the document's embeddings.
6. Upsert that centroid into the global doc-summary namespace.

Current query flow:

1. Embed the user query.
2. Search the global doc-summary namespace first.
3. If the top match is strong enough, search only that one document namespace.
4. Otherwise search the top few matched document namespaces.
5. Rerank and pack final context from those selected docs.

So the basic architecture already matches the intended two-stage strategy.

## What Is Good

- Per-document namespaces are already in place.
- Index-versioned namespaces make reingest safer.
- Global doc-selection is separate from per-document retrieval.
- Failed-ingest cleanup is much better than before.
- Tree indexing is materially better than flat chunking for long structured PDFs.

## Main Issues

## 1. The Doc-Selection Vector Is Too Crude

Right now the global document vector is just a centroid of all embeddings from the document.

That is the biggest quality risk in the current design.

Why this is weak:

- long documents get averaged into one blurred topic vector
- repeated boilerplate can dominate the centroid
- tree docs mix heading-summary vectors and paragraph vectors into the same average
- multi-topic documents become harder to route correctly at stage 1

This can cause the wrong document to be chosen before the system ever searches the right namespace.

## 2. Duplicate Documents Can Pollute Doc Selection

`doc_id` is derived from `slug + checksum-prefix`, not checksum alone.

That means the same PDF content ingested under different filenames can create multiple logical
documents with separate namespaces and separate doc-summary vectors.

That weakens stage-1 routing because duplicate docs can crowd out other candidates.

## 3. First-Pass Doc Selection Is Weaker Than It Looks

The pipeline supports query rewriting for doc selection, but the initial chat path currently disables
rewrite on the first selection pass and only retries later when retrieval confidence is already low.

That means the most important routing decision often uses only the raw user query against one centroid
per doc.

## 4. Empty Or Stale Doc Summaries Fall Back To "Most Recent Docs"

If doc-summary selection returns nothing, the fallback is to search the most recently created docs.

That is operationally convenient, but it is a weak relevance fallback for a growing corpus and can
produce low-quality or misleading retrieval.

## 5. Standard Chunking Is Still Too Structure-Blind

For documents below the tree threshold, chunking is paragraph-packing without heading-aware boundaries.

That means:

- section information is mostly reduced to page labels
- short structured PDFs lose good heading signals
- OCR lists, forms, brochures, and tables are still only partially preserved

This is likely the second-largest ingestion-quality risk after the doc-summary centroid.

## 6. The Tree Route Still Prunes Aggressively

Tree retrieval first finds headings, then usually narrows to only a few sections before paragraph
search.

That saves cost, but it can miss relevant evidence when:

- the query spans multiple sections
- the right heading is not in the top heading set
- OCR noise weakens heading matches

This is more of a retrieval issue than an ingest issue, but it directly affects whether the per-doc
namespace strategy works reliably.

## 7. OCR Markdown Structure Is Underused

The system already gets Markdown-style OCR output from Mistral, but the standard path mostly reduces
that into paragraph blocks and token windows.

There is room to preserve more of:

- headings
- lists
- tables
- key-value/form structure
- page-level OCR quality signals

Those would improve both embeddings and downstream citations.

## Recommended Changes

## 1. Replace The Single Centroid With A Better Doc Profile

Best next change:

- generate a dedicated document-profile text from title, filename, source URL, TOC headings, and a
  short abstract
- embed that profile directly instead of averaging every chunk/node vector

Even better:

- store 2-4 doc-selection vectors per doc
- examples: abstract vector, headings/TOC vector, product/topic keywords vector

That keeps stage-1 cheap while making document routing much sharper.

## 2. Weight Heading-Level Signals More Than Raw Paragraphs

If centroids are kept, they should not treat every vector equally.

Prefer a weighted doc summary such as:

- heavier weight for title / TOC / top headings
- medium weight for heading summaries
- lighter weight for paragraph vectors
- optional downweight for repeated boilerplate chunks

## 3. Deduplicate By Checksum Or Source Identity

Add a stronger duplicate policy so the same underlying PDF does not create multiple logical docs just
because the filename changed.

At minimum:

- detect same checksum across filenames
- decide whether to reuse the existing doc or mark it as an alias/update

## 4. Make First-Pass Doc Selection Smarter

Improve the first routing step by doing one or more of:

- allow rewrite on the initial doc-selection pass
- add a lexical fallback over filename, slug, source URL, and headings
- use a second doc-selection signal beyond one vector similarity score

## 5. Lower The Tree Threshold Or Use Structure-Aware Routing

The current page-count threshold is a rough heuristic.

Better routing would consider:

- presence of a TOC
- heading density
- layout complexity
- list/table density

Many sub-50-page PDFs still want the tree path.

## 6. Upgrade Standard Chunking

For the non-tree path, preserve more structure during chunk creation:

- heading-aware chunk boundaries
- list-aware chunking
- table/key-value chunk markers
- chunk metadata such as `chunk_type`, `structure_confidence`, and `ocr_layout_mode`

## 7. Loosen Tree Retrieval When Confidence Is Low

When tree heading selection is weak, do not prune too early.

Safer fallback behavior:

- search more than four sections
- expand to a broader paragraph search when heading confidence is low
- allow a whole-doc paragraph fallback inside the selected doc namespace

## Suggested Priority Order

1. Improve the doc-summary vector design.
2. Deduplicate same-content documents.
3. Make first-pass doc selection smarter.
4. Lower or replace the page-count-only tree threshold.
5. Improve standard chunking for structure-heavy short documents.
6. Relax overly aggressive tree pruning when confidence is low.

## Bottom Line

The overall architecture is good and already close to the intended design:

- global cheap doc selection first
- then search only the chosen document namespaces

The biggest current weakness is not namespace design. It is that the stage-1 document vector is too
lossy. If that one vector is improved, the rest of the pipeline has a much better chance of landing in
the right document before per-doc retrieval begins.
