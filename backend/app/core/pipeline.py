from __future__ import annotations

import logging
import queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException

from app.adapters.embeddings import embed_texts
from app.adapters.pinecone_store import PineconeVectorStore
from app.config import Settings
from app.core.cache import JsonCache, make_cache_key, normalize_query
from app.core.context import select_context
from app.core.doc_summaries import (
    doc_summary_record_ids,
    lexical_doc_score,
    query_doc_summaries,
    select_doc_ids_from_matches,
)
from app.core.generation import generate_answer, review_answer
from app.core.retrieval import (
    RetrievalItem,
    retrieval_item_from_dict,
    retrieval_item_to_dict,
    retrieve_standard_for_doc,
    retrieve_tree_for_doc,
)
from app.core.rerank import rerank_items
from app.core.rewrite import rewrite_query_for_doc_selection
from app.storage.registry import DocMeta, DocRegistry


logger = logging.getLogger(__name__)

_INLINE_CITATION_RE = re.compile(r"\[([0-9,\s]+)\]")
_BROAD_QUERY_PATTERNS = (
    "what is",
    "what are",
    "what counts as",
    "why ",
    "how ",
    "when ",
    "common elements",
    "what should be included",
    "what questions should",
    "what rights",
    "what are the main challenges",
    "definition",
    "definitions",
    "examples",
    "categories",
    "challenges",
    "rights",
    "used",
    "matter",
    "include",
    "inventory",
    "besides",
)


_STAGE_LABELS: Dict[str, str] = {
    "registry_lookup": "Loading documents",
    "response_cache_key": "Checking response cache",
    "response_cache": "Using cached response",
    "embed_query": "Embedding query",
    "doc_selection": "Routing documents",
    "retrieve": "Retrieving evidence",
    "retrieve_cache": "Using cached retrieval",
    "rerank_fetch": "Loading retrieval details",
    "rerank": "Ranking evidence",
    "rerank_skip": "Skipping rerank",
    "rewrite_query": "Rewriting query",
    "doc_selection_rewrite": "Retrying document routing",
    "rewrite_retrieve": "Retrieving evidence",
    "rewrite_retrieve_cache": "Using cached retrieval",
    "rewrite_rerank_fetch": "Loading retrieval details",
    "rewrite_rerank": "Ranking evidence",
    "rewrite_rerank_skip": "Skipping rerank",
    "context_select": "Packing context",
    "generate": "Writing answer",
}


def _humanize_stage(stage: str) -> str:
    key = str(stage or "").strip()
    if not key:
        return "Working"
    label = _STAGE_LABELS.get(key)
    if label:
        return label
    return key.replace("_", " ").strip().title()


def _with_stage_metadata(payload: Dict[str, object]) -> Dict[str, object]:
    if "stage" not in payload:
        return payload
    stage = str(payload.get("stage") or "").strip()
    enriched = dict(payload)
    enriched.setdefault("label", _humanize_stage(stage))
    if "message" not in enriched:
        if enriched.get("cache_hit"):
            enriched["message"] = f"{enriched['label']}."
        elif enriched.get("skipped"):
            enriched["message"] = f"{enriched['label']}."
        else:
            enriched["message"] = enriched["label"]
    return enriched


def _dump_model(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _emit(
    on_event: Optional[Callable[[Dict[str, object]], None]],
    event: str,
    payload: Dict[str, object],
) -> None:
    if on_event is None:
        return
    on_event({"event": event, "data": _with_stage_metadata(payload)})


def _stage_start(on_event: Optional[Callable[[Dict[str, object]], None]], stage: str) -> None:
    _emit(on_event, "stage_start", {"stage": stage})


def _stage_end(
    on_event: Optional[Callable[[Dict[str, object]], None]], stage: str, elapsed_ms: int
) -> None:
    _emit(on_event, "stage_end", {"stage": stage, "elapsed_ms": elapsed_ms})


def _sort_metas_for_fallback(metas: List[DocMeta]) -> List[DocMeta]:
    def key(m: DocMeta):
        return (m.created_at or "", m.filename.lower())

    return sorted(metas, key=key, reverse=True)


def _build_lexical_doc_scores(
    *,
    metas: List[DocMeta],
    queries: List[str],
    summary_fields_by_doc_id: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    non_empty_queries = [q for q in queries if (q or "").strip()]
    if not non_empty_queries:
        return scores
    summary_fields_by_doc_id = summary_fields_by_doc_id or {}
    for meta in metas:
        fields = [
            meta.filename,
            meta.slug,
            meta.source_url,
            *summary_fields_by_doc_id.get(meta.doc_id, []),
        ]
        score = max(lexical_doc_score(query, fields) for query in non_empty_queries)
        if score > 0:
            scores[meta.doc_id] = score
    return scores


def _load_doc_summary_lexical_fields(
    *,
    metas: List[DocMeta],
    settings: Settings,
    vector_store: PineconeVectorStore,
) -> Dict[str, List[str]]:
    fetch_records = getattr(vector_store, "fetch_records", None)
    if not callable(fetch_records):
        return {}

    ids: List[str] = []
    for meta in metas:
        ids.extend(doc_summary_record_ids(meta.doc_id))
    if not ids:
        return {}

    try:
        records = fetch_records(ids=ids, namespace=settings.doc_summary_namespace) or {}
    except Exception:
        logger.exception("doc_summary_lexical_fields_load_failed")
        return {}

    summary_fields_by_doc_id: Dict[str, List[str]] = {}
    for record in records.values():
        metadata = record.get("metadata", {}) or {}
        doc_id = str(metadata.get("doc_id") or record.get("id") or "").strip()
        summary_text = str(metadata.get("summary_text") or "").strip()
        if not doc_id or not summary_text:
            continue
        summary_fields_by_doc_id.setdefault(doc_id, []).append(summary_text)
    return summary_fields_by_doc_id


def _should_skip_rerank(
    *,
    ordered_items: List[RetrievalItem],
    strong_doc_match: bool,
    low_confidence_threshold: float,
) -> bool:
    """
    Production-safe fast path:
    if retrieval is already very confident, skip vector fetch + rerank.
    """
    if not strong_doc_match or not ordered_items:
        return False
    top1 = float(ordered_items[0].score)
    top2 = float(ordered_items[1].score) if len(ordered_items) > 1 else 0.0
    min_top = max(float(low_confidence_threshold) + 0.15, 0.55)
    min_ratio = 1.08
    if top1 < min_top:
        return False
    if top2 > 0 and top1 < (top2 * min_ratio):
        return False
    return True


def _is_broad_synthesis_query(query: str) -> bool:
    normalized = " ".join(str(query or "").lower().split())
    if len(normalized.split()) < 6:
        return False
    return any(pattern in normalized for pattern in _BROAD_QUERY_PATTERNS)


def _expand_strong_doc_selection(
    *,
    query: str,
    ranked: List[Dict[str, Any]],
    settings: Settings,
) -> List[str]:
    if not ranked:
        return []

    top_doc_id = str(ranked[0].get("doc_id") or "").strip()
    if not top_doc_id:
        return []
    if not _is_broad_synthesis_query(query):
        return [top_doc_id]

    selected = [top_doc_id]
    top_aggregate = float(ranked[0].get("aggregate_score", 0.0) or 0.0)
    top_vector = float(ranked[0].get("best_vector_score", 0.0) or 0.0)
    max_docs = min(3, max(2, int(getattr(settings, "doc_summary_top_k", 3) or 3)))

    for candidate in ranked[1:]:
        if len(selected) >= max_docs:
            break
        candidate_doc_id = str(candidate.get("doc_id") or "").strip()
        if not candidate_doc_id:
            continue
        aggregate = float(candidate.get("aggregate_score", 0.0) or 0.0)
        vector_score = float(candidate.get("best_vector_score", 0.0) or 0.0)
        lexical_score = float(candidate.get("lexical_score", 0.0) or 0.0)

        if aggregate < max(0.18, top_aggregate * 0.82):
            continue
        if vector_score > 0.0 and vector_score < max(0.22, top_vector * 0.72):
            continue
        if vector_score <= 0.0 and lexical_score < 0.20:
            continue
        selected.append(candidate_doc_id)

    return selected


def _build_chunk_citations(
    *,
    items: List[RetrievalItem],
    source_url_by_doc_id: Dict[str, str],
) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    seen: set[str] = set()
    per_doc_count: Dict[str, int] = {}
    for item in items:
        doc_id = str(item.doc_id or "").strip()
        if not doc_id:
            continue
        source_id = str(item.source_id or "").strip()
        dedupe_key = source_id or "|".join(
            [
                doc_id,
                str(int(item.page_start)),
                str(int(item.page_end)),
                str(item.section_title or ""),
            ]
        )
        if dedupe_key in seen:
            continue
        if per_doc_count.get(doc_id, 0) >= 3:
            continue

        citations.append(
            {
                "doc_id": doc_id,
                "filename": item.filename,
                "source_url": item.source_url or source_url_by_doc_id.get(doc_id, ""),
                "source_id": source_id or f"{doc_id}:citation:{per_doc_count.get(doc_id, 0) + 1}",
                "page_start": int(item.page_start),
                "page_end": int(item.page_end),
                "section_title": item.section_title,
            }
        )
        seen.add(dedupe_key)
        per_doc_count[doc_id] = per_doc_count.get(doc_id, 0) + 1

        if len(citations) >= 12:
            break

    return citations


def _context_doc_number_map(context_items: List[Dict[str, Any]]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    seen_doc_ids: set[str] = set()

    for item in context_items:
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        mapping[len(mapping) + 1] = doc_id

    return mapping


def _extract_answer_citation_numbers(answer: str) -> List[int]:
    seen: set[int] = set()
    ordered: List[int] = []

    for match in _INLINE_CITATION_RE.finditer(str(answer or "")):
        raw = str(match.group(1) or "")
        for token in raw.split(","):
            cleaned = token.strip()
            if not cleaned.isdigit():
                continue
            number = int(cleaned)
            if number <= 0 or number in seen:
                continue
            seen.add(number)
            ordered.append(number)

    return ordered


def _citation_keywords(text: str) -> set[str]:
    return {
        token
        for token in re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).split()
        if len(token) >= 4
    }


def _filter_citations_for_answer(
    *,
    citations: List[Dict[str, Any]],
    answer: str,
    context_items: List[Dict[str, Any]],
    selected_items: List[RetrievalItem],
) -> List[Dict[str, Any]]:
    if "[" not in answer or "]" not in answer:
        return []

    cited_numbers = _extract_answer_citation_numbers(answer)
    if not cited_numbers:
        return citations

    doc_number_map = _context_doc_number_map(context_items)
    cited_doc_ids = {
        doc_number_map[number]
        for number in cited_numbers
        if number in doc_number_map
    }
    if not cited_doc_ids:
        return citations

    source_item_by_id = {
        str(item.source_id or "").strip(): item
        for item in selected_items
        if str(item.source_id or "").strip()
    }
    answer_terms = _citation_keywords(answer)
    filtered = [citation for citation in citations if citation.get("doc_id") in cited_doc_ids]
    grouped: Dict[str, List[tuple[float, Dict[str, Any]]]] = {}

    for citation in filtered:
        source_id = str(citation.get("source_id") or "").strip()
        item = source_item_by_id.get(source_id)
        citation_terms = _citation_keywords(
            " ".join(
                [
                    str(citation.get("section_title") or ""),
                    str(getattr(item, "text", "") or ""),
                ]
            )
        )
        overlap = len(answer_terms.intersection(citation_terms))
        score = float(overlap)
        if item is not None:
            score += float(item.score)
        grouped.setdefault(str(citation.get("doc_id") or ""), []).append((score, citation))

    narrowed: List[Dict[str, Any]] = []
    for doc_id, doc_citations in grouped.items():
        ranked = sorted(
            doc_citations,
            key=lambda entry: (
                entry[0],
                -int(entry[1].get("page_start") or 0),
                -int(entry[1].get("page_end") or 0),
            ),
            reverse=True,
        )
        keep_count = min(2, len(ranked))
        narrowed.extend(citation for _score, citation in ranked[:keep_count])

    return narrowed


def _build_chunk_payload(items: List[RetrievalItem]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for item in items:
        chunks.append(
            {
                "source_id": item.source_id,
                "doc_id": item.doc_id,
                "filename": item.filename,
                "source_url": item.source_url,
                "text": item.text,
                "score": float(item.score),
                "page_start": int(item.page_start),
                "page_end": int(item.page_end),
                "section_title": item.section_title,
                "route": item.route,
            }
        )
    return chunks


def _mark_retrieval_degraded(
    *,
    debug_info: Optional[Dict[str, Any]],
    stage: str,
    details: Dict[str, Any],
) -> None:
    if debug_info is None:
        return
    degraded = debug_info.setdefault("retrieval_degraded", {"stages": []})
    stages = degraded.setdefault("stages", [])
    stages.append({"stage": stage, **details})


def select_docs_with_rewrite_retry(
    *,
    query: str,
    query_embedding: List[float],
    metas: List[DocMeta],
    settings: Settings,
    vector_store: PineconeVectorStore,
    debug: Optional[Dict[str, Any]] = None,
    allow_rewrite: bool = True,
) -> Tuple[List[str], bool, str, List[float]]:
    rewritten_query = query
    rewritten_embedding = query_embedding
    combined_matches: List[Dict[str, Any]] = []

    matches = query_doc_summaries(
        query_embedding=query_embedding,
        settings=settings,
        vector_store=vector_store,
        top_k=getattr(settings, "doc_summary_match_top_k", settings.doc_summary_top_k),
    )
    combined_matches.extend([{**match, "query_variant": "raw"} for match in matches])

    rewrite_attempts = max(0, int(settings.doc_rewrite_max_attempts)) if allow_rewrite else 0
    for attempt in range(rewrite_attempts):
        rewritten_query = rewrite_query_for_doc_selection(query, settings)
        if rewritten_query.strip() == query.strip():
            break
        rewritten_embedding = embed_texts([rewritten_query], settings)[0]
        matches2 = query_doc_summaries(
            query_embedding=rewritten_embedding,
            settings=settings,
            vector_store=vector_store,
            top_k=getattr(settings, "doc_summary_match_top_k", settings.doc_summary_top_k),
        )
        combined_matches.extend([{**match, "query_variant": "rewrite"} for match in matches2])
        break

    lexical_queries = [query, rewritten_query] if rewritten_query != query else [query]
    lexical_scores = _build_lexical_doc_scores(
        metas=metas,
        queries=lexical_queries,
    )
    selected_ids, strong, info = select_doc_ids_from_matches(
        combined_matches,
        settings=settings,
        lexical_scores=lexical_scores,
        top_k_fallback=min(4, max(1, int(settings.doc_summary_top_k))),
    )

    # Summary-text lexical fallback is the most expensive doc-selection fallback.
    # Only load it when the cheaper vector + metadata signals do not already yield a strong winner.
    if not strong:
        summary_fields_by_doc_id = _load_doc_summary_lexical_fields(
            metas=metas,
            settings=settings,
            vector_store=vector_store,
        )
        if summary_fields_by_doc_id:
            lexical_scores = _build_lexical_doc_scores(
                metas=metas,
                queries=lexical_queries,
                summary_fields_by_doc_id=summary_fields_by_doc_id,
            )
            selected_ids, strong, info = select_doc_ids_from_matches(
                combined_matches,
                settings=settings,
                lexical_scores=lexical_scores,
                top_k_fallback=min(4, max(1, int(settings.doc_summary_top_k))),
            )

    if debug is not None:
        debug["doc_selection"] = {
            "query_used": query,
            "rewritten_query": rewritten_query if rewritten_query != query else None,
            "lexical_scores": lexical_scores,
            **info,
            "selected_doc_ids": selected_ids,
        }

    if strong and selected_ids:
        ranked = list(info.get("ranked") or [])
        expanded_selected_ids = _expand_strong_doc_selection(
            query=query,
            ranked=ranked,
            settings=settings,
        )
        if expanded_selected_ids:
            selected_ids = expanded_selected_ids
        if debug is not None:
            debug["doc_selection"]["fallback"] = None
            debug["doc_selection"]["selected_doc_ids"] = selected_ids
            debug["doc_selection"]["expanded_for_breadth"] = len(selected_ids) > 1
        if rewritten_query != query:
            return selected_ids, True, rewritten_query, rewritten_embedding
        return selected_ids, True, query, query_embedding

    if not selected_ids:
        fallback = [
            doc_id
            for doc_id, _score in sorted(
                lexical_scores.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )[:3]
            if _score > 0
        ]
        if not fallback and len(metas) <= 4:
            fallback = [m.doc_id for m in sorted(metas, key=lambda m: m.filename.lower())]
        if not fallback:
            fallback = [m.doc_id for m in _sort_metas_for_fallback(metas)[:3]]
        if debug is not None:
            debug["doc_selection_fallback"] = {
                "selected_doc_ids": fallback,
                "strategy": "lexical_then_small_corpus_then_recency",
            }
        return fallback, False, query, query_embedding

    if debug is not None:
        debug["doc_selection"]["fallback"] = None
    if rewritten_query != query:
        return selected_ids[:4], False, rewritten_query, rewritten_embedding
    return selected_ids[:4], False, query, query_embedding


def _build_broadened_doc_candidates(
    *,
    metas: List[DocMeta],
    already_selected_doc_ids: List[str],
    query: str,
    limit: int,
) -> List[DocMeta]:
    limit = max(1, int(limit))
    meta_by_id = {m.doc_id: m for m in metas}
    selected: List[DocMeta] = [
        meta_by_id[doc_id]
        for doc_id in already_selected_doc_ids
        if doc_id in meta_by_id
    ]
    if len(selected) >= limit:
        return selected[:limit]

    selected_ids = {m.doc_id for m in selected}
    if len(metas) <= 4:
        remaining = [
            meta
            for meta in sorted(metas, key=lambda m: m.filename.lower())
            if meta.doc_id not in selected_ids
        ]
        return [*selected, *remaining][:limit]

    lexical_scores = _build_lexical_doc_scores(metas=metas, queries=[query])
    backup_ids: List[str] = [
        doc_id
        for doc_id, score in sorted(
            lexical_scores.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        if score > 0 and doc_id not in selected_ids
    ]
    if len(backup_ids) < limit:
        for meta in _sort_metas_for_fallback(metas):
            if meta.doc_id in selected_ids or meta.doc_id in backup_ids:
                continue
            backup_ids.append(meta.doc_id)
            if len(selected) + len(backup_ids) >= limit:
                break

    backups = [meta_by_id[doc_id] for doc_id in backup_ids if doc_id in meta_by_id]
    return [*selected, *backups][:limit]


def run_chat(
    *,
    query: str,
    debug_enabled: bool,
    settings: Settings,
    registry: DocRegistry,
    vector_store: PineconeVectorStore,
    retrieval_cache: JsonCache,
    response_cache: JsonCache,
    tree_dir,  # Path-like, only used for on-disk artifacts; retrieval is Pinecone-only
    on_event: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[str, Any]:
    started = perf_counter()
    debug_info: Optional[Dict[str, Any]] = {"stage_order": []} if debug_enabled else None

    t_lookup = perf_counter()
    _stage_start(on_event, "registry_lookup")
    metas = registry.list()
    _stage_end(on_event, "registry_lookup", int((perf_counter() - t_lookup) * 1000))
    if not metas:
        raise HTTPException(status_code=404, detail="No documents uploaded. Upload PDFs first.")

    normalized = normalize_query(query)
    doc_versions = sorted(f"{m.doc_id}:{m.index_version}:{m.route}" for m in metas)

    t_resp_key = perf_counter()
    _stage_start(on_event, "response_cache_key")
    response_key = make_cache_key(
        {
            "kind": "response",
            "query": normalized,
            "doc_versions": doc_versions,
            "generation_model": settings.openai_generation_model,
            "doc_summary_top_k": settings.doc_summary_top_k,
            "doc_summary_match_top_k": getattr(
                settings, "doc_summary_match_top_k", settings.doc_summary_top_k
            ),
            "doc_summary_strategy_version": getattr(
                settings, "doc_summary_strategy_version", "multi_vector_v1"
            ),
            "doc_strong_min_score": settings.doc_strong_min_score,
            "doc_strong_min_ratio": settings.doc_strong_min_ratio,
            "max_context_tokens": settings.max_context_tokens,
            "context_max_item_tokens": settings.context_max_item_tokens,
            "context_use_mmr": settings.context_use_mmr,
            "context_mmr_lambda": settings.context_mmr_lambda,
            "rerank_top_k": settings.rerank_top_k,
            "rerank_candidate_k": settings.rerank_candidate_k,
            "low_confidence_threshold": settings.low_confidence_threshold,
            "rewrite_model": settings.openai_rewrite_model,
            "rewrite_attempts": settings.doc_rewrite_max_attempts,
            "citation_mode": "chunk_filtered_v3",
            "generate_answers_enabled": settings.rag_generate_answers_enabled,
            "generation_max_completion_tokens": getattr(
                settings, "openai_generation_max_completion_tokens", 1400
            ),
            "answer_review": True,
            "debug": debug_enabled,
        }
    )
    _stage_end(on_event, "response_cache_key", int((perf_counter() - t_resp_key) * 1000))

    cached = response_cache.get(response_key)
    if cached:
        _emit(on_event, "info", {"stage": "response_cache", "cache_hit": True})
        payload = cached.value or {}
        return payload

    # Embed original query once.
    t_embed = perf_counter()
    _stage_start(on_event, "embed_query")
    query_embedding = embed_texts([query], settings)[0]
    _stage_end(on_event, "embed_query", int((perf_counter() - t_embed) * 1000))

    # Always doc-select via doc summaries first.
    t_select = perf_counter()
    _stage_start(on_event, "doc_selection")
    selected_doc_ids, strong, retrieval_query, retrieval_embedding = select_docs_with_rewrite_retry(
        query=query,
        query_embedding=query_embedding,
        metas=metas,
        settings=settings,
        vector_store=vector_store,
        debug=debug_info,
        allow_rewrite=True,
    )
    _stage_end(on_event, "doc_selection", int((perf_counter() - t_select) * 1000))

    meta_by_id = {m.doc_id: m for m in metas}
    selected_metas = [meta_by_id[d] for d in selected_doc_ids if d in meta_by_id]
    if not selected_metas:
        raise HTTPException(status_code=404, detail="Selected documents not found in registry.")

    def _retrieve_and_rerank(
        *,
        selected_metas_local: List[DocMeta],
        retrieval_query_local: str,
        retrieval_embedding_local: List[float],
        strong_doc_match: bool,
        stage_prefix: str = "",
    ) -> tuple[List[RetrievalItem], bool]:
        stage_retrieve = f"{stage_prefix}retrieve"
        stage_rerank_fetch = f"{stage_prefix}rerank_fetch"
        stage_rerank = f"{stage_prefix}rerank"
        stage_retrieve_cache = f"{stage_prefix}retrieve_cache"
        debug_retrieval_key = "retrieval" if not stage_prefix else f"{stage_prefix}retrieval".rstrip("_")
        degraded = False

        selected_versions = sorted(
            f"{m.doc_id}:{m.index_version}:{m.route}" for m in selected_metas_local
        )
        retrieval_key = make_cache_key(
            {
                "kind": "retrieval",
                "query": normalize_query(retrieval_query_local),
                "orig_query": normalized,
                "selected_doc_versions": selected_versions,
                "retrieve_top_k": settings.retrieve_top_k,
                "tree_section_top_k": settings.tree_section_top_k,
                "tree_node_selection_mode": getattr(settings, "tree_node_selection_mode", "vector_only"),
                "tree_node_selection_candidate_k": getattr(settings, "tree_node_selection_candidate_k", 24),
                "tree_node_selection_top_n": getattr(settings, "tree_node_selection_top_n", 6),
                "doc_strong_min_score": settings.doc_strong_min_score,
                "doc_strong_min_ratio": settings.doc_strong_min_ratio,
                "tree_node_selection_model": getattr(settings, "openai_tree_search_model", None),
            }
        )

        results_local: List[RetrievalItem] = []
        cached_r = retrieval_cache.get(retrieval_key)
        if cached_r:
            _emit(on_event, "info", {"stage": stage_retrieve_cache, "cache_hit": True})
            results_local = [
                retrieval_item_from_dict(item)
                for item in (cached_r.value or {}).get("items", [])
            ]
            if debug_info is not None:
                debug_info["stage_order"].append(stage_retrieve_cache)
                debug_info[debug_retrieval_key] = {"cache_hit": True, "count": len(results_local)}
        else:
            t_retrieve = perf_counter()
            _stage_start(on_event, stage_retrieve)
            retrieval_debug_by_doc: Dict[str, Dict[str, Any]] = {}
            failed_doc_ids: List[str] = []
            degraded_doc_ids: List[str] = []

            def _retrieve_one(meta: DocMeta):
                local_debug: Dict[str, Any] = {}
                if meta.route == "tree":
                    items = retrieve_tree_for_doc(
                        doc_id=meta.doc_id,
                        query=retrieval_query_local,
                        query_embedding=retrieval_embedding_local,
                        settings=settings,
                        vector_store=vector_store,
                        top_k=settings.retrieve_top_k,
                        debug=local_debug,
                        tree_dir=tree_dir,
                        index_version=meta.index_version,
                    )
                else:
                    items = retrieve_standard_for_doc(
                        doc_id=meta.doc_id,
                        query_embedding=retrieval_embedding_local,
                        settings=settings,
                        vector_store=vector_store,
                        index_version=meta.index_version,
                        top_k=settings.retrieve_top_k,
                    )
                return meta.doc_id, items, local_debug

            with ThreadPoolExecutor(max_workers=min(6, max(1, len(selected_metas_local)))) as executor:
                futures = {
                    executor.submit(_retrieve_one, m): m.doc_id for m in selected_metas_local
                }
                for future in as_completed(futures):
                    doc_id_hint = futures[future]
                    try:
                        doc_id, items, local_debug = future.result()
                    except Exception:
                        degraded = True
                        failed_doc_ids.append(doc_id_hint)
                        continue
                    results_local.extend(items or [])
                    if local_debug.get("degraded"):
                        degraded = True
                        degraded_doc_ids.append(doc_id)
                    if debug_info is not None and local_debug:
                        retrieval_debug_by_doc[doc_id] = local_debug
            _stage_end(on_event, stage_retrieve, int((perf_counter() - t_retrieve) * 1000))
            if failed_doc_ids:
                _mark_retrieval_degraded(
                    debug_info=debug_info,
                    stage=stage_retrieve,
                    details={
                        "failed_doc_ids": failed_doc_ids,
                    },
                )
            if degraded_doc_ids:
                _mark_retrieval_degraded(
                    debug_info=debug_info,
                    stage=stage_retrieve,
                    details={
                        "degraded_doc_ids": sorted(degraded_doc_ids),
                    },
                )
            if not degraded:
                retrieval_cache.set(
                    retrieval_key,
                    {"items": [retrieval_item_to_dict(item) for item in results_local]},
                )
            if debug_info is not None:
                debug_info["stage_order"].append(stage_retrieve)
                debug_info.setdefault(debug_retrieval_key, {})["count"] = len(results_local)
                debug_info.setdefault(debug_retrieval_key, {})["degraded"] = degraded
                if retrieval_debug_by_doc:
                    key = "retrieval_by_doc" if not stage_prefix else f"{stage_prefix}retrieval_by_doc".rstrip("_")
                    debug_info[key] = retrieval_debug_by_doc

        if not results_local:
            return [], degraded

        ordered_for_fetch = sorted(results_local, key=lambda x: x.score, reverse=True)
        if _should_skip_rerank(
            ordered_items=ordered_for_fetch,
            strong_doc_match=strong_doc_match,
            low_confidence_threshold=settings.low_confidence_threshold,
        ):
            _emit(
                on_event,
                "info",
                {
                    "stage": stage_rerank,
                    "skipped": True,
                    "reason": "high_confidence_retrieval",
                },
            )
            reranked_local = ordered_for_fetch[: max(1, int(settings.rerank_top_k))]
            if debug_info is not None:
                debug_info["stage_order"].append(f"{stage_rerank}_skip")
                debug_info["top_score"] = reranked_local[0].score if reranked_local else -1.0
                debug_info["low_confidence_threshold"] = settings.low_confidence_threshold
            return reranked_local, degraded

        # Attach stored vector values for rerank candidates (bounded, grouped by namespace/doc_id).
        t_fetch = perf_counter()
        _stage_start(on_event, stage_rerank_fetch)
        candidates = ordered_for_fetch[: max(0, int(settings.rerank_candidate_k))]
        ids_by_namespace: Dict[str, List[str]] = {}
        for item in candidates:
            namespace = item.vector_namespace or item.doc_id
            if namespace and item.source_id:
                ids_by_namespace.setdefault(namespace, []).append(item.source_id)
        embedding_by_key: Dict[tuple[str, str], List[float]] = {}
        failed_fetch_namespaces: List[str] = []
        if ids_by_namespace:
            fetch_fn = getattr(vector_store, "fetch", None)
            if not callable(fetch_fn):
                degraded = True
                failed_fetch_namespaces.extend(sorted(ids_by_namespace))
            else:
                with ThreadPoolExecutor(max_workers=min(6, len(ids_by_namespace))) as executor:
                    futures = {
                        executor.submit(fetch_fn, ids=ids, namespace=namespace): namespace
                        for namespace, ids in ids_by_namespace.items()
                    }
                    for future in as_completed(futures):
                        namespace = futures[future]
                        try:
                            id_to_values = future.result() or {}
                        except Exception:
                            degraded = True
                            failed_fetch_namespaces.append(namespace)
                            continue
                        for source_id, emb in id_to_values.items():
                            if emb:
                                embedding_by_key[(namespace, source_id)] = emb
        if failed_fetch_namespaces:
            _mark_retrieval_degraded(
                debug_info=debug_info,
                stage=stage_rerank_fetch,
                details={"failed_namespaces": failed_fetch_namespaces},
            )

        if embedding_by_key:
            next_results: List[RetrievalItem] = []
            for item in results_local:
                namespace = item.vector_namespace or item.doc_id
                emb = embedding_by_key.get((namespace, item.source_id))
                if emb is not None:
                    next_results.append(
                        RetrievalItem(
                            source_id=item.source_id,
                            doc_id=item.doc_id,
                            filename=item.filename,
                            file_url=item.file_url,
                            source_url=item.source_url,
                            text=item.text,
                            score=item.score,
                            page_start=item.page_start,
                            page_end=item.page_end,
                            section_title=item.section_title,
                            route=item.route,
                            vector_namespace=item.vector_namespace,
                            embedding=emb,
                        )
                    )
                else:
                    next_results.append(item)
            results_local = next_results
        _stage_end(on_event, stage_rerank_fetch, int((perf_counter() - t_fetch) * 1000))

        # Rerank deterministically with cosine(query, stored_vector).
        t_rerank = perf_counter()
        _stage_start(on_event, stage_rerank)
        reranked_local = rerank_items(
            query_embedding=retrieval_embedding_local,
            items=results_local,
            top_k=settings.rerank_top_k,
            candidate_k=settings.rerank_candidate_k,
        )
        _stage_end(on_event, stage_rerank, int((perf_counter() - t_rerank) * 1000))

        if debug_info is not None:
            debug_info["stage_order"].append(stage_rerank)
            debug_info["top_score"] = reranked_local[0].score if reranked_local else -1.0
            debug_info["low_confidence_threshold"] = settings.low_confidence_threshold
        return reranked_local, degraded

    reranked, retrieval_degraded = _retrieve_and_rerank(
        selected_metas_local=selected_metas,
        retrieval_query_local=retrieval_query,
        retrieval_embedding_local=retrieval_embedding,
        strong_doc_match=strong,
        stage_prefix="",
    )
    if not reranked:
        raise HTTPException(status_code=404, detail="No relevant context found.")

    # Strict rewrite policy: only retry once we see low confidence after retrieval/rerank.
    if reranked[0].score < settings.low_confidence_threshold and int(settings.doc_rewrite_max_attempts) > 0:
        rewritten_query = retrieval_query
        rewritten_embedding = retrieval_embedding
        reused_initial_rewrite = rewritten_query.strip() != query.strip()

        if not reused_initial_rewrite:
            t_rewrite = perf_counter()
            _stage_start(on_event, "rewrite_query")
            rewritten_query = rewrite_query_for_doc_selection(query, settings)
            rewritten_embedding = embed_texts([rewritten_query], settings)[0]
            _stage_end(on_event, "rewrite_query", int((perf_counter() - t_rewrite) * 1000))
        else:
            _emit(
                on_event,
                "info",
                {
                    "stage": "rewrite_query",
                    "skipped": True,
                    "reason": "reused_initial_rewrite",
                },
            )

        t_select_retry = perf_counter()
        _stage_start(on_event, "doc_selection_rewrite")
        retry_doc_ids, retry_strong, retry_retrieval_query, retry_retrieval_embedding = (
            select_docs_with_rewrite_retry(
                query=rewritten_query,
                query_embedding=rewritten_embedding,
                metas=metas,
                settings=settings,
                vector_store=vector_store,
                debug=None,
                allow_rewrite=False,
            )
        )
        _stage_end(on_event, "doc_selection_rewrite", int((perf_counter() - t_select_retry) * 1000))

        retry_selected_metas = [meta_by_id[d] for d in retry_doc_ids if d in meta_by_id]
        retry_reranked: List[RetrievalItem] = []
        retry_degraded = False
        if retry_selected_metas:
            retry_reranked, retry_degraded = _retrieve_and_rerank(
                selected_metas_local=retry_selected_metas,
                retrieval_query_local=retry_retrieval_query,
                retrieval_embedding_local=retry_retrieval_embedding,
                strong_doc_match=retry_strong,
                stage_prefix="rewrite_",
            )

        base_top = reranked[0].score if reranked else -1.0
        retry_top = retry_reranked[0].score if retry_reranked else -1.0
        used_rewrite_retry = bool(retry_reranked and retry_top > base_top)
        if used_rewrite_retry:
            selected_metas = retry_selected_metas
            reranked = retry_reranked
            strong = retry_strong
            retrieval_degraded = retry_degraded
        if debug_info is not None:
            debug_info["rewrite_retry"] = {
                "query_used": rewritten_query,
                "retry_selected_doc_ids": retry_doc_ids,
                "base_top_score": base_top,
                "retry_top_score": retry_top,
                "applied": used_rewrite_retry,
                "retry_degraded": retry_degraded,
                "reused_initial_rewrite": reused_initial_rewrite,
            }

    if reranked and reranked[0].score < settings.low_confidence_threshold:
        broadened_limit = min(
            max(2, len(selected_metas) + 1),
            max(2, min(4, len(metas))),
        )
        broadened_metas = _build_broadened_doc_candidates(
            metas=metas,
            already_selected_doc_ids=[m.doc_id for m in selected_metas],
            query=query,
            limit=broadened_limit,
        )
        broadened_doc_ids = [m.doc_id for m in broadened_metas]
        should_retry_broader = (
            len(broadened_doc_ids) > len(selected_metas)
            and broadened_doc_ids != [m.doc_id for m in selected_metas]
        )
        broadened_reranked: List[RetrievalItem] = []
        broadened_degraded = False
        if should_retry_broader:
            broadened_reranked, broadened_degraded = _retrieve_and_rerank(
                selected_metas_local=broadened_metas,
                retrieval_query_local=query,
                retrieval_embedding_local=query_embedding,
                strong_doc_match=False,
                stage_prefix="broad_",
            )

        base_top = reranked[0].score if reranked else -1.0
        broader_top = broadened_reranked[0].score if broadened_reranked else -1.0
        used_broader_retry = bool(broadened_reranked and broader_top > base_top)
        if used_broader_retry:
            selected_metas = broadened_metas
            reranked = broadened_reranked
            retrieval_degraded = broadened_degraded
        if debug_info is not None:
            debug_info["broad_retrieval_retry"] = {
                "candidate_doc_ids": broadened_doc_ids,
                "base_top_score": base_top,
                "retry_top_score": broader_top,
                "applied": used_broader_retry,
                "retry_degraded": broadened_degraded,
            }

    generation_enabled = bool(getattr(settings, "rag_generate_answers_enabled", True))

    # Context budget selection for either retrieval-only or grounded generation.
    t_context = perf_counter()
    _stage_start(on_event, "context_select")
    doc_labels = {m.doc_id: m.filename for m in selected_metas if m.filename}
    selected, context_items = select_context(
        items=reranked, settings=settings, doc_labels=doc_labels or None, query=query
    )
    _stage_end(on_event, "context_select", int((perf_counter() - t_context) * 1000))

    source_url_by_doc_id = {m.doc_id: getattr(m, "source_url", "") for m in selected_metas}

    if not generation_enabled:
        _emit(
            on_event,
            "info",
            {
                "stage": "generate",
                "skipped": True,
                "reason": "rag_generate_answers_disabled",
            },
        )
        payload = {
            "query": query,
            "chunks": _build_chunk_payload(selected),
            "selected_doc_ids": [m.doc_id for m in selected_metas],
            "route": _route_label(selected_metas),
            "used_context_count": len(selected),
            "debug": debug_info,
        }
        if not retrieval_degraded:
            response_cache.set(response_key, payload)
        logger.info(
            "chat_complete generation_enabled=%s selected_docs=%d used_context=%d elapsed_ms=%d",
            generation_enabled,
            len(selected_metas),
            len(selected),
            int((perf_counter() - started) * 1000),
        )
        return payload

    citations = _build_chunk_citations(
        items=selected,
        source_url_by_doc_id=source_url_by_doc_id,
    )

    t_generate = perf_counter()
    _stage_start(on_event, "generate")
    answer = generate_answer(
        query=query,  # answer should respond to the user's original question
        context_items=context_items,
        settings=settings,
    )
    answer = review_answer(
        query=query,
        draft_answer=answer,
        context_items=context_items,
        settings=settings,
    )
    _stage_end(on_event, "generate", int((perf_counter() - t_generate) * 1000))
    citations_for_answer = _filter_citations_for_answer(
        citations=citations,
        answer=answer,
        context_items=context_items,
        selected_items=selected,
    )

    payload = {
        "answer": answer,
        "citations": citations_for_answer,
        "chunks": None,
        "selected_doc_ids": [m.doc_id for m in selected_metas],
        "route": _route_label(selected_metas),
        "used_context_count": len(selected),
        "debug": debug_info,
    }
    if not retrieval_degraded:
        response_cache.set(response_key, payload)
    logger.info(
        "chat_complete strong_doc_match=%s selected_docs=%d used_context=%d elapsed_ms=%d",
        strong,
        len(selected_metas),
        len(selected),
        int((perf_counter() - started) * 1000),
    )
    return payload


def _route_label(metas: List[DocMeta]) -> str:
    has_tree = any(m.route == "tree" for m in metas)
    has_std = any(m.route != "tree" for m in metas)
    if has_tree and has_std:
        return "hybrid"
    return "tree" if has_tree else "standard"


def format_sse(event: str, data: Dict[str, object]) -> str:
    import json

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"


def stream_chat(
    *,
    query: str,
    debug_enabled: bool,
    settings: Settings,
    registry: DocRegistry,
    vector_store: PineconeVectorStore,
    retrieval_cache: JsonCache,
    response_cache: JsonCache,
    tree_dir,
):
    event_queue: "queue.Queue[Dict[str, object]]" = queue.Queue()
    done_marker: Dict[str, object] = {"event": "_done", "data": {}}

    def on_event(payload: Dict[str, object]) -> None:
        event_queue.put(payload)

    def run() -> None:
        try:
            payload = run_chat(
                query=query,
                debug_enabled=debug_enabled,
                settings=settings,
                registry=registry,
                vector_store=vector_store,
                retrieval_cache=retrieval_cache,
                response_cache=response_cache,
                tree_dir=tree_dir,
                on_event=on_event,
            )
            event_queue.put({"event": "result", "data": payload})
        except HTTPException as exc:
            event_queue.put(
                {"event": "error", "data": {"status_code": exc.status_code, "detail": exc.detail}}
            )
        except Exception as exc:  # pragma: no cover
            event_queue.put({"event": "error", "data": {"status_code": 500, "detail": str(exc)}})
        finally:
            event_queue.put(done_marker)

    threading.Thread(target=run, daemon=True).start()

    def generator():
        while True:
            try:
                payload = event_queue.get(timeout=float(settings.sse_heartbeat_seconds))
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if payload is done_marker or payload.get("event") == "_done":
                break
            yield format_sse(payload["event"], payload["data"])

    return generator
