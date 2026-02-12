from __future__ import annotations

import logging
import queue
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
    query_doc_summaries,
    select_doc_ids_from_matches,
)
from app.core.generation import generate_answer_and_summary
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
    on_event({"event": event, "data": payload})


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


def _consolidate_citations(
    *,
    items: List[RetrievalItem],
    source_url_by_doc_id: Dict[str, str],
) -> List[Dict[str, Any]]:
    by_doc: Dict[str, Dict[str, Any]] = {}
    doc_order: List[str] = []

    for item in items:
        doc_id = str(item.doc_id or "").strip()
        if not doc_id:
            continue
        if doc_id not in by_doc:
            by_doc[doc_id] = {
                "doc_id": doc_id,
                "filename": item.filename,
                "file_url": item.file_url,
                "source_url": item.source_url or source_url_by_doc_id.get(doc_id, ""),
                "source_id": f"{doc_id}:consolidated",
                "page_start": int(item.page_start),
                "page_end": int(item.page_end),
                "section_title": item.section_title,
            }
            doc_order.append(doc_id)
            continue

        cur = by_doc[doc_id]
        cur["page_start"] = min(int(cur["page_start"]), int(item.page_start))
        cur["page_end"] = max(int(cur["page_end"]), int(item.page_end))
        if not cur.get("filename") and item.filename:
            cur["filename"] = item.filename
        if not cur.get("file_url") and item.file_url:
            cur["file_url"] = item.file_url
        if not cur.get("source_url"):
            cur["source_url"] = item.source_url or source_url_by_doc_id.get(doc_id, "")
        if not cur.get("section_title") and item.section_title:
            cur["section_title"] = item.section_title

    return [by_doc[doc_id] for doc_id in doc_order]


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
    # Pass 1: doc summaries with original query
    matches = query_doc_summaries(
        query_embedding=query_embedding, settings=settings, vector_store=vector_store
    )
    selected_ids, strong, info = select_doc_ids_from_matches(matches, settings=settings)
    if debug is not None:
        debug["doc_selection"] = {
            "pass": 1,
            "query_used": query,
            **info,
            "selected_doc_ids": selected_ids,
        }

    if strong:
        return selected_ids, True, query, query_embedding

    # Rewrite-and-retry (no query-shape heuristics)
    rewritten_query = query
    rewritten_embedding = query_embedding
    for attempt in range(max(0, int(settings.doc_rewrite_max_attempts)) if allow_rewrite else 0):
        rewritten_query = rewrite_query_for_doc_selection(query, settings)
        rewritten_embedding = embed_texts([rewritten_query], settings)[0]
        matches2 = query_doc_summaries(
            query_embedding=rewritten_embedding,
            settings=settings,
            vector_store=vector_store,
        )
        selected2, strong2, info2 = select_doc_ids_from_matches(matches2, settings=settings)
        if debug is not None:
            debug["doc_selection_retry"] = {
                "pass": 2,
                "attempt": attempt + 1,
                "query_used": rewritten_query,
                **info2,
                "selected_doc_ids": selected2,
            }
            debug["rewritten_query"] = rewritten_query
        if selected2:
            selected_ids = selected2
        strong = strong2
        break

    # If doc summaries are empty/stale, fall back to top 3 by recency.
    if not selected_ids:
        fallback = [m.doc_id for m in _sort_metas_for_fallback(metas)[:3]]
        if debug is not None:
            debug["doc_selection_fallback"] = {"selected_doc_ids": fallback}
        return fallback, False, query, query_embedding

    if strong and selected_ids:
        return [selected_ids[0]], True, rewritten_query, rewritten_embedding
    return selected_ids[:3], False, rewritten_query, rewritten_embedding


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
            "max_context_tokens": settings.max_context_tokens,
            "rerank_top_k": settings.rerank_top_k,
            "rerank_candidate_k": settings.rerank_candidate_k,
            "rewrite_model": settings.openai_rewrite_model,
            "rewrite_attempts": settings.doc_rewrite_max_attempts,
            "citation_mode": "doc_consolidated_v1",
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
        allow_rewrite=False,
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
    ) -> List[RetrievalItem]:
        stage_retrieve = f"{stage_prefix}retrieve"
        stage_rerank_fetch = f"{stage_prefix}rerank_fetch"
        stage_rerank = f"{stage_prefix}rerank"
        stage_retrieve_cache = f"{stage_prefix}retrieve_cache"
        debug_retrieval_key = "retrieval" if not stage_prefix else f"{stage_prefix}retrieval".rstrip("_")

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
                        debug=local_debug if debug_info else None,
                        tree_dir=tree_dir,
                        index_version=meta.index_version,
                    )
                else:
                    items = retrieve_standard_for_doc(
                        doc_id=meta.doc_id,
                        query_embedding=retrieval_embedding_local,
                        settings=settings,
                        vector_store=vector_store,
                        top_k=settings.retrieve_top_k,
                    )
                return meta.doc_id, items, local_debug

            with ThreadPoolExecutor(max_workers=min(6, max(1, len(selected_metas_local)))) as executor:
                futures = [executor.submit(_retrieve_one, m) for m in selected_metas_local]
                for future in as_completed(futures):
                    try:
                        doc_id, items, local_debug = future.result()
                    except Exception:
                        continue
                    results_local.extend(items or [])
                    if debug_info is not None and local_debug:
                        retrieval_debug_by_doc[doc_id] = local_debug
            _stage_end(on_event, stage_retrieve, int((perf_counter() - t_retrieve) * 1000))
            retrieval_cache.set(
                retrieval_key,
                {"items": [retrieval_item_to_dict(item) for item in results_local]},
            )
            if debug_info is not None:
                debug_info["stage_order"].append(stage_retrieve)
                debug_info.setdefault(debug_retrieval_key, {})["count"] = len(results_local)
                if retrieval_debug_by_doc:
                    key = "retrieval_by_doc" if not stage_prefix else f"{stage_prefix}retrieval_by_doc".rstrip("_")
                    debug_info[key] = retrieval_debug_by_doc

        if not results_local:
            return []

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
            return reranked_local

        # Attach stored vector values for rerank candidates (bounded, grouped by namespace/doc_id).
        t_fetch = perf_counter()
        _stage_start(on_event, stage_rerank_fetch)
        candidates = ordered_for_fetch[: max(0, int(settings.rerank_candidate_k))]
        ids_by_doc: Dict[str, List[str]] = {}
        for item in candidates:
            if item.doc_id and item.source_id:
                ids_by_doc.setdefault(item.doc_id, []).append(item.source_id)
        embedding_by_key: Dict[tuple[str, str], List[float]] = {}
        if ids_by_doc:
            with ThreadPoolExecutor(max_workers=min(6, len(ids_by_doc))) as executor:
                futures = {
                    executor.submit(vector_store.fetch, ids=ids, namespace=doc_id): doc_id
                    for doc_id, ids in ids_by_doc.items()
                }
                for future in as_completed(futures):
                    doc_id = futures[future]
                    try:
                        id_to_values = future.result() or {}
                    except Exception:
                        continue
                    for source_id, emb in id_to_values.items():
                        if emb:
                            embedding_by_key[(doc_id, source_id)] = emb

        if embedding_by_key:
            next_results: List[RetrievalItem] = []
            for item in results_local:
                emb = embedding_by_key.get((item.doc_id, item.source_id))
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
        return reranked_local

    reranked = _retrieve_and_rerank(
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
        t_rewrite = perf_counter()
        _stage_start(on_event, "rewrite_query")
        rewritten_query = rewrite_query_for_doc_selection(query, settings)
        rewritten_embedding = embed_texts([rewritten_query], settings)[0]
        _stage_end(on_event, "rewrite_query", int((perf_counter() - t_rewrite) * 1000))

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
        if retry_selected_metas:
            retry_reranked = _retrieve_and_rerank(
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
        if debug_info is not None:
            debug_info["rewrite_retry"] = {
                "query_used": rewritten_query,
                "retry_selected_doc_ids": retry_doc_ids,
                "base_top_score": base_top,
                "retry_top_score": retry_top,
                "applied": used_rewrite_retry,
            }

    if not reranked or reranked[0].score < settings.low_confidence_threshold:
        payload = {
            "answer": (
                "I could not find strong evidence in the uploaded documents. "
                "Please provide a more specific question or a keyword unique to the document you mean."
            ),
            "summary": None,
            "citations": [],
            "selected_doc_ids": [m.doc_id for m in selected_metas],
            "route": _route_label(selected_metas),
            "used_context_count": 0,
            "debug": debug_info,
        }
        response_cache.set(response_key, payload)
        return payload

    # Context budget selection + grounded generation.
    t_context = perf_counter()
    _stage_start(on_event, "context_select")
    doc_labels = {m.doc_id: m.filename for m in selected_metas if m.filename}
    selected, context_items = select_context(
        items=reranked, settings=settings, doc_labels=doc_labels or None
    )
    _stage_end(on_event, "context_select", int((perf_counter() - t_context) * 1000))

    source_url_by_doc_id = {m.doc_id: getattr(m, "source_url", "") for m in selected_metas}

    t_generate = perf_counter()
    _stage_start(on_event, "generate")
    answer, summary = generate_answer_and_summary(
        query=query,  # answer should respond to the user's original question
        context_items=context_items,
        settings=settings,
    )
    _stage_end(on_event, "generate", int((perf_counter() - t_generate) * 1000))

    citations = _consolidate_citations(
        items=selected,
        source_url_by_doc_id=source_url_by_doc_id,
    )

    payload = {
        "answer": answer,
        "summary": summary or None,
        "citations": citations,
        "selected_doc_ids": [m.doc_id for m in selected_metas],
        "route": _route_label(selected_metas),
        "used_context_count": len(selected),
        "debug": debug_info,
    }
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
