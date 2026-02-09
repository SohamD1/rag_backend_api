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


def select_docs_with_rewrite_retry(
    *,
    query: str,
    query_embedding: List[float],
    metas: List[DocMeta],
    settings: Settings,
    vector_store: PineconeVectorStore,
    debug: Optional[Dict[str, Any]] = None,
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
    for attempt in range(max(0, int(settings.doc_rewrite_max_attempts))):
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
    )
    _stage_end(on_event, "doc_selection", int((perf_counter() - t_select) * 1000))

    meta_by_id = {m.doc_id: m for m in metas}
    selected_metas = [meta_by_id[d] for d in selected_doc_ids if d in meta_by_id]
    if not selected_metas:
        raise HTTPException(status_code=404, detail="Selected documents not found in registry.")

    # Retrieval cache is keyed by rewritten query + selected doc versions.
    selected_versions = sorted(f"{m.doc_id}:{m.index_version}:{m.route}" for m in selected_metas)
    retrieval_key = make_cache_key(
        {
            "kind": "retrieval",
            "query": normalize_query(retrieval_query),
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

    results: List[RetrievalItem] = []
    cached_r = retrieval_cache.get(retrieval_key)
    if cached_r:
        _emit(on_event, "info", {"stage": "retrieval_cache", "cache_hit": True})
        results = [
            retrieval_item_from_dict(item)
            for item in (cached_r.value or {}).get("items", [])
        ]
        if debug_info is not None:
            debug_info["stage_order"].append("retrieve_cache")
            debug_info["retrieval"] = {"cache_hit": True, "count": len(results)}
    else:
        t_retrieve = perf_counter()
        _stage_start(on_event, "retrieve")
        retrieval_debug_by_doc: Dict[str, Dict[str, Any]] = {}

        def _retrieve_one(meta: DocMeta):
            local_debug: Dict[str, Any] = {}
            if meta.route == "tree":
                items = retrieve_tree_for_doc(
                    doc_id=meta.doc_id,
                    query=query,
                    query_embedding=retrieval_embedding,
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
                    query_embedding=retrieval_embedding,
                    settings=settings,
                    vector_store=vector_store,
                    top_k=settings.retrieve_top_k,
                )
            return meta.doc_id, items, local_debug

        with ThreadPoolExecutor(max_workers=min(6, max(1, len(selected_metas)))) as executor:
            futures = [executor.submit(_retrieve_one, m) for m in selected_metas]
            for future in as_completed(futures):
                try:
                    doc_id, items, local_debug = future.result()
                except Exception:
                    continue
                results.extend(items or [])
                if debug_info is not None and local_debug:
                    retrieval_debug_by_doc[doc_id] = local_debug
        _stage_end(on_event, "retrieve", int((perf_counter() - t_retrieve) * 1000))
        retrieval_cache.set(
            retrieval_key,
            {"items": [retrieval_item_to_dict(item) for item in results]},
        )
        if debug_info is not None:
            debug_info["stage_order"].append("retrieve")
            debug_info.setdefault("retrieval", {})["count"] = len(results)
            if retrieval_debug_by_doc:
                debug_info["retrieval_by_doc"] = retrieval_debug_by_doc

    if not results:
        raise HTTPException(status_code=404, detail="No relevant context found.")

    # Attach stored vector values for rerank candidates (bounded, grouped by namespace/doc_id).
    t_fetch = perf_counter()
    _stage_start(on_event, "rerank_fetch")
    ordered_for_fetch = sorted(results, key=lambda x: x.score, reverse=True)
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
        for item in results:
            emb = embedding_by_key.get((item.doc_id, item.source_id))
            if emb is not None:
                next_results.append(
                    RetrievalItem(
                        source_id=item.source_id,
                        doc_id=item.doc_id,
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
        results = next_results
    _stage_end(on_event, "rerank_fetch", int((perf_counter() - t_fetch) * 1000))

    # Rerank deterministically with cosine(query, stored_vector).
    t_rerank = perf_counter()
    _stage_start(on_event, "rerank")
    reranked = rerank_items(
        query_embedding=retrieval_embedding,
        items=results,
        top_k=settings.rerank_top_k,
        candidate_k=settings.rerank_candidate_k,
    )
    _stage_end(on_event, "rerank", int((perf_counter() - t_rerank) * 1000))

    if debug_info is not None:
        debug_info["stage_order"].append("rerank")
        debug_info["top_score"] = reranked[0].score if reranked else -1.0
        debug_info["low_confidence_threshold"] = settings.low_confidence_threshold

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

    t_generate = perf_counter()
    _stage_start(on_event, "generate")
    answer, summary = generate_answer_and_summary(
        query=query,  # answer should respond to the user's original question
        context_items=context_items,
        settings=settings,
    )
    _stage_end(on_event, "generate", int((perf_counter() - t_generate) * 1000))

    citations = []
    for item in selected:
        citations.append(
            {
                "doc_id": item.doc_id,
                "source_id": item.source_id,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "section_title": item.section_title,
            }
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
