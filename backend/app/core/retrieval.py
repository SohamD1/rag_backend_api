from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from app.adapters.pinecone_store import PineconeVectorStore
from app.config import Settings
from app.core.tree_search import select_tree_nodes_with_llm
from app.services.tree_index import load_headings


@dataclass(frozen=True)
class RetrievalItem:
    source_id: str
    doc_id: str
    text: str
    score: float
    page_start: int
    page_end: int
    section_title: Optional[str]
    route: str
    embedding: Optional[List[float]] = None


def retrieval_item_to_dict(item: RetrievalItem) -> Dict:
    return {
        "source_id": item.source_id,
        "doc_id": item.doc_id,
        "text": item.text,
        "score": item.score,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "section_title": item.section_title,
        "route": item.route,
    }


def retrieval_item_from_dict(payload: Dict) -> RetrievalItem:
    return RetrievalItem(
        source_id=str(payload.get("source_id", "")),
        doc_id=str(payload.get("doc_id", "")),
        text=str(payload.get("text", "")),
        score=float(payload.get("score", 0.0)),
        page_start=int(payload.get("page_start", 1)),
        page_end=int(payload.get("page_end", 1)),
        section_title=payload.get("section_title"),
        route=str(payload.get("route", "standard")),
    )


def retrieve_standard_for_doc(
    *,
    doc_id: str,
    query_embedding: List[float],
    settings: Settings,
    vector_store: PineconeVectorStore,
    top_k: Optional[int] = None,
) -> List[RetrievalItem]:
    limit = int(top_k or settings.retrieve_top_k)
    matches = vector_store.query(
        vector=query_embedding,
        top_k=limit,
        namespace=doc_id,
    )
    items: List[RetrievalItem] = []
    for match in matches:
        meta = match.get("metadata", {}) or {}
        items.append(
            RetrievalItem(
                source_id=str(meta.get("chunk_id") or match.get("id") or ""),
                doc_id=str(meta.get("doc_id") or doc_id),
                text=str(meta.get("text") or ""),
                score=float(match.get("score", 0.0)),
                page_start=int(meta.get("page_start", 1)),
                page_end=int(meta.get("page_end", 1)),
                section_title=meta.get("section_title"),
                route=str(meta.get("route") or "standard"),
            )
        )
    return items


def retrieve_tree_for_doc(
    *,
    doc_id: str,
    query: str,
    query_embedding: List[float],
    settings: Settings,
    vector_store: PineconeVectorStore,
    top_k: Optional[int] = None,
    debug: Optional[Dict] = None,
    tree_dir: Optional[Path] = None,
    index_version: Optional[str] = None,
) -> List[RetrievalItem]:
    limit = int(top_k or settings.retrieve_top_k)
    mode = str(getattr(settings, "tree_node_selection_mode", "vector_only") or "vector_only").strip().lower()
    if mode not in {"vector_only", "vector_then_llm", "llm_then_vector"}:
        mode = "vector_only"

    heading_levels = ["section", "subsection", "subsubsection"]
    heading_top_k = max(
        int(getattr(settings, "tree_section_top_k", 12)),
        int(getattr(settings, "tree_node_selection_candidate_k", 24)),
    )
    heading_top_k = max(1, heading_top_k)

    def query_headings() -> List[Dict]:
        heading_results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=min(6, len(heading_levels))) as executor:
            futures = [
                executor.submit(
                    vector_store.query,
                    vector=query_embedding,
                    top_k=int(heading_top_k),
                    namespace=doc_id,
                    filter={"level": {"$eq": level}},
                )
                for level in heading_levels
            ]
            for future in as_completed(futures):
                try:
                    heading_results.extend(future.result() or [])
                except Exception:
                    continue
        return sorted(heading_results, key=lambda x: x.get("score", 0.0), reverse=True)

    def vector_only() -> List[RetrievalItem]:
        heading_results = query_headings()

        section_ids: List[str] = []
        for match in heading_results:
            meta = match.get("metadata", {}) or {}
            section_id = meta.get("section_id") or meta.get("node_id") or match.get("id")
            if not section_id:
                continue
            section_id = str(section_id)
            if section_id in section_ids:
                continue
            section_ids.append(section_id)
            if len(section_ids) >= 4:
                break

        paragraph_results: List[Dict] = []
        if section_ids:
            per_section_top_k = max(limit * 5, 25)
            with ThreadPoolExecutor(max_workers=min(8, len(section_ids))) as executor:
                futures = [
                    executor.submit(
                        vector_store.query,
                        vector=query_embedding,
                        top_k=int(per_section_top_k),
                        namespace=doc_id,
                        filter={"level": {"$eq": "paragraph"}, "section_id": {"$eq": sid}},
                    )
                    for sid in section_ids
                ]
                for future in as_completed(futures):
                    try:
                        paragraph_results.extend(future.result() or [])
                    except Exception:
                        continue
        else:
            paragraph_results = vector_store.query(
                vector=query_embedding,
                top_k=limit,
                namespace=doc_id,
                filter={"level": {"$eq": "paragraph"}},
            )

        paragraph_results = sorted(paragraph_results, key=lambda x: x.get("score", 0.0), reverse=True)[:limit]

        items: List[RetrievalItem] = []
        for match in paragraph_results:
            meta = match.get("metadata", {}) or {}
            node_id = meta.get("node_id") or match.get("id") or ""
            items.append(
                RetrievalItem(
                    source_id=str(node_id),
                    doc_id=str(meta.get("doc_id") or doc_id),
                    text=str(meta.get("text") or ""),
                    score=float(match.get("score", 0.0)),
                    page_start=int(meta.get("page_start", 1)),
                    page_end=int(meta.get("page_end", 1)),
                    section_title=meta.get("breadcrumb") or meta.get("title"),
                    route="tree",
                )
            )

        if debug is not None:
            debug["tree"] = {
                "mode": "vector_only",
                "heading_count": len(heading_results),
                "section_id_count": len(section_ids),
                "paragraph_count": len(paragraph_results),
                "section_ids": section_ids,
            }
        return items

    # Hybrid modes need headings artifact to expand descendants and/or drive LLM selection.
    headings: Dict[str, Dict] = {}
    if tree_dir is not None and index_version:
        headings = load_headings(doc_id, tree_dir, index_version)

    def breadcrumb_for(nid: str) -> str:
        if not headings or nid not in headings:
            return ""
        titles: List[str] = []
        cur = nid
        seen: set[str] = set()
        while cur and cur not in seen:
            seen.add(cur)
            node = headings.get(cur) or {}
            t = str(node.get("title") or "").strip()
            if t:
                titles.append(t)
            cur = str(node.get("parent_id") or "")
            if not cur:
                break
        return " > ".join(reversed(titles))

    def descendants(root_ids: List[str]) -> List[str]:
        if not headings:
            return [i for i in root_ids if i]
        out: List[str] = []
        seen: set[str] = set()
        stack = [i for i in root_ids if i]
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            out.append(nid)
            kids = headings.get(nid, {}).get("child_ids") or []
            if isinstance(kids, list):
                for k in kids:
                    kk = str(k or "").strip()
                    if kk and kk not in seen:
                        stack.append(kk)
        return out

    def section_ids_for(node_ids: List[str]) -> List[str]:
        ids: List[str] = []
        seen: set[str] = set()
        for nid in node_ids:
            sid = None
            if headings:
                sid = headings.get(nid, {}).get("section_id")
            if sid:
                sid = str(sid)
            if not sid:
                continue
            if sid in seen:
                continue
            seen.add(sid)
            ids.append(sid)
            if len(ids) >= 4:
                break
        return ids

    def query_paragraphs_by_section(section_ids: List[str]) -> List[Dict]:
        if not section_ids:
            return vector_store.query(
                vector=query_embedding,
                top_k=limit,
                namespace=doc_id,
                filter={"level": {"$eq": "paragraph"}},
            )
        per_section_top_k = max(limit * 5, 25)
        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=min(8, len(section_ids))) as executor:
            futures = [
                executor.submit(
                    vector_store.query,
                    vector=query_embedding,
                    top_k=int(per_section_top_k),
                    namespace=doc_id,
                    filter={"level": {"$eq": "paragraph"}, "section_id": {"$eq": sid}},
                )
                for sid in section_ids
            ]
            for future in as_completed(futures):
                try:
                    results.extend(future.result() or [])
                except Exception:
                    continue
        return results

    def query_paragraphs_by_parent_ids(parent_ids: List[str]) -> List[Dict]:
        parent_ids = [p for p in (parent_ids or []) if p]
        if not parent_ids:
            return []
        # Keep Pinecone metadata filter size bounded.
        if len(parent_ids) > 200:
            return []
        per_query_top_k = max(limit * 5, 25)
        try:
            return vector_store.query(
                vector=query_embedding,
                top_k=int(per_query_top_k),
                namespace=doc_id,
                filter={"level": {"$eq": "paragraph"}, "parent_id": {"$in": parent_ids}},
            )
        except Exception:
            # Fall back to per-parent queries if $in isn't supported.
            results: List[Dict] = []
            with ThreadPoolExecutor(max_workers=min(8, len(parent_ids))) as executor:
                futures = [
                    executor.submit(
                        vector_store.query,
                        vector=query_embedding,
                        top_k=int(max(10, limit)),
                        namespace=doc_id,
                        filter={"level": {"$eq": "paragraph"}, "parent_id": {"$eq": pid}},
                    )
                    for pid in parent_ids[:50]
                ]
                for future in as_completed(futures):
                    try:
                        results.extend(future.result() or [])
                    except Exception:
                        continue
            return results

    def llm_select_from_candidates(candidates: List[Dict]) -> tuple[list[str], Optional[str]]:
        top_n = int(getattr(settings, "tree_node_selection_top_n", 6))
        return select_tree_nodes_with_llm(query=query, candidates=candidates, settings=settings, top_n=top_n)

    if mode == "vector_only":
        return vector_only()

    # Build candidate list for the LLM.
    candidates: List[Dict] = []
    if mode == "vector_then_llm":
        heading_results = query_headings()
        by_id: Dict[str, Dict] = {}
        for match in heading_results:
            meta = match.get("metadata", {}) or {}
            nid = str(meta.get("node_id") or match.get("id") or "").strip()
            if not nid:
                continue
            cand = {
                "node_id": nid,
                "level": str(meta.get("level") or ""),
                "title": str(meta.get("title") or ""),
                "breadcrumb": str(meta.get("breadcrumb") or ""),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "summary": str(meta.get("summary") or ""),
                "score": float(match.get("score", 0.0)),
            }
            prev = by_id.get(nid)
            if prev is None or cand["score"] > float(prev.get("score", 0.0)):
                by_id[nid] = cand
        candidates = sorted(by_id.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
        candidates = candidates[: max(1, int(getattr(settings, "tree_node_selection_candidate_k", 24)))]
    elif mode == "llm_then_vector":
        if not headings:
            return vector_only()
        # Prefer higher-level nodes; keep prompt bounded.
        all_nodes = list(headings.values())
        all_nodes.sort(key=lambda x: (x.get("page_start", 0), x.get("level", ""), x.get("node_id", "")))
        max_nodes = max(1, int(getattr(settings, "tree_node_selection_max_tree_nodes", 80)))
        # First take sections, then subsections, then the rest.
        ordered = []
        for level in ("section", "subsection", "subsubsection"):
            for n in all_nodes:
                if str(n.get("level")) == level:
                    ordered.append(n)
        if not ordered:
            ordered = all_nodes
        for n in ordered[:max_nodes]:
            nid = str(n.get("node_id") or "").strip()
            if not nid:
                continue
            candidates.append(
                {
                    "node_id": nid,
                    "level": str(n.get("level") or ""),
                    "title": str(n.get("title") or ""),
                    "breadcrumb": breadcrumb_for(nid),
                    "page_start": n.get("page_start"),
                    "page_end": n.get("page_end"),
                    "summary": str(n.get("summary") or ""),
                }
            )

    selected_ids, thinking = llm_select_from_candidates(candidates)
    if not selected_ids:
        return vector_only()

    allowed_parents = descendants(selected_ids)
    paragraph_results = query_paragraphs_by_parent_ids(allowed_parents)
    used_section_ids: List[str] = []
    if not paragraph_results:
        used_section_ids = section_ids_for(selected_ids)
        paragraph_results = query_paragraphs_by_section(used_section_ids)

    paragraph_results = sorted(paragraph_results, key=lambda x: x.get("score", 0.0), reverse=True)[:limit]

    items: List[RetrievalItem] = []
    for match in paragraph_results:
        meta = match.get("metadata", {}) or {}
        node_id = meta.get("node_id") or match.get("id") or ""
        items.append(
            RetrievalItem(
                source_id=str(node_id),
                doc_id=str(meta.get("doc_id") or doc_id),
                text=str(meta.get("text") or ""),
                score=float(match.get("score", 0.0)),
                page_start=int(meta.get("page_start", 1)),
                page_end=int(meta.get("page_end", 1)),
                section_title=meta.get("breadcrumb") or meta.get("title"),
                route="tree",
            )
        )

    if debug is not None:
        debug["tree"] = {
            "mode": mode,
            "candidate_count": len(candidates),
            "selected_node_ids": selected_ids,
            "thinking": thinking,
            "allowed_parent_count": len(allowed_parents),
            "fallback_section_ids": used_section_ids,
            "paragraph_count": len(paragraph_results),
        }

    return items
