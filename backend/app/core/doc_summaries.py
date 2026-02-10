from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.adapters.pinecone_store import PineconeVectorStore
from app.config import Settings


def _l2_normalize(vec: List[float]) -> List[float]:
    if not vec:
        return []
    norm = sum(v * v for v in vec) ** 0.5
    if norm <= 0:
        return []
    return [v / norm for v in vec]


def compute_centroid(embeddings: List[List[float]]) -> List[float]:
    if not embeddings:
        return []
    dim = len(embeddings[0])
    if dim <= 0:
        return []
    accum = [0.0] * dim
    count = 0
    for vec in embeddings:
        if not vec or len(vec) != dim:
            continue
        v = _l2_normalize(vec)
        if not v:
            continue
        for i, value in enumerate(v):
            accum[i] += value
        count += 1
    if count <= 0:
        return []
    centroid = [v / count for v in accum]
    return _l2_normalize(centroid)


def upsert_doc_centroid(
    *,
    doc_id: str,
    slug: str,
    filename: str,
    source_url: str,
    route: str,
    page_count: int,
    token_count: int,
    index_version: str,
    centroid: List[float],
    settings: Settings,
    vector_store: PineconeVectorStore,
) -> None:
    if not centroid:
        return
    metadata: Dict = {
        "doc_id": doc_id,
        "slug": slug,
        "filename": filename,
        "file_url": f"/api/v1/documents/{doc_id}/file",
        "source_url": source_url,
        "route": route,
        "page_count": page_count,
        "token_count": token_count,
        "index_version": index_version,
    }
    vector_store.upsert(
        [{"id": doc_id, "values": centroid, "metadata": metadata}],
        namespace=settings.doc_summary_namespace,
    )


def query_doc_summaries(
    *,
    query_embedding: List[float],
    settings: Settings,
    vector_store: PineconeVectorStore,
    top_k: Optional[int] = None,
) -> List[Dict]:
    k = int(top_k or settings.doc_summary_top_k)
    k = max(1, k)
    return vector_store.query(
        vector=query_embedding,
        top_k=k,
        namespace=settings.doc_summary_namespace,
    )


def select_doc_ids_from_matches(
    matches: List[Dict],
    *,
    settings: Settings,
    top_k_fallback: int = 3,
) -> Tuple[List[str], bool, Dict]:
    ranked: List[Dict] = []
    for match in matches or []:
        meta = match.get("metadata", {}) or {}
        doc_id = meta.get("doc_id") or match.get("id")
        if not doc_id:
            continue
        ranked.append({"doc_id": str(doc_id), "score": float(match.get("score", 0.0))})
    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)

    debug = {"ranked": ranked[:10]}
    if not ranked:
        return [], False, debug

    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None
    strong = bool(top1["score"] >= settings.doc_strong_min_score)
    if strong and top2 is not None and top2["score"] > 0:
        strong = bool(top1["score"] >= top2["score"] * settings.doc_strong_min_ratio)

    if strong:
        return [top1["doc_id"]], True, {**debug, "strong": True}

    selected = [item["doc_id"] for item in ranked[: max(1, top_k_fallback)]]
    return selected, False, {**debug, "strong": False}
