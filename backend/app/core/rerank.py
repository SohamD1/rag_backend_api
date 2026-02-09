from __future__ import annotations

from typing import List, Optional

from app.core.retrieval import RetrievalItem
from app.core.vector_math import cosine_similarity


def rerank_items(
    *,
    query_embedding: List[float],
    items: List[RetrievalItem],
    top_k: int,
    candidate_k: int,
) -> List[RetrievalItem]:
    if not items:
        return []
    ordered = sorted(items, key=lambda x: x.score, reverse=True)
    candidates = ordered[: max(0, int(candidate_k))] if candidate_k > 0 else ordered

    scored: List[tuple[float, RetrievalItem]] = []
    for item in candidates:
        score = item.score
        if item.embedding is not None:
            score = cosine_similarity(query_embedding, item.embedding)
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[RetrievalItem] = []
    for score, item in scored[: max(1, int(top_k))]:
        out.append(
            RetrievalItem(
                source_id=item.source_id,
                doc_id=item.doc_id,
                text=item.text,
                score=float(score),
                page_start=item.page_start,
                page_end=item.page_end,
                section_title=item.section_title,
                route=item.route,
                embedding=item.embedding,
            )
        )
    return out
