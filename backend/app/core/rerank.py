from __future__ import annotations

import math
from typing import List, Optional

from app.core.retrieval import RetrievalItem
from app.core.vector_math import cosine_similarity


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _semantic_score(query_embedding: List[float], item: RetrievalItem) -> float:
    if item.embedding is None:
        return _clamp01(float(item.score))
    return _clamp01(cosine_similarity(query_embedding, item.embedding))


def _structure_bonus(item: RetrievalItem) -> float:
    bonus = 0.0
    if item.section_title:
        bonus += 0.04
    page_span = max(1, int(item.page_end) - int(item.page_start) + 1)
    if page_span <= 1:
        bonus += 0.05
    elif page_span <= 3:
        bonus += 0.03
    else:
        bonus += 0.01
    if item.route == "tree":
        bonus += 0.02
    return bonus


def _base_candidate_score(query_embedding: List[float], item: RetrievalItem) -> float:
    semantic = _semantic_score(query_embedding, item)
    prior = _clamp01(float(item.score))
    return (0.8 * semantic) + (0.2 * prior) + _structure_bonus(item)


def _redundancy_penalty(candidate: RetrievalItem, selected: List[RetrievalItem]) -> float:
    if candidate.embedding is None or not selected:
        return 0.0
    max_sim = 0.0
    for chosen in selected:
        if chosen.embedding is None:
            continue
        max_sim = max(max_sim, _clamp01(cosine_similarity(candidate.embedding, chosen.embedding)))
    return max_sim


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

    weighted: List[tuple[float, RetrievalItem]] = [
        (_base_candidate_score(query_embedding, item), item) for item in candidates
    ]
    weighted.sort(key=lambda x: x[0], reverse=True)

    selected: List[RetrievalItem] = []
    selected_copy: List[RetrievalItem] = []
    remaining = list(weighted)
    limit = max(1, int(top_k))
    while remaining and len(selected) < limit:
        best_idx = 0
        best_score = None
        for idx, (base_score, item) in enumerate(remaining):
            redundancy = _redundancy_penalty(item, selected_copy)
            adjusted_score = base_score - (0.22 * redundancy)
            if best_score is None or adjusted_score > best_score:
                best_score = adjusted_score
                best_idx = idx
        base_score, item = remaining.pop(best_idx)
        redundancy = _redundancy_penalty(item, selected_copy)
        adjusted_score = base_score - (0.22 * redundancy)
        selected.append(
            RetrievalItem(
                source_id=item.source_id,
                doc_id=item.doc_id,
                filename=item.filename,
                file_url=item.file_url,
                source_url=item.source_url,
                text=item.text,
                score=float(adjusted_score),
                page_start=item.page_start,
                page_end=item.page_end,
                section_title=item.section_title,
                route=item.route,
                embedding=item.embedding,
            )
        )
        selected_copy.append(item)
    return selected
