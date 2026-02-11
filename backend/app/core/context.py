from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from app.config import Settings
from app.core.retrieval import RetrievalItem
from app.core.tokens import estimate_tokens, truncate_to_tokens
from app.core.vector_math import cosine_similarity


def _dedupe_by_text(items: List[RetrievalItem]) -> List[RetrievalItem]:
    seen: set[str] = set()
    out: List[RetrievalItem] = []
    for item in items:
        key = " ".join((item.text or "").split())
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _mmr_reorder(
    items: List[RetrievalItem],
    *,
    lambda_mult: float,
) -> List[RetrievalItem]:
    """
    Maximal Marginal Relevance:
    - reward relevance (item.score)
    - penalize redundancy (cosine(item.embedding, selected.embedding))
    """
    if not items:
        return []
    lam = float(lambda_mult)
    if lam < 0.0:
        lam = 0.0
    if lam > 1.0:
        lam = 1.0

    # If we don't have embeddings, MMR can't do anything useful.
    if sum(1 for i in items if i.embedding) < 2:
        return items

    remaining = list(items)
    selected: List[RetrievalItem] = [remaining.pop(0)]
    while remaining:
        best_idx = 0
        best_score = None
        for idx, item in enumerate(remaining):
            rel = float(item.score)
            max_sim = 0.0
            if item.embedding:
                for chosen in selected:
                    if chosen.embedding:
                        max_sim = max(
                            max_sim, cosine_similarity(item.embedding, chosen.embedding)
                        )
            mmr = (lam * rel) - ((1.0 - lam) * max_sim)
            if best_score is None or mmr > best_score:
                best_score = mmr
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return selected


def select_context(
    *,
    items: List[RetrievalItem],
    settings: Settings,
    doc_labels: Optional[Dict[str, str]] = None,
) -> Tuple[List[RetrievalItem], List[Dict]]:
    """
    Select top items under a strict token budget.
    Optionally reorders items (MMR) to reduce redundancy before applying the token budget.
    Returns (selected_items, context_items_for_generation).
    """
    items = [i for i in (items or []) if i.text]
    items = _dedupe_by_text(items)
    if getattr(settings, "context_use_mmr", False):
        items = _mmr_reorder(items, lambda_mult=float(getattr(settings, "context_mmr_lambda", 0.7)))

    selected: List[RetrievalItem] = []
    context_items: List[Dict] = []
    budget = int(settings.max_context_tokens)
    per_item_cap = int(getattr(settings, "context_max_item_tokens", 0) or 0)
    used = 0
    for item in items:
        text = item.text
        # Ensure one huge passage doesn't consume the entire budget.
        if per_item_cap > 0:
            try:
                if estimate_tokens(text, settings.openai_generation_model) > per_item_cap:
                    text = truncate_to_tokens(text, settings.openai_generation_model, per_item_cap)
            except Exception:
                # If tokenization fails for any reason, fall back to original text.
                text = item.text

        cost = estimate_tokens(text, settings.openai_generation_model)
        if selected and used + cost > budget:
            break
        doc_label = item.doc_id
        if doc_labels and item.doc_id in doc_labels:
            doc_label = doc_labels[item.doc_id]
        header = f"doc={doc_label} pages={item.page_start}-{item.page_end}"
        if item.section_title:
            header += f" section={item.section_title}"
        selected.append(item)
        context_items.append(
            {
                "header": header,
                "text": text,
                "doc_id": item.doc_id,
                "filename": item.filename,
                "page_start": item.page_start,
                "page_end": item.page_end,
            }
        )
        used += cost
    return selected, context_items
