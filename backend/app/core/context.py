from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from app.config import Settings
from app.core.retrieval import RetrievalItem
from app.core.tokens import estimate_tokens, truncate_to_tokens
from app.core.vector_math import cosine_similarity


_NON_WORD_RE = re.compile(r"[^a-z0-9]+")
_BROAD_QUERY_PATTERNS = (
    "what is",
    "what are",
    "what counts as",
    "why ",
    "how ",
    "when ",
    "rights",
    "challenges",
    "definition",
    "examples",
    "categories",
    "included",
    "inventory",
    "besides",
)


def _dedupe_within_doc(items: List[RetrievalItem]) -> List[RetrievalItem]:
    """
    Keep exact duplicates out of a single document, but preserve the same
    passage when it appears in different docs as corroborating evidence.
    """
    seen: set[tuple[str, str]] = set()
    out: List[RetrievalItem] = []
    for item in items:
        key = " ".join((item.text or "").split())
        if not key:
            continue
        dedupe_key = (item.doc_id, key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
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


def _normalize_lexical(text: str) -> str:
    return _NON_WORD_RE.sub(" ", str(text or "").lower()).strip()


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in _normalize_lexical(text).split()
        if len(token) >= 3
    }


def _is_broad_query(query: str) -> bool:
    normalized = " ".join(str(query or "").lower().split())
    if len(normalized.split()) < 6:
        return False
    return any(pattern in normalized for pattern in _BROAD_QUERY_PATTERNS)


def _build_query_facets(query: str) -> List[Tuple[str, set[str]]]:
    normalized = " ".join(str(query or "").lower().split())
    facets: List[Tuple[str, set[str]]] = []

    def add(name: str, terms: set[str]) -> None:
        if not terms:
            return
        if any(existing_name == name for existing_name, _ in facets):
            return
        facets.append((name, terms))

    if "asset" in normalized:
        add("assets", {"asset", "assets", "property", "investments", "accounts"})
    if "liabilit" in normalized or "debt" in normalized:
        add("liabilities", {"liability", "liabilities", "debt", "debts", "mortgage", "mortgages"})
    if "definition" in normalized or "what is" in normalized or "what are" in normalized or "what counts as" in normalized:
        add("definition", {"definition", "defined", "means", "refers", "is", "are"})
    if "categor" in normalized:
        add("categories", {"category", "categories", "types", "class", "classes"})
    if "example" in normalized:
        add("examples", {"example", "examples", "include", "including", "such"})
    if "right" in normalized:
        add("rights", {"right", "rights", "access", "authority", "entitled"})
    if "challenge" in normalized:
        add("challenges", {"challenge", "challenges", "difficult", "difficulty", "problem", "problems", "restrict"})
    if "used" in normalized or "when" in normalized:
        add("uses", {"used", "use", "common", "typically", "when"})
    if "tax" in normalized:
        add("tax", {"tax", "taxes", "deemed", "disposition", "capital"})

    return facets


def _facet_hits(item: RetrievalItem, query: str) -> set[str]:
    facets = _build_query_facets(query)
    if not facets:
        return set()

    haystack_tokens = _tokenize(" ".join([str(item.section_title or ""), str(item.text or "")]))
    hits: set[str] = set()
    for name, terms in facets:
        if haystack_tokens.intersection(terms):
            hits.add(name)
    return hits


def select_context(
    *,
    items: List[RetrievalItem],
    settings: Settings,
    doc_labels: Optional[Dict[str, str]] = None,
    query: str = "",
) -> Tuple[List[RetrievalItem], List[Dict]]:
    """
    Select top items under a strict token budget.
    Optionally reorders items (MMR) to reduce redundancy before applying the token budget.
    Returns (selected_items, context_items_for_generation).
    """
    items = [i for i in (items or []) if i.text]
    items = _dedupe_within_doc(items)
    if getattr(settings, "context_use_mmr", False):
        items = _mmr_reorder(items, lambda_mult=float(getattr(settings, "context_mmr_lambda", 0.7)))

    selected: List[RetrievalItem] = []
    context_items: List[Dict] = []
    budget = int(settings.max_context_tokens)
    per_item_cap = int(getattr(settings, "context_max_item_tokens", 0) or 0)
    used = 0

    remaining = list(items)
    selected_doc_ids: set[str] = set()
    uncovered_facets = {name for name, _terms in _build_query_facets(query)}

    def prepare_item(item: RetrievalItem) -> tuple[str, int]:
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
        return text, cost

    def add_item(item: RetrievalItem, text: str, cost: int) -> None:
        nonlocal used
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
        selected_doc_ids.add(item.doc_id)

    # First, greedily cover distinct requested facets when the query is multi-part.
    if uncovered_facets:
        while uncovered_facets:
            best_index = None
            best_item: RetrievalItem | None = None
            best_text = ""
            best_cost = 0
            best_gain = 0
            for idx, item in enumerate(remaining):
                text, cost = prepare_item(item)
                if used + cost > budget:
                    continue
                hits = _facet_hits(item, query).intersection(uncovered_facets)
                if not hits:
                    continue
                gain = len(hits)
                if best_item is None:
                    best_index = idx
                    best_item = item
                    best_text = text
                    best_cost = cost
                    best_gain = gain
                    continue
                if gain > best_gain or (
                    gain == best_gain
                    and (
                        (item.doc_id not in selected_doc_ids and best_item.doc_id in selected_doc_ids)
                        or float(item.score) > float(best_item.score)
                    )
                ):
                    best_index = idx
                    best_item = item
                    best_text = text
                    best_cost = cost
                    best_gain = gain
            if best_item is None or best_index is None:
                break
            add_item(best_item, best_text, best_cost)
            remaining.pop(best_index)
            uncovered_facets.difference_update(_facet_hits(best_item, query))

    # Then fill the remaining budget, preferring new docs for broad questions.
    broad_query = _is_broad_query(query)
    ordered_remaining = sorted(
        remaining,
        key=lambda item: (
            1 if broad_query and item.doc_id not in selected_doc_ids else 0,
            float(item.score),
        ),
        reverse=True,
    )
    for item in ordered_remaining:
        text, cost = prepare_item(item)
        if used + cost > budget:
            continue
        add_item(item, text, cost)
    return selected, context_items
