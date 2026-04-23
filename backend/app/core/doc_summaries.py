from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib.parse import urlparse

from app.adapters.embeddings import embed_texts
from app.adapters.pinecone_store import PineconeVectorStore
from app.config import Settings


DOC_SUMMARY_KINDS: Tuple[str, ...] = ("profile", "headings", "keywords")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_HEADING_PREFIX_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[A-Z]|[IVXLC]+)[).:\-\s]+")
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def doc_summary_record_ids(doc_id: str) -> List[str]:
    if not doc_id:
        return []
    return [f"{doc_id}:{kind}" for kind in DOC_SUMMARY_KINDS]


def upsert_doc_summary_records(
    *,
    records: Sequence[Dict[str, Any]],
    settings: Settings,
    vector_store: PineconeVectorStore,
) -> None:
    items: List[Dict[str, Any]] = []
    for record in records or []:
        values = list(record.get("values") or [])
        record_id = str(record.get("id") or "").strip()
        if not record_id or not values:
            continue
        items.append(
            {
                "id": record_id,
                "values": values,
                "metadata": dict(record.get("metadata", {}) or {}),
            }
        )
    if not items:
        return
    vector_store.upsert(items, namespace=settings.doc_summary_namespace)


def delete_doc_summary_records(
    *,
    doc_id: str,
    settings: Settings,
    vector_store: PineconeVectorStore,
) -> None:
    ids = doc_summary_record_ids(doc_id)
    if not ids:
        return
    vector_store.delete_ids(ids, namespace=settings.doc_summary_namespace)


def _normalize_space(text: str) -> str:
    return " ".join((text or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _truncate_chars(text: str, limit: int) -> str:
    cleaned = _normalize_space(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        value = _normalize_space(item)
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _normalize_lexical(text: str) -> str:
    return _NON_WORD_RE.sub(" ", (text or "").lower()).strip()


def _tokenize_lexical(text: str) -> List[str]:
    raw = _normalize_lexical(text)
    if not raw:
        return []
    return [
        token
        for token in raw.split()
        if len(token) >= 2 and token not in _STOPWORDS
    ]


def _extract_source_url_terms(source_url: str) -> List[str]:
    try:
        parsed = urlparse(source_url or "")
    except Exception:
        return _tokenize_lexical(source_url)
    return _tokenize_lexical(" ".join(part for part in [parsed.netloc, parsed.path] if part))


def lexical_doc_score(query: str, fields: Sequence[str]) -> float:
    query_tokens = _tokenize_lexical(query)
    if not query_tokens:
        return 0.0
    haystack = " ".join(_normalize_space(part) for part in fields if part)
    normalized_haystack = _normalize_lexical(haystack)
    if not normalized_haystack:
        return 0.0

    hay_tokens = set(normalized_haystack.split())
    overlap = sum(1 for token in set(query_tokens) if token in hay_tokens)
    long_overlap = sum(1 for token in set(query_tokens) if len(token) >= 5 and token in hay_tokens)
    exact_phrase = _normalize_lexical(query)

    score = 0.0
    if exact_phrase and exact_phrase in normalized_haystack:
        score += 0.22
    score += 0.18 * (overlap / max(1, len(set(query_tokens))))
    score += min(0.10, 0.03 * long_overlap)
    return min(score, 0.45)


def extract_heading_candidates_from_texts(
    texts: Sequence[str],
    *,
    max_headings: int = 18,
) -> List[str]:
    candidates: List[str] = []
    for text in texts or []:
        for raw_line in (text or "").replace("\r", "\n").split("\n"):
            line = " ".join(raw_line.split()).strip(" -:\t")
            if not line:
                continue
            if len(line) < 4 or len(line) > 120:
                continue
            if len(line.split()) > 14:
                continue
            if line.count("|") >= 2:
                continue
            if re.fullmatch(r"[\d\W]+", line):
                continue
            normalized = _HEADING_PREFIX_RE.sub("", line).strip()
            if len(normalized) < 4:
                continue
            alpha_chars = sum(1 for ch in normalized if ch.isalpha())
            if alpha_chars < max(3, len(normalized) // 4):
                continue
            title_ratio = sum(1 for word in normalized.split() if word[:1].isupper()) / max(
                1, len(normalized.split())
            )
            looks_heading = (
                normalized.isupper()
                or title_ratio >= 0.6
                or bool(re.match(r"^\d+(?:\.\d+)*\s+\S+", line))
                or normalized.endswith(":")
            )
            if not looks_heading:
                continue
            candidates.append(normalized.rstrip(":"))
            if len(candidates) >= max_headings * 2:
                break
        if len(candidates) >= max_headings * 2:
            break
    return _dedupe_preserve(candidates)[:max_headings]


def build_doc_summary_texts(
    *,
    slug: str,
    filename: str,
    source_url: str,
    route: str,
    page_count: int,
    token_count: int,
    abstract_fragments: Sequence[str],
    headings: Sequence[str],
) -> Dict[str, str]:
    cleaned_headings = _dedupe_preserve(headings)[:16]
    abstract = " ".join(
        _truncate_chars(fragment, 280)
        for fragment in abstract_fragments
        if _normalize_space(fragment)
    ).strip()
    if not abstract:
        abstract = "No abstract extracted."

    keyword_weights: Counter[str] = Counter()
    for token in _tokenize_lexical(filename):
        keyword_weights[token] += 3
    for token in _tokenize_lexical(slug):
        keyword_weights[token] += 3
    for token in _extract_source_url_terms(source_url):
        keyword_weights[token] += 2
    for heading in cleaned_headings:
        for token in _tokenize_lexical(heading):
            keyword_weights[token] += 4
    for fragment in abstract_fragments:
        for token in _tokenize_lexical(fragment):
            keyword_weights[token] += 1

    ranked_keywords = [
        token
        for token, _score in sorted(
            keyword_weights.items(),
            key=lambda item: (-item[1], -len(item[0]), item[0]),
        )[:24]
    ]

    profile_lines = [
        "Document profile",
        f"Filename: {filename or slug or 'document'}",
        f"Slug: {slug or 'document'}",
        f"Source URL: {source_url or 'n/a'}",
        f"Route: {route}",
        f"Pages: {int(page_count)}",
        f"Estimated tokens: {int(token_count)}",
        f"Abstract: {abstract}",
    ]
    headings_lines = [
        "Document headings",
        f"Filename: {filename or slug or 'document'}",
        f"Title: {slug or filename or 'document'}",
    ]
    if cleaned_headings:
        headings_lines.append("Headings: " + " | ".join(cleaned_headings))
    else:
        headings_lines.append("Headings: none extracted")
    headings_lines.append(f"Abstract: {abstract}")
    keywords_lines = [
        "Document keywords",
        f"Filename: {filename or slug or 'document'}",
        f"Title: {slug or filename or 'document'}",
        "Keywords: "
        + (" | ".join(ranked_keywords) if ranked_keywords else "none extracted"),
    ]
    if cleaned_headings:
        keywords_lines.append("Key headings: " + " | ".join(cleaned_headings[:8]))

    return {
        "profile": "\n".join(profile_lines).strip(),
        "headings": "\n".join(headings_lines).strip(),
        "keywords": "\n".join(keywords_lines).strip(),
    }


def build_doc_summary_records(
    *,
    doc_id: str,
    slug: str,
    filename: str,
    source_url: str,
    route: str,
    page_count: int,
    token_count: int,
    index_version: str,
    summary_texts: Mapping[str, str],
    settings: Settings,
) -> List[Dict[str, Any]]:
    ordered: List[Tuple[str, str]] = []
    for kind in DOC_SUMMARY_KINDS:
        text = _normalize_space(str(summary_texts.get(kind) or ""))
        if text:
            ordered.append((kind, text))
    if not ordered:
        return []

    embeddings = embed_texts([text for _kind, text in ordered], settings)
    items: List[Dict[str, Any]] = []
    for (kind, text), embedding in zip(ordered, embeddings):
        items.append(
            {
                "id": f"{doc_id}:{kind}",
                "values": embedding,
                "metadata": {
                    "doc_id": doc_id,
                    "slug": slug,
                    "filename": filename,
                    "source_url": source_url,
                    "route": route,
                    "page_count": page_count,
                    "token_count": token_count,
                    "index_version": index_version,
                    "summary_kind": kind,
                    "summary_text": text,
                    "doc_summary_strategy": getattr(
                        settings, "doc_summary_strategy_version", "multi_vector_v1"
                    ),
                },
            }
        )
    return items


def upsert_doc_summaries(
    *,
    doc_id: str,
    slug: str,
    filename: str,
    source_url: str,
    route: str,
    page_count: int,
    token_count: int,
    index_version: str,
    summary_texts: Mapping[str, str],
    settings: Settings,
    vector_store: PineconeVectorStore,
) -> None:
    records = build_doc_summary_records(
        doc_id=doc_id,
        slug=slug,
        filename=filename,
        source_url=source_url,
        route=route,
        page_count=page_count,
        token_count=token_count,
        index_version=index_version,
        summary_texts=summary_texts,
        settings=settings,
    )
    upsert_doc_summary_records(records=records, settings=settings, vector_store=vector_store)


def query_doc_summaries(
    *,
    query_embedding: List[float],
    settings: Settings,
    vector_store: PineconeVectorStore,
    top_k: Optional[int] = None,
) -> List[Dict]:
    k = int(top_k or getattr(settings, "doc_summary_match_top_k", settings.doc_summary_top_k))
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
    lexical_scores: Optional[Mapping[str, float]] = None,
) -> Tuple[List[str], bool, Dict]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for match in matches or []:
        meta = match.get("metadata", {}) or {}
        doc_id = str(meta.get("doc_id") or match.get("id") or "").strip()
        if not doc_id:
            continue
        summary_kind = str(meta.get("summary_kind") or "profile").strip() or "profile"
        score = float(match.get("score", 0.0))
        query_variant = str(match.get("query_variant") or "raw")
        entry = grouped.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "kind_scores": {},
                "match_count": 0,
                "queries": set(),
                "summary_hits": [],
                "filename": str(meta.get("filename") or ""),
                "slug": str(meta.get("slug") or ""),
                "source_url": str(meta.get("source_url") or ""),
            },
        )
        entry["match_count"] += 1
        entry["queries"].add(query_variant)
        existing = float(entry["kind_scores"].get(summary_kind, 0.0))
        if score >= existing:
            entry["kind_scores"][summary_kind] = score
        entry["summary_hits"].append(
            {
                "kind": summary_kind,
                "score": score,
                "query_variant": query_variant,
                "summary_text": str(meta.get("summary_text") or ""),
            }
        )

    lexical_scores = dict(lexical_scores or {})
    ranked: List[Dict[str, Any]] = []
    for doc_id, entry in grouped.items():
        kind_scores = sorted(
            (float(value) for value in (entry.get("kind_scores") or {}).values()),
            reverse=True,
        )
        base_score = sum(kind_scores[:2])
        multi_kind_bonus = 0.04 if len(kind_scores) >= 2 else 0.0
        lexical_score = float(lexical_scores.get(doc_id, 0.0))
        aggregate_score = base_score + multi_kind_bonus + lexical_score
        ranked.append(
            {
                "doc_id": doc_id,
                "aggregate_score": aggregate_score,
                "best_vector_score": kind_scores[0] if kind_scores else 0.0,
                "multi_kind_bonus": multi_kind_bonus,
                "lexical_score": lexical_score,
                "matched_kinds": sorted((entry.get("kind_scores") or {}).keys()),
                "kind_scores": dict(sorted((entry.get("kind_scores") or {}).items())),
                "match_count": int(entry.get("match_count") or 0),
                "query_variants": sorted(entry.get("queries") or []),
            }
        )

    for doc_id, lexical_score in lexical_scores.items():
        if doc_id in grouped or lexical_score <= 0:
            continue
        ranked.append(
            {
                "doc_id": doc_id,
                "aggregate_score": float(lexical_score),
                "best_vector_score": 0.0,
                "multi_kind_bonus": 0.0,
                "lexical_score": float(lexical_score),
                "matched_kinds": [],
                "kind_scores": {},
                "match_count": 0,
                "query_variants": [],
            }
        )

    ranked = sorted(
        ranked,
        key=lambda item: (
            float(item["aggregate_score"]),
            float(item["best_vector_score"]),
            len(item["matched_kinds"]),
            item["doc_id"],
        ),
        reverse=True,
    )

    debug = {"ranked": ranked[:10]}
    if not ranked:
        return [], False, debug

    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None
    strong = bool(top1["best_vector_score"] >= settings.doc_strong_min_score)
    if strong:
        if top2 is not None and float(top2["aggregate_score"]) > 0:
            best_vector_lead = bool(
                float(top1["best_vector_score"]) >= float(top2["best_vector_score"])
            )
            strong = bool(
                top1["aggregate_score"] >= top2["aggregate_score"] * settings.doc_strong_min_ratio
                and best_vector_lead
            )
        else:
            strong = True

    if strong:
        return [top1["doc_id"]], True, {**debug, "strong": True}

    selected = [item["doc_id"] for item in ranked[: max(1, top_k_fallback)]]
    return selected, False, {**debug, "strong": False}
