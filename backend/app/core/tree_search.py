from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.adapters.openai_client import get_openai_client
from app.config import Settings
from app.core.json_utils import extract_json_object


def _clamp_int(value: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        n = int(value)
    except Exception:
        return default
    if n < min_value:
        return min_value
    if n > max_value:
        return max_value
    return n


def select_tree_nodes_with_llm(
    *,
    query: str,
    candidates: List[Dict[str, Any]],
    settings: Settings,
    top_n: int,
) -> Tuple[List[str], Optional[str]]:
    """
    Return (node_ids, thinking). If the model output can't be parsed, returns ([], None).
    """
    query = (query or "").strip()
    if not query or not candidates:
        return [], None

    top_n = _clamp_int(top_n, default=6, min_value=1, max_value=20)

    # Keep prompts bounded and deterministic.
    compact = []
    for c in candidates:
        node_id = str(c.get("node_id") or c.get("id") or "").strip()
        if not node_id:
            continue
        compact.append(
            {
                "node_id": node_id,
                "level": str(c.get("level") or ""),
                "title": str(c.get("title") or ""),
                "breadcrumb": str(c.get("breadcrumb") or ""),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "summary": str(c.get("summary") or ""),
            }
        )

    if not compact:
        return [], None

    system_prompt = (
        "You are a document navigation assistant.\n"
        "Given a user question and candidate nodes from a document tree, pick the node_ids most likely to contain the answer.\n"
        "Rules:\n"
        "- Do NOT answer the question.\n"
        "- Select up to top_n node_ids.\n"
        "- Prefer nodes whose titles/summaries directly match the query intent.\n"
        "- Return STRICT JSON only.\n"
        'Schema: {"thinking": "...", "node_list": ["node_id", ...]}\n'
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"top_n: {top_n}\n\n"
        f"Candidates:\n{compact}\n\n"
        "JSON:"
    )

    try:
        client = get_openai_client(settings)
        response = client.chat.completions.create(
            model=getattr(settings, "openai_tree_search_model", settings.openai_generation_model),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
        payload = extract_json_object(raw)
        thinking = str(payload.get("thinking") or "").strip() or None
        node_list = payload.get("node_list") or []
        if not isinstance(node_list, list):
            node_list = []
        out: List[str] = []
        seen: set[str] = set()
        for item in node_list:
            nid = str(item or "").strip()
            if not nid or nid in seen:
                continue
            seen.add(nid)
            out.append(nid)
            if len(out) >= top_n:
                break
        return out, thinking
    except Exception:
        return [], None

