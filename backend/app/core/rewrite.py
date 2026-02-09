from __future__ import annotations

from app.adapters.openai_client import get_openai_client
from app.config import Settings
from app.core.json_utils import extract_json_object


def rewrite_query_for_doc_selection(query: str, settings: Settings) -> str:
    """
    No query-shape heuristics. This is invoked only when doc selection is not confident.
    Output is used for doc-summary selection (and then retrieval).
    """
    text = (query or "").strip()
    if not text:
        return query

    system_prompt = (
        "You rewrite questions into a concise retrieval query for document selection.\n"
        "Goal: help identify which uploaded document contains the answer.\n"
        "Rules:\n"
        "- Do NOT answer the question.\n"
        "- Do NOT add facts not present in the question.\n"
        "- Extract key entities, product names, acronyms (expand if obvious), time ranges, and unique keywords.\n"
        "- Keep it short and keyword-dense.\n"
        "Return STRICT JSON: {\"rewritten_query\": \"...\"}."
    )
    user_prompt = f"Question:\n{text}\n\nJSON:"

    try:
        client = get_openai_client(settings)
        response = client.chat.completions.create(
            model=settings.openai_rewrite_model or settings.openai_generation_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
        payload = extract_json_object(raw)
        rewritten = str(payload.get("rewritten_query") or "").strip()
        return rewritten or query
    except Exception:
        return query

