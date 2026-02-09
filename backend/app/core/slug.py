import re


def slugify(text: str, *, max_len: int = 60) -> str:
    """
    Deterministic, readable slug:
    - lowercase
    - spaces/dashes -> underscores
    - strip non [a-z0-9_]
    - collapse repeated underscores
    """
    raw = (text or "").strip().lower()
    if not raw:
        return "document"

    raw = re.sub(r"[\s\-]+", "_", raw)
    raw = re.sub(r"[^a-z0-9_]+", "", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    if not raw:
        return "document"
    if len(raw) > max_len:
        raw = raw[:max_len].rstrip("_")
    return raw or "document"

