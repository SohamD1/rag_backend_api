from __future__ import annotations

from functools import lru_cache

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


@lru_cache(maxsize=8)
def get_token_encoder(model: str):
    if tiktoken is None:
        raise RuntimeError("tiktoken is required for token-based chunking.")
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str, model: str) -> int:
    if not text:
        return 0
    enc = get_token_encoder(model)
    return len(enc.encode(text))


def truncate_to_tokens(text: str, model: str, max_tokens: int) -> str:
    """
    Deterministically truncate to at most max_tokens for the given model.
    Uses tiktoken for correctness; raises if tiktoken is unavailable.
    """
    if not text:
        return ""
    limit = int(max_tokens)
    if limit <= 0:
        return ""
    enc = get_token_encoder(model)
    toks = enc.encode(text)
    if len(toks) <= limit:
        return text
    return enc.decode(toks[:limit])
