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

