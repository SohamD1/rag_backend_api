from __future__ import annotations

from typing import List

from app.config import Settings
from app.adapters.openai_client import get_openai_client


def embed_texts(texts: List[str], settings: Settings) -> List[List[float]]:
    if not texts:
        return []
    client = get_openai_client(settings)
    kwargs = {}
    if settings.openai_embedding_dimensions is not None:
        kwargs["dimensions"] = settings.openai_embedding_dimensions

    # Avoid sending arbitrarily large requests (can exceed provider limits).
    batch_size = max(1, int(getattr(settings, "openai_embedding_batch_size", 128)))
    out: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
            **kwargs,
        )
        out.extend([item.embedding for item in response.data])
    return out
