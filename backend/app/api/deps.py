from __future__ import annotations

from threading import Lock

from fastapi import HTTPException

from app.adapters.pinecone_store import PineconeVectorStore
from app.config import settings


_vector_store: PineconeVectorStore | None = None
_vector_store_lock = Lock()


def get_vector_store() -> PineconeVectorStore:
    """
    Lazily initialize Pinecone so app startup is not blocked by external
    dependency issues. This keeps /api/health responsive during deploys.
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    with _vector_store_lock:
        if _vector_store is not None:
            return _vector_store
        try:
            _vector_store = PineconeVectorStore(settings)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Vector store unavailable: {exc}") from exc
        return _vector_store
