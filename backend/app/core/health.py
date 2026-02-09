from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Dict

from pinecone import Pinecone

from app.adapters.embeddings import embed_texts
from app.config import Settings


def _run_check(check: Callable[[], Dict[str, Any] | None]) -> Dict[str, Any]:
    start = perf_counter()
    try:
        info = check() or {}
    except Exception as exc:  # pragma: no cover
        return {
            "ok": False,
            "latency_ms": int((perf_counter() - start) * 1000),
            "detail": str(exc),
        }
    return {
        "ok": True,
        "latency_ms": int((perf_counter() - start) * 1000),
        "info": info,
    }


def check_dependencies(settings: Settings) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, Any]] = {}

    def openai_check() -> Dict[str, Any]:
        emb = embed_texts(["ping"], settings)[0]
        return {
            "embedding_model": settings.openai_embedding_model,
            "embedding_dimensions": len(emb),
            "requested_dimensions": settings.openai_embedding_dimensions,
        }

    def pinecone_check() -> Dict[str, Any]:
        client = Pinecone(api_key=settings.pinecone_api_key)
        if settings.pinecone_host:
            index = client.Index(settings.pinecone_index, host=settings.pinecone_host)
        else:
            index = client.Index(settings.pinecone_index)
        stats = index.describe_index_stats(
            _request_timeout=(5.0, float(settings.pinecone_timeout_seconds))
        )
        return {
            "index": settings.pinecone_index,
            "host": settings.pinecone_host,
            "index_dimension": getattr(stats, "dimension", None),
            "total_vector_count": getattr(stats, "total_vector_count", None),
        }

    checks["openai"] = _run_check(openai_check)
    checks["pinecone"] = _run_check(pinecone_check)

    if checks["openai"]["ok"] and checks["pinecone"]["ok"]:
        openai_dim = checks["openai"]["info"].get("embedding_dimensions")
        pinecone_dim = checks["pinecone"]["info"].get("index_dimension")
        if isinstance(openai_dim, int) and isinstance(pinecone_dim, int):
            checks["dimension_match"] = {
                "ok": openai_dim == pinecone_dim,
                "detail": f"OpenAI embedding dim={openai_dim} Pinecone index dim={pinecone_dim}",
            }

    overall_ok = all(result.get("ok") for result in checks.values())
    return {"ok": overall_ok, "checks": checks}

