from fastapi import APIRouter

from app.api.security import KBAdminAuthDep
from app.config import settings


router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/health/deps", dependencies=[KBAdminAuthDep])
def health_dependencies():
    return {
        "ok": bool(
            settings.openai_api_key
            and settings.pinecone_api_key
            and settings.pinecone_index
        ),
        "checks": {
            "openai": {
                "configured": bool(settings.openai_api_key),
                "embedding_model": settings.openai_embedding_model,
                "embedding_dimensions": settings.openai_embedding_dimensions,
            },
            "pinecone": {
                "configured": bool(settings.pinecone_api_key and settings.pinecone_index),
                "index": settings.pinecone_index,
                "host": settings.pinecone_host,
            },
        },
    }
