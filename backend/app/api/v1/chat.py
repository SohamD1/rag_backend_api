import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.adapters.pinecone_store import PineconeVectorStore
from app.api.schemas import ChatRequest, ChatResponse
from app.config import resolve_cache_dir, resolve_docs_dir, resolve_tree_dir, settings
from app.core.cache import JsonCache
from app.core.pipeline import run_chat, stream_chat
from app.storage.registry import DocRegistry


router = APIRouter()
logger = logging.getLogger(__name__)

DOCS_DIR = resolve_docs_dir(settings)
TREE_DIR = resolve_tree_dir(settings)
CACHE_DIR = resolve_cache_dir(settings)

vector_store = PineconeVectorStore(settings)
registry = DocRegistry(DOCS_DIR)
retrieval_cache = JsonCache(CACHE_DIR / "retrieval")
response_cache = JsonCache(CACHE_DIR / "response")


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    payload = run_chat(
        query=req.query,
        debug_enabled=bool(req.debug or settings.rag_debug),
        settings=settings,
        registry=registry,
        vector_store=vector_store,
        retrieval_cache=retrieval_cache,
        response_cache=response_cache,
        tree_dir=TREE_DIR,
        on_event=None,
    )
    return ChatResponse(**payload)


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    gen = stream_chat(
        query=req.query,
        debug_enabled=bool(req.debug or settings.rag_debug),
        settings=settings,
        registry=registry,
        vector_store=vector_store,
        retrieval_cache=retrieval_cache,
        response_cache=response_cache,
        tree_dir=TREE_DIR,
    )
    return StreamingResponse(gen(), media_type="text/event-stream")

