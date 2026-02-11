import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from fastapi import Depends

from app.api.deps import get_vector_store
from app.api.schemas import ChatRequest, ChatResponse
from app.api.security import KBAppAuthDep
from app.api.rate_limit import chat_rate_limit, make_rate_limit_dep
from app.config import resolve_cache_dir, resolve_docs_dir, resolve_tree_dir, settings
from app.core.cache import JsonCache
from app.core.pipeline import run_chat, stream_chat
from app.storage.registry import DocRegistry


router = APIRouter(dependencies=[KBAppAuthDep, Depends(make_rate_limit_dep(chat_rate_limit))])
logger = logging.getLogger(__name__)

DOCS_DIR = resolve_docs_dir(settings)
TREE_DIR = resolve_tree_dir(settings)
CACHE_DIR = resolve_cache_dir(settings)

registry = DocRegistry(DOCS_DIR)
retrieval_cache = JsonCache(CACHE_DIR / "retrieval")
response_cache = JsonCache(CACHE_DIR / "response")


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if settings.log_payloads or settings.rag_debug or bool(req.debug):
        logger.info("chat_request %s", {"query": req.query, "debug": bool(req.debug)})
    payload = run_chat(
        query=req.query,
        debug_enabled=bool(req.debug or settings.rag_debug),
        settings=settings,
        registry=registry,
        vector_store=get_vector_store(),
        retrieval_cache=retrieval_cache,
        response_cache=response_cache,
        tree_dir=TREE_DIR,
        on_event=None,
    )
    if settings.log_chat_answer:
        logger.info(
            "chat_answer %s",
            {
                "answer": payload.get("answer", ""),
                "citations": payload.get("citations", []),
                "selected_doc_ids": payload.get("selected_doc_ids", []),
                "route": payload.get("route", ""),
                "used_context_count": payload.get("used_context_count", 0),
            },
        )
    elif settings.log_payloads or settings.rag_debug or bool(req.debug):
        # Log the full response for easier end-to-end debugging.
        logger.info("chat_response %s", payload)
    return ChatResponse(**payload)


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if settings.log_payloads or settings.rag_debug or bool(req.debug):
        logger.info("chat_stream_request %s", {"query": req.query, "debug": bool(req.debug)})
    gen = stream_chat(
        query=req.query,
        debug_enabled=bool(req.debug or settings.rag_debug),
        settings=settings,
        registry=registry,
        vector_store=get_vector_store(),
        retrieval_cache=retrieval_cache,
        response_cache=response_cache,
        tree_dir=TREE_DIR,
    )
    return StreamingResponse(gen(), media_type="text/event-stream")
