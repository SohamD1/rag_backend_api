import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.api.admin import router as admin_router
from app.api.health import router as health_router
from app.api.v1.chat import router as chat_v1_router
from app.config import ROOT_DIR, settings, validate_settings


def _configure_logging() -> None:
    """
    Uvicorn config doesn't always set a permissive level for non-uvicorn loggers.
    Ensure our app logs (logger.info/...) show up in the server terminal.
    """
    level_name = str(os.getenv("APP_LOG_LEVEL", "INFO")).upper().strip()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if root.level > level:
        root.setLevel(level)


_configure_logging()
validate_settings(settings)

app = FastAPI(
    title="RAG Backend API",
    version="1.0.0",
    docs_url="/docs" if settings.kb_enable_docs else None,
    redoc_url="/redoc" if settings.kb_enable_docs else None,
    openapi_url="/openapi.json" if settings.kb_enable_docs else None,
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; base-uri 'none'; form-action 'self'; "
            "frame-ancestors 'none'; img-src 'self' data:; "
            "script-src 'self'; style-src 'self'; connect-src 'self'",
        )
        if settings.app_env not in {"development", "dev", "local", "test"}:
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )
        return response


class UploadSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method == "POST" and request.url.path == "/api/admin/documents":
            raw_length = request.headers.get("content-length")
            if raw_length:
                try:
                    content_length = int(raw_length)
                except ValueError:
                    content_length = 0
                multipart_overhead = 1024 * 1024
                if content_length > max(1, int(settings.max_upload_bytes)) + multipart_overhead:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "PDF upload is too large."},
                    )
        return await call_next(request)


app.add_middleware(UploadSizeLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(chat_v1_router, prefix="/api/v1")

FRONTEND_DIR = ROOT_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/admin", StaticFiles(directory=FRONTEND_DIR, html=True), name="admin")


@app.get("/", include_in_schema=False)
def root() -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "RAG Backend API",
        "version": "1.0.0",
        "health_url": "/api/health",
    }
    if settings.kb_enable_docs:
        payload["docs_url"] = "/docs"
    return payload
