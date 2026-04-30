import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

app = FastAPI(title="RAG Backend API", version="1.0.0")

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
    return {
        "name": "RAG Backend API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/api/health",
    }
