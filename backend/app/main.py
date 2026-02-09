from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.v1.chat import router as chat_v1_router
from app.api.v1.documents import router as documents_v1_router


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
app.include_router(documents_v1_router, prefix="/api/v1")
app.include_router(chat_v1_router, prefix="/api/v1")

