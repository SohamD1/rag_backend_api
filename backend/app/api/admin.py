from __future__ import annotations

from urllib.parse import urlparse

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.api.dashboard_auth import (
    DashboardAuthDep,
    create_dashboard_token,
    verify_dashboard_totp,
)
from app.api.rate_limit import admin_rate_limit, make_rate_limit_dep
from app.api.schemas import DocumentCreateResponse, DocumentListResponse
from app.api.v1.documents import create_document, delete_document, list_documents
from app.config import settings

router = APIRouter(prefix="/admin", tags=["admin-dashboard"])


class DashboardLoginRequest(BaseModel):
    code: str = Field(..., min_length=6, max_length=16)


@router.post("/login")
def login_dashboard(
    req: DashboardLoginRequest,
    _rate_limit: None = Depends(make_rate_limit_dep(admin_rate_limit)),
):
    verify_dashboard_totp(req.code)
    return {"ok": True, "token": create_dashboard_token(settings)}


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    dependencies=[
        DashboardAuthDep,
        Depends(make_rate_limit_dep(admin_rate_limit)),
    ],
)
def dashboard_list_documents():
    return list_documents()


@router.post(
    "/documents",
    response_model=DocumentCreateResponse,
    dependencies=[
        DashboardAuthDep,
        Depends(make_rate_limit_dep(admin_rate_limit)),
    ],
)
def dashboard_create_document(file: UploadFile = File(...), source_url: str = Form(...)):
    parsed = urlparse(str(source_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="source_url must be an http(s) URL")
    return create_document(file=file, source_url=source_url)


@router.delete(
    "/documents/{doc_id}",
    dependencies=[
        DashboardAuthDep,
        Depends(make_rate_limit_dep(admin_rate_limit)),
    ],
)
def dashboard_delete_document(doc_id: str):
    return delete_document(doc_id)
