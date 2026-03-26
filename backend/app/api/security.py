from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Depends, HTTPException, Request

from app.config import settings


def _get_bearer_token(req: Request) -> Optional[str]:
    auth = req.headers.get("authorization") or ""
    if not auth:
        return None
    parts = auth.split(" ", 1)
    if len(parts) != 2:
        return None
    scheme, token = parts[0].strip().lower(), parts[1].strip()
    if scheme != "bearer" or not token:
        return None
    return token


def _require_token(*, req: Request, expected: Optional[str], kind: str) -> None:
    expected = (expected or "").strip()
    if not expected:
        raise HTTPException(
            status_code=500,
            detail=f"Authentication is required but the {kind} token is not configured",
        )
    got = _get_bearer_token(req) or ""
    if not got or not hmac.compare_digest(got, expected):
        raise HTTPException(status_code=401, detail=f"Missing or invalid {kind} token")


def require_kb_app_token(req: Request) -> None:
    if not settings.kb_require_auth:
        return
    _require_token(req=req, expected=settings.kb_app_token, kind="app")


def require_kb_admin_token(req: Request) -> None:
    if not settings.kb_require_auth:
        return
    _require_token(req=req, expected=settings.kb_admin_token, kind="admin")


KBAppAuthDep = Depends(require_kb_app_token)
KBAdminAuthDep = Depends(require_kb_admin_token)
