from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import Depends, HTTPException, Request


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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes"}


def _require_token(*, req: Request, expected: Optional[str], kind: str) -> None:
    """
    If expected token is configured, require `Authorization: Bearer <token>`.
    If expected is unset/empty, allow all (local dev).
    """
    expected = (expected or "").strip()
    if not expected:
        return
    got = _get_bearer_token(req) or ""
    if not got or not hmac.compare_digest(got, expected):
        raise HTTPException(status_code=401, detail=f"Missing or invalid {kind} token")


def require_kb_app_token(req: Request) -> None:
    if not _bool_env("KB_REQUIRE_AUTH", False):
        return

    expected = (os.getenv("KB_APP_TOKEN") or "").strip()
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="KB_REQUIRE_AUTH is enabled but KB_APP_TOKEN is not set",
        )
    _require_token(req=req, expected=expected, kind="app")


def require_kb_admin_token(req: Request) -> None:
    if not _bool_env("KB_REQUIRE_AUTH", False):
        return

    expected = (os.getenv("KB_ADMIN_TOKEN") or "").strip()
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="KB_REQUIRE_AUTH is enabled but KB_ADMIN_TOKEN is not set",
        )
    _require_token(req=req, expected=expected, kind="admin")


KBAppAuthDep = Depends(require_kb_app_token)
KBAdminAuthDep = Depends(require_kb_admin_token)
