from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from typing import Any

from fastapi import Depends, HTTPException, Request

from app.config import DASHBOARD_TOKEN_SECRET_MIN_LENGTH, Settings, settings


def _decode_totp_secret(secret: str) -> bytes:
    cleaned = "".join(str(secret or "").strip().upper().split())
    if not cleaned:
        raise ValueError("empty TOTP secret")
    padding = "=" * (-len(cleaned) % 8)
    try:
        return base64.b32decode(cleaned + padding, casefold=True)
    except binascii.Error as exc:
        raise ValueError("invalid TOTP secret") from exc


def totp_code(secret: str, *, for_time: int | None = None, step_seconds: int = 30) -> str:
    key = _decode_totp_secret(secret)
    counter = int((time.time() if for_time is None else for_time) // step_seconds)
    msg = counter.to_bytes(8, "big")
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    value = int.from_bytes(digest[offset : offset + 4], "big") & 0x7FFFFFFF
    return f"{value % 1_000_000:06d}"


def verify_totp_code(
    code: str,
    secret: str,
    *,
    at_time: int | None = None,
    window: int = 1,
    step_seconds: int = 30,
) -> bool:
    cleaned = "".join(ch for ch in str(code or "") if ch.isdigit())
    if len(cleaned) != 6:
        return False
    now = int(time.time() if at_time is None else at_time)
    for offset in range(-window, window + 1):
        candidate_time = now + (offset * step_seconds)
        if hmac.compare_digest(cleaned, totp_code(secret, for_time=candidate_time)):
            return True
    return False


def verify_dashboard_totp(code: str) -> None:
    secret = str(settings.dashboard_totp_secret or "").strip()
    if not secret:
        raise HTTPException(status_code=500, detail="Dashboard TOTP secret is not configured")
    try:
        valid_code = verify_totp_code(code, secret)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="Dashboard TOTP secret is invalid. Use a Base32 setup key, not a numeric test code.",
        ) from exc

    if not valid_code:
        raise HTTPException(status_code=401, detail="Invalid verification code")


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _dashboard_token_secret(s: Settings) -> str:
    secret = str(s.dashboard_token_secret or "").strip()
    if len(secret) < DASHBOARD_TOKEN_SECRET_MIN_LENGTH:
        raise HTTPException(
            status_code=500,
            detail="Dashboard token secret is not configured",
        )
    return secret


def create_dashboard_token(s: Settings, *, now: int | None = None) -> str:
    issued_at = int(time.time() if now is None else now)
    payload = {
        "sub": "admin",
        "iat": issued_at,
        "exp": issued_at + max(60, int(s.dashboard_token_ttl_seconds)),
    }
    payload_raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url_encode(payload_raw)
    signature = hmac.new(
        _dashboard_token_secret(s).encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{payload_b64}.{_b64url_encode(signature)}"


def verify_dashboard_token(token: str, s: Settings, *, now: int | None = None) -> bool:
    token = str(token or "").strip()
    if "." not in token:
        return False
    payload_b64, signature_b64 = token.split(".", 1)
    if not payload_b64 or not signature_b64:
        return False

    expected_signature = hmac.new(
        _dashboard_token_secret(s).encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    try:
        got_signature = _b64url_decode(signature_b64)
        payload: dict[str, Any] = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return False

    if not hmac.compare_digest(got_signature, expected_signature):
        return False
    if payload.get("sub") != "admin":
        return False
    return int(payload.get("exp") or 0) >= int(time.time() if now is None else now)


def require_dashboard_token(req: Request) -> None:
    auth = str(req.headers.get("Authorization") or "").strip()
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not verify_dashboard_token(token, settings):
        raise HTTPException(status_code=401, detail="Missing or invalid dashboard token")


DashboardAuthDep = Depends(require_dashboard_token)
