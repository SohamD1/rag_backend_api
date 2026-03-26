from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import HTTPException, Request

from app.config import settings

# When the KB is called from friedmann-prod, it includes a stable advisor identity header.
# We only trust this header when the request is authenticated with the app token.
_ADVISOR_HEADER = "x-friedmann-user-id"
_ADVISOR_ID_RE = re.compile(r"^[A-Za-z0-9:_-]{1,128}$")


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _client_ip(req: Request) -> str:
    # Only trust proxy headers when explicitly enabled.
    if str(os.getenv("RAG_TRUST_X_FORWARDED_FOR") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        xff = (req.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if xff:
            return xff
    return (req.client.host if req.client else "") or "unknown"


def _bearer(req: Request) -> Optional[str]:
    auth = req.headers.get("authorization") or ""
    if not auth:
        return None
    parts = auth.split(" ", 1)
    if len(parts) != 2:
        return None
    if parts[0].strip().lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def _expected_kb_app_token() -> str:
    return settings.kb_app_token or ""


def _is_trusted_app_request(req: Request) -> bool:
    # Only trust identity headers when auth is enabled and the bearer token matches.
    if not settings.kb_require_auth:
        return False
    expected = _expected_kb_app_token()
    if not expected:
        return False
    got = _bearer(req) or ""
    if not got:
        return False
    return hmac.compare_digest(got, expected)


def _advisor_user_id(req: Request) -> Optional[str]:
    raw = (req.headers.get(_ADVISOR_HEADER) or "").strip()
    if not raw:
        return None
    # Defensive caps: avoid unbounded memory growth / pathological header values.
    if len(raw) > 128:
        return None
    if not _ADVISOR_ID_RE.match(raw):
        return None
    return raw


def _key(req: Request) -> str:
    # IMPORTANT: Do NOT accept arbitrary bearer tokens as a rate limit key.
    # If KB_REQUIRE_AUTH is off (dev) or the token is invalid, fall back to IP.
    if _is_trusted_app_request(req):
        advisor_id = _advisor_user_id(req)
        if advisor_id:
            return f"advisor:{advisor_id}"

        token = _bearer(req)
        if token:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            return f"token:{digest}"

    return f"ip:{_client_ip(req)}"


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class TokenBucketLimiter:
    def __init__(
        self,
        *,
        rpm: int,
        burst: int,
        max_buckets: int = 10_000,
        bucket_ttl_seconds: int = 3600,
    ) -> None:
        self.rpm = max(0, int(rpm))
        self.burst = max(1, int(burst))
        self.max_buckets = max(0, int(max_buckets))
        self.bucket_ttl_seconds = max(60, int(bucket_ttl_seconds))
        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}

    def _prune(self, now: float) -> None:
        if self.max_buckets <= 0:
            return
        if len(self._buckets) <= self.max_buckets:
            return

        cutoff = now - float(self.bucket_ttl_seconds)
        if cutoff > 0:
            stale = [k for k, b in self._buckets.items() if b.updated_at < cutoff]
            for k in stale:
                self._buckets.pop(k, None)

        if len(self._buckets) <= self.max_buckets:
            return

        # Still too many keys: drop the oldest to cap memory.
        extra = len(self._buckets) - self.max_buckets
        if extra <= 0:
            return
        oldest = sorted(self._buckets.items(), key=lambda kv: kv[1].updated_at)[:extra]
        for k, _ in oldest:
            self._buckets.pop(k, None)

    def allow(self, key: str) -> tuple[bool, int]:
        """
        Returns (allowed, retry_after_seconds).
        retry_after_seconds is 0 when allowed.
        """
        if self.rpm <= 0:
            return True, 0

        now = time.monotonic()
        refill_per_sec = self.rpm / 60.0
        with self._lock:
            # Memory safety: limit unbounded key growth.
            self._prune(now)

            b = self._buckets.get(key)
            if b is None:
                b = _Bucket(tokens=float(self.burst), updated_at=now)
                self._buckets[key] = b

            elapsed = max(0.0, now - b.updated_at)
            if elapsed > 0:
                b.tokens = min(float(self.burst), b.tokens + elapsed * refill_per_sec)
                b.updated_at = now

            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return True, 0

            needed = 1.0 - b.tokens
            retry = int(math.ceil(needed / refill_per_sec)) if refill_per_sec > 0 else 60
            return False, max(1, retry)


_chat_limiter: Optional[TokenBucketLimiter] = None
_admin_limiter: Optional[TokenBucketLimiter] = None


def chat_rate_limit(req: Request) -> None:
    global _chat_limiter
    if _chat_limiter is None:
        rpm = _get_int_env("RAG_CHAT_RPM", 120)
        burst = _get_int_env("RAG_CHAT_BURST", 30)
        max_buckets = _get_int_env("RAG_RATE_LIMIT_MAX_BUCKETS", 10_000)
        ttl = _get_int_env("RAG_RATE_LIMIT_BUCKET_TTL_SECONDS", 3600)
        _chat_limiter = TokenBucketLimiter(
            rpm=rpm, burst=burst, max_buckets=max_buckets, bucket_ttl_seconds=ttl
        )

    allowed, retry_after = _chat_limiter.allow(_key(req))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )


def admin_rate_limit(req: Request) -> None:
    global _admin_limiter
    if _admin_limiter is None:
        rpm = _get_int_env("RAG_ADMIN_RPM", 30)
        burst = _get_int_env("RAG_ADMIN_BURST", 10)
        max_buckets = _get_int_env("RAG_RATE_LIMIT_MAX_BUCKETS", 10_000)
        ttl = _get_int_env("RAG_RATE_LIMIT_BUCKET_TTL_SECONDS", 3600)
        _admin_limiter = TokenBucketLimiter(
            rpm=rpm, burst=burst, max_buckets=max_buckets, bucket_ttl_seconds=ttl
        )

    allowed, retry_after = _admin_limiter.allow(_key(req))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )

def make_rate_limit_dep(fn: Callable[[Request], None]) -> Callable[[Request], None]:
    # Convenience for `Depends(...)` call sites.
    return fn
