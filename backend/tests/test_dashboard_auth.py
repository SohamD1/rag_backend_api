from __future__ import annotations

from app.api.dashboard_auth import (
    create_dashboard_token,
    totp_code,
    verify_dashboard_token,
    verify_totp_code,
)
from app.api.v1 import documents
from app.config import settings


def test_totp_verification_accepts_current_code():
    secret = "JBSWY3DPEHPK3PXP"
    code = totp_code(secret, for_time=1_700_000_000)

    assert verify_totp_code(code, secret, at_time=1_700_000_000)
    assert not verify_totp_code("000000", secret, at_time=1_700_000_000)


def test_dashboard_login_accepts_current_totp(client):
    old_totp = settings.dashboard_totp_secret
    old_token_secret = settings.dashboard_token_secret
    try:
        object.__setattr__(settings, "dashboard_totp_secret", "JBSWY3DPEHPK3PXP")
        object.__setattr__(settings, "dashboard_token_secret", "x" * 48)

        response = client.post(
            "/api/admin/login",
            json={"code": totp_code("JBSWY3DPEHPK3PXP")},
        )
    finally:
        object.__setattr__(settings, "dashboard_totp_secret", old_totp)
        object.__setattr__(settings, "dashboard_token_secret", old_token_secret)

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["token"]


def test_dashboard_token_expires():
    old_token_secret = settings.dashboard_token_secret
    old_ttl = settings.dashboard_token_ttl_seconds
    try:
        object.__setattr__(settings, "dashboard_token_secret", "x" * 48)
        object.__setattr__(settings, "dashboard_token_ttl_seconds", 60)

        token = create_dashboard_token(settings, now=1000)

        assert verify_dashboard_token(token, settings, now=1059)
        assert not verify_dashboard_token(token, settings, now=1061)
    finally:
        object.__setattr__(settings, "dashboard_token_secret", old_token_secret)
        object.__setattr__(settings, "dashboard_token_ttl_seconds", old_ttl)


def test_dashboard_login_rejects_invalid_totp_secret_cleanly(client):
    old_totp = settings.dashboard_totp_secret
    try:
        object.__setattr__(settings, "dashboard_totp_secret", "1234")

        response = client.post("/api/admin/login", json={"code": "123456"})
    finally:
        object.__setattr__(settings, "dashboard_totp_secret", old_totp)

    assert response.status_code == 500
    assert "Base32 setup key" in response.json()["detail"]


def test_dashboard_documents_require_token(client):
    response = client.get("/api/admin/documents")

    assert response.status_code == 401


def test_dashboard_documents_accept_bearer_token(client, monkeypatch):
    class EmptyRegistry:
        def list(self):
            return []

    old_token_secret = settings.dashboard_token_secret
    try:
        object.__setattr__(settings, "dashboard_token_secret", "x" * 48)
        monkeypatch.setattr(documents, "registry", EmptyRegistry())
        token = create_dashboard_token(settings)

        response = client.get(
            "/api/admin/documents",
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        object.__setattr__(settings, "dashboard_token_secret", old_token_secret)

    assert response.status_code == 200
    assert response.json() == {"docs": []}


def test_v1_documents_route_is_not_registered(client):
    response = client.get("/api/v1/documents")

    assert response.status_code == 404
