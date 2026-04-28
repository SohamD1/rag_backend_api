from __future__ import annotations

from app.config import settings


def test_health_check_remains_public(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dependency_health_requires_admin_auth(client):
    old_require_auth = settings.kb_require_auth
    old_admin_token = settings.kb_admin_token
    try:
        object.__setattr__(settings, "kb_require_auth", True)
        object.__setattr__(settings, "kb_admin_token", "admin-token")

        unauthenticated = client.get("/api/health/deps")
        authenticated = client.get(
            "/api/health/deps",
            headers={"Authorization": "Bearer admin-token"},
        )
    finally:
        object.__setattr__(settings, "kb_require_auth", old_require_auth)
        object.__setattr__(settings, "kb_admin_token", old_admin_token)

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200
