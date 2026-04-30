from __future__ import annotations

from app.api.dashboard_auth import create_dashboard_token
from app.config import settings


def test_health_check_remains_public(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dependency_health_requires_dashboard_auth(client):
    old_require_auth = settings.kb_require_auth
    old_token_secret = settings.dashboard_token_secret
    try:
        object.__setattr__(settings, "kb_require_auth", True)
        object.__setattr__(settings, "dashboard_token_secret", "x" * 48)
        token = create_dashboard_token(settings)

        unauthenticated = client.get("/api/health/deps")
        authenticated = client.get(
            "/api/health/deps",
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        object.__setattr__(settings, "kb_require_auth", old_require_auth)
        object.__setattr__(settings, "dashboard_token_secret", old_token_secret)

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200
