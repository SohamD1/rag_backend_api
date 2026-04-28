from __future__ import annotations

import pytest

from app.config import Settings, validate_settings


def test_auth_defaults_required_when_app_env_is_omitted(monkeypatch):
    monkeypatch.delenv("KB_REQUIRE_AUTH", raising=False)

    settings = Settings(app_env="")

    assert settings.app_env == "production"
    assert settings.kb_require_auth is True


def test_development_env_is_explicit_local_auth_opt_out(monkeypatch):
    monkeypatch.delenv("KB_REQUIRE_AUTH", raising=False)

    settings = Settings(app_env="development")

    assert settings.kb_require_auth is False


def test_kb_require_auth_can_explicitly_opt_out(monkeypatch):
    monkeypatch.setenv("KB_REQUIRE_AUTH", "false")

    settings = Settings(app_env="production")

    assert settings.kb_require_auth is False
    assert settings.kb_require_auth_explicit is True


def test_validate_settings_fails_closed_without_tokens(monkeypatch):
    monkeypatch.delenv("KB_REQUIRE_AUTH", raising=False)
    settings = Settings(app_env="", kb_app_token=None, kb_admin_token=None)

    with pytest.raises(RuntimeError, match="Authentication is required"):
        validate_settings(settings)
