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
    monkeypatch.delenv("KB_ENABLE_DOCS", raising=False)

    settings = Settings(app_env="development")

    assert settings.kb_require_auth is False
    assert settings.kb_enable_docs is True


def test_docs_default_off_in_production(monkeypatch):
    monkeypatch.delenv("KB_ENABLE_DOCS", raising=False)

    settings = Settings(app_env="production")

    assert settings.kb_enable_docs is False


def test_docs_can_be_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("KB_ENABLE_DOCS", "true")

    settings = Settings(app_env="production")

    assert settings.kb_enable_docs is True


def test_kb_require_auth_can_explicitly_opt_out(monkeypatch):
    monkeypatch.setenv("KB_REQUIRE_AUTH", "false")

    settings = Settings(app_env="production")

    assert settings.kb_require_auth is False
    assert settings.kb_require_auth_explicit is True


def test_validate_settings_fails_closed_without_tokens(monkeypatch):
    monkeypatch.delenv("KB_REQUIRE_AUTH", raising=False)
    settings = Settings(
        app_env="",
        kb_app_token=None,
        dashboard_totp_secret=None,
        dashboard_token_secret=None,
    )

    with pytest.raises(RuntimeError, match="Authentication is required"):
        validate_settings(settings)


def test_validate_settings_accepts_dashboard_admin_auth(monkeypatch):
    monkeypatch.delenv("KB_REQUIRE_AUTH", raising=False)
    settings = Settings(
        app_env="",
        kb_app_token="app-token",
        dashboard_totp_secret="JBSWY3DPEHPK3PXP",
        dashboard_token_secret="x" * 48,
    )

    validate_settings(settings)


def test_validate_settings_rejects_short_dashboard_token_secret(monkeypatch):
    monkeypatch.delenv("KB_REQUIRE_AUTH", raising=False)
    settings = Settings(
        app_env="",
        kb_app_token="app-token",
        dashboard_totp_secret="JBSWY3DPEHPK3PXP",
        dashboard_token_secret="short",
    )

    with pytest.raises(RuntimeError, match="DASHBOARD_TOKEN_SECRET must be at least 32"):
        validate_settings(settings)
