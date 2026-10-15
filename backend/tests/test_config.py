"""Secret validation (Phase 6A): placeholders and weak values are refused in production."""

from __future__ import annotations

import secrets

import pytest

from app.config import _INSECURE_JWT_DEFAULT, Settings, is_weak_secret, looks_like_placeholder

STRONG = secrets.token_urlsafe(48)
# Tests pass the secret explicitly: a JWT_SECRET_KEY in the environment (CI, a developer .env)
# would otherwise override the built-in default and change what is being tested.

PLACEHOLDERS = [
    "change-me-in-production-use-a-long-random-string",  # built-in default
    "replace-with-a-long-random-string-of-32-plus-chars",  # backend/.env.example
    "your_secret_key_at_least_32_chars",  # infrastructure/.env.example
    "REPLACE-WITH-A-LONG-RANDOM-STRING-OF-32-PLUS-CHARS",  # case-insensitive
    "this-is-an-example-secret-value-that-is-long-enough",  # marker inside a long string
    "a" * 64,  # one repeated character
    "abcabcabcabcabcabcabcabcabcabcabcabc",  # too few distinct characters
    "short",  # under 32 characters
]


@pytest.mark.parametrize("value", PLACEHOLDERS)
def test_weak_secrets_are_detected(value: str) -> None:
    assert is_weak_secret(value)


def test_strong_secret_is_accepted() -> None:
    assert not is_weak_secret(STRONG)
    assert not looks_like_placeholder(STRONG)
    settings = Settings(ENVIRONMENT="production", JWT_SECRET_KEY=STRONG)
    assert settings.jwt_secret_is_weak is False


@pytest.mark.parametrize("value", [p for p in PLACEHOLDERS if len(p) >= 32])
def test_production_refuses_placeholder_secrets_without_logging_them(value: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        Settings(ENVIRONMENT="production", JWT_SECRET_KEY=value)
    message = str(excinfo.value)
    assert "JWT_SECRET_KEY" in message
    assert value not in message and value.lower() not in message.lower()


def test_development_allows_placeholder_but_flags_it() -> None:
    settings = Settings(ENVIRONMENT="development", JWT_SECRET_KEY=_INSECURE_JWT_DEFAULT)
    assert settings.jwt_secret_is_weak is True
    assert Settings(ENVIRONMENT="development", JWT_SECRET_KEY=STRONG).jwt_secret_is_weak is False


def test_short_secret_is_rejected_everywhere() -> None:
    with pytest.raises(ValueError, match="32 characters"):
        Settings(JWT_SECRET_KEY="too-short")


def test_startup_warns_about_weak_secret_without_printing_it(caplog: pytest.LogCaptureFixture) -> None:
    from app.main import _log_configuration

    settings = Settings(ENVIRONMENT="development", JWT_SECRET_KEY=_INSECURE_JWT_DEFAULT)
    with caplog.at_level("WARNING", logger="app"):
        _log_configuration(settings)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("JWT_SECRET_KEY" in w for w in warnings)
    assert all(settings.JWT_SECRET_KEY not in w for w in warnings)
    assert all(settings.JWT_SECRET_KEY not in r.getMessage() for r in caplog.records)
