"""Application-level tests: metadata, middleware order, error format, docs, shutdown cleanup."""

from __future__ import annotations

import logging

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app, create_app
from app.utils.audio import UPLOAD_TEMP_DIR, cleanup_temp_files

AUTH = "/api/v1/auth"


def test_app_metadata(client: TestClient) -> None:
    assert app.title == "Gender Voice Recognition API"
    assert app.description == "API for gender classification from voice"
    assert app.version == "1.0.0"
    assert app.docs_url == "/docs" and app.redoc_url == "/redoc"
    body = client.get("/health").json()
    assert body["version"] == "1.0.0" and body["status"] in ("ok", "degraded")
    root = client.get("/").json()
    assert root["docs"] == "/docs" and root["api"] == "/api/v1" and root["version"] == "1.0.0"


def test_docs_and_openapi(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == "Gender Voice Recognition API"
    assert {t["name"] for t in spec["tags"]} == {"system", "authentication", "predictions", "analytics"}
    for path, operations in spec["paths"].items():
        for method, op in operations.items():
            assert op.get("summary") or op.get("description"), f"{method.upper()} {path} lacks documentation"
    schemas = spec["components"]["schemas"]
    assert "examples" in schemas["UserRegister"] and "examples" in schemas["PredictionResponse"]
    assert "examples" in schemas["HealthResponse"] and "examples" in schemas["StatisticsResponse"]


def test_cors_configuration(client: TestClient) -> None:
    preflight = client.options(
        "/api/v1/predictions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, Content-Type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert preflight.headers["access-control-allow-credentials"] == "true"
    assert preflight.headers["access-control-max-age"] == "3600"
    allowed = {m.strip() for m in preflight.headers["access-control-allow-methods"].split(",")}
    assert allowed == {"GET", "POST", "DELETE", "OPTIONS"}
    assert "authorization" in preflight.headers["access-control-allow-headers"].lower()

    # A foreign origin gets no CORS grant
    other = client.options(
        "/health", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"}
    )
    assert "access-control-allow-origin" not in other.headers


def test_cors_headers_on_error_responses(client: TestClient) -> None:
    """CORS is the outermost middleware, so even 404/401 responses carry the headers."""
    r = client.get("/does-not-exist", headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 404
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert r.json() == {"detail": "Not Found"}

    r = client.get("/api/v1/auth/me", headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 401 and r.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_consistent_error_format(client: TestClient) -> None:
    # HTTPException -> string detail (and WWW-Authenticate preserved)
    r = client.get("/api/v1/auth/me")
    assert r.json() == {"detail": "Not authenticated"} and r.headers["www-authenticate"] == "Bearer"

    # Validation on auth routes -> 400 with a string detail
    r = client.post(f"{AUTH}/login", json={"email": "bad", "password": ""})
    assert r.status_code == 400 and isinstance(r.json()["detail"], str)

    # Validation elsewhere -> 422 with a string detail plus the field errors
    client.post(
        f"{AUTH}/register",
        json={
            "username": "mainuser",
            "email": "main@example.com",
            "password": "s3cretpassword",
            "password_confirm": "s3cretpassword",
        },
    )
    token = client.post(
        f"{AUTH}/login", json={"email": "main@example.com", "password": "s3cretpassword"}
    ).json()
    r = client.get(
        "/api/v1/me/predictions",
        params={"limit": "abc"},
        headers={"Authorization": f"Bearer {token['access_token']}"},
    )
    assert r.status_code == 422
    body = r.json()
    assert isinstance(body["detail"], str) and "limit" in body["detail"] and isinstance(body["errors"], list)


def test_global_exception_handler_hides_internals(caplog: pytest.LogCaptureFixture) -> None:
    boom = APIRouter()

    @boom.get("/__boom")
    async def explode() -> None:
        raise RuntimeError("secret internal failure")

    test_app = create_app(get_settings())
    test_app.include_router(boom)
    with caplog.at_level(logging.ERROR, logger="app"):
        with TestClient(test_app, raise_server_exceptions=False) as c:
            r = c.get("/__boom", headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error"}
    assert "secret internal failure" not in r.text
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert any("Unhandled error" in rec.message for rec in caplog.records)
    assert any(rec.exc_info for rec in caplog.records)  # traceback logged


def test_logging_middleware_filters_sensitive_paths(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="app"):
        client.post(f"{AUTH}/login?debug=1", json={"email": "x@example.com", "password": "nope-nope"})
        client.get("/health?probe=1")
    messages = [rec.getMessage() for rec in caplog.records if rec.name == "app"]
    login_lines = [m for m in messages if "/auth/login" in m]
    assert login_lines and all("[sensitive]" in m and "debug=1" not in m for m in login_lines)
    assert any("/health?probe=1" in m for m in messages)
    assert not any("nope-nope" in m for m in messages)


def test_configuration_logged_with_secrets_masked(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app"):
        with TestClient(create_app(get_settings())) as c:
            assert c.get("/health").status_code == 200
    config_lines = [
        rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("Configuration:")
    ]
    assert config_lines
    line = config_lines[0]
    assert get_settings().JWT_SECRET_KEY not in line and "***" in line
    assert "Startup complete" in " ".join(rec.getMessage() for rec in caplog.records)


def test_shutdown_removes_temp_files() -> None:
    UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    leftover = UPLOAD_TEMP_DIR / "upload_leftover.wav"
    leftover.write_bytes(b"x")
    with TestClient(create_app(get_settings())) as c:
        assert c.get("/").status_code == 200
    assert not leftover.exists()
    assert cleanup_temp_files() == 0
