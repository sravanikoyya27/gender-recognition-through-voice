"""Smoke tests for application startup, health endpoints and middleware."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main as app_main


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200  # database (SQLite override) and model both available
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "connected" and body["model_loaded"] is True
    assert body["version"]
    assert "timestamp" in body


def test_health_is_503_when_the_database_is_unavailable(client: TestClient) -> None:
    with patch.object(app_main, "check_database_connection", return_value=False):
        response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded" and body["database"] == "unavailable"
    assert body["model_loaded"] is True
    # The probe result is what the prediction routes use to decide on persistence.
    assert client.app.state.database_available is False
    # Restore readiness for the other tests (the conftest patch returns True again).
    assert client.get("/health").status_code == 200
    assert client.app.state.database_available is True


def test_health_is_503_when_no_model_is_loaded(client: TestClient) -> None:
    inference = client.app.state.inference
    model = inference._model
    inference._model = None  # simulate missing artifacts
    try:
        response = client.get("/health")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "degraded" and body["model_loaded"] is False
        assert body["database"] == "connected"
    finally:
        inference._model = model
    assert client.get("/health").status_code == 200


def test_liveness_endpoint_is_always_200(client: TestClient) -> None:
    inference = client.app.state.inference
    model = inference._model
    inference._model = None
    try:
        with patch.object(app_main, "check_database_connection", return_value=False):
            response = client.get("/health/live")
    finally:
        inference._model = model
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_root_lists_entry_points(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["health"] == "/health"


def test_logging_middleware_adds_headers(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "test-123"})
    assert response.headers["X-Request-ID"] == "test-123"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_cors_allows_frontend_origin(client: TestClient) -> None:
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_openapi_includes_versioned_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/health" in paths and "/health/live" in paths
    assert "/api/v1/auth/login" in paths
    assert "/api/v1/predictions" in paths
