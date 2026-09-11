"""Smoke tests for application startup, health endpoint and middleware."""

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["database"] in {"connected", "unavailable"}
    assert isinstance(body["model_loaded"], bool)
    assert body["version"]
    assert "timestamp" in body


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
    assert "/health" in paths
    assert "/api/v1/auth/login" in paths
    assert "/api/v1/predictions" in paths
