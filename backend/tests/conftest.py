"""Shared pytest fixtures.

API tests run against the FastAPI app with the database dependency redirected to an
in-memory SQLite database, so they never touch the configured PostgreSQL instance.
The application lifespan runs once per session, so the real MLP artifacts are loaded
(``app.state.inference``) exactly as in production.

Fixtures
--------
client / test_client   TestClient with lifespan events (same object under both names)
test_engine            in-memory SQLite engine shared with the app
db_session             SQLAlchemy session bound to that engine
test_user              a registered user (dict with id, username, email, password)
valid_token            a JWT access token for ``test_user``
auth_headers_for_user  ``Authorization`` header built from ``valid_token``
test_audio_file        bytes of ``tests/fixtures/sample_audio.wav`` (3 s, 16 kHz, mono)
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import main as app_main
from app.main import app
from app.models.database import Base, get_db

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_AUDIO_PATH = FIXTURES_DIR / "sample_audio.wav"
TEST_PASSWORD = "s3cretpassword"

_user_counter = itertools.count(1)


@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _record):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def client(test_engine) -> Iterator[TestClient]:
    """TestClient with lifespan events running (startup probes DB and loads the model)."""
    TestingSession = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        # Persistence goes to the SQLite override. The /health route re-probes the configured
        # PostgreSQL and would switch persistence off when it is not running, so the probe is
        # pinned to "connected" for the test session (the lifespan already ran with the real probe).
        with patch.object(app_main, "check_database_connection", return_value=True):
            app.state.database_available = True
            yield c
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(scope="session")
def test_client(client: TestClient) -> TestClient:
    """Alias required by the Phase 5 specification (same client as ``client``)."""
    return client


@pytest.fixture
def db_session(test_engine) -> Iterator[Session]:
    """A session on the shared in-memory database (sees rows created through the API)."""
    session = Session(bind=test_engine, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_user(client: TestClient) -> dict:
    """Register a fresh user through the real endpoint. Returns id, username, email and password."""
    n = next(_user_counter)
    username = f"testuser{n}"
    email = f"{username}@example.com"
    r = client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "email": email,
            "password": TEST_PASSWORD,
            "password_confirm": TEST_PASSWORD,
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()["user"]
    return {"id": body["id"], "username": username, "email": email, "password": TEST_PASSWORD}


@pytest.fixture
def valid_token(client: TestClient, test_user: dict) -> str:
    """JWT access token obtained by logging ``test_user`` in."""
    credentials = {"email": test_user["email"], "password": test_user["password"]}
    r = client.post("/api/v1/auth/login", json=credentials)
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.fixture
def auth_headers_for_user(valid_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {valid_token}"}


@pytest.fixture(scope="session")
def test_audio_file() -> bytes:
    """The committed 3-second sample recording (regenerate with tests/fixtures/generate_sample_audio.py)."""
    if not SAMPLE_AUDIO_PATH.exists():
        from tests.fixtures.generate_sample_audio import main as generate

        generate()
    return SAMPLE_AUDIO_PATH.read_bytes()
