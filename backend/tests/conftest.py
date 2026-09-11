"""Shared pytest fixtures.

API tests run against the FastAPI app with the database dependency redirected to an
in-memory SQLite database, so they never touch the configured PostgreSQL instance.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.models.database import Base, get_db


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
        app.state.database_available = True  # persistence goes to the SQLite override
        yield c
    app.dependency_overrides.pop(get_db, None)
