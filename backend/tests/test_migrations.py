"""Alembic migration tests: always against a temporary SQLite file, and against PostgreSQL when reachable."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config

from alembic import command
from app.config import BACKEND_DIR, get_settings
from app.models.database import Base, run_migrations

EXPECTED_TABLES = {"users", "predictions", "acoustic_features", "model_metrics"}


def _alembic_config(url: str) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def _assert_schema_matches_models(url: str) -> None:
    engine = sa.create_engine(url)
    insp = sa.inspect(engine)
    assert EXPECTED_TABLES <= set(insp.get_table_names())
    for table in EXPECTED_TABLES:
        db_cols = {c["name"] for c in insp.get_columns(table)}
        model_cols = {c.name for c in Base.metadata.tables[table].columns}
        assert db_cols == model_cols, table
    with engine.connect() as conn:
        assert conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
    engine.dispose()


def test_migrations_upgrade_and_downgrade_sqlite(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'migrations.db'}"
    run_migrations(url)  # the same helper the app uses at startup
    _assert_schema_matches_models(url)

    command.downgrade(_alembic_config(url), "base")
    engine = sa.create_engine(url)
    assert not EXPECTED_TABLES & set(sa.inspect(engine).get_table_names())
    engine.dispose()


def test_migration_history_is_linear() -> None:
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(_alembic_config("sqlite://"))
    heads = script.get_heads()
    assert len(heads) == 1, f"multiple heads: {heads}"
    assert script.get_revision(heads[0]) is not None


@pytest.fixture
def postgres_temp_db() -> str:
    """A throw-away PostgreSQL database next to the configured one; skips when unreachable."""
    url = sa.engine.make_url(get_settings().DATABASE_URL)
    if url.get_backend_name() != "postgresql":
        pytest.skip("DATABASE_URL is not PostgreSQL")
    admin = sa.create_engine(
        url.set(database="postgres"), isolation_level="AUTOCOMMIT", connect_args={"connect_timeout": 3}
    )
    name = f"voice_gender_test_{uuid.uuid4().hex[:8]}"
    try:
        with admin.connect() as conn:
            conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    except sa.exc.OperationalError:
        pytest.skip("PostgreSQL not reachable")
    temp_url = url.set(database=name).render_as_string(hide_password=False)
    try:
        yield temp_url
    finally:
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def test_migrations_against_postgres(postgres_temp_db: str) -> None:
    run_migrations(postgres_temp_db)
    _assert_schema_matches_models(postgres_temp_db)

    engine = sa.create_engine(postgres_temp_db)
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO users (username, email, password_hash) VALUES ('pg', 'pg@example.com', 'hash')"
            )
        )
        conn.execute(
            sa.text(
                "INSERT INTO predictions (user_id, predicted_gender, confidence_score, model_used) "
                "VALUES ((SELECT id FROM users WHERE username='pg'), 'female', 0.9, 'MLP')"
            )
        )
        conn.execute(
            sa.text(
                "INSERT INTO acoustic_features (prediction_id, meanfun) "
                "VALUES ((SELECT max(id) FROM predictions), 0.21)"
            )
        )
        conn.execute(sa.text("DELETE FROM users WHERE username='pg'"))  # ON DELETE CASCADE
        assert conn.execute(sa.text("SELECT count(*) FROM acoustic_features")).scalar() == 0
        with pytest.raises(sa.exc.DBAPIError):
            conn.execute(
                sa.text("INSERT INTO predictions (predicted_gender, confidence_score) VALUES ('Male', 0.5)")
            )
    engine.dispose()
