"""SQLAlchemy 2.0 ORM: engine, session factory, models, table creation and migrations.

Tables
------
users              accounts (username/email unique, bcrypt password hash)
predictions        one row per inference request (cascade-deleted with the user)
acoustic_features  the 22 specan measurements of a prediction (cascade-deleted with it)
model_metrics      benchmark metrics of trained models

Conventions
-----------
* ``predicted_gender`` stores the manifest's class names ``"male"`` / ``"female"``
  (lower case) so the database, the API and ``model_config.json`` agree.
* ``acoustic_features`` columns are snake_case versions of the feature names in
  ``docs/acoustic_features.md`` (``Q25`` -> ``q25``, ``sp.ent`` -> ``sp_ent``). All
  22 specan measurements are stored: the 20 model inputs plus ``duration`` and
  ``peakf``.
* Schema changes go through Alembic (``backend/alembic``); ``init_db()`` applies
  migrations at startup and only falls back to ``create_all`` when Alembic is
  unavailable.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Generator
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
    inspect,
    text,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from app.config import BACKEND_DIR, get_settings
from app.ml.features import FEATURE_NAMES
from app.utils.security import hash_password, verify_password

logger = logging.getLogger(__name__)

settings = get_settings()


def _connect_args(url: str) -> dict[str, Any]:
    if url.startswith("postgresql"):
        return {"connect_timeout": 3}  # bound connection attempts (seconds)
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


# Creating the engine does not open a connection; connections are made lazily.
engine = create_engine(
    settings.DATABASE_URL, pool_pre_ping=True, future=True, connect_args=_connect_args(settings.DATABASE_URL)
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

_probe_lock = threading.Lock()
_probe_cache: tuple[float, bool] = (0.0, False)  # (timestamp, result)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )

    def set_password(self, password: str) -> None:
        self.password_hash = hash_password(password)

    def check_password(self, password: str) -> bool:
        return verify_password(password, self.password_hash)

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r} email={self.email!r}>"


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("predicted_gender IN ('male', 'female')", name="ck_predictions_gender"),
        CheckConstraint(
            "confidence_score >= 0.0 AND confidence_score <= 1.0", name="ck_predictions_confidence"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Nullable so that anonymous predictions are possible until authentication is enforced.
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=True
    )

    audio_file_path: Mapped[str | None] = mapped_column(String(255))  # local path or S3 key
    audio_duration: Mapped[float | None] = mapped_column(Float)  # seconds
    predicted_gender: Mapped[str] = mapped_column(String(10), nullable=False)  # "male" | "female"
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0.0 .. 1.0
    probabilities: Mapped[dict | None] = mapped_column(JSON)  # {"male": p, "female": p}
    model_used: Mapped[str] = mapped_column(String(50), nullable=False, default="MLP")
    model_version: Mapped[str | None] = mapped_column(String(50))
    inference_time_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True, nullable=False
    )

    user: Mapped[User | None] = relationship(back_populates="predictions")
    acoustic_features: Mapped[list[AcousticFeatures]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} user_id={self.user_id} gender={self.predicted_gender!r} "
            f"confidence={self.confidence_score:.3f} model={self.model_used!r}>"
        )


def feature_column_name(feature: str) -> str:
    """Canonical feature name -> column name (``Q25`` -> ``q25``, ``sp.ent`` -> ``sp_ent``)."""
    return feature.lower().replace(".", "_")


#: All 22 specan measurements in specan order (20 model inputs + duration + peakf).
SPECAN_COLUMNS: tuple[str, ...] = (
    "duration",
    *FEATURE_NAMES[:12],  # meanfreq .. centroid
    "peakf",
    *FEATURE_NAMES[12:],  # meanfun .. modindx
)
FEATURE_COLUMN_MAP: dict[str, str] = {name: feature_column_name(name) for name in SPECAN_COLUMNS}


class AcousticFeatures(Base):
    __tablename__ = "acoustic_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), index=True, nullable=False
    )

    duration: Mapped[float | None] = mapped_column(Float)
    meanfreq: Mapped[float | None] = mapped_column(Float)
    sd: Mapped[float | None] = mapped_column(Float)
    median: Mapped[float | None] = mapped_column(Float)
    q25: Mapped[float | None] = mapped_column(Float)
    q75: Mapped[float | None] = mapped_column(Float)
    iqr: Mapped[float | None] = mapped_column(Float)
    skew: Mapped[float | None] = mapped_column(Float)
    kurt: Mapped[float | None] = mapped_column(Float)
    sp_ent: Mapped[float | None] = mapped_column(Float)
    sfm: Mapped[float | None] = mapped_column(Float)
    mode: Mapped[float | None] = mapped_column(Float)
    centroid: Mapped[float | None] = mapped_column(Float)
    peakf: Mapped[float | None] = mapped_column(Float)
    meanfun: Mapped[float | None] = mapped_column(Float)
    minfun: Mapped[float | None] = mapped_column(Float)
    maxfun: Mapped[float | None] = mapped_column(Float)
    meandom: Mapped[float | None] = mapped_column(Float)
    mindom: Mapped[float | None] = mapped_column(Float)
    maxdom: Mapped[float | None] = mapped_column(Float)
    dfrange: Mapped[float | None] = mapped_column(Float)
    modindx: Mapped[float | None] = mapped_column(Float)

    prediction: Mapped[Prediction] = relationship(back_populates="acoustic_features")

    @classmethod
    def from_features(cls, features: dict[str, float]) -> AcousticFeatures:
        """Build a row from a feature dict keyed by canonical names (``Q25``, ``sp.ent``, ...)."""
        values = {column: features.get(name) for name, column in FEATURE_COLUMN_MAP.items()}
        return cls(**{k: (None if v is None else float(v)) for k, v in values.items()})

    def to_dict(self, model_features_only: bool = False) -> dict[str, float | None]:
        """Features keyed by canonical names, in specan order (or only the 20 model inputs)."""
        names = FEATURE_NAMES if model_features_only else SPECAN_COLUMNS
        return {name: getattr(self, FEATURE_COLUMN_MAP[name]) for name in names}

    def __repr__(self) -> str:
        return f"<AcousticFeatures id={self.id} prediction_id={self.prediction_id} meanfun={self.meanfun}>"


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(
        String(50), index=True, nullable=False
    )  # "MLP", "RandomForest"...
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float | None] = mapped_column(Float)
    recall: Mapped[float | None] = mapped_column(Float)
    f1_score: Mapped[float | None] = mapped_column(Float)
    total_predictions: Mapped[int | None] = mapped_column(Integer)  # evaluation (test) set size
    training_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    model_version: Mapped[str | None] = mapped_column(String(20))
    inference_time_ms: Mapped[float | None] = mapped_column(Float)  # measured single-call latency
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<ModelMetrics id={self.id} model={self.model_name!r} version={self.model_version!r} "
            f"accuracy={self.accuracy:.4f}>"
        )


# ---------------------------------------------------------------------------
# Session and lifecycle helpers
# ---------------------------------------------------------------------------


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a request-scoped session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection(cache_seconds: float = 5.0) -> bool:
    """Return True if ``SELECT 1`` succeeds, False otherwise (never raises).

    The result is cached for ``cache_seconds`` so frequent health polls do not
    each pay for a failing connection attempt. Pass 0 to force a fresh probe.
    """
    global _probe_cache
    now = time.monotonic()
    with _probe_lock:
        ts, cached = _probe_cache
        if cache_seconds > 0 and now - ts < cache_seconds:
            return cached
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            result = True
        except SQLAlchemyError as exc:
            logger.warning("Database unavailable: %s", exc.__class__.__name__)
            result = False
        _probe_cache = (now, result)
        return result


def create_tables(bind=None) -> None:
    """``Base.metadata.create_all`` - creates missing tables without migrations."""
    Base.metadata.create_all(bind=bind or engine)


def run_migrations(database_url: str | None = None, revision: str = "head") -> None:
    """Apply Alembic migrations programmatically (``alembic upgrade head``)."""
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", (database_url or settings.DATABASE_URL).replace("%", "%%"))
    cfg.attributes["configure_logging"] = False  # leave the application's logging configuration intact
    command.upgrade(cfg, revision)


def init_db(use_migrations: bool = True) -> str:
    """Create/upgrade the schema at startup. Returns a short description of what happened.

    Prefers Alembic migrations (the versioned source of truth). Falls back to
    ``create_all`` when Alembic or its configuration is not available.
    """
    if use_migrations and (BACKEND_DIR / "alembic.ini").exists():
        try:
            run_migrations()
            return "migrations applied (alembic upgrade head)"
        except ImportError:
            logger.warning("Alembic not installed; falling back to create_all")
    create_tables()
    return "tables created (create_all)"


def table_names(bind=None) -> list[str]:
    return sorted(inspect(bind or engine).get_table_names())
