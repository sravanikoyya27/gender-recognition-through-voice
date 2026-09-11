"""ORM model tests on an in-memory SQLite database (no PostgreSQL required)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ml.features import FEATURE_NAMES
from app.models.database import (
    FEATURE_COLUMN_MAP,
    SPECAN_COLUMNS,
    AcousticFeatures,
    Base,
    ModelMetrics,
    Prediction,
    User,
    create_tables,
)

SAMPLE_FEATURES = {
    "duration": 2.5,
    "meanfreq": 0.18,
    "sd": 0.05,
    "median": 0.17,
    "Q25": 0.16,
    "Q75": 0.22,
    "IQR": 0.06,
    "skew": 15.2,
    "kurt": 270.0,
    "sp.ent": 0.54,
    "sfm": 0.18,
    "mode": 0.17,
    "centroid": 0.18,
    "peakf": 0.5,
    "meanfun": 0.12,
    "minfun": 0.11,
    "maxfun": 0.13,
    "meandom": 0.46,
    "mindom": 0.39,
    "maxdom": 0.56,
    "dfrange": 0.17,
    "modindx": 0.52,
}


@pytest.fixture
def db() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _record):  # SQLite enforces ON DELETE CASCADE only with this pragma
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    create_tables(bind=engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _user(db: Session, name: str = "alice") -> User:
    user = User(username=name, email=f"{name}@example.com")
    user.set_password("correct horse battery staple")
    db.add(user)
    db.commit()
    return user


def test_user_password_hashing(db: Session) -> None:
    user = _user(db)
    assert user.password_hash != "correct horse battery staple"
    assert user.password_hash.startswith("$2b$")
    assert user.check_password("correct horse battery staple")
    assert not user.check_password("wrong")
    assert user.created_at is not None and user.updated_at is not None
    assert "alice" in repr(user)


def test_user_unique_constraints(db: Session) -> None:
    _user(db, "bob")
    db.add(User(username="bob", email="other@example.com", password_hash="x"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    db.add(User(username="bob2", email="bob@example.com", password_hash="x"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_prediction_with_features_roundtrip(db: Session) -> None:
    user = _user(db)
    prediction = Prediction(
        user_id=user.id,
        audio_file_path="uploads/clip.wav",
        audio_duration=2.5,
        predicted_gender="female",
        confidence_score=0.93,
        probabilities={"male": 0.07, "female": 0.93},
        model_used="MLP",
        model_version="2026-09-11",
        inference_time_ms=6.4,
    )
    prediction.acoustic_features.append(AcousticFeatures.from_features(SAMPLE_FEATURES))
    db.add(prediction)
    db.commit()

    stored = db.get(Prediction, prediction.id)
    assert stored is not None and stored.user.username == "alice"
    assert stored.created_at is not None
    assert "female" in repr(stored)
    features = stored.acoustic_features[0]
    assert features.to_dict() == SAMPLE_FEATURES
    assert list(features.to_dict()) == list(SPECAN_COLUMNS)
    assert list(features.to_dict(model_features_only=True)) == list(FEATURE_NAMES)
    assert features.q25 == 0.16 and features.sp_ent == 0.54  # snake_case columns
    assert "prediction_id" in repr(features)


def test_feature_columns_cover_all_specan_measurements() -> None:
    assert len(SPECAN_COLUMNS) == 22
    assert set(FEATURE_NAMES) <= set(SPECAN_COLUMNS)
    assert {"duration", "peakf", "modindx"} <= set(SPECAN_COLUMNS)
    columns = {c.name for c in AcousticFeatures.__table__.columns}
    assert set(FEATURE_COLUMN_MAP.values()) <= columns
    assert FEATURE_COLUMN_MAP["Q25"] == "q25" and FEATURE_COLUMN_MAP["sp.ent"] == "sp_ent"


def test_cascade_delete_user_removes_predictions_and_features(db: Session) -> None:
    user = _user(db)
    prediction = Prediction(user_id=user.id, predicted_gender="male", confidence_score=0.8)
    prediction.acoustic_features.append(AcousticFeatures.from_features(SAMPLE_FEATURES))
    db.add(prediction)
    db.commit()
    assert db.query(AcousticFeatures).count() == 1

    db.delete(user)
    db.commit()
    assert db.query(Prediction).count() == 0
    assert db.query(AcousticFeatures).count() == 0


def test_anonymous_prediction_allowed(db: Session) -> None:
    db.add(Prediction(predicted_gender="male", confidence_score=0.6))
    db.commit()
    assert db.query(Prediction).filter(Prediction.user_id.is_(None)).count() == 1


@pytest.mark.parametrize(
    "gender, confidence", [("Male", 0.5), ("unknown", 0.5), ("male", 1.5), ("female", -0.1)]
)
def test_prediction_check_constraints(db: Session, gender: str, confidence: float) -> None:
    db.add(Prediction(predicted_gender=gender, confidence_score=confidence))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_model_metrics(db: Session) -> None:
    metrics = ModelMetrics(
        model_name="MLP",
        accuracy=0.9811,
        precision=0.9812,
        recall=0.9811,
        f1_score=0.9811,
        total_predictions=476,
        training_timestamp=datetime(2026, 9, 11, tzinfo=UTC),
        model_version="v1.0",
    )
    db.add(metrics)
    db.commit()
    assert "0.9811" in repr(metrics) and "MLP" in repr(metrics)


def test_indexes_and_tables_exist(db: Session) -> None:
    insp = inspect(db.get_bind())
    assert set(insp.get_table_names()) == {"users", "predictions", "acoustic_features", "model_metrics"}

    def indexed(table: str) -> set[str]:
        return {col for idx in insp.get_indexes(table) for col in idx["column_names"]}

    assert {"username", "email"} <= indexed("users")
    assert {"user_id", "created_at"} <= indexed("predictions")
    assert "prediction_id" in indexed("acoustic_features")
    assert "model_name" in indexed("model_metrics")
    unique = {idx["name"] for idx in insp.get_indexes("users") if idx["unique"]}
    assert len(unique) == 2
    assert set(Base.metadata.tables) == {"users", "predictions", "acoustic_features", "model_metrics"}
