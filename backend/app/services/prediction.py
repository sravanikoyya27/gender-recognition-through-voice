"""Prediction pipeline: temp file -> decode -> features -> scaler + model -> persist -> S3 -> cleanup.

Every step is timed. Persistence is skipped (with ``prediction_id`` null) when the
database is unavailable; a failing commit raises ``PersistenceError`` (-> HTTP 500).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.ml.features import FEATURE_NAMES, FeatureExtractionError, extract_all_features, features_to_vector
from app.ml.inference import InferenceService
from app.models.database import AcousticFeatures, Prediction
from app.services.analytics import invalidate_statistics
from app.services.storage import upload_to_s3
from app.utils.audio import DecodedAudio

logger = logging.getLogger("app.predictions")


class PersistenceError(RuntimeError):
    """The prediction ran but could not be stored."""


class InferenceError(RuntimeError):
    """The model failed to produce a prediction."""


# ---------------------------------------------------------------------------
# Running statistics (simple performance monitoring, logged periodically)
# ---------------------------------------------------------------------------


@dataclass
class PredictionStats:
    count: int = 0
    by_gender: dict[str, int] = field(default_factory=lambda: {"male": 0, "female": 0})
    confidence_sum: float = 0.0
    inference_ms_sum: float = 0.0
    features_ms_sum: float = 0.0
    total_ms_sum: float = 0.0
    log_every: int = 25
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self, gender: str, confidence: float, inference_ms: float, features_ms: float, total_ms: float
    ) -> None:
        with self._lock:
            self.count += 1
            self.by_gender[gender] = self.by_gender.get(gender, 0) + 1
            self.confidence_sum += confidence
            self.inference_ms_sum += inference_ms
            self.features_ms_sum += features_ms
            self.total_ms_sum += total_ms
            if self.count % self.log_every == 0:
                logger.info("Prediction stats: %s", self.summary())

    def summary(self) -> dict[str, Any]:
        n = max(self.count, 1)
        return {
            "count": self.count,
            "by_gender": dict(self.by_gender),
            "mean_confidence": round(self.confidence_sum / n, 4),
            "mean_inference_ms": round(self.inference_ms_sum / n, 2),
            "mean_feature_extraction_ms": round(self.features_ms_sum / n, 2),
            "mean_total_ms": round(self.total_ms_sum / n, 2),
        }


stats = PredictionStats()


def warm_up_feature_extraction() -> float:
    """Run the extractor once on a synthetic signal so the first request does not pay JIT/import costs.

    Returns the time taken in milliseconds.
    """
    start = time.perf_counter()
    sr = 16_000
    t = np.arange(int(sr * 1.5)) / sr
    signal = sum(np.sin(2 * np.pi * k * 140.0 * t) / k for k in range(1, 12)).astype(np.float32)
    try:
        extract_all_features(0.5 * signal / np.max(np.abs(signal)), sr)
    except FeatureExtractionError:  # pragma: no cover - the synthetic signal is always voiced
        pass
    elapsed = (time.perf_counter() - start) * 1000
    logger.info("Feature extraction warmed up in %.0f ms", elapsed)
    return elapsed


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def extract(audio: DecodedAudio) -> tuple[dict[str, float], float]:
    """All 22 specan measurements plus the extraction time in ms."""
    start = time.perf_counter()
    features = extract_all_features(audio.samples, audio.sample_rate)
    return features, (time.perf_counter() - start) * 1000


def infer(inference: InferenceService, features: dict[str, float]) -> dict[str, Any]:
    try:
        return inference.predict(features_to_vector(features))
    except Exception as exc:  # noqa: BLE001 - any model failure is a 500 with traceback logged
        logger.exception("Model inference failed")
        raise InferenceError(f"Model inference failed: {exc.__class__.__name__}") from exc


def persist(
    db: Session | None,
    *,
    user_id: int | None,
    audio_file_path: str | None,
    audio: DecodedAudio,
    features: dict[str, float],
    result: dict[str, Any],
) -> int | None:
    """Store Prediction + AcousticFeatures rows. Returns the prediction id (None when db is None)."""
    if db is None:
        return None
    try:
        row = Prediction(
            user_id=user_id,
            audio_file_path=audio_file_path,
            audio_duration=audio.duration_seconds,
            predicted_gender=result["predicted_gender"],
            confidence_score=result["confidence"],
            probabilities=result["probabilities"],
            model_used=result["model_used"],
            model_version=result["model_version"],
            inference_time_ms=result["inference_time_ms"],
        )
        row.acoustic_features.append(AcousticFeatures.from_features(features))
        db.add(row)
        db.commit()
        db.refresh(row)
        invalidate_statistics(user_id)  # cached per-user statistics are stale now
        return row.id
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Database error while storing prediction")
        raise PersistenceError("Database error while storing the prediction") from exc


def build_response(
    *,
    prediction_id: int | None,
    result: dict[str, Any],
    audio: DecodedAudio,
    features: dict[str, float],
    features_ms: float,
    total_ms: float,
    filename: str | None,
    audio_file_path: str | None,
    user_id: int | None,
) -> dict[str, Any]:
    model_features = {name: float(features[name]) for name in FEATURE_NAMES}
    return {
        "prediction_id": prediction_id,
        "gender": result["predicted_gender"],
        "predicted_gender": result["predicted_gender"],
        "confidence": result["confidence"],
        "confidence_percentage": round(result["confidence"] * 100, 2),
        "probabilities": result["probabilities"],
        "inference_time_ms": round(result["inference_time_ms"], 3),
        "feature_extraction_time_ms": round(features_ms, 3),
        "total_time_ms": round(total_ms, 3),
        "audio_duration": round(audio.duration_seconds, 3),
        "duration_seconds": round(audio.duration_seconds, 3),
        "sample_rate": audio.sample_rate,
        "audio_filename": filename,
        "audio_file_path": audio_file_path,
        "audio_features": model_features,
        "features": model_features,
        "model_used": result["model_used"],
        "model_version": result["model_version"],
        "user_id": user_id,
    }


def run_prediction(
    audio: DecodedAudio,
    inference: InferenceService,
    *,
    db: Session | None,
    user_id: int | None,
    filename: str | None,
    temp_path: Path | None,
    started_at: float,
) -> dict[str, Any]:
    """Full pipeline for one decoded recording.

    Raises FeatureExtractionError, InferenceError or PersistenceError.
    """
    features, features_ms = extract(audio)
    result = infer(inference, features)

    storage_key = upload_to_s3(temp_path, user_id, filename) if temp_path is not None else None
    audio_file_path = storage_key or filename
    prediction_id = persist(
        db, user_id=user_id, audio_file_path=audio_file_path, audio=audio, features=features, result=result
    )

    total_ms = (time.perf_counter() - started_at) * 1000
    stats.record(
        result["predicted_gender"], result["confidence"], result["inference_time_ms"], features_ms, total_ms
    )
    logger.info(
        "prediction id=%s user_id=%s gender=%s confidence=%.4f | features %.1f ms, "
        "inference %.2f ms, total %.1f ms | %s",
        prediction_id,
        user_id,
        result["predicted_gender"],
        result["confidence"],
        features_ms,
        result["inference_time_ms"],
        total_ms,
        filename,
    )
    return build_response(
        prediction_id=prediction_id,
        result=result,
        audio=audio,
        features=features,
        features_ms=features_ms,
        total_ms=total_ms,
        filename=filename,
        audio_file_path=audio_file_path,
        user_id=user_id,
    )


def run_batch(
    items: list[tuple[str | None, DecodedAudio, Path | None]],
    inference: InferenceService,
    *,
    db: Session | None,
    user_id: int | None,
) -> list[dict[str, Any]]:
    """Feature extraction per file, then one batched model call, then per-file persistence.

    Returns one dict per item: ``{"status": "ok", "result": ...}`` or ``{"status": "error", "error": ...}``.
    """
    started = time.perf_counter()
    extracted: list[tuple[int, dict[str, float], float]] = []
    outcomes: list[dict[str, Any]] = [{} for _ in items]
    for idx, (_filename, audio, _path) in enumerate(items):
        try:
            features, features_ms = extract(audio)
            extracted.append((idx, features, features_ms))
        except FeatureExtractionError as exc:
            outcomes[idx] = {"status": "error", "error": str(exc)}

    if extracted:
        matrix = np.vstack([features_to_vector(f) for _, f, _ in extracted])
        try:
            results = inference.predict_batch(matrix)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Batched model inference failed")
            raise InferenceError(f"Model inference failed: {exc.__class__.__name__}") from exc
        for (idx, features, features_ms), result in zip(extracted, results, strict=True):
            filename, audio, temp_path = items[idx]
            storage_key = upload_to_s3(temp_path, user_id, filename) if temp_path is not None else None
            audio_file_path = storage_key or filename
            prediction_id = persist(
                db,
                user_id=user_id,
                audio_file_path=audio_file_path,
                audio=audio,
                features=features,
                result=result,
            )
            total_ms = (time.perf_counter() - started) * 1000
            stats.record(
                result["predicted_gender"],
                result["confidence"],
                result["inference_time_ms"],
                features_ms,
                total_ms,
            )
            outcomes[idx] = {
                "status": "ok",
                "result": build_response(
                    prediction_id=prediction_id,
                    result=result,
                    audio=audio,
                    features=features,
                    features_ms=features_ms,
                    total_ms=total_ms,
                    filename=filename,
                    audio_file_path=audio_file_path,
                    user_id=user_id,
                ),
            }
    logger.info(
        "batch prediction user_id=%s files=%d ok=%d",
        user_id,
        len(items),
        sum(o.get("status") == "ok" for o in outcomes),
    )
    return outcomes
