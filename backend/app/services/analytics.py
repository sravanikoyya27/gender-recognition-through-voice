"""History, statistics and model-performance queries."""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.config import get_settings
from app.models.database import ModelMetrics, Prediction
from app.services.storage import delete_from_s3, is_s3_key
from app.utils.cache import get_cache, statistics_key

logger = logging.getLogger("app.analytics")

SORT_COLUMNS = {
    "date": Prediction.created_at,
    "confidence": Prediction.confidence_score,
    "duration": Prediction.audio_duration,
}
GENDER_FILTERS = {"all", "male", "female"}


class InvalidQueryError(ValueError):
    """Bad pagination / sort / filter parameters (-> HTTP 400)."""


class NotOwnerError(PermissionError):
    """The prediction belongs to another user (-> HTTP 403)."""


class StorageDeleteError(RuntimeError):
    """The audio object could not be removed from S3 (-> HTTP 500)."""


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def parse_sort(sort_by: str) -> tuple[str, bool]:
    """'date' | '-date' | 'confidence' | ... -> (column key, descending)."""
    descending = sort_by.startswith("-")
    key = sort_by.lstrip("-").lower()
    if key not in SORT_COLUMNS:
        raise InvalidQueryError(f"sort_by must be one of {sorted(SORT_COLUMNS)} (prefix '-' for descending)")
    return key, descending


def normalise_gender_filter(value: str | None) -> str:
    gender = (value or "all").strip().lower()
    if gender not in GENDER_FILTERS:
        raise InvalidQueryError("gender_filter must be 'Male', 'Female' or 'all'")
    return gender


def list_predictions(
    db: Session, user_id: int, *, limit: int, offset: int, sort_by: str, gender_filter: str | None
) -> tuple[int, list[Prediction]]:
    if limit < 1 or limit > 100:
        raise InvalidQueryError("limit must be between 1 and 100")
    if offset < 0:
        raise InvalidQueryError("offset must be >= 0")
    key, descending = parse_sort(sort_by)
    gender = normalise_gender_filter(gender_filter)

    base = select(Prediction).where(Prediction.user_id == user_id)
    if gender != "all":
        base = base.where(Prediction.predicted_gender == gender)
    total = db.execute(select(func.count()).select_from(base.subquery())).scalar_one()

    column = SORT_COLUMNS[key]
    # Tie-break on id in the same direction (timestamps can collide within one second)
    tiebreak = Prediction.id.desc() if descending else Prediction.id.asc()
    order = [column.desc() if descending else column.asc(), tiebreak]
    rows = db.execute(base.order_by(*order).offset(offset).limit(limit)).scalars().all()
    logger.info(
        "history user_id=%s limit=%d offset=%d sort=%s gender=%s -> %d/%d",
        user_id,
        limit,
        offset,
        sort_by,
        gender,
        len(rows),
        total,
    )
    return int(total), list(rows)


def get_prediction(db: Session, user_id: int, prediction_id: int) -> Prediction | None:
    """The prediction with its features; raises NotOwnerError when it belongs to someone else."""
    row = db.execute(
        select(Prediction)
        .options(selectinload(Prediction.acoustic_features))
        .where(Prediction.id == prediction_id)
    ).scalar_one_or_none()
    if row is None:
        return None
    if row.user_id != user_id:
        raise NotOwnerError
    return row


def features_of(row: Prediction) -> dict[str, float | None]:
    if row.acoustic_features:
        return row.acoustic_features[0].to_dict()
    return {}


def delete_prediction(db: Session, user_id: int, prediction_id: int) -> bool:
    """Delete a prediction (features cascade) and its S3 object. Returns False when not found."""
    row = get_prediction(db, user_id, prediction_id)
    if row is None:
        return False
    if row.audio_file_path and is_s3_key(row.audio_file_path):
        if not delete_from_s3(row.audio_file_path):
            raise StorageDeleteError("Error deleting the audio file from S3")
    db.delete(row)
    db.commit()
    get_cache().delete(statistics_key(user_id))
    logger.info("prediction deleted id=%s user_id=%s", prediction_id, user_id)
    return True


# ---------------------------------------------------------------------------
# Statistics (cached per user)
# ---------------------------------------------------------------------------


def compute_statistics(db: Session, user_id: int, days: int = 30) -> dict[str, Any]:
    base = select(Prediction).where(Prediction.user_id == user_id).subquery()
    p = base.c
    totals = db.execute(
        select(
            func.count(p.id),
            func.avg(p.confidence_score),
            func.avg(p.audio_duration),
            func.min(p.audio_duration),
            func.max(p.audio_duration),
            func.avg(p.audio_duration * p.audio_duration),
        )
    ).one()
    count, avg_conf, avg_dur, min_dur, max_dur, avg_sq_dur = totals
    count = int(count or 0)

    by_gender = {
        gender: (int(n), (float(conf) if conf is not None else None))
        for gender, n, conf in db.execute(
            select(p.predicted_gender, func.count(p.id), func.avg(p.confidence_score)).group_by(
                p.predicted_gender
            )
        ).all()
    }
    male_n, male_conf = by_gender.get("male", (0, None))
    female_n, female_conf = by_gender.get("female", (0, None))

    std = None
    if count and avg_dur is not None and avg_sq_dur is not None:
        variance = max(float(avg_sq_dur) - float(avg_dur) ** 2, 0.0)
        std = math.sqrt(variance)

    since = datetime.now(UTC) - timedelta(days=days)
    day = func.date(Prediction.created_at)
    by_date = db.execute(
        select(day, func.count(Prediction.id))
        .where(Prediction.user_id == user_id, Prediction.created_at >= since)
        .group_by(day)
        .order_by(day.desc())
    ).all()

    def pct(n: int) -> float:
        return round(100.0 * n / count, 2) if count else 0.0

    return {
        "total_predictions": count,
        "gender_distribution": {
            "male": male_n,
            "female": female_n,
            "male_percentage": pct(male_n),
            "female_percentage": pct(female_n),
        },
        "average_confidence": round(float(avg_conf), 4) if avg_conf is not None else None,
        "confidence_by_gender": {
            "male": round(male_conf, 4) if male_conf is not None else None,
            "female": round(female_conf, 4) if female_conf is not None else None,
        },
        "average_audio_duration": round(float(avg_dur), 3) if avg_dur is not None else None,
        "duration_range": {
            "min": round(float(min_dur), 3) if min_dur is not None else None,
            "max": round(float(max_dur), 3) if max_dur is not None else None,
            "mean": round(float(avg_dur), 3) if avg_dur is not None else None,
            "std": round(std, 3) if std is not None else None,
        },
        "predictions_by_date": [{"date": str(d), "count": int(n)} for d, n in by_date],
        "generated_at": datetime.now(UTC).isoformat(),
        "cached": False,
    }


def get_statistics(db: Session, user_id: int) -> dict[str, Any]:
    cache = get_cache()
    key = statistics_key(user_id)
    cached = cache.get_json(key)
    if cached is not None:
        cached["cached"] = True
        return cached
    stats = compute_statistics(db, user_id)
    cache.set_json(key, stats)
    logger.info(
        "statistics computed user_id=%s total=%d cache=%s", user_id, stats["total_predictions"], cache.stats()
    )
    return stats


def invalidate_statistics(user_id: int | None) -> None:
    if user_id is not None:
        get_cache().delete(statistics_key(user_id))


# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    "mlp": ("MLP Neural Network", "MLP"),
    "random_forest": ("Random Forest", "RandomForest"),
    "svm": ("Support Vector Machine (RBF)", "SVM"),
    "gradient_boosting": ("Gradient Boosting", "GradientBoosting"),
    "decision_tree": ("Decision Tree (baseline)", "DecisionTree"),
}


def _metrics_from_manifest(manifest: dict[str, Any]) -> dict[str, Any] | None:
    selected = manifest.get("selected_model", {})
    bench = manifest.get("benchmark_metrics", {})
    test = bench.get("test", {})
    if not test:
        return None
    name, key = MODEL_DISPLAY.get(selected.get("key", "mlp"), (selected.get("label", "MLP"), "MLP"))
    return {
        "model_name": key,
        "display_name": name,
        "accuracy": test.get("accuracy"),
        "precision": test.get("precision_macro"),
        "recall": test.get("recall_macro"),
        "f1_score": test.get("f1_macro"),
        "total_predictions": test.get("n_samples"),
        "inference_time_ms": bench.get("inference", {}).get("single_call_mean_ms"),
        "model_version": (selected.get("model_version") or "")[:19] or None,
        "training_timestamp": selected.get("model_version"),
    }


def _metrics_from_comparison_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model in report.get("models", []):
        if model.get("variant") != "base":
            continue
        name, key = MODEL_DISPLAY.get(model["key"], (model["label"], model["key"]))
        test = model.get("test", {})
        rows.append(
            {
                "model_name": key,
                "display_name": name,
                "accuracy": test.get("accuracy"),
                "precision": test.get("precision_macro"),
                "recall": test.get("recall_macro"),
                "f1_score": test.get("f1_macro"),
                "total_predictions": test.get("n_samples"),
                "inference_time_ms": model.get("resources", {}).get("single_call_mean_ms"),
                "model_version": (report.get("generated_at") or "")[:19] or None,
                "training_timestamp": report.get("generated_at"),
            }
        )
    return rows


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def seed_model_metrics(
    db: Session, manifest_path: Path | None = None, report_path: Path | None = None
) -> int:
    """Insert benchmark rows from the deployment manifest and the comparison report (idempotent)."""
    settings = get_settings()
    manifest_path = manifest_path or settings.MODEL_MANIFEST_PATH
    report_path = report_path or settings.MODEL_COMPARISON_REPORT_PATH

    candidates: list[dict[str, Any]] = []
    if report_path.exists():
        candidates.extend(_metrics_from_comparison_report(json.loads(report_path.read_text())))
    if manifest_path.exists() and not any(c["model_name"] == "MLP" for c in candidates):
        entry = _metrics_from_manifest(json.loads(manifest_path.read_text()))
        if entry:
            candidates.append(entry)

    inserted = 0
    for c in candidates:
        if c.get("accuracy") is None:
            continue
        exists = db.execute(
            select(ModelMetrics.id).where(
                ModelMetrics.model_name == c["model_name"], ModelMetrics.model_version == c["model_version"]
            )
        ).first()
        if exists:
            continue
        db.add(
            ModelMetrics(
                model_name=c["model_name"],
                accuracy=float(c["accuracy"]),
                precision=c.get("precision"),
                recall=c.get("recall"),
                f1_score=c.get("f1_score"),
                total_predictions=c.get("total_predictions"),
                training_timestamp=_parse_ts(c.get("training_timestamp")),
                model_version=c.get("model_version"),
                inference_time_ms=c.get("inference_time_ms"),
            )
        )
        inserted += 1
    if inserted:
        db.commit()
        logger.info("Seeded %d model_metrics rows", inserted)
    return inserted


def model_performance(db: Session) -> dict[str, Any]:
    """Latest row per model from model_metrics; falls back to the manifest when the table is empty."""
    settings = get_settings()
    rows = (
        db.execute(select(ModelMetrics).order_by(ModelMetrics.model_name, ModelMetrics.created_at.desc()))
        .scalars()
        .all()
    )
    latest: dict[str, ModelMetrics] = {}
    for row in rows:
        latest.setdefault(row.model_name, row)

    current = "MLP"
    if settings.MODEL_MANIFEST_PATH.exists():
        manifest = json.loads(settings.MODEL_MANIFEST_PATH.read_text())
        current = MODEL_DISPLAY.get(manifest.get("selected_model", {}).get("key", "mlp"), ("", "MLP"))[1]
        if not latest:
            entry = _metrics_from_manifest(manifest)
            if entry:
                latest[entry["model_name"]] = ModelMetrics(
                    model_name=entry["model_name"],
                    accuracy=entry["accuracy"],
                    precision=entry["precision"],
                    recall=entry["recall"],
                    f1_score=entry["f1_score"],
                    total_predictions=entry["total_predictions"],
                    training_timestamp=_parse_ts(entry["training_timestamp"]),
                    model_version=entry["model_version"],
                    inference_time_ms=entry["inference_time_ms"],
                )

    display = {key: name for name, key in MODEL_DISPLAY.values()}
    models = []
    for key, row in sorted(latest.items(), key=lambda kv: (kv[0] != current, -kv[1].accuracy)):
        models.append(
            {
                "name": display.get(key, key),
                "model_key": key,
                "accuracy": row.accuracy,
                "precision": row.precision,
                "recall": row.recall,
                "f1_score": row.f1_score,
                "inference_time_ms": row.inference_time_ms,
                "version": row.model_version,
                "training_date": row.training_timestamp.date().isoformat()
                if row.training_timestamp
                else None,
                "total_predictions": row.total_predictions,
            }
        )
    logger.info("model performance requested: %d models, current=%s", len(models), current)
    return {
        "models": models,
        "current_model": current,
        "note": (
            "Metrics were measured on the historical voice.csv test split; "
            "live-audio accuracy is unvalidated."
        ),
    }
