"""Prediction schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Gender = Literal["male", "female"]


class PredictionResponse(BaseModel):
    """Result of one prediction.

    ``gender``/``confidence``/``confidence_percentage``/``audio_features`` follow the
    API specification; ``predicted_gender``, ``features``, ``duration_seconds`` are
    aliases kept for the existing frontend client.
    """

    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "prediction_id": 123,
                    "gender": "female",
                    "predicted_gender": "female",
                    "confidence": 0.9673,
                    "confidence_percentage": 96.73,
                    "probabilities": {"male": 0.0327, "female": 0.9673},
                    "inference_time_ms": 5.1,
                    "feature_extraction_time_ms": 14.8,
                    "total_time_ms": 41.2,
                    "audio_duration": 3.5,
                    "duration_seconds": 3.5,
                    "sample_rate": 16000,
                    "audio_filename": "clip.wav",
                    "audio_file_path": "clip.wav",
                    "audio_features": {"meanfreq": 0.183, "sd": 0.056, "meanfun": 0.171, "modindx": 0.53},
                    "features": {"meanfreq": 0.183, "sd": 0.056, "meanfun": 0.171, "modindx": 0.53},
                    "model_used": "MLP",
                    "model_version": "2026-09-11T19:36:37",
                    "user_id": 1,
                }
            ]
        },
    )

    prediction_id: int | None = Field(default=None, description="Database id; null when persistence is off")
    gender: Gender
    predicted_gender: Gender
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_percentage: float = Field(ge=0.0, le=100.0)
    probabilities: dict[str, float]

    inference_time_ms: float
    feature_extraction_time_ms: float | None = None
    total_time_ms: float | None = None

    audio_duration: float
    duration_seconds: float
    sample_rate: int
    audio_filename: str | None = None
    audio_file_path: str | None = None

    audio_features: dict[str, float] = Field(description="The 20 acoustic model features (kHz-based units)")
    features: dict[str, float]

    model_used: str
    model_version: str | None = None
    user_id: int | None = None
    created_at: datetime | None = None


class BatchPredictionItem(BaseModel):
    filename: str | None
    status: Literal["ok", "error"]
    error: str | None = None
    result: PredictionResponse | None = None


class BatchPredictionResponse(BaseModel):
    count: int
    succeeded: int
    failed: int
    total_time_ms: float
    results: list[BatchPredictionItem]


class PredictionListResponse(BaseModel):
    items: list[PredictionResponse]
    total: int
