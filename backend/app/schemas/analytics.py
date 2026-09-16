"""Analytics and history schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Gender = Literal["male", "female"]


class PredictionSummary(BaseModel):
    """Basic prediction info for history listings."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    gender: Gender
    confidence: float
    audio_duration: float | None
    created_at: datetime
    model_used: str


class PredictionHistoryResponse(BaseModel):
    total: int
    page_size: int
    page_offset: int
    sort_by: str
    gender_filter: str
    predictions: list[PredictionSummary]


class PredictionDetailResponse(PredictionSummary):
    """A prediction with its stored acoustic features (all 22 specan measurements)."""

    inference_time_ms: float | None
    model_version: str | None
    probabilities: dict[str, float] | None
    audio_file_path: str | None
    audio_features: dict[str, float | None]


class GenderDistribution(BaseModel):
    male: int
    female: int
    male_percentage: float
    female_percentage: float


class DurationRange(BaseModel):
    min: float | None
    max: float | None
    mean: float | None
    std: float | None


class DateCount(BaseModel):
    date: str
    count: int


class StatisticsResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "total_predictions": 150,
                    "gender_distribution": {
                        "male": 75,
                        "female": 75,
                        "male_percentage": 50.0,
                        "female_percentage": 50.0,
                    },
                    "average_confidence": 0.9564,
                    "confidence_by_gender": {"male": 0.9673, "female": 0.9455},
                    "average_audio_duration": 3.2,
                    "duration_range": {"min": 1.0, "max": 9.8, "mean": 3.2, "std": 1.5},
                    "predictions_by_date": [
                        {"date": "2026-09-11", "count": 5},
                        {"date": "2026-09-10", "count": 3},
                    ],
                    "generated_at": "2026-09-11T20:00:00Z",
                    "cached": False,
                }
            ]
        }
    )

    total_predictions: int
    gender_distribution: GenderDistribution
    average_confidence: float | None
    confidence_by_gender: dict[str, float | None]
    average_audio_duration: float | None
    duration_range: DurationRange
    predictions_by_date: list[DateCount] = Field(
        description="Daily counts for the last 30 days, newest first"
    )
    generated_at: datetime
    cached: bool = False


class ModelPerformanceEntry(BaseModel):
    name: str
    model_key: str
    accuracy: float
    precision: float | None
    recall: float | None
    f1_score: float | None
    inference_time_ms: float | None
    version: str | None
    training_date: str | None
    total_predictions: int | None = Field(default=None, description="Evaluation (test) set size")
    evaluation: str = "historical voice.csv feature space (see docs/dataset_provenance.md)"


class ModelPerformanceResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "models": [
                        {
                            "name": "MLP Neural Network",
                            "model_key": "MLP",
                            "accuracy": 0.9811,
                            "precision": 0.9812,
                            "recall": 0.9811,
                            "f1_score": 0.9811,
                            "inference_time_ms": 5.3,
                            "version": "2026-09-11T19:36:37",
                            "training_date": "2026-09-11",
                            "total_predictions": 476,
                            "evaluation": "historical voice.csv feature space "
                            "(see docs/dataset_provenance.md)",
                        }
                    ],
                    "current_model": "MLP",
                    "note": "Metrics were measured on the historical voice.csv test split; "
                    "live-audio accuracy is unvalidated.",
                }
            ]
        }
    )

    models: list[ModelPerformanceEntry]
    current_model: str
    note: str


class FeatureImportance(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_key: str
    label: str
    method: str
    values: dict[str, float] = Field(description="Importance per feature, sorted descending")


class FeatureReference(BaseModel):
    source: str
    mean: dict[str, float]
    std: dict[str, float]


class ModelFeaturesResponse(BaseModel):
    feature_order: list[str]
    importance: FeatureImportance | None
    reference: FeatureReference | None
    note: str


class MessageResponse(BaseModel):
    message: str
