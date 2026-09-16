"""Analytics and history routes (mounted under the API prefix ``/api/v1``).

GET    /me/predictions                 paginated history of the current user
GET    /me/predictions/{prediction_id} one prediction with its acoustic features
DELETE /me/predictions/{prediction_id} delete a prediction (and its S3 object)
GET    /me/statistics                  cached per-user statistics (1 h TTL)
GET    /models/performance             public: benchmark metrics of the trained models
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.models.database import User, get_db
from app.schemas.analytics import (
    MessageResponse,
    ModelFeaturesResponse,
    ModelPerformanceResponse,
    PredictionDetailResponse,
    PredictionHistoryResponse,
    PredictionSummary,
    StatisticsResponse,
)
from app.services import analytics as service
from app.services.analytics import InvalidQueryError, NotOwnerError, StorageDeleteError

logger = logging.getLogger("app.analytics")

router = APIRouter(tags=["analytics"])


def _summary(row) -> PredictionSummary:
    return PredictionSummary(
        id=row.id,
        gender=row.predicted_gender,
        confidence=row.confidence_score,
        audio_duration=row.audio_duration,
        created_at=row.created_at,
        model_used=row.model_used,
    )


@router.get("/me/predictions", response_model=PredictionHistoryResponse, summary="My prediction history")
def my_predictions(
    limit: int = Query(20, description="Page size (1-100)"),
    offset: int = Query(0, description="Rows to skip"),
    sort_by: str = Query(
        "-date", description="date | -date | confidence | -confidence | duration | -duration"
    ),
    gender_filter: str | None = Query(None, description="Male, Female or all"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PredictionHistoryResponse:
    """Paginated history of the current user's predictions, newest first by default."""
    try:
        total, rows = service.list_predictions(
            db, user.id, limit=limit, offset=offset, sort_by=sort_by, gender_filter=gender_filter
        )
    except InvalidQueryError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return PredictionHistoryResponse(
        total=total,
        page_size=limit,
        page_offset=offset,
        sort_by=sort_by,
        gender_filter=service.normalise_gender_filter(gender_filter),
        predictions=[_summary(r) for r in rows],
    )


@router.get(
    "/me/predictions/{prediction_id}",
    response_model=PredictionDetailResponse,
    summary="One of my predictions with its acoustic features",
    responses={403: {"description": "Not owned by the current user"}, 404: {"description": "Not found"}},
)
def my_prediction_detail(
    prediction_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> PredictionDetailResponse:
    """One prediction with probabilities and all 22 stored acoustic measurements."""
    try:
        row = service.get_prediction(db, user.id, prediction_id)
    except NotOwnerError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Prediction not owned by you"
        ) from exc
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return PredictionDetailResponse(
        **_summary(row).model_dump(),
        inference_time_ms=row.inference_time_ms,
        model_version=row.model_version,
        probabilities=row.probabilities,
        audio_file_path=row.audio_file_path,
        audio_features=service.features_of(row),
    )


@router.delete(
    "/me/predictions/{prediction_id}",
    response_model=MessageResponse,
    summary="Delete one of my predictions",
    responses={
        403: {"description": "Not owned"},
        404: {"description": "Not found"},
        500: {"description": "S3 delete failed"},
    },
)
def delete_my_prediction(
    prediction_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> MessageResponse:
    """Delete a prediction, its features and (when stored) its S3 audio object."""
    try:
        deleted = service.delete_prediction(db, user.id, prediction_id)
    except NotOwnerError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Prediction not owned by you"
        ) from exc
    except StorageDeleteError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return MessageResponse(message="Prediction deleted")


@router.get(
    "/me/statistics", response_model=StatisticsResponse, summary="My prediction statistics (cached 1 h)"
)
def my_statistics(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> StatisticsResponse:
    """Aggregated statistics of the current user's predictions (cached for one hour)."""
    return StatisticsResponse(**service.get_statistics(db, user.id))


@router.get(
    "/models/performance",
    response_model=ModelPerformanceResponse,
    summary="Benchmark metrics of the available models (public)",
)
def models_performance(db: Session = Depends(get_db)) -> ModelPerformanceResponse:
    """Benchmark metrics of the trained models and the current production model (no authentication)."""
    return ModelPerformanceResponse(**service.model_performance(db))


@router.get(
    "/models/features",
    response_model=ModelFeaturesResponse,
    summary="Feature order, measured importances and training reference statistics (public)",
)
def models_features() -> ModelFeaturesResponse:
    """Metadata the results view uses to rank, highlight and normalise the 20 acoustic features."""
    return ModelFeaturesResponse(**service.model_features())
