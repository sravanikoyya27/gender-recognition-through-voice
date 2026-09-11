"""API routers."""

from fastapi import APIRouter

from app.api.analytics import router as analytics_router
from app.api.auth import router as auth_router
from app.api.predictions import router as predictions_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(predictions_router)
api_router.include_router(analytics_router)

__all__ = ["api_router"]
