"""Authentication routes: register, login, refresh, logout, current user.

Mounted under the API prefix (``/api/v1/auth``). Error bodies are always
``{"detail": "<message>"}``. Status codes: 201 created, 400 validation,
401 authentication failure, 404 user missing, 409 conflict.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.models.database import User, get_db
from app.schemas.user import (
    MessageResponse,
    RefreshRequest,
    RegisterResponse,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)
from app.services import auth as auth_service
from app.services.auth import TokenError, UserAlreadyExistsError

logger = logging.getLogger("app.auth")

router = APIRouter(prefix="/auth", tags=["authentication"])
bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the bearer token to a User. 401 for missing/invalid/expired tokens, 404 if deleted."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = auth_service.decode_token(credentials.credentials, expected_type="access")
    except TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.message,
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    try:
        user_id = int(payload.sub)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    user = auth_service.get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    """Like get_current_user but returns None when no token is supplied (anonymous access)."""
    if credentials is None or not credentials.credentials:
        return None
    return get_current_user(credentials, db)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"description": "Validation error"}, 409: {"description": "User already exists"}},
)
def register(payload: UserRegister, db: Session = Depends(get_db)) -> RegisterResponse:
    """Create an account. Username and email must be unused; passwords must match (8+ characters)."""
    try:
        user = auth_service.register_user(db, payload)
    except UserAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=exc.message) from exc
    return RegisterResponse(message="Registration successful", user=UserResponse.model_validate(user))


@router.post(
    "/login",
    response_model=TokenResponse,
    responses={401: {"description": "Invalid email or password"}},
)
def login(payload: UserLogin, db: Session = Depends(get_db)) -> TokenResponse:
    """Exchange email and password for a bearer access token (30 days) and a refresh token (90 days)."""
    user = auth_service.authenticate_user(db, str(payload.email), payload.password)
    if user is None:
        # One message for unknown email and wrong password: no account enumeration.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth_service.issue_tokens(user)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    responses={401: {"description": "Invalid or expired refresh token"}},
)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Issue a new token pair from a valid refresh token."""
    try:
        data = auth_service.decode_token(payload.refresh_token, expected_type="refresh")
    except TokenError as exc:
        detail = "Refresh token has expired" if exc.reason == "expired" else "Invalid refresh token"
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail) from exc
    user = auth_service.get_user_by_id(db, int(data.sub))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    logger.info("Access token refreshed user_id=%s", user.id)
    return auth_service.issue_tokens(user)


@router.post("/logout", response_model=MessageResponse)
def logout(current_user: User = Depends(get_current_user)) -> MessageResponse:
    """Stateless JWTs cannot be revoked server-side; the client discards its tokens."""
    logger.info("Logout user_id=%s", current_user.id)
    return MessageResponse(message="Logged out successfully")


@router.get("/me", response_model=UserResponse, responses={401: {"description": "Not authenticated"}})
def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """The account that owns the bearer token."""
    return UserResponse.model_validate(current_user)
