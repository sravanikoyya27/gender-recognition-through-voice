"""Authentication business logic: registration, credential checks and JWT issuing/validation.

Token payloads: ``{"sub": "<user id>", "exp": <unix ts>, "iat": <unix ts>, "type": "access" | "refresh"}``.
Refresh tokens additionally carry a random ``jti`` so they are unique per issue.
"""

from __future__ import annotations

import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Literal

from jose import ExpiredSignatureError, JWTError, jwt
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.database import User
from app.schemas.user import TokenPayload, TokenResponse, UserRegister
from app.utils.security import hash_password, verify_password

logger = logging.getLogger("app.auth")

TokenType = Literal["access", "refresh"]

# A real bcrypt hash used to keep login timing similar for unknown emails.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


class TokenError(Exception):
    """A token could not be accepted. ``reason`` is 'expired', 'invalid' or 'wrong_type'."""

    def __init__(self, reason: Literal["expired", "invalid", "wrong_type"]) -> None:
        super().__init__(reason)
        self.reason = reason

    @property
    def message(self) -> str:
        return {
            "expired": "Token has expired",
            "invalid": "Invalid token",
            "wrong_type": "Wrong token type",
        }[self.reason]


class UserAlreadyExistsError(Exception):
    def __init__(self, field: Literal["username", "email"]) -> None:
        super().__init__(field)
        self.field = field

    @property
    def message(self) -> str:
        return "Username is already taken" if self.field == "username" else "Email is already registered"


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def _encode(subject: str | int, token_type: TokenType, lifetime: timedelta) -> tuple[str, int]:
    settings = get_settings()
    now = datetime.now(UTC)
    expire = now + lifetime
    payload: dict[str, object] = {
        "sub": str(subject),
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "type": token_type,
    }
    if token_type == "refresh":
        payload["jti"] = secrets.token_hex(16)
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token, int(lifetime.total_seconds())


def create_access_token(subject: str | int, lifetime: timedelta | None = None) -> tuple[str, int]:
    """Return ``(token, expires_in_seconds)``; default lifetime is ACCESS_TOKEN_EXPIRE_DAYS."""
    lifetime = lifetime if lifetime is not None else timedelta(days=get_settings().ACCESS_TOKEN_EXPIRE_DAYS)
    return _encode(subject, "access", lifetime)


def create_refresh_token(subject: str | int, lifetime: timedelta | None = None) -> str:
    lifetime = lifetime if lifetime is not None else timedelta(days=get_settings().REFRESH_TOKEN_EXPIRE_DAYS)
    return _encode(subject, "refresh", lifetime)[0]


def decode_token(token: str, expected_type: TokenType = "access") -> TokenPayload:
    """Validate signature, expiry and token type. Raises ``TokenError``."""
    settings = get_settings()
    try:
        data = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        payload = TokenPayload(**data)
    except ExpiredSignatureError as exc:
        raise TokenError("expired") from exc
    except (JWTError, ValueError) as exc:
        raise TokenError("invalid") from exc
    if payload.type != expected_type:
        raise TokenError("wrong_type")
    return payload


def issue_tokens(user: User) -> TokenResponse:
    access, expires_in = create_access_token(user.id)
    return TokenResponse(
        access_token=access,
        token_type="bearer",
        expires_in=expires_in,
        refresh_token=create_refresh_token(user.id),
    )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.execute(
        select(User).where(func.lower(User.email) == email.strip().lower())
    ).scalar_one_or_none()


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.execute(select(User).where(func.lower(User.username) == username.lower())).scalar_one_or_none()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def register_user(db: Session, payload: UserRegister) -> User:
    """Create a user after checking username/email uniqueness (case-insensitive)."""
    logger.info("Registration attempt username=%s email=%s", payload.username, payload.email)
    if get_user_by_username(db, payload.username) is not None:
        raise UserAlreadyExistsError("username")
    if get_user_by_email(db, str(payload.email)) is not None:
        raise UserAlreadyExistsError("email")

    user = User(username=payload.username, email=str(payload.email).strip().lower())
    user.set_password(payload.password)
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("User registered user_id=%s username=%s", user.id, user.username)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    """Return the user when the credentials are valid, else None (timing-equalised)."""
    user = get_user_by_email(db, email)
    if user is None:
        verify_password(password, _DUMMY_HASH)  # same cost as a real check
        logger.warning("Failed login: unknown email=%s", email)
        return None
    if not user.check_password(password):
        logger.warning("Failed login: wrong password user_id=%s email=%s", user.id, email)
        return None
    logger.info("Successful login user_id=%s", user.id)
    return user
