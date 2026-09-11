"""User and authentication schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class UserRegister(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "username": "alice42",
                    "email": "alice@example.com",
                    "password": "correct-horse-battery",
                    "password_confirm": "correct-horse-battery",
                }
            ]
        }
    )

    username: str = Field(
        min_length=3,
        max_length=50,
        pattern=r"^[A-Za-z0-9]+$",
        description="3-50 alphanumeric characters",
        examples=["alice42"],
    )
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    password_confirm: str = Field(min_length=8, max_length=128)

    @model_validator(mode="after")
    def _passwords_match(self) -> "UserRegister":
        if self.password != self.password_confirm:
            raise ValueError("Passwords do not match")
        return self


class UserLogin(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"email": "alice@example.com", "password": "correct-horse-battery"}]}
    )

    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: EmailStr
    created_at: datetime


class RegisterResponse(BaseModel):
    message: str
    user: UserResponse


class TokenResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "access_token": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature",
                    "token_type": "bearer",
                    "expires_in": 2592000,
                    "refresh_token": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIiwidHlwZSI6InJlZnJlc2gifQ.signature",
                }
            ]
        }
    )

    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until the access token expires
    refresh_token: str | None = None


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenPayload(BaseModel):
    sub: str  # user id
    exp: int  # unix timestamp
    iat: int | None = None
    type: str = "access"  # "access" | "refresh"
    jti: str | None = None


class MessageResponse(BaseModel):
    message: str
