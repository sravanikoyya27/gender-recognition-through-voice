"""Authentication API tests (register, login, refresh, logout, current user)."""

from __future__ import annotations

from datetime import timedelta

from fastapi.testclient import TestClient
from jose import jwt

from app.config import get_settings
from app.services.auth import create_access_token, create_refresh_token

AUTH = "/api/v1/auth"


def _register(client: TestClient, username: str, email: str, password: str = "s3cretpassword") -> dict:
    return client.post(
        f"{AUTH}/register",
        json={"username": username, "email": email, "password": password, "password_confirm": password},
    ).json()


def _login(client: TestClient, email: str, password: str = "s3cretpassword"):
    return client.post(f"{AUTH}/login", json={"email": email, "password": password})


def test_register_success(client: TestClient) -> None:
    r = client.post(
        f"{AUTH}/register",
        json={
            "username": "alice",
            "email": "Alice@Example.com",
            "password": "s3cretpassword",
            "password_confirm": "s3cretpassword",
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["message"] == "Registration successful"
    assert body["user"]["username"] == "alice"
    assert body["user"]["email"] == "alice@example.com"  # normalised
    assert "password" not in body["user"] and "password_hash" not in body["user"]
    assert body["user"]["id"] > 0 and body["user"]["created_at"]


def test_register_conflicts(client: TestClient) -> None:
    _register(client, "bob", "bob@example.com")
    r = client.post(
        f"{AUTH}/register",
        json={
            "username": "BOB",
            "email": "other@example.com",
            "password": "s3cretpassword",
            "password_confirm": "s3cretpassword",
        },
    )
    assert r.status_code == 409 and r.json()["detail"] == "Username is already taken"
    r = client.post(
        f"{AUTH}/register",
        json={
            "username": "bob2",
            "email": "bob@example.com",
            "password": "s3cretpassword",
            "password_confirm": "s3cretpassword",
        },
    )
    assert r.status_code == 409 and r.json()["detail"] == "Email is already registered"


def test_register_validation_errors_are_400(client: TestClient) -> None:
    base = {"username": "carol", "email": "carol@example.com", "password": "s3cretpassword"}
    cases = [
        ({**base, "password_confirm": "different-one"}, "Passwords do not match"),
        ({**base, "email": "not-an-email", "password_confirm": base["password"]}, "email"),
        ({**base, "password": "short", "password_confirm": "short"}, "password"),
        ({**base, "username": "no spaces!", "password_confirm": base["password"]}, "username"),
        ({**base, "username": "ab", "password_confirm": base["password"]}, "username"),
    ]
    for payload, fragment in cases:
        r = client.post(f"{AUTH}/register", json=payload)
        assert r.status_code == 400, payload
        assert isinstance(r.json()["detail"], str) and fragment in r.json()["detail"], r.json()


def test_non_auth_routes_keep_fastapi_validation_format(client: TestClient) -> None:
    # A malformed body on a non-auth route is not rewritten to the auth 400 format.
    r = client.post("/api/v1/predictions/batch", json={"nope": 1}, headers={"Authorization": "Bearer x.y.z"})
    assert r.status_code in (400, 401, 422)
    assert "detail" in r.json()


def test_login_returns_jwt(client: TestClient) -> None:
    user = _register(client, "dave", "dave@example.com")["user"]
    r = _login(client, "dave@example.com")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == get_settings().ACCESS_TOKEN_EXPIRE_DAYS * 86400
    settings = get_settings()
    payload = jwt.decode(body["access_token"], settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    assert payload["sub"] == str(user["id"]) and payload["type"] == "access"
    assert payload["exp"] - payload["iat"] == body["expires_in"]
    refresh = jwt.decode(body["refresh_token"], settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    assert refresh["type"] == "refresh" and refresh["jti"]


def test_login_failures_are_401_without_enumeration(client: TestClient) -> None:
    _register(client, "erin", "erin@example.com")
    wrong = _login(client, "erin@example.com", "wrong-password")
    unknown = _login(client, "nobody@example.com", "whatever1")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"] == "Invalid email or password"
    assert wrong.headers["www-authenticate"] == "Bearer"


def test_current_user_dependency(client: TestClient) -> None:
    user = _register(client, "frank", "frank@example.com")["user"]
    token = _login(client, "frank@example.com").json()["access_token"]

    ok = client.get(f"{AUTH}/me", headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200 and ok.json()["id"] == user["id"]

    assert client.get(f"{AUTH}/me").status_code == 401
    assert client.get(f"{AUTH}/me").json()["detail"] == "Not authenticated"

    bad = client.get(f"{AUTH}/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert bad.status_code == 401 and bad.json()["detail"] == "Invalid token"

    expired, _ = create_access_token(user["id"], lifetime=timedelta(seconds=-5))
    r = client.get(f"{AUTH}/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401 and r.json()["detail"] == "Token has expired"

    refresh_as_access = create_refresh_token(user["id"])
    r = client.get(f"{AUTH}/me", headers={"Authorization": f"Bearer {refresh_as_access}"})
    assert r.status_code == 401 and r.json()["detail"] == "Wrong token type"

    ghost, _ = create_access_token(999_999)
    r = client.get(f"{AUTH}/me", headers={"Authorization": f"Bearer {ghost}"})
    assert r.status_code == 404 and r.json()["detail"] == "User not found"


def test_refresh_token_flow(client: TestClient) -> None:
    _register(client, "grace", "grace@example.com")
    tokens = _login(client, "grace@example.com").json()

    r = client.post(f"{AUTH}/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert r.status_code == 200, r.text
    new_access = r.json()["access_token"]
    assert client.get(f"{AUTH}/me", headers={"Authorization": f"Bearer {new_access}"}).status_code == 200

    r = client.post(f"{AUTH}/refresh", json={"refresh_token": tokens["access_token"]})
    assert r.status_code == 401 and r.json()["detail"] == "Invalid refresh token"

    r = client.post(f"{AUTH}/refresh", json={"refresh_token": "garbage"})
    assert r.status_code == 401 and r.json()["detail"] == "Invalid refresh token"

    settings = get_settings()
    sub = jwt.decode(tokens["refresh_token"], settings.JWT_SECRET_KEY, algorithms=["HS256"])["sub"]
    expired = create_refresh_token(sub, lifetime=timedelta(seconds=-5))
    r = client.post(f"{AUTH}/refresh", json={"refresh_token": expired})
    assert r.status_code == 401 and r.json()["detail"] == "Refresh token has expired"


def test_logout(client: TestClient) -> None:
    _register(client, "heidi", "heidi@example.com")
    token = _login(client, "heidi@example.com").json()["access_token"]
    r = client.post(f"{AUTH}/logout", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200 and r.json() == {"message": "Logged out successfully"}
    assert client.post(f"{AUTH}/logout").status_code == 401


def test_secret_key_length_is_enforced() -> None:
    import pytest

    from app.config import Settings

    with pytest.raises(ValueError, match="32 characters"):
        Settings(JWT_SECRET_KEY="too-short")
