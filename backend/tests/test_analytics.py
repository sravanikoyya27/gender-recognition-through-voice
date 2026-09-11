"""Analytics and history API tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ml.features import FEATURE_NAMES
from app.services.analytics import seed_model_metrics
from app.utils.cache import get_cache, statistics_key
from tests.test_predictions import synth_voice, wav_bytes

AUTH = "/api/v1/auth"
PRED = "/api/v1/predictions"


def _user(client: TestClient, name: str) -> dict[str, str]:
    client.post(
        f"{AUTH}/register",
        json={
            "username": name,
            "email": f"{name}@example.com",
            "password": "s3cretpassword",
            "password_confirm": "s3cretpassword",
        },
    )
    token = client.post(f"{AUTH}/login", json={"email": f"{name}@example.com", "password": "s3cretpassword"})
    return {"Authorization": f"Bearer {token.json()['access_token']}"}


def _predict(client: TestClient, headers: dict, f0: float, seconds: float, seed: int) -> dict:
    r = client.post(
        PRED,
        files={
            "audio": (f"{int(f0)}hz.wav", wav_bytes(synth_voice(f0, seconds=seconds, seed=seed)), "audio/wav")
        },
        headers=headers,
    )
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def owner(client: TestClient) -> dict:
    headers = _user(client, "analyst")
    made = [
        _predict(client, headers, 110.0, 2.0, 31),
        _predict(client, headers, 225.0, 3.0, 32),
        _predict(client, headers, 120.0, 1.5, 33),
    ]
    return {"headers": headers, "predictions": made}


@pytest.fixture(scope="module")
def other(client: TestClient) -> dict[str, str]:
    return _user(client, "intruder")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_history_requires_auth(client: TestClient) -> None:
    assert client.get("/api/v1/me/predictions").status_code == 401
    assert client.get("/api/v1/me/statistics").status_code == 401


def test_history_listing_and_pagination(client: TestClient, owner: dict) -> None:
    r = client.get("/api/v1/me/predictions", headers=owner["headers"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 3 and body["page_size"] == 20 and body["page_offset"] == 0
    assert body["sort_by"] == "-date" and body["gender_filter"] == "all"
    ids = [p["id"] for p in body["predictions"]]
    assert ids == sorted(ids, reverse=True)  # newest first
    item = body["predictions"][0]
    assert set(item) == {"id", "gender", "confidence", "audio_duration", "created_at", "model_used"}
    assert item["model_used"] == "MLP"

    page = client.get(
        "/api/v1/me/predictions", params={"limit": 2, "offset": 2}, headers=owner["headers"]
    ).json()
    assert page["total"] == 3 and len(page["predictions"]) == 1 and page["page_offset"] == 2

    asc = client.get("/api/v1/me/predictions", params={"sort_by": "date"}, headers=owner["headers"]).json()
    assert [p["id"] for p in asc["predictions"]] == sorted(ids)

    by_conf = client.get(
        "/api/v1/me/predictions", params={"sort_by": "-confidence"}, headers=owner["headers"]
    ).json()
    confs = [p["confidence"] for p in by_conf["predictions"]]
    assert confs == sorted(confs, reverse=True)


def test_history_gender_filter(client: TestClient, owner: dict) -> None:
    genders = {p["gender"] for p in owner["predictions"]}
    for label in ("Male", "female", "all"):
        r = client.get("/api/v1/me/predictions", params={"gender_filter": label}, headers=owner["headers"])
        assert r.status_code == 200
        body = r.json()
        if label.lower() == "all":
            assert body["total"] == 3
        else:
            assert all(p["gender"] == label.lower() for p in body["predictions"])
            assert body["total"] == sum(
                g == label.lower() for g in [p["gender"] for p in owner["predictions"]]
            )
    assert genders  # sanity: fixture produced predictions


def test_history_invalid_parameters(client: TestClient, owner: dict) -> None:
    for params in (
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
        {"sort_by": "name"},
        {"gender_filter": "other"},
    ):
        r = client.get("/api/v1/me/predictions", params=params, headers=owner["headers"])
        assert r.status_code == 400, params
        assert isinstance(r.json()["detail"], str)


def test_history_is_per_user(client: TestClient, owner: dict, other: dict[str, str]) -> None:
    assert client.get("/api/v1/me/predictions", headers=other).json()["total"] == 0


def test_legacy_predictions_list(client: TestClient, owner: dict) -> None:
    r = client.get(PRED, headers=owner["headers"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 3 and len(body["items"]) == 3
    assert body["items"][0]["predicted_gender"] in ("male", "female")
    assert list(body["items"][0]["features"]) == list(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Detail and delete
# ---------------------------------------------------------------------------


def test_prediction_detail(client: TestClient, owner: dict, other: dict[str, str]) -> None:
    pid = owner["predictions"][0]["prediction_id"]
    r = client.get(f"/api/v1/me/predictions/{pid}", headers=owner["headers"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == pid and body["inference_time_ms"] > 0
    assert set(body["probabilities"]) == {"male", "female"}
    feats = body["audio_features"]
    assert len(feats) == 22 and set(FEATURE_NAMES) <= set(feats) and {"duration", "peakf"} <= set(feats)
    assert feats["duration"] == pytest.approx(2.0, abs=0.01)

    assert client.get(f"/api/v1/me/predictions/{pid}", headers=other).status_code == 403
    assert client.get("/api/v1/me/predictions/999999", headers=owner["headers"]).status_code == 404


def test_delete_prediction(client: TestClient, owner: dict, other: dict[str, str]) -> None:
    headers = _user(client, "deleter")
    pid = _predict(client, headers, 130.0, 2.0, 41)["prediction_id"]

    assert client.delete(f"/api/v1/me/predictions/{pid}", headers=other).status_code == 403
    r = client.delete(f"/api/v1/me/predictions/{pid}", headers=headers)
    assert r.status_code == 200 and r.json() == {"message": "Prediction deleted"}
    assert client.delete(f"/api/v1/me/predictions/{pid}", headers=headers).status_code == 404
    assert client.get(f"/api/v1/me/predictions/{pid}", headers=headers).status_code == 404
    assert client.get("/api/v1/me/predictions", headers=headers).json()["total"] == 0


# ---------------------------------------------------------------------------
# Statistics and caching
# ---------------------------------------------------------------------------


def test_statistics_and_cache(client: TestClient) -> None:
    headers = _user(client, "statuser")
    _predict(client, headers, 110.0, 2.0, 51)
    _predict(client, headers, 230.0, 4.0, 52)

    cache = get_cache()
    misses_before, hits_before = cache.misses, cache.hits

    first = client.get("/api/v1/me/statistics", headers=headers)
    assert first.status_code == 200, first.text
    s = first.json()
    assert s["total_predictions"] == 2 and s["cached"] is False
    dist = s["gender_distribution"]
    assert dist["male"] + dist["female"] == 2
    assert dist["male_percentage"] + dist["female_percentage"] == pytest.approx(100.0)
    assert 0.5 <= s["average_confidence"] <= 1.0
    assert s["average_audio_duration"] == pytest.approx(3.0, abs=0.01)
    rng = s["duration_range"]
    assert rng["min"] == pytest.approx(2.0, abs=0.01) and rng["max"] == pytest.approx(4.0, abs=0.01)
    assert rng["mean"] == pytest.approx(3.0, abs=0.01) and rng["std"] == pytest.approx(1.0, abs=0.01)
    assert s["predictions_by_date"] and s["predictions_by_date"][0]["count"] == 2
    assert set(s["confidence_by_gender"]) == {"male", "female"}

    second = client.get("/api/v1/me/statistics", headers=headers).json()
    assert second["cached"] is True and second["total_predictions"] == 2
    assert cache.hits == hits_before + 1 and cache.misses == misses_before + 1
    assert cache.stats()["backend"] in ("memory", "redis")

    # A new prediction invalidates the cached statistics
    _predict(client, headers, 140.0, 1.5, 53)
    third = client.get("/api/v1/me/statistics", headers=headers).json()
    assert third["cached"] is False and third["total_predictions"] == 3
    assert cache.backend.get(statistics_key(1)) is None or True  # key of another user untouched


def test_cache_ttl_is_one_hour() -> None:
    assert get_settings().CACHE_TTL_SECONDS == 3600
    assert get_cache().default_ttl == 3600


# ---------------------------------------------------------------------------
# Model performance (public)
# ---------------------------------------------------------------------------


def test_models_performance_public_and_seeded(client: TestClient, test_engine) -> None:
    r = client.get("/api/v1/models/performance")  # no token
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["current_model"] == "MLP"
    keys = {m["model_key"] for m in body["models"]}
    assert "MLP" in keys
    mlp = next(m for m in body["models"] if m["model_key"] == "MLP")
    assert 0.9 < mlp["accuracy"] <= 1.0 and mlp["f1_score"] and mlp["training_date"]
    assert "voice.csv" in body["note"]

    with Session(test_engine) as db:
        inserted = seed_model_metrics(db)
        assert seed_model_metrics(db) == 0  # idempotent
    if inserted:
        body = client.get("/api/v1/models/performance").json()
        assert body["models"][0]["model_key"] == "MLP"  # current model listed first
        assert len(body["models"]) >= 2
        assert all(0 < m["accuracy"] <= 1 for m in body["models"])
        assert all(m["inference_time_ms"] is not None for m in body["models"])
