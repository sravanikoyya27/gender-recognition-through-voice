"""GET /api/v1/models/features serves measured importances and reference statistics when present."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import get_settings
from app.ml.features import FEATURE_NAMES
from app.services.analytics import model_features


def test_models_features_endpoint(client: TestClient) -> None:
    r = client.get("/api/v1/models/features")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["feature_order"] == list(FEATURE_NAMES)
    assert "note" in body
    settings = get_settings()
    if settings.FEATURE_IMPORTANCE_PATH.exists():
        imp = body["importance"]
        assert imp and set(imp["values"]) <= set(FEATURE_NAMES)
        values = list(imp["values"].values())
        assert values == sorted(values, reverse=True)
        assert imp["model_key"] and "Phase 2C" in imp["method"]
    if settings.TRAINING_METRICS_PATH.exists():
        ref = body["reference"]
        assert ref and set(ref["mean"]) == set(FEATURE_NAMES) and set(ref["std"]) == set(FEATURE_NAMES)


def test_model_features_degrades_without_artifacts(tmp_path: Path) -> None:
    body = model_features(tmp_path / "missing.json", tmp_path / "missing2.json")
    assert body["importance"] is None and body["reference"] is None
    assert body["feature_order"] == list(FEATURE_NAMES)


def test_model_features_reads_custom_files(tmp_path: Path) -> None:
    importance = tmp_path / "fi.json"
    importance.write_text(
        json.dumps(
            {
                "svm": {
                    "method": "permutation importance",
                    "importance": {"meanfun": 0.5, "IQR": 0.2, "not_a_feature": 9.0},
                }
            }
        )
    )
    metrics = tmp_path / "m.json"
    metrics.write_text(
        json.dumps(
            {
                "data": {
                    "statistics": {
                        "train_feature_mean": {n: 0.1 for n in FEATURE_NAMES},
                        "train_feature_std": {n: 0.05 for n in FEATURE_NAMES},
                    }
                }
            }
        )
    )
    body = model_features(importance, metrics)
    assert body["importance"]["model_key"] == "SVM"
    assert list(body["importance"]["values"]) == ["meanfun", "IQR"]  # sorted, unknown names dropped
    assert body["reference"]["mean"]["meanfun"] == 0.1
