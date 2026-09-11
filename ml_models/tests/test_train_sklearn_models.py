"""Tests for train_sklearn_models.py: specification compliance and an end-to-end run."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from evaluation import CLASS_TO_INDEX, labels_to_index, metrics_from_probabilities
from train_sklearn_models import MODEL_SPECS, make_classifier, make_pipeline, run
from voice_features import MODEL_FEATURE_NAMES


def test_class_mapping_is_project_wide() -> None:
    assert CLASS_TO_INDEX == {"male": 0, "female": 1}
    np.testing.assert_array_equal(labels_to_index(["Male", "female", "MALE"]), [0, 1, 0])


def test_metrics_from_probabilities_shape() -> None:
    y = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]])
    m = metrics_from_probabilities(y, probs, "toy")
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["confusion_matrix"]["matrix"] == [[1, 1], [1, 1]]
    assert 0 <= m["roc_auc"] <= 1
    assert set(m["per_class"]) == {"male", "female"}


def test_estimators_match_specification() -> None:
    rf = make_classifier("random_forest").get_params()
    assert (rf["n_estimators"], rf["max_depth"], rf["min_samples_split"], rf["min_samples_leaf"]) == (
        200,
        None,
        2,
        1,
    )
    assert (rf["max_features"], rf["n_jobs"], rf["random_state"], rf["class_weight"]) == (
        "sqrt",
        -1,
        42,
        "balanced",
    )

    svm = make_classifier("svm").get_params()
    assert (svm["kernel"], svm["C"], svm["gamma"], svm["probability"]) == ("rbf", 1.0, "scale", True)
    assert (svm["class_weight"], svm["random_state"]) == ("balanced", 42)

    gb = make_classifier("gradient_boosting").get_params()
    assert (gb["n_estimators"], gb["learning_rate"], gb["max_depth"]) == (200, 0.1, 5)
    assert (gb["min_samples_split"], gb["min_samples_leaf"], gb["subsample"], gb["random_state"]) == (
        5,
        2,
        0.8,
        42,
    )

    dt = make_classifier("decision_tree").get_params()
    assert (dt["random_state"], dt["class_weight"]) == (42, "balanced")


def test_pipeline_structure() -> None:
    for spec in MODEL_SPECS:
        pipeline = make_pipeline(spec.key)
        assert isinstance(pipeline, Pipeline)
        assert [name for name, _ in pipeline.steps] == ["scaler", "model"]
        assert pipeline.named_steps["scaler"].__class__.__name__ == "StandardScaler"


def test_grids_match_specification() -> None:
    grids = {spec.key: spec.grid for spec in MODEL_SPECS}
    assert grids["random_forest"] == {"model__n_estimators": [100, 200, 300]}
    assert grids["svm"] == {"model__C": [0.1, 1, 10]}
    assert grids["gradient_boosting"] == {"model__learning_rate": [0.01, 0.1, 0.2]}


def test_end_to_end_run_writes_report_and_artifacts(synthetic_splits: Path, tmp_path: Path) -> None:
    out = tmp_path / "sklearn"
    report = run(synthetic_splits, out, seed=1, tune=False, cv_folds=3)

    keys = ["random_forest", "svm", "gradient_boosting", "decision_tree"]
    assert list(report["models"]) == keys
    for key in keys:
        assert (out / f"{key}_model.pkl").exists()
        assert (out / f"{key}_scaler.pkl").exists()
        assert (out / f"feature_importance_{key}.png").exists()
        entry = report["models"][key]
        for split in ("train", "validation", "test"):
            assert {
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "roc_auc",
                "confusion_matrix",
            } <= set(entry["metrics"][split])
        assert entry["metrics"]["test"]["accuracy"] > 0.8  # separable clusters
        assert entry["cross_validation_train"]["folds"] == 3
        assert entry["inference_timing"]["n_predictions"] == 1000
        assert entry["model_size_bytes"]["file"] > 0
        assert list(entry["feature_importance_top10"])[:1]
    assert report["models"]["svm"]["feature_importance_method"].startswith("permutation")
    assert report["models"]["random_forest"]["feature_importance_method"].startswith("feature_importances_")

    expected_files = (
        "comparison_report.json",
        "comparison_plots.png",
        "feature_importance.json",
        "accuracy_comparison.png",
        "confusion_matrices.png",
        "roc_curves.png",
        "feature_importance_top15.png",
        "training_log.txt",
    )
    for name in expected_files:
        assert (out / name).exists(), name

    saved = json.loads((out / "comparison_report.json").read_text())
    assert saved["data"]["preprocessing"]["feature_order"] == list(MODEL_FEATURE_NAMES)
    assert saved["data"]["preprocessing"]["class_mapping"] == {"male": 0, "female": 1}
    assert (
        saved["ranking_by_test_accuracy"][0]["test_accuracy"]
        >= saved["ranking_by_test_accuracy"][-1]["test_accuracy"]
    )

    # Saved pipeline reloads and predicts with the scaler inside
    with (out / "svm_model.pkl").open("rb") as fh:
        pipeline = pickle.load(fh)
    probs = pipeline.predict_proba(np.zeros((3, 20)))
    assert probs.shape == (3, 2)


def test_tuning_saves_best_params(synthetic_splits: Path, tmp_path: Path) -> None:
    out = tmp_path / "tuned"
    report = run(synthetic_splits, out, seed=1, tune=True, model_keys=["svm"], cv_folds=3)
    tuning = report["tuning"]["svm"]
    assert tuning["best_params"]["model__C"] in (0.1, 1, 10)
    assert len(tuning["results"]) == 3
    assert (out / "svm_tuned_model.pkl").exists()
    assert (out / "tuning_results.json").exists()
