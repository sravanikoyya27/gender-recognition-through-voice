"""Tests for model_comparison.py using freshly trained models on the synthetic splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from model_comparison import (
    HISTORICAL_RESULTS,
    TIE_BAND_PP,
    discover_models,
    run,
    select_production_model,
)
from train_neural_network import TrainingConfig, train
from train_sklearn_models import run as run_sklearn
from voice_features import MODEL_FEATURE_NAMES

tf = pytest.importorskip("tensorflow")


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Train the MLP (few epochs) and the scikit-learn models on shared synthetic splits."""
    import numpy as np

    base = tmp_path_factory.mktemp("artifacts")
    data_dir = base / "splits"
    data_dir.mkdir()
    rng = np.random.default_rng(0)

    def make(n: int) -> pd.DataFrame:
        parts = []
        for label, centre in (("male", -1.0), ("female", 1.0)):
            X = rng.normal(loc=centre, scale=1.0, size=(n, len(MODEL_FEATURE_NAMES)))
            df = pd.DataFrame(X, columns=list(MODEL_FEATURE_NAMES))
            df["label"] = label
            parts.append(df)
        return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=0)

    make(120).to_csv(data_dir / "train.csv", index=False)
    make(20).to_csv(data_dir / "val.csv", index=False)
    make(20).to_csv(data_dir / "test.csv", index=False)

    models_dir = base / "models"
    train(TrainingConfig(data_dir=data_dir, output_dir=models_dir, epochs=4, batch_size=16, seed=1))
    run_sklearn(
        data_dir, models_dir / "sklearn", seed=1, tune=True, model_keys=["svm", "decision_tree"], cv_folds=3
    )
    return {"data_dir": data_dir, "models_dir": models_dir, "base": base}


def test_discover_models_finds_base_and_tuned(trained_artifacts: dict[str, Path]) -> None:
    entries = discover_models(trained_artifacts["models_dir"])
    keys = {e.key for e in entries}
    assert {"mlp", "svm", "decision_tree", "svm_tuned", "decision_tree_tuned"} <= keys
    mlp = next(e for e in entries if e.key == "mlp")
    assert mlp.kind == "keras" and mlp.meets_nn_objective
    assert all(e.variant in ("base", "tuned") for e in entries)


def test_selection_policy_uses_validation_and_tiebreakers() -> None:
    def row(key, val_acc, nn, ms, mb, test_acc=0.5):
        return {
            "key": key,
            "label": key,
            "variant": "base",
            "kind": "keras" if nn else "sklearn",
            "meets_nn_objective": nn,
            "validation": {"accuracy": val_acc, "f1_macro": val_acc, "roc_auc": val_acc, "n_samples": 100},
            "test": {"accuracy": test_acc},
            "resources": {"single_call_mean_ms": ms, "memory_footprint_mb": mb, "artifact_bytes": 1000},
        }

    # NN tied with a faster classical model -> NN wins the tie-break, engineering alternative recorded
    rows = [
        row("mlp", 0.980, True, 1.0, 500.0, test_acc=0.97),
        row("svm", 0.985, False, 0.2, 50.0, test_acc=0.99),
    ]
    sel = select_production_model(rows)
    assert sel["selected"] == "mlp"
    assert sel["engineering_alternative"] == "svm"
    assert sel["policy"]["test_set_used_for_selection"] is False

    # NN clearly worse on validation (outside the tie band) -> the better model wins despite the objective
    rows = [
        row("mlp", 0.90, True, 1.0, 500.0),
        row("svm", 0.90 + (TIE_BAND_PP + 0.5) / 100, False, 0.2, 50.0),
    ]
    assert select_production_model(rows)["selected"] == "svm"


def test_end_to_end_comparison_and_manifest(trained_artifacts: dict[str, Path]) -> None:
    out = trained_artifacts["base"] / "comparison"
    deploy = trained_artifacts["base"] / "deploy"
    report = run(
        models_dir=trained_artifacts["models_dir"],
        data_dir=trained_artifacts["data_dir"],
        output_dir=out,
        deploy_dir=deploy,
        isolate=False,
    )

    for name in (
        "comparison_report.md",
        "comparison_report.json",
        "comparison_metrics.csv",
        "comparison_plots.png",
        "accuracy_comparison.png",
        "confusion_matrices.png",
        "roc_curves.png",
        "accuracy_vs_speed.png",
        "accuracy_vs_memory.png",
        "comparison_log.txt",
    ):
        assert (out / name).exists(), name

    table = pd.read_csv(out / "comparison_metrics.csv")
    assert list(table.columns)[:8] == [
        "Model Name",
        "Variant",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "AUC-ROC",
        "Inf. Time (ms)",
    ]
    assert table["Accuracy"].is_monotonic_decreasing
    assert set(table["Variant"]) == {"base", "tuned"}

    assert report["selection"]["selected"] in {r["key"] for r in report["models"]}
    assert report["analysis"]["best"]["accuracy"]["models"]
    assert all("probabilities_female" not in r["test"] for r in report["models"])
    assert len(report["historical_reference"]) == len(HISTORICAL_RESULTS)
    assert report["methodology_flags"]

    manifest = json.loads((deploy / "model_config.json").read_text())
    assert manifest["input"]["dimension"] == 20
    assert manifest["input"]["feature_order"] == list(MODEL_FEATURE_NAMES)
    assert manifest["output"]["class_mapping"] == {"0": "male", "1": "female"}
    assert manifest["validity"]["live_audio_end_to_end_validated"] is False
    assert (deploy / manifest["artifacts"]["model_file"]).exists()
    if manifest["artifacts"]["scaler_file"]:
        assert (deploy / manifest["artifacts"]["scaler_file"]).exists()
    assert manifest["selection"]["selected"] == report["selection"]["selected"]

    md = (out / "comparison_report.md").read_text(encoding="utf-8")
    assert md.startswith("# Model Selection Analysis")
    for heading in (
        "## Executive summary",
        "## Metrics table",
        "## Recommendations",
        "## Implementation notes",
        "## Deployment requirements",
        "## Historical / reference results",
    ):
        assert heading in md, heading
    assert "96.74" in md  # listed as historical, not as a result
