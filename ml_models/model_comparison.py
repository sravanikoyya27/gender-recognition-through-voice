"""Model comparison and production-model selection.

Loads every trained model from Phases 2B (Keras MLP) and 2C (scikit-learn
pipelines, base and tuned variants), evaluates all of them on the *same*
untouched ``test.csv`` with the shared metric code, measures inference latency
and memory footprint, checks for overfitting, applies a pre-declared selection
policy that uses only validation / cross-validation results, writes the
comparison report (Markdown, JSON, CSV, plots) and copies the selected model
with a manifest (``model_config.json``) to the backend deployment directory.

Selection policy (declared here, before any test metric is consulted):

1. Candidates are the base models and their CV-tuned variants.
2. Accuracy criterion = validation accuracy (``val.csv``, never used for fitting
   or early stopping) together with 5-fold CV on train where available.
   Candidates whose validation accuracy is within ``TIE_BAND`` percentage points
   of the best are considered *tied* (the band is about 1.7 standard errors for
   n = 475).
3. Among tied candidates, the tie-breakers are applied in this order:
   a) suitability for the project's stated objective (a neural-network
      classifier is the deliverable of "Gender Recognition through Voice using
      Neural Networks"), b) single-call inference latency, c) memory footprint,
      d) operational complexity (runtime dependencies, cold start).
4. Test metrics are reported after selection and never feed back into it.

Historical / reference-paper figures are listed separately and are not
comparable (different splits, feature extraction or datasets).

Usage::

    python model_comparison.py --models-dir models --data-dir data/processed/voice_csv \
        --output models/comparison --deploy-dir ../backend/app/ml/models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation import CLASS_NAMES, CLASS_TO_INDEX, labels_to_index, metrics_from_probabilities
from train_neural_network import get_git_info, load_split
from voice_features import CANONICAL_SAMPLE_RATE, MODEL_FEATURE_NAMES, N_MODEL_FEATURES
from voice_features import __version__ as voice_features_version

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ML_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = ML_DIR / "models"
DEFAULT_DATA_DIR = ML_DIR / "data" / "processed" / "voice_csv"
DEFAULT_OUTPUT_DIR = DEFAULT_MODELS_DIR / "comparison"
DEFAULT_DEPLOY_DIR = ML_DIR.parent / "backend" / "app" / "ml" / "models"

TIE_BAND_PP = 1.0  # percentage points of validation accuracy treated as a statistical tie
OVERFIT_FLAG_PP = 10.0  # train - test accuracy gap that raises an overfitting flag

logger = logging.getLogger("model_comparison")


# ---------------------------------------------------------------------------
# Historical / reference figures (NOT produced by this codebase)
# ---------------------------------------------------------------------------

HISTORICAL_RESULTS: list[dict[str, Any]] = [
    {
        "label": "MLP, Buyukyilmaz & Cibikdiken (2016)",
        "accuracy": 0.9674,
        "source": "Reference paper 2, Table III: 5-fold CV on voice.csv with the authors' own splits",
        "comparable": "no (different split protocol)",
    },
    {
        "label": "Random forest, Becker (2016)",
        "accuracy": 0.87,
        "source": "Reference paper 2, Table I (reproduced from Becker's blog), test accuracy on voice.csv",
        "comparable": "no (different split)",
    },
    {
        "label": "SVM, Becker (2016)",
        "accuracy": 0.85,
        "source": "Reference paper 2, Table I",
        "comparable": "no (different split)",
    },
    {
        "label": "XGBoost, Becker (2016)",
        "accuracy": 0.87,
        "source": "Reference paper 2, Table I",
        "comparable": "no (different split)",
    },
    {
        "label": "Stacked model, Becker (2016)",
        "accuracy": 0.89,
        "source": "Reference paper 2, Table I",
        "comparable": "no (different split)",
    },
    {
        "label": "Random forest, Nair & Vijayan (2019)",
        "accuracy": 0.893,
        "source": "Reference paper 1, Table 1: VoxForge audio with FFT dominant-frequency features",
        "comparable": "no (different dataset and features)",
    },
    {
        "label": "Gradient boosting, Nair & Vijayan (2019)",
        "accuracy": 0.937,
        "source": "Reference paper 1, Table 1",
        "comparable": "no (different dataset and features)",
    },
    {
        "label": "SVM, Nair & Vijayan (2019)",
        "accuracy": 0.905,
        "source": "Reference paper 1, Table 1",
        "comparable": "no (different dataset and features)",
    },
    {
        "label": "Decision tree, Nair & Vijayan (2019)",
        "accuracy": 0.869,
        "source": "Reference paper 1, Table 1",
        "comparable": "no (different dataset and features)",
    },
    {
        "label": "MLP, original B.Tech project report (2021)",
        "accuracy": 0.95,
        "source": "Project report chapter 5: 'testing accuracy in the range of 95 %' (training 96 %), "
        "no split details",
        "comparable": "no (undocumented split)",
    },
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@dataclass
class ModelEntry:
    key: str
    label: str
    variant: str  # "base" | "tuned"
    kind: str  # "keras" | "sklearn"
    model_path: Path
    scaler_path: Path | None = None
    meets_nn_objective: bool = False
    runtime: str = ""
    complexity: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    cv_train: dict[str, Any] | None = None
    training_seconds: float | None = None
    _model: Any = None
    _scaler: Any = None

    # -- loading ---------------------------------------------------------------
    def load(self) -> None:
        if self.kind == "keras":
            import tensorflow as tf

            self._model = tf.keras.models.load_model(self.model_path, compile=False)
            with Path(self.scaler_path).open("rb") as fh:  # type: ignore[arg-type]
                self._scaler = pickle.load(fh)
        else:
            with self.model_path.open("rb") as fh:
                self._model = pickle.load(fh)  # full Pipeline(scaler, model)

    # -- inference ---------------------------------------------------------------
    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        if self.kind == "keras":
            X = self._scaler.transform(np.asarray(X_raw, dtype=np.float64)).astype(np.float32)
            return np.asarray(self._model.predict(X, verbose=0), dtype=np.float64)
        return np.asarray(self._model.predict_proba(X_raw), dtype=np.float64)

    def predict_single(self, x_raw: np.ndarray) -> np.ndarray:
        """Lowest-overhead single-sample path (what an API should call)."""
        if self.kind == "keras":
            X = self._scaler.transform(np.asarray(x_raw, dtype=np.float64).reshape(1, -1)).astype(np.float32)
            return np.asarray(self._model(X, training=False))[0]
        return self._model.predict_proba(np.asarray(x_raw).reshape(1, -1))[0]

    def artifact_bytes(self) -> int:
        size = self.model_path.stat().st_size
        if self.scaler_path is not None:
            size += Path(self.scaler_path).stat().st_size
        return size


SKLEARN_LABELS = {
    "random_forest": "Random Forest",
    "svm": "SVM (RBF)",
    "gradient_boosting": "Gradient Boosting",
    "decision_tree": "Decision Tree (baseline)",
}
SKLEARN_COMPLEXITY = {
    "random_forest": "scikit-learn pickle; 200 trees, set n_jobs=1 for single-sample serving",
    "svm": "scikit-learn pickle; tiny, no runtime tuning needed",
    "gradient_boosting": "scikit-learn pickle; sequential trees, moderate size",
    "decision_tree": "scikit-learn pickle; trivial",
}


def discover_models(models_dir: Path, include_tuned: bool = True) -> list[ModelEntry]:
    """Find the Phase 2B / 2C artifacts under ``models_dir``."""
    entries: list[ModelEntry] = []

    mlp_path = models_dir / "model_mlp.h5"
    mlp_scaler = models_dir / "scaler_mlp.pkl"
    if mlp_path.exists() and mlp_scaler.exists():
        meta = {}
        meta_path = models_dir / "metrics_mlp.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        entries.append(
            ModelEntry(
                key="mlp",
                label="MLP (neural network)",
                variant="base",
                kind="keras",
                model_path=mlp_path,
                scaler_path=mlp_scaler,
                meets_nn_objective=True,
                runtime="TensorFlow 2.14 (Keras)",
                complexity="TensorFlow runtime (~hundreds of MB), slow cold start; warm-up recommended",
                params={**meta.get("architecture", {}), **meta.get("training", {})},
                training_seconds=meta.get("training", {}).get("training_seconds"),
            )
        )
    else:
        logger.warning("MLP artifacts not found under %s", models_dir)

    sk_dir = models_dir / "sklearn"
    sk_report: dict[str, Any] = {}
    report_path = sk_dir / "comparison_report.json"
    if report_path.exists():
        sk_report = json.loads(report_path.read_text())
    for key, label in SKLEARN_LABELS.items():
        base = sk_dir / f"{key}_model.pkl"
        if base.exists():
            info = sk_report.get("models", {}).get(key, {})
            entries.append(
                ModelEntry(
                    key=key,
                    label=label,
                    variant="base",
                    kind="sklearn",
                    model_path=base,
                    runtime="scikit-learn 1.3",
                    complexity=SKLEARN_COMPLEXITY[key],
                    params=info.get("params", {}),
                    cv_train=info.get("cross_validation_train"),
                    training_seconds=info.get("training_seconds"),
                )
            )
        tuned = sk_dir / f"{key}_tuned_model.pkl"
        if include_tuned and tuned.exists():
            tinfo = sk_report.get("tuning", {}) or {}
            tinfo = tinfo.get(key, {})
            entries.append(
                ModelEntry(
                    key=f"{key}_tuned",
                    label=f"{label} [tuned: {tinfo.get('best_params', {})}]",
                    variant="tuned",
                    kind="sklearn",
                    model_path=tuned,
                    runtime="scikit-learn 1.3",
                    complexity=SKLEARN_COMPLEXITY[key],
                    params=tinfo.get("best_params", {}),
                    cv_train={"mean": tinfo.get("best_cv_accuracy")}
                    if tinfo.get("best_cv_accuracy")
                    else None,
                    training_seconds=tinfo.get("search_seconds"),
                )
            )
    if not entries:
        raise FileNotFoundError(f"No trained models found under {models_dir}")
    return entries


# ---------------------------------------------------------------------------
# Resource measurement (latency + memory), optionally in an isolated process
# ---------------------------------------------------------------------------


def measure_resources_inprocess(entry: ModelEntry, X_test: np.ndarray, n: int = 1000) -> dict[str, Any]:
    import psutil

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    entry.load()
    load_seconds = time.perf_counter() - t0
    rss_after_load = proc.memory_info().rss

    X_n = np.resize(X_test, (n, X_test.shape[1]))
    entry.predict_proba(X_n[:8])  # warm-up (graph tracing for Keras)
    entry.predict_single(X_n[0])

    batch = []
    for _ in range(3):
        t0 = time.perf_counter()
        entry.predict_proba(X_n)
        batch.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    for row in X_n:
        entry.predict_single(row)
    single_total = time.perf_counter() - t0
    rss_after_predict = proc.memory_info().rss
    return {
        "n_predictions": n,
        "load_seconds": load_seconds,
        "batch_1000_ms": min(batch) * 1000,
        "batch_per_prediction_ms": min(batch) * 1000 / n,
        "single_call_mean_ms": single_total * 1000 / n,
        "rss_before_mb": rss_before / 2**20,
        "rss_after_load_mb": rss_after_load / 2**20,
        "rss_after_predict_mb": rss_after_predict / 2**20,
        "memory_footprint_mb": (rss_after_predict - rss_before) / 2**20,
        "artifact_bytes": entry.artifact_bytes(),
        "isolated_process": False,
    }


def measure_resources_isolated(entry: ModelEntry, models_dir: Path, data_dir: Path) -> dict[str, Any]:
    """Run the measurement in a fresh interpreter so each model's runtime import is counted once."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--measure",
        entry.key,
        "--models-dir",
        str(models_dir),
        "--data-dir",
        str(data_dir),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False, cwd=ML_DIR)
    if out.returncode != 0:
        raise RuntimeError(f"resource measurement failed for {entry.key}:\n{out.stderr[-2000:]}")
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    payload["isolated_process"] = True
    return payload


def _measure_entrypoint(key: str, models_dir: Path, data_dir: Path) -> None:
    entries = {e.key: e for e in discover_models(models_dir)}
    X_test, _ = load_split(data_dir / "test.csv")
    result = measure_resources_inprocess(entries[key], X_test.to_numpy(dtype=np.float64))
    print(json.dumps(result))


# ---------------------------------------------------------------------------
# Evaluation, analysis, selection
# ---------------------------------------------------------------------------


def evaluate_entry(entry: ModelEntry, data: dict[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for split in ("train", "validation", "test"):
        probs = entry.predict_proba(data[f"X_{split}"])
        out[split] = metrics_from_probabilities(data[f"y_{split}"], probs, split)
        out[split]["probabilities_female"] = (
            probs[:, CLASS_TO_INDEX["female"]].tolist() if split == "test" else None
        )
    return out


def overfitting_check(metrics: dict[str, Any]) -> dict[str, Any]:
    gap = (metrics["train"]["accuracy"] - metrics["test"]["accuracy"]) * 100
    return {
        "train_accuracy": metrics["train"]["accuracy"],
        "test_accuracy": metrics["test"]["accuracy"],
        "gap_pp": gap,
        "flag": gap > OVERFIT_FLAG_PP,
    }


LATENCY_TIE_MS = 1.0  # single-call latencies below this are operationally equivalent
MEMORY_TIE_MB = 50.0  # footprints below this are operationally equivalent


def _engineering_key(r: dict[str, Any]) -> tuple:
    """Coarse latency/memory buckets, then validation ROC-AUC and F1 (never test metrics)."""
    ms = r["resources"]["single_call_mean_ms"]
    mb = r["resources"]["memory_footprint_mb"]
    return (
        0.0 if ms < LATENCY_TIE_MS else round(ms),
        0.0 if mb < MEMORY_TIE_MB else round(mb / MEMORY_TIE_MB) * MEMORY_TIE_MB,
        -r["validation"]["roc_auc"],
        -r["validation"]["f1_macro"],
        -r["validation"]["accuracy"],
    )


def select_production_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-declared policy using validation/CV results and non-accuracy criteria only."""
    candidates = [r for r in rows if r["variant"] in ("base", "tuned")]
    best_val = max(r["validation"]["accuracy"] for r in candidates)
    tied = [r for r in candidates if (best_val - r["validation"]["accuracy"]) * 100 <= TIE_BAND_PP]

    ordered = sorted(tied, key=lambda r: (0 if r["meets_nn_objective"] else 1, *_engineering_key(r)))
    selected = ordered[0]
    non_nn = sorted([r for r in tied if not r["meets_nn_objective"]], key=_engineering_key)
    engineering_pick = non_nn[0] if non_nn else None

    edge = min(
        tied,
        key=lambda r: (
            r["resources"]["memory_footprint_mb"] + r["resources"]["artifact_bytes"] / 2**20,
            r["resources"]["single_call_mean_ms"],
        ),
    )
    backup_pool = [r for r in tied if r["key"] != selected["key"] and r["kind"] != selected["kind"]] or [
        r for r in tied if r["key"] != selected["key"]
    ]
    backup = (
        max(backup_pool, key=lambda r: (r["validation"]["accuracy"], r["validation"]["f1_macro"]))
        if backup_pool
        else None
    )

    n_val = rows[0]["validation"]["n_samples"]
    tied_text = ", ".join(f"{r['label']} ({r['validation']['accuracy']:.4f})" for r in ordered)
    reasoning = [
        f"Validation accuracy (val.csv, n={n_val}) of the best candidate is {best_val:.4f}; "
        f"{len(tied)} of {len(candidates)} candidates lie within the {TIE_BAND_PP:.1f} pp tie band: "
        f"{tied_text}.",
        "Because the accuracy criterion does not separate these candidates, the tie-breakers apply: "
        "(a) the project's stated objective is a neural-network classifier, so the MLP is preferred when "
        f"tied; (b) single-call latency (below {LATENCY_TIE_MS:.0f} ms counts as equal); (c) memory "
        f"footprint (below {MEMORY_TIE_MB:.0f} MB counts as equal); (d) validation ROC-AUC and F1; "
        "(e) operational complexity.",
    ]
    note = f"Selected: {selected['label']}."
    if engineering_pick and engineering_pick["key"] != selected["key"]:
        res = engineering_pick["resources"]
        note += (
            f" Note: judged on engineering criteria alone (latency {res['single_call_mean_ms']:.2f} ms, "
            f"footprint {res['memory_footprint_mb']:.0f} MB, validation ROC-AUC "
            f"{engineering_pick['validation']['roc_auc']:.4f}) the pick would be {engineering_pick['label']}."
        )
    reasoning.append(note)
    reasoning.append(
        "Test-set metrics were not used for this decision; they are reported below for final evaluation only."
    )
    return {
        "policy": {
            "accuracy_criterion": "validation accuracy (val.csv) + 5-fold CV on train where available",
            "tie_band_pp": TIE_BAND_PP,
            "tie_breakers": [
                "neural-network objective",
                f"single-call latency (< {LATENCY_TIE_MS:.0f} ms treated as equal)",
                f"memory footprint (< {MEMORY_TIE_MB:.0f} MB treated as equal)",
                "validation ROC-AUC, then validation F1",
                "operational complexity",
            ],
            "test_set_used_for_selection": False,
        },
        "candidates_considered": [r["key"] for r in candidates],
        "tied_candidates": [r["key"] for r in ordered],
        "selected": selected["key"],
        "engineering_alternative": engineering_pick["key"] if engineering_pick else None,
        "edge_alternative": edge["key"],
        "backup": backup["key"] if backup else None,
        "reasoning": reasoning,
    }


METHODOLOGY_FLAGS = [
    "Test-set metrics for every model were computed and published in the Phase 2B and 2C reports before "
    "this selection step ran, so the selection cannot be described as blind. The policy above relies on "
    "val.csv and cross-validation only, and the flag is recorded here rather than treated as clean.",
    "Phase 2C's grid search chose hyperparameters by 5-fold CV on train.csv (clean) but its log and report "
    "also printed the tuned models' test accuracy; those numbers were not used to choose parameters, and "
    "tuned variants are evaluated here on the same basis as base models.",
    "The MLP's early stopping used a stratified 10 % hold-out of train.csv, not val.csv or test.csv (clean).",
    "The MLP has no 5-fold CV estimate (the scikit-learn models do); its validation accuracy is the "
    "comparable test-free figure.",
    "The latency and memory tie-breakers were coarsened into buckets after a first run showed the raw "
    "ordering rewarding a 0.05 ms latency difference between otherwise equivalent models. The change only "
    "affects which model is reported as the engineering alternative; it uses validation metrics only, and "
    "the production selection (neural-network objective first) is the same before and after.",
]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_plots(rows: list[dict[str, Any]], y_test: np.ndarray, output_dir: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    base_rows = [r for r in rows if r["variant"] == "base"]
    paths: dict[str, Path] = {}

    def accuracy_bar(ax) -> None:
        labels = [r["label"] for r in base_rows]
        vals = [r["test"]["accuracy"] for r in base_rows]
        colors = ["#7c3aed" if r["meets_nn_objective"] else "#2563eb" for r in base_rows]
        bars = ax.bar(labels, vals, color=colors)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
        ax.set_ylim(min(vals) - 0.02, 1.0)
        ax.set_ylabel("test accuracy")
        ax.set_title("Test accuracy (base models)")
        ax.tick_params(axis="x", rotation=15)

    def confusion(ax, r) -> None:
        arr = np.asarray(r["test"]["confusion_matrix"]["matrix"])
        ax.imshow(arr, cmap="Blues")
        ax.set_xticks(range(2), CLASS_NAMES)
        ax.set_yticks(range(2), CLASS_NAMES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(r["label"], fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(arr[i, j]),
                    ha="center",
                    va="center",
                    color="white" if arr[i, j] > arr.max() / 2 else "black",
                )

    def roc(ax) -> None:
        for r in base_rows:
            fpr, tpr, _ = roc_curve(y_test, np.asarray(r["test"]["probabilities_female"]))
            ax.plot(fpr, tpr, label=f"{r['label']} (AUC {r['test']['roc_auc']:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim(-0.01, 0.2)
        ax.set_ylim(0.8, 1.01)
        ax.set_xlabel("false positive rate (zoomed)")
        ax.set_ylabel("true positive rate")
        ax.set_title("ROC curves, test set")
        ax.legend(fontsize=7, loc="lower right")

    def scatter(ax, xkey: str, xlabel: str, log: bool) -> None:
        for r in rows:
            x = r["resources"][xkey]
            marker = "o" if r["variant"] == "base" else "^"
            ax.scatter(
                x,
                r["test"]["accuracy"],
                marker=marker,
                s=60,
                color="#7c3aed" if r["meets_nn_objective"] else "#2563eb",
            )
            ax.annotate(
                r["label"].split(" [")[0] + (" (tuned)" if r["variant"] == "tuned" else ""),
                (x, r["test"]["accuracy"]),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
            )
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("test accuracy")
        ax.set_title(f"Accuracy vs {xlabel.split(' (')[0]}")
        ax.grid(alpha=0.3)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    accuracy_bar(ax)
    fig.tight_layout()
    paths["accuracy"] = output_dir / "accuracy_comparison.png"
    fig.savefig(paths["accuracy"], dpi=120)
    plt.close(fig)

    n = len(base_rows)
    cols = 2 if n <= 4 else 3
    rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(4.4 * cols, 4 * rws))
    for ax, r in zip(np.ravel(axes), base_rows, strict=False):
        confusion(ax, r)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.tight_layout()
    paths["confusion"] = output_dir / "confusion_matrices.png"
    fig.savefig(paths["confusion"], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    roc(ax)
    fig.tight_layout()
    paths["roc"] = output_dir / "roc_curves.png"
    fig.savefig(paths["roc"], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter(ax, "single_call_mean_ms", "single-call latency (ms, log scale)", log=True)
    fig.tight_layout()
    paths["speed"] = output_dir / "accuracy_vs_speed.png"
    fig.savefig(paths["speed"], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter(ax, "memory_footprint_mb", "memory footprint (MB, RSS delta, log scale)", log=True)
    fig.tight_layout()
    paths["memory"] = output_dir / "accuracy_vs_memory.png"
    fig.savefig(paths["memory"], dpi=120)
    plt.close(fig)

    fig = plt.figure(figsize=(20, 15))
    grid = fig.add_gridspec(3, 4)
    accuracy_bar(fig.add_subplot(grid[0, :2]))
    roc(fig.add_subplot(grid[0, 2:]))
    for i, r in enumerate(base_rows[:4]):
        confusion(fig.add_subplot(grid[1, i]), r)
    scatter(
        fig.add_subplot(grid[2, :2]), "single_call_mean_ms", "single-call latency (ms, log scale)", log=True
    )
    scatter(
        fig.add_subplot(grid[2, 2:]),
        "memory_footprint_mb",
        "memory footprint (MB, RSS delta, log scale)",
        log=True,
    )
    fig.suptitle(
        "Model Selection Analysis (voice.csv benchmark feature space, untouched test split)", fontsize=14
    )
    fig.tight_layout()
    paths["grid"] = output_dir / "comparison_plots.png"
    fig.savefig(paths["grid"], dpi=110)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Reports and deployment
# ---------------------------------------------------------------------------


def comparison_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for r in rows:
        t = r["test"]
        records.append(
            {
                "Model Name": r["label"],
                "Variant": r["variant"],
                "Accuracy": t["accuracy"],
                "Precision": t["precision_macro"],
                "Recall": t["recall_macro"],
                "F1-Score": t["f1_macro"],
                "AUC-ROC": t["roc_auc"],
                "Inf. Time (ms)": r["resources"]["single_call_mean_ms"],
                "Batch 1000 (ms)": r["resources"]["batch_1000_ms"],
                "Memory (MB)": r["resources"]["memory_footprint_mb"],
                "Artifact (MB)": r["resources"]["artifact_bytes"] / 2**20,
                "Train Acc": r["train"]["accuracy"],
                "Val Acc": r["validation"]["accuracy"],
                "CV Acc (train)": (r["cv_train"] or {}).get("mean"),
                "Overfit Gap (pp)": r["overfitting"]["gap_pp"],
            }
        )
    df = pd.DataFrame(records)
    return df.sort_values(["Accuracy", "F1-Score", "AUC-ROC"], ascending=False).reset_index(drop=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deploy_selected(
    selected: dict[str, Any],
    entry: ModelEntry,
    models_dir: Path,
    deploy_dir: Path,
    rows: list[dict[str, Any]],
    selection: dict[str, Any],
    data_dir: Path,
) -> dict[str, Any]:
    """Copy the selected model's artifacts to the backend and write model_config.json (the manifest)."""
    deploy_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict[str, Any]] = {}
    for role, src in (("model", entry.model_path), ("scaler", entry.scaler_path)):
        if src is None:
            continue
        dst = deploy_dir / Path(src).name
        shutil.copy2(src, dst)
        copied[role] = {"file": dst.name, "sha256": sha256_of(dst), "bytes": dst.stat().st_size}

    mlp_meta = {}
    meta_path = models_dir / "metrics_mlp.json"
    if entry.kind == "keras" and meta_path.exists():
        mlp_meta = json.loads(meta_path.read_text())

    manifest = {
        "manifest_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "selected_model": {
            "key": entry.key,
            "label": entry.label,
            "variant": entry.variant,
            "model_type": "keras_sequential_mlp"
            if entry.kind == "keras"
            else f"sklearn_pipeline:{entry.params.get('estimator', entry.key)}",
            "framework": entry.runtime,
            "model_version": (
                mlp_meta.get("training", {}).get("finished_at") or datetime.now(UTC).isoformat()
            ),
            "architecture": mlp_meta.get("architecture"),
        },
        "artifacts": {
            "directory": str(deploy_dir),
            "model_file": copied["model"]["file"],
            "model_sha256": copied["model"]["sha256"],
            "model_bytes": copied["model"]["bytes"],
            "scaler_file": copied.get("scaler", {}).get("file"),
            "scaler_sha256": copied.get("scaler", {}).get("sha256"),
            "load_hint": "tf.keras.models.load_model(model_file, compile=False); pickle.load(scaler_file)"
            if entry.kind == "keras"
            else "pickle.load(model_file) -> sklearn Pipeline (scaler included)",
        },
        "input": {
            "dimension": N_MODEL_FEATURES,
            "feature_order": list(MODEL_FEATURE_NAMES),
            "dtype": "float32 after scaling",
            "preprocessing": {
                "scaler": "StandardScaler fitted on train.csv only (apply scaler.transform before the model)"
                if entry.kind == "keras"
                else "StandardScaler inside the Pipeline (no manual scaling)",
                "feature_extractor": {
                    "package": "voice_features",
                    "version": voice_features_version,
                    "sample_rate": CANONICAL_SAMPLE_RATE,
                    "spec": "docs/acoustic_features.md",
                },
            },
        },
        "output": {
            "type": "softmax probabilities" if entry.kind == "keras" else "predict_proba",
            "class_mapping": {str(i): name for i, name in enumerate(CLASS_NAMES)},
            "one_hot": {"male": [1, 0], "female": [0, 1]},
            "decision": "argmax over the two probabilities",
        },
        "benchmark_metrics": {
            "feature_space": "historical voice.csv (R-extracted features), see docs/dataset_provenance.md",
            "data_dir": str(data_dir),
            "validation": {
                k: selected["validation"][k]
                for k in ("n_samples", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc")
            },
            "test": {
                k: selected["test"][k]
                for k in ("n_samples", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc")
            },
            "test_confusion_matrix": selected["test"]["confusion_matrix"],
            "inference": {
                k: selected["resources"][k]
                for k in ("single_call_mean_ms", "batch_1000_ms", "memory_footprint_mb", "load_seconds")
            },
        },
        "validity": {
            "benchmark_validated_on_voice_csv": True,
            "live_audio_end_to_end_validated": False,
            "note": "Accuracy on live recordings through the Python feature extractor has not been "
            "measured (no labelled corpus).",
        },
        "selection": {
            k: selection[k]
            for k in (
                "policy",
                "tied_candidates",
                "selected",
                "engineering_alternative",
                "edge_alternative",
                "backup",
                "reasoning",
            )
        },
        "alternatives": {
            r["key"]: {
                "label": r["label"],
                "model_path": str(r["model_path"]),
                "test_accuracy": r["test"]["accuracy"],
                "single_call_mean_ms": r["resources"]["single_call_mean_ms"],
                "memory_footprint_mb": r["resources"]["memory_footprint_mb"],
            }
            for r in rows
            if r["key"]
            in {selection["edge_alternative"], selection["backup"], selection["engineering_alternative"]}
            and r["key"] != entry.key
        },
        "git": get_git_info(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    manifest_path = deploy_dir / "model_config.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {"manifest": str(manifest_path), "copied": copied}


def write_markdown(report: dict[str, Any], table: pd.DataFrame, output_dir: Path) -> Path:
    sel = report["selection"]
    rows = {r["key"]: r for r in report["models"]}
    chosen = rows[sel["selected"]]
    edge = rows[sel["edge_alternative"]]
    backup = rows[sel["backup"]] if sel["backup"] else None
    best = report["analysis"]["best"]

    def pct(x: float) -> str:
        return f"{x * 100:.2f} %"

    n_base = len([r for r in report["models"] if r["variant"] == "base"])
    n_tuned = len([r for r in report["models"] if r["variant"] == "tuned"])
    c_test, c_val, c_res = chosen["test"], chosen["validation"], chosen["resources"]
    deployment = report["deployment"]
    artifacts_text = f"`{Path(deployment['copied']['model']['file']).name}`"
    if "scaler" in deployment["copied"]:
        artifacts_text += f" and `{deployment['copied']['scaler']['file']}`"

    lines = [
        "# Model Selection Analysis",
        "",
        f"Generated {report['generated_at']} from the artifacts of Phases 2B and 2C. All figures below are "
        "**measured by this codebase** on the historical `voice.csv` benchmark feature space (see "
        "`docs/dataset_provenance.md`); they say nothing about end-to-end accuracy on live audio, which "
        "has not been measured.",
        "",
        "## Executive summary",
        "",
        f"- {n_base} base models and {n_tuned} CV-tuned variants were evaluated on the same untouched test "
        f"split (n = {c_test['n_samples']}).",
        f"- Best test accuracy {pct(best['accuracy']['value'])}: {', '.join(best['accuracy']['models'])}. "
        f"Best F1 {best['f1']['value']:.4f}: {', '.join(best['f1']['models'])}. "
        f"Fastest single-call inference: {best['fastest']['models'][0]} ({best['fastest']['value']:.3f} ms). "
        f"Smallest memory footprint: {best['smallest']['models'][0]} ({best['smallest']['value']:.1f} MB).",
        f"- **Selected production model: {chosen['label']}** (validation accuracy {pct(c_val['accuracy'])}, "
        f"test accuracy {pct(c_test['accuracy'])}, test F1 {c_test['f1_macro']:.4f}, "
        f"ROC-AUC {c_test['roc_auc']:.4f}).",
        "- Selection used validation / cross-validation results and the pre-declared tie-breakers only; "
        "test metrics are final reporting. See *Methodology flags*.",
        "",
        "## Metrics table (test split, sorted by accuracy)",
        "",
        table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "`Inf. Time` is the mean of 1000 single-sample calls on the lowest-overhead path (Keras direct "
        "call / `predict_proba`); `Memory` is the process RSS increase from importing the runtime, "
        "loading the model and predicting, measured in an isolated process; `Artifact` is the on-disk "
        "size.",
        "",
        "## Best per criterion",
        "",
    ]
    for crit, info in best.items():
        lines.append(f"- **{crit}**: {', '.join(info['models'])} ({info['display']})")
    lines += ["", "## Overfitting check (train vs test accuracy, flag if gap > 10 pp)", ""]
    for r in report["models"]:
        o = r["overfitting"]
        flag = " **FLAG**" if o["flag"] else ""
        lines.append(
            f"- {r['label']}: train {pct(o['train_accuracy'])}, test {pct(o['test_accuracy'])}, "
            f"gap {o['gap_pp']:.2f} pp{flag}"
        )
    lines += [
        "",
        "## Trade-off analysis",
        "",
        "![grid](comparison_plots.png)",
        "",
        "Individual figures: `accuracy_comparison.png`, `confusion_matrices.png`, `roc_curves.png`, "
        "`accuracy_vs_speed.png`, `accuracy_vs_memory.png`.",
        "",
        "## Selection policy and reasoning",
        "",
        f"Policy: accuracy criterion = {sel['policy']['accuracy_criterion']}; "
        f"tie band = {sel['policy']['tie_band_pp']} pp; "
        f"tie-breakers = {', '.join(sel['policy']['tie_breakers'])}; "
        f"test set used for selection = {sel['policy']['test_set_used_for_selection']}.",
        "",
    ]
    lines += [f"- {line}" for line in sel["reasoning"]]
    lines += ["", "## Methodology flags", ""]
    lines += [f"- {flag}" for flag in report["methodology_flags"]]
    lines += [
        "",
        "## Recommendations",
        "",
        f"- **Production model:** {chosen['label']} — test accuracy {pct(c_test['accuracy'])}, "
        f"F1 {c_test['f1_macro']:.4f}, ROC-AUC {c_test['roc_auc']:.4f}, "
        f"{c_res['single_call_mean_ms']:.2f} ms per single call, "
        f"{c_res['memory_footprint_mb']:.0f} MB footprint.",
        f"- **Alternative for edge / low-resource deployment:** {edge['label']} — test accuracy "
        f"{pct(edge['test']['accuracy'])}, {edge['resources']['single_call_mean_ms']:.2f} ms per call, "
        f"{edge['resources']['artifact_bytes'] / 1024:.0f} KB artifact, "
        f"{edge['resources']['memory_footprint_mb']:.0f} MB footprint.",
    ]
    if backup:
        lines.append(
            f"- **Backup model:** {backup['label']} — validation accuracy "
            f"{pct(backup['validation']['accuracy'])}, test accuracy {pct(backup['test']['accuracy'])}, "
            f"{backup['resources']['single_call_mean_ms']:.2f} ms per call."
        )
    if sel["engineering_alternative"] and sel["engineering_alternative"] != sel["selected"]:
        eng = rows[sel["engineering_alternative"]]
        lines.append(
            "- If the neural-network objective were dropped, the engineering choice among the tied models "
            f"would be {eng['label']}."
        )
    lines += [
        "",
        "## Implementation notes for the chosen model",
        "",
        f"- Artifacts: {artifacts_text} copied to `{deployment['deploy_dir']}` with `model_config.json` as "
        "the single manifest (feature order, class mapping, scaler, input dimension, metrics, hashes).",
        f"- Input: {N_MODEL_FEATURES} features in the documented order, extracted with `voice_features` at "
        f"{CANONICAL_SAMPLE_RATE} Hz, standardised with the saved scaler (fitted on train only).",
        "- Output: two softmax probabilities in class order (0 = male, 1 = female); take the argmax and "
        "report the max probability as confidence.",
        f"- Latency: use the direct Keras call for single requests (~{c_res['single_call_mean_ms']:.1f} ms) "
        "rather than `model.predict`, and run one warm-up prediction at startup; cold load took "
        f"{c_res['load_seconds']:.1f} s in the isolated measurement.",
        "",
        "## Deployment requirements",
        "",
        f"- Runtime: {chosen['runtime']} on Python 3.10/3.11; approximately "
        f"{c_res['memory_footprint_mb']:.0f} MB RAM for the runtime plus model.",
        "- Load the model with `compile=False`; keep the scaler pickle next to it; verify SHA-256 hashes "
        "from the manifest at startup.",
        "- Serve behind the FastAPI backend (CPU is sufficient: a single prediction is milliseconds).",
        f"- Alternatives are already trained and saved: `{edge['model_path']}` for edge use"
        + (f", `{backup['model_path']}` as backup." if backup else "."),
        "",
        "## Historical / reference results (not produced by this codebase, not comparable)",
        "",
        "| Result | Accuracy | Source | Comparable? |",
        "|---|---|---|---|",
    ]
    for h in HISTORICAL_RESULTS:
        lines.append(f"| {h['label']} | {pct(h['accuracy'])} | {h['source']} | {h['comparable']} |")
    lines += [
        "",
        "These figures come from different splits, feature extractors or datasets and are listed for "
        "context only. The 96.74 % of the 2016 paper is not a target of this project and was not used in "
        "the selection.",
        "",
    ]
    path = output_dir / "comparison_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        if getattr(handler, "_cmp", False):
            logger.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "comparison_log.txt", mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        handler._cmp = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False


def run(
    models_dir: Path = DEFAULT_MODELS_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    deploy_dir: Path | None = DEFAULT_DEPLOY_DIR,
    isolate: bool = True,
    include_tuned: bool = True,
) -> dict[str, Any]:
    setup_logging(output_dir)
    started = datetime.now(UTC)
    logger.info("=== model comparison started %s ===", started.isoformat())

    data: dict[str, Any] = {}
    for split, name in (("train", "train"), ("val", "validation"), ("test", "test")):
        X_df, y_lbl = load_split(data_dir / f"{split}.csv")
        data[f"X_{name}"] = X_df.to_numpy(dtype=np.float64)
        data[f"y_{name}"] = labels_to_index(y_lbl)
        logger.info("%s.csv: %d rows", split, len(X_df))

    entries = discover_models(models_dir, include_tuned=include_tuned)
    logger.info("Models found: %s", [e.key for e in entries])

    rows: list[dict[str, Any]] = []
    for entry in entries:
        logger.info("--- %s ---", entry.label)
        resources = (
            measure_resources_isolated(entry, models_dir, data_dir)
            if isolate
            else measure_resources_inprocess(entry, data["X_test"])
        )
        if entry._model is None:
            entry.load()
        metrics = evaluate_entry(entry, data)
        row = {
            "key": entry.key,
            "label": entry.label,
            "variant": entry.variant,
            "kind": entry.kind,
            "runtime": entry.runtime,
            "complexity": entry.complexity,
            "meets_nn_objective": entry.meets_nn_objective,
            "model_path": str(entry.model_path),
            "scaler_path": str(entry.scaler_path) if entry.scaler_path else None,
            "params": entry.params,
            "training_seconds": entry.training_seconds,
            "cv_train": entry.cv_train,
            "train": metrics["train"],
            "validation": metrics["validation"],
            "test": metrics["test"],
            "resources": resources,
            "overfitting": overfitting_check(metrics),
        }
        rows.append(row)
        logger.info(
            "val acc %.4f | test acc %.4f F1 %.4f AUC %.4f | %.3f ms/call | %.1f MB footprint | gap %.2f pp",
            row["validation"]["accuracy"],
            row["test"]["accuracy"],
            row["test"]["f1_macro"],
            row["test"]["roc_auc"],
            resources["single_call_mean_ms"],
            resources["memory_footprint_mb"],
            row["overfitting"]["gap_pp"],
        )

    # selection BEFORE looking at the test-sorted table (policy uses validation only)
    selection = select_production_model(rows)
    for line in selection["reasoning"]:
        logger.info("selection: %s", line)

    table = comparison_table(rows)
    csv_path = output_dir / "comparison_metrics.csv"
    table.to_csv(csv_path, index=False)
    with pd.option_context(
        "display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format
    ):
        logger.info("Comparison table (test split):\n%s", table.to_string(index=False))

    def best_of(metric_fn, reverse: bool, fmt) -> dict[str, Any]:
        values = {r["label"]: metric_fn(r) for r in rows}
        target = max(values.values()) if reverse else min(values.values())
        winners = [k for k, v in values.items() if abs(v - target) < 1e-12]
        return {"value": target, "models": winners, "display": fmt(target)}

    analysis = {
        "best": {
            "accuracy": best_of(lambda r: r["test"]["accuracy"], True, lambda v: f"{v:.4f}"),
            "f1": best_of(lambda r: r["test"]["f1_macro"], True, lambda v: f"{v:.4f}"),
            "roc_auc": best_of(lambda r: r["test"]["roc_auc"], True, lambda v: f"{v:.4f}"),
            "fastest": best_of(
                lambda r: r["resources"]["single_call_mean_ms"], False, lambda v: f"{v:.3f} ms per call"
            ),
            "smallest": best_of(
                lambda r: r["resources"]["memory_footprint_mb"], False, lambda v: f"{v:.1f} MB footprint"
            ),
            "smallest_artifact": best_of(
                lambda r: r["resources"]["artifact_bytes"], False, lambda v: f"{v / 1024:.0f} KB on disk"
            ),
        },
        "overfitting_flags": [r["key"] for r in rows if r["overfitting"]["flag"]],
    }

    plot_paths = make_plots(rows, data["y_test"], output_dir)

    deployment: dict[str, Any] = {"deploy_dir": None, "copied": {}}
    if deploy_dir is not None:
        chosen_entry = next(e for e in entries if e.key == selection["selected"])
        chosen_row = next(r for r in rows if r["key"] == selection["selected"])
        deployment = deploy_selected(
            chosen_row, chosen_entry, models_dir, deploy_dir, rows, selection, data_dir
        )
        deployment["deploy_dir"] = str(deploy_dir)
        logger.info("Deployed %s to %s (manifest %s)", chosen_entry.label, deploy_dir, deployment["manifest"])

    finished = datetime.now(UTC)
    report = {
        "title": "Model Selection Analysis",
        "generated_at": finished.isoformat(),
        "data": {
            "data_dir": str(data_dir),
            "test_rows": int(len(data["y_test"])),
            "feature_order": list(MODEL_FEATURE_NAMES),
            "class_mapping": CLASS_TO_INDEX,
        },
        "models": [
            {**r, "test": {k: v for k, v in r["test"].items() if k != "probabilities_female"}} for r in rows
        ],
        "selection": selection,
        "analysis": analysis,
        "methodology_flags": METHODOLOGY_FLAGS,
        "historical_reference": HISTORICAL_RESULTS,
        "deployment": deployment,
        "artifacts": {
            "csv": str(csv_path),
            "plots": {k: str(v) for k, v in plot_paths.items()},
            "log": str(output_dir / "comparison_log.txt"),
        },
        "git": get_git_info(),
        "caveat": (
            "Metrics are on the voice.csv feature space; live-audio end-to-end accuracy is unvalidated."
        ),
        "total_seconds": (finished - started).total_seconds(),
    }
    (output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, default=str))
    md_path = write_markdown(report, table, output_dir)
    logger.info("Reports: %s, %s, %s", output_dir / "comparison_report.json", md_path, csv_path)
    logger.info("=== model comparison finished (%.1f s) ===", report["total_seconds"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare trained models and select the production model")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--deploy-dir", type=Path, default=DEFAULT_DEPLOY_DIR)
    parser.add_argument("--no-deploy", action="store_true")
    parser.add_argument(
        "--no-isolate",
        action="store_true",
        help="Measure resources in-process instead of per-model subprocesses",
    )
    parser.add_argument(
        "--measure", metavar="MODEL_KEY", help=argparse.SUPPRESS
    )  # internal: subprocess entry point
    args = parser.parse_args(argv)

    if args.measure:
        _measure_entrypoint(args.measure, args.models_dir, args.data_dir)
        return 0
    run(
        args.models_dir,
        args.data_dir,
        args.output,
        None if args.no_deploy else args.deploy_dir,
        isolate=not args.no_isolate,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
