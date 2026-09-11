"""Train and compare scikit-learn classifiers on the same splits as the MLP.

Models (hyperparameters fixed by the project specification):

* Random Forest, Support Vector Machine (RBF), Gradient Boosting, and a
  Decision Tree baseline.

Every model is a ``Pipeline([("scaler", StandardScaler()), ("model", clf)])``
fitted on ``train.csv`` only, so the scaler never sees validation or test rows
(it is verified to equal the Phase 2A scaler). Labels are integers with the
project-wide mapping male = 0, female = 1. Evaluation uses the same
``val.csv`` / ``test.csv`` files as the MLP and the same metric code
(``evaluation.py``), so the comparison is like-for-like.

Usage::

    python train_sklearn_models.py --data-dir data/processed/voice_csv --output models/sklearn [--tune]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import platform
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import CLASS_NAMES, CLASS_TO_INDEX, labels_to_index, log_metrics, metrics_from_probabilities
from train_neural_network import get_git_info, load_split
from voice_features import MODEL_FEATURE_NAMES, N_MODEL_FEATURES

ML_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ML_DIR / "data" / "processed"
DEFAULT_OUTPUT_DIR = ML_DIR / "models" / "sklearn"
MLP_METRICS_PATH = ML_DIR / "models" / "metrics_mlp.json"

logger = logging.getLogger("train_sklearn_models")


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    grid: dict[str, list[Any]]
    baseline: bool = False


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("random_forest", "Random Forest", {"model__n_estimators": [100, 200, 300]}),
    ModelSpec("svm", "SVM (RBF)", {"model__C": [0.1, 1, 10]}),
    ModelSpec("gradient_boosting", "Gradient Boosting", {"model__learning_rate": [0.01, 0.1, 0.2]}),
    ModelSpec(
        "decision_tree", "Decision Tree (baseline)", {"model__max_depth": [None, 5, 10, 20]}, baseline=True
    ),
)
SPEC_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


def make_classifier(key: str, seed: int = 42):
    """The exact estimator configuration from the specification."""
    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced",
        )
    if key == "svm":
        return SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed
        )
    if key == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=seed,
        )
    if key == "decision_tree":
        # Baseline: an unconstrained single tree, balanced classes, same seed.
        return DecisionTreeClassifier(random_state=seed, class_weight="balanced")
    raise KeyError(key)


def make_pipeline(key: str, seed: int = 42) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", make_classifier(key, seed))])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data(data_dir: Path) -> dict[str, Any]:
    """train/val/test features (raw, in MODEL_FEATURE_NAMES order) and 0/1 labels."""
    out: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        X_df, y_lbl = load_split(data_dir / f"{split}.csv")
        out[f"X_{split}"] = X_df.to_numpy(dtype=np.float64)
        out[f"y_{split}"] = labels_to_index(y_lbl)
        out[f"{split}_stats"] = {
            "rows": int(len(X_df)),
            "labels": {name: int((out[f"y_{split}"] == i).sum()) for name, i in CLASS_TO_INDEX.items()},
        }
    return out


def check_scaler_matches_phase_2a(pipeline: Pipeline, data_dir: Path) -> str:
    """Confirm the pipeline's scaler (fit on train.csv) equals the Phase 2A scaler.joblib."""
    existing = data_dir / "scaler.joblib"
    if not existing.exists():
        return "no Phase 2A scaler.joblib found; pipeline scaler fitted on train.csv"
    reference = joblib.load(existing)
    fitted = pipeline.named_steps["scaler"]
    if np.allclose(reference.mean_, fitted.mean_) and np.allclose(reference.scale_, fitted.scale_):
        return f"pipeline scaler fitted on train.csv is identical to {existing.name} (Phase 2A)"
    return f"WARNING: pipeline scaler differs from {existing.name}"


# ---------------------------------------------------------------------------
# Training, cross-validation, timing, importance
# ---------------------------------------------------------------------------


def fit_one(key: str, X: np.ndarray, y: np.ndarray, seed: int) -> tuple[str, Pipeline, float]:
    """Fit one pipeline (executed in a worker process)."""
    pipeline = make_pipeline(key, seed)
    t0 = time.perf_counter()
    pipeline.fit(X, y)
    return key, pipeline, time.perf_counter() - t0


def fit_all_parallel(
    keys: list[str], X: np.ndarray, y: np.ndarray, seed: int, n_jobs: int
) -> dict[str, tuple[Pipeline, float]]:
    """Train all models concurrently (one process per model)."""
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(fit_one)(key, X, y, seed) for key in keys)
    return {key: (pipeline, seconds) for key, pipeline, seconds in results}


def cross_validate(key: str, X: np.ndarray, y: np.ndarray, seed: int, folds: int = 5) -> dict[str, Any]:
    """5-fold stratified CV accuracy on the training split with a 95 % t-interval."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(make_pipeline(key, seed), X, y, cv=cv, scoring="accuracy", n_jobs=1)
    mean, std = float(scores.mean()), float(scores.std(ddof=1))
    half_width = float(stats.t.ppf(0.975, df=folds - 1) * std / np.sqrt(folds))
    return {
        "folds": folds,
        "scores": [float(s) for s in scores],
        "mean": mean,
        "std": std,
        "ci95": [mean - half_width, mean + half_width],
    }


def time_inference(pipeline: Pipeline, X: np.ndarray, n: int = 1000, repeats: int = 3) -> dict[str, float]:
    """Batch latency for n rows and per-sample latency for n single-row calls (milliseconds)."""
    X_n = np.resize(X, (n, X.shape[1]))
    batch_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        pipeline.predict_proba(X_n)
        batch_times.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    for row in X_n:
        pipeline.predict_proba(row[None, :])
    single_total = time.perf_counter() - t0
    return {
        "n_predictions": n,
        "batch_total_ms": float(min(batch_times) * 1000),
        "batch_per_prediction_ms": float(min(batch_times) * 1000 / n),
        "single_call_total_ms": float(single_total * 1000),
        "single_call_mean_ms": float(single_total * 1000 / n),
    }


def feature_importance(
    key: str, pipeline: Pipeline, X_val: np.ndarray, y_val: np.ndarray, seed: int
) -> tuple[dict[str, float], str]:
    """Native feature_importances_ where the estimator has them; permutation importance otherwise (SVM)."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=np.float64)
        method = "feature_importances_ (impurity-based)"
    else:
        result = permutation_importance(
            pipeline, X_val, y_val, n_repeats=10, random_state=seed, scoring="accuracy", n_jobs=1
        )
        values = np.asarray(result.importances_mean, dtype=np.float64)
        method = "permutation importance on val.csv (accuracy drop, 10 repeats)"
    ordered = sorted(zip(MODEL_FEATURE_NAMES, values, strict=True), key=lambda kv: kv[1], reverse=True)
    return {name: float(v) for name, v in ordered}, method


def pickle_size(obj: Any) -> int:
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def tune_one(key: str, X: np.ndarray, y: np.ndarray, seed: int, folds: int = 5) -> dict[str, Any]:
    spec = SPEC_BY_KEY[key]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        make_pipeline(key, seed), spec.grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    seconds = time.perf_counter() - t0
    table = [
        {
            "params": {k: (None if v is None else v) for k, v in p.items()},
            "mean_cv_accuracy": float(m),
            "std": float(s),
        }
        for p, m, s in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
            strict=True,
        )
    ]
    return {
        "grid": spec.grid,
        "best_params": search.best_params_,
        "best_cv_accuracy": float(search.best_score_),
        "search_seconds": seconds,
        "results": table,
        "estimator": search.best_estimator_,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_accuracy(report_models: dict[str, Any], ax) -> None:
    keys = list(report_models)
    labels = [report_models[k]["label"] for k in keys]
    train = [report_models[k]["metrics"]["train"]["accuracy"] for k in keys]
    test = [report_models[k]["metrics"]["test"]["accuracy"] for k in keys]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, train, 0.4, label="train", color="#a78bfa")
    ax.bar(x + 0.2, test, 0.4, label="test", color="#7c3aed")
    for i, (tr, te) in enumerate(zip(train, test, strict=True)):
        ax.text(i - 0.2, tr + 0.002, f"{tr:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, te + 0.002, f"{te:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylim(min(min(test), 0.9) - 0.02, 1.005)
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy: train vs test")
    ax.legend(loc="lower right")


def plot_confusion(cm: list[list[int]], label: str, ax) -> None:
    arr = np.asarray(cm)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(2), CLASS_NAMES)
    ax.set_yticks(range(2), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{label} (test)")
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
    return im


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], ax) -> None:
    for label, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{label} (AUC {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves (test set, positive class = female)")
    ax.legend(loc="lower right", fontsize=8)


def plot_importance(importance: dict[str, float], label: str, ax, top: int = 15) -> None:
    items = list(importance.items())[:top][::-1]
    names = [k for k, _ in items]
    values = [v for _, v in items]
    ax.barh(names, values, color="#2563eb")
    ax.set_title(f"{label}: top {min(top, len(items))} features")
    ax.tick_params(labelsize=8)


def save_all_plots(
    report_models: dict[str, Any], curves: dict, importances: dict[str, dict[str, float]], output_dir: Path
) -> dict[str, Path]:
    plt = _plt()
    paths: dict[str, Path] = {}
    keys = list(report_models)
    n = len(keys)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_accuracy(report_models, ax)
    fig.tight_layout()
    paths["accuracy"] = output_dir / "accuracy_comparison.png"
    fig.savefig(paths["accuracy"], dpi=120)
    plt.close(fig)

    cols = 2 if n <= 4 else 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4 * rows))
    for ax, key in zip(np.ravel(axes), keys, strict=False):
        plot_confusion(
            report_models[key]["metrics"]["test"]["confusion_matrix"]["matrix"],
            report_models[key]["label"],
            ax,
        )
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.tight_layout()
    paths["confusion"] = output_dir / "confusion_matrices.png"
    fig.savefig(paths["confusion"], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_roc(curves, ax)
    fig.tight_layout()
    paths["roc"] = output_dir / "roc_curves.png"
    fig.savefig(paths["roc"], dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    for ax, key in zip(np.ravel(axes), keys, strict=False):
        plot_importance(importances[key], report_models[key]["label"], ax, top=15)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.tight_layout()
    paths["importance"] = output_dir / "feature_importance_top15.png"
    fig.savefig(paths["importance"], dpi=120)
    plt.close(fig)

    for key in keys:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        plot_importance(importances[key], report_models[key]["label"], ax, top=10)
        fig.tight_layout()
        p = output_dir / f"feature_importance_{key}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths[f"importance_{key}"] = p

    # Combined grid: row 1 accuracy + ROC, row 2 confusion matrices, row 3 importances
    fig = plt.figure(figsize=(5.2 * max(n, 2), 15))
    grid = fig.add_gridspec(3, max(n, 2))
    plot_accuracy(report_models, fig.add_subplot(grid[0, : max(n // 2, 1)]))
    plot_roc(curves, fig.add_subplot(grid[0, max(n // 2, 1) :]))
    for i, key in enumerate(keys):
        plot_confusion(
            report_models[key]["metrics"]["test"]["confusion_matrix"]["matrix"],
            report_models[key]["label"],
            fig.add_subplot(grid[1, i]),
        )
        plot_importance(importances[key], report_models[key]["label"], fig.add_subplot(grid[2, i]), top=15)
    fig.suptitle("scikit-learn model comparison (voice.csv benchmark feature space)", fontsize=14)
    fig.tight_layout()
    paths["grid"] = output_dir / "comparison_plots.png"
    fig.savefig(paths["grid"], dpi=110)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        if getattr(handler, "_sk", False):
            logger.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "training_log.txt", mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        handler._sk = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False


def mlp_reference(path: Path = MLP_METRICS_PATH) -> dict[str, Any] | None:
    """Phase 2B MLP results on the same val/test files, for the side-by-side table."""
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    return {
        "label": "MLP (Phase 2B, TensorFlow)",
        "source": str(path),
        "data_dir": m["data"]["data_dir"],
        "note": "Trained on 90 % of train.csv (10 % held out for early stopping); same val.csv / test.csv.",
        "training_seconds": m["training"]["training_seconds"],
        "metrics": {"validation": m["metrics"]["validation"], "test": m["metrics"]["test"]},
    }


def run(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    tune: bool = False,
    model_keys: list[str] | None = None,
    n_jobs: int = -1,
    cv_folds: int = 5,
) -> dict[str, Any]:
    setup_logging(output_dir)
    started = datetime.now(UTC)
    keys = model_keys or [spec.key for spec in MODEL_SPECS]
    logger.info("=== scikit-learn training started %s ===", started.isoformat())
    logger.info("Models: %s | seed %d | data %s | tune=%s", keys, seed, data_dir, tune)
    logger.info(
        "Versions: scikit-learn %s, numpy %s, python %s",
        sklearn.__version__,
        np.__version__,
        platform.python_version(),
    )

    data = load_data(data_dir)
    for split in ("train", "val", "test"):
        logger.info("%s.csv: %s", split, data[f"{split}_stats"])
    logger.info("Feature order (%d): %s", N_MODEL_FEATURES, list(MODEL_FEATURE_NAMES))
    logger.info("Class mapping: %s", CLASS_TO_INDEX)

    # --- parallel training ---------------------------------------------------
    t0 = time.perf_counter()
    fitted = fit_all_parallel(
        keys, data["X_train"], data["y_train"], seed, n_jobs=min(len(keys), 4) if n_jobs == -1 else n_jobs
    )
    logger.info("All models trained in %.1f s wall time (parallel)", time.perf_counter() - t0)
    scaler_note = check_scaler_matches_phase_2a(fitted[keys[0]][0], data_dir)
    logger.info("Scaler: %s", scaler_note)

    report_models: dict[str, Any] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    importances: dict[str, dict[str, float]] = {}
    for key in keys:
        spec = SPEC_BY_KEY[key]
        pipeline, train_seconds = fitted[key]
        logger.info("--- %s (trained in %.2f s) ---", spec.label, train_seconds)

        metrics = {}
        for split in ("train", "val", "test"):
            probs = pipeline.predict_proba(data[f"X_{split}"])
            name = {"train": "train", "val": "validation", "test": "test"}[split]
            metrics[name] = metrics_from_probabilities(data[f"y_{split}"], probs, name)
            log_metrics(metrics[name], logger, prefix=f"{spec.label} ")
            if split == "test":
                fpr, tpr, _ = roc_curve(data["y_test"], probs[:, CLASS_TO_INDEX["female"]])
                curves[spec.label] = (fpr, tpr, metrics[name]["roc_auc"])

        cv = cross_validate(key, data["X_train"], data["y_train"], seed, cv_folds)
        logger.info(
            "%s %d-fold CV accuracy: %.4f +/- %.4f (95%% CI %.4f-%.4f)",
            spec.label,
            cv_folds,
            cv["mean"],
            cv["std"],
            *cv["ci95"],
        )

        timing = time_inference(pipeline, data["X_test"])
        logger.info(
            "%s inference: %.2f ms per 1000-row batch, %.3f ms per single call",
            spec.label,
            timing["batch_total_ms"],
            timing["single_call_mean_ms"],
        )

        importance, method = feature_importance(key, pipeline, data["X_val"], data["y_val"], seed)
        importances[key] = importance
        logger.info("%s top-5 features (%s): %s", spec.label, method, list(importance)[:5])

        model_path = output_dir / f"{key}_model.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(pipeline, fh, protocol=pickle.HIGHEST_PROTOCOL)
        scaler_path = output_dir / f"{key}_scaler.pkl"
        with scaler_path.open("wb") as fh:
            pickle.dump(pipeline.named_steps["scaler"], fh, protocol=pickle.HIGHEST_PROTOCOL)

        report_models[key] = {
            "label": spec.label,
            "baseline": spec.baseline,
            "estimator": pipeline.named_steps["model"].__class__.__name__,
            "params": {
                k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                for k, v in pipeline.named_steps["model"].get_params().items()
            },
            "training_seconds": train_seconds,
            "cross_validation_train": cv,
            "metrics": metrics,
            "inference_timing": timing,
            "model_size_bytes": {"pipeline_pickle": pickle_size(pipeline), "file": model_path.stat().st_size},
            "feature_importance_method": method,
            "feature_importance_top10": dict(list(importance.items())[:10]),
            "artifacts": {"model": str(model_path), "scaler": str(scaler_path)},
        }

    (output_dir / "feature_importance.json").write_text(
        json.dumps(
            {
                key: {
                    "method": report_models[key]["feature_importance_method"],
                    "importance": importances[key],
                }
                for key in keys
            },
            indent=2,
        )
    )

    # --- optional tuning -----------------------------------------------------
    tuning: dict[str, Any] | None = None
    if tune:
        tuning = {}
        for key in keys:
            spec = SPEC_BY_KEY[key]
            logger.info("Grid search %s over %s", spec.label, spec.grid)
            result = tune_one(key, data["X_train"], data["y_train"], seed, cv_folds)
            best = result.pop("estimator")
            probs = best.predict_proba(data["X_test"])
            test_metrics = metrics_from_probabilities(data["y_test"], probs, "test")
            tuned_path = output_dir / f"{key}_tuned_model.pkl"
            with tuned_path.open("wb") as fh:
                pickle.dump(best, fh, protocol=pickle.HIGHEST_PROTOCOL)
            result["test_metrics"] = test_metrics
            result["artifact"] = str(tuned_path)
            tuning[key] = result
            logger.info(
                "%s best params %s (CV %.4f) -> test accuracy %.4f in %.1f s",
                spec.label,
                result["best_params"],
                result["best_cv_accuracy"],
                test_metrics["accuracy"],
                result["search_seconds"],
            )
        (output_dir / "tuning_results.json").write_text(json.dumps(tuning, indent=2, default=str))

    # --- plots and report ----------------------------------------------------
    plot_paths = save_all_plots(report_models, curves, importances, output_dir)

    ranking = sorted(keys, key=lambda k: report_models[k]["metrics"]["test"]["accuracy"], reverse=True)
    finished = datetime.now(UTC)
    report = {
        "generated_at": finished.isoformat(),
        "total_seconds": (finished - started).total_seconds(),
        "data": {
            "data_dir": str(data_dir),
            "splits": {s: data[f"{s}_stats"] for s in ("train", "val", "test")},
            "preprocessing": {
                "scaler": "StandardScaler inside each Pipeline, fitted on train.csv only",
                "scaler_check": scaler_note,
                "feature_order": list(MODEL_FEATURE_NAMES),
                "class_mapping": CLASS_TO_INDEX,
                "labels": "integers, male=0, female=1 (no one-hot)",
            },
            "provenance": "docs/dataset_provenance.md (historical voice.csv, R-extracted features)",
        },
        "models": report_models,
        "ranking_by_test_accuracy": [
            {
                "model": k,
                "label": report_models[k]["label"],
                "test_accuracy": report_models[k]["metrics"]["test"]["accuracy"],
            }
            for k in ranking
        ],
        "mlp_reference": mlp_reference(),
        "tuning": tuning,
        "plots": {k: str(v) for k, v in plot_paths.items()},
        "git": get_git_info(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "caveat": (
            "All metrics are measured on the historical voice.csv feature space (features extracted "
            "in R by the dataset author). They do not measure end-to-end accuracy on live audio "
            "through the Python extractor."
        ),
    }
    (output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, default=str))

    logger.info(
        "Ranking by test accuracy: %s",
        [(r["label"], round(r["test_accuracy"], 4)) for r in report["ranking_by_test_accuracy"]],
    )
    if report["mlp_reference"]:
        logger.info(
            "MLP (Phase 2B) test accuracy for comparison: %.4f",
            report["mlp_reference"]["metrics"]["test"]["accuracy"],
        )
    logger.info("Report written to %s", output_dir / "comparison_report.json")
    logger.info(
        "=== scikit-learn training finished %s (total %.1f s) ===",
        finished.isoformat(),
        report["total_seconds"],
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and compare scikit-learn classifiers on the 20 acoustic features"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune", action="store_true", help="Run GridSearchCV with the specified grids")
    parser.add_argument("--models", nargs="*", choices=[s.key for s in MODEL_SPECS], help="Subset of models")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args(argv)
    run(
        args.data_dir,
        args.output,
        seed=args.seed,
        tune=args.tune,
        model_keys=args.models,
        cv_folds=args.cv_folds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
