"""Train the gender-recognition MLP on the 20 acoustic features.

Architecture (fixed by the project specification, inherited from the original
B.Tech project and Buyukyilmaz & Cibikdiken 2016):

    Input(20) -> Dense(64, relu) -> Dropout(0.25)
              -> Dense(256, relu) -> Dropout(0.25)
              -> Dense(256, relu) -> Dropout(0.25)
              -> Dense(64, relu) -> Dropout(0.25)
              -> Dense(2, softmax)

Class mapping: index 0 = male (one-hot [1, 0]), index 1 = female ([0, 1]).

Usage::

    python train_neural_network.py --data-dir data/processed/voice_csv --output models

Inputs are the ``train.csv`` / ``val.csv`` / ``test.csv`` files written by
``data_preparation.py`` (raw feature values). Every artefact needed for
inference is written to ``--output``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import voice_features
from evaluation import CLASS_NAMES, CLASS_TO_INDEX, log_metrics, metrics_from_probabilities
from voice_features import MODEL_FEATURE_NAMES, N_MODEL_FEATURES

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TF info/warning spam before import

ML_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ML_DIR / "data" / "processed"
DEFAULT_OUTPUT_DIR = ML_DIR / "models"

# Output index order comes from evaluation.py (project-wide): male -> [1, 0], female -> [0, 1]
__all__ = ["CLASS_NAMES", "CLASS_TO_INDEX", "TrainingConfig", "build_model", "compile_model", "train"]

logger = logging.getLogger("train_neural_network")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    epochs: int = 150
    batch_size: int = 32
    validation_split: float = 0.10  # taken from the training CSV, stratified
    learning_rate: float = 1e-3
    beta_1: float = 0.9
    beta_2: float = 0.999
    dropout_rate: float = 0.25
    hidden_units: tuple[int, ...] = (64, 256, 256, 64)
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    seed: int = 42
    deterministic: bool = True
    dataset_description: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)


# ---------------------------------------------------------------------------
# Reproducibility, logging, metadata helpers
# ---------------------------------------------------------------------------


def set_seeds(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed python, numpy and tensorflow. Returns a serialisable snapshot of the RNG state."""
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()

    np_state = np.random.get_state()
    return {
        "seed": seed,
        "deterministic_ops": deterministic,
        "python_random_state_hash": hash(random.getstate()),
        "numpy_state": {
            "algorithm": str(np_state[0]),
            "keys": [int(v) for v in np_state[1]],
            "pos": int(np_state[2]),
        },
    }


def get_git_info(repo_dir: Path = ML_DIR) -> dict[str, str | bool]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=10, check=False
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit or "no commits yet",
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(status) if status is not None else True,
    }


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        if getattr(handler, "_train_nn", False):
            logger.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console._train_nn = True  # type: ignore[attr-defined]
    file_handler = logging.FileHandler(output_dir / "training_log.txt", mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler._train_nn = True  # type: ignore[attr-defined]
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read a split CSV and validate that it has exactly the 20 features (in order) plus label."""
    df = pd.read_csv(path)
    expected = [*MODEL_FEATURE_NAMES, "label"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    features = df[list(MODEL_FEATURE_NAMES)]
    if list(features.columns) != list(MODEL_FEATURE_NAMES):
        raise ValueError("Feature order mismatch")
    labels = df["label"].astype(str).str.lower()
    unknown = sorted(set(labels) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"{path} has unknown labels {unknown}")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path} contains NaN/inf feature values")
    return features, labels


def one_hot(labels: pd.Series | np.ndarray) -> np.ndarray:
    """male -> [1, 0], female -> [0, 1]."""
    idx = np.asarray([CLASS_TO_INDEX[str(lbl).lower()] for lbl in labels], dtype=np.int64)
    out = np.zeros((len(idx), len(CLASS_NAMES)), dtype=np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out


def load_or_fit_scaler(X_train: np.ndarray, data_dir: Path) -> tuple[StandardScaler, str]:
    """Reuse the scaler fitted by data_preparation.save_splits() (train only) when present.

    If ``scaler.joblib`` exists it is loaded and checked against a fresh fit on the
    same training data; otherwise a new StandardScaler is fitted on ``X_train``.
    Either way the scaler has only ever seen training rows.
    """
    fresh = StandardScaler().fit(X_train)
    existing = data_dir / "scaler.joblib"
    if existing.exists():
        scaler = joblib.load(existing)
        if np.allclose(scaler.mean_, fresh.mean_) and np.allclose(scaler.scale_, fresh.scale_):
            return scaler, f"reused {existing} (verified identical to a fit on train.csv)"
        logger.warning("%s does not match a fit on train.csv; using a fresh fit", existing)
    return fresh, "fitted on train.csv"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(config: TrainingConfig, n_features: int = N_MODEL_FEATURES):
    """The exact MLP: 20 -> 64 -> 256 -> 256 -> 64 -> 2, ReLU, Dropout 0.25 after each hidden layer."""
    import tensorflow as tf

    layers = tf.keras.layers
    model = tf.keras.Sequential(name="gender_mlp")
    model.add(tf.keras.Input(shape=(n_features,), name="acoustic_features"))
    for i, units in enumerate(config.hidden_units, start=1):
        model.add(layers.Dense(units, activation="relu", name=f"dense_{i}"))
        model.add(layers.Dropout(config.dropout_rate, seed=config.seed + i, name=f"dropout_{i}"))
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax", name="output"))
    return model


def compile_model(model, config: TrainingConfig):
    import tensorflow as tf

    optimizer = tf.keras.optimizers.Nadam(
        learning_rate=config.learning_rate, beta_1=config.beta_1, beta_2=config.beta_2
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def make_callbacks(config: TrainingConfig, checkpoint_path: Path) -> list:
    import tensorflow as tf

    class EpochLogger(tf.keras.callbacks.Callback):
        """Log every epoch's metrics and the current learning rate to the log file."""

        def __init__(self) -> None:
            super().__init__()
            self.learning_rates: list[float] = []
            self.epoch_times: list[float] = []
            self._start = 0.0

        def on_epoch_begin(self, epoch, logs=None):  # noqa: ANN001
            self._start = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):  # noqa: ANN001
            logs = logs or {}
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            self.learning_rates.append(lr)
            self.epoch_times.append(time.perf_counter() - self._start)
            metrics = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
            logger.info("epoch %3d/%d | %s | lr=%.2e", epoch + 1, config.epochs, metrics, lr)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.early_stopping_patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=config.reduce_lr_patience,
            factor=config.reduce_lr_factor,
            min_lr=config.min_learning_rate,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=0
        ),
        EpochLogger(),
    ]


# ---------------------------------------------------------------------------
# Evaluation and plots
# ---------------------------------------------------------------------------


def evaluate_split(model, X: np.ndarray, y_idx: np.ndarray, split_name: str) -> dict[str, Any]:
    """Accuracy, precision, recall, F1 (per class, macro, weighted), ROC-AUC and confusion matrix.

    Uses the shared ``evaluation.metrics_from_probabilities`` so the MLP and the
    scikit-learn models are scored by identical code.
    """
    probs = np.asarray(model.predict(X, verbose=0), dtype=np.float64)
    metrics = metrics_from_probabilities(y_idx, probs, split_name)
    log_metrics(metrics, logger)
    return metrics


def plot_history(history: dict[str, list[float]], output_dir: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {}
    for key, title, filename in (
        ("loss", "Loss", "training_loss.png"),
        ("accuracy", "Accuracy", "training_accuracy.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(history[key], label=f"train {key}")
        ax.plot(history[f"val_{key}"], label=f"validation {key}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(f"MLP training {title}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths[key] = path
    return paths


def plot_confusion_matrix(cm: list[list[int]], output_dir: Path, split_name: str = "test") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix ({split_name}, n={arr.sum()})")
    threshold = arr.max() / 2
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j,
                i,
                str(arr[i, j]),
                ha="center",
                va="center",
                color="white" if arr[i, j] > threshold else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = output_dir / f"confusion_matrix_{split_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------


def train(config: TrainingConfig) -> dict[str, Any]:
    """Run the full pipeline. Returns the metrics/metadata dict that is also written to disk."""
    import tensorflow as tf

    output_dir = config.output_dir
    setup_logging(output_dir)
    started = datetime.now(UTC)
    logger.info("=== MLP training started %s ===", started.isoformat())
    logger.info("Hyperparameters: %s", json.dumps({k: str(v) for k, v in asdict(config).items()}))
    logger.info(
        "Versions: tensorflow %s, numpy %s, voice_features %s, python %s",
        tf.__version__,
        np.__version__,
        voice_features.__version__,
        platform.python_version(),
    )

    rng_state = set_seeds(config.seed, config.deterministic)
    (output_dir / "random_state.json").write_text(json.dumps(rng_state, indent=2))

    # --- data ----------------------------------------------------------------
    X_train_df, y_train_lbl = load_split(config.data_dir / "train.csv")
    X_val_df, y_val_lbl = load_split(config.data_dir / "val.csv")
    X_test_df, y_test_lbl = load_split(config.data_dir / "test.csv")
    logger.info("Feature order (%d): %s", N_MODEL_FEATURES, list(MODEL_FEATURE_NAMES))

    dataset_stats = {}
    for name, X, y in (
        ("train", X_train_df, y_train_lbl),
        ("val", X_val_df, y_val_lbl),
        ("test", X_test_df, y_test_lbl),
    ):
        counts = y.value_counts().to_dict()
        dataset_stats[name] = {"rows": int(len(X)), "labels": {k: int(v) for k, v in counts.items()}}
        logger.info("%s.csv: %d rows, labels %s", name, len(X), counts)
    dataset_stats["train_feature_mean"] = {c: float(v) for c, v in X_train_df.mean().items()}
    dataset_stats["train_feature_std"] = {c: float(v) for c, v in X_train_df.std().items()}

    scaler, scaler_source = load_or_fit_scaler(X_train_df.to_numpy(dtype=np.float64), config.data_dir)
    logger.info("Scaler: %s", scaler_source)

    def transform(df: pd.DataFrame) -> np.ndarray:
        return scaler.transform(df.to_numpy(dtype=np.float64)).astype(np.float32)

    X_train_all = transform(X_train_df)
    y_train_all = one_hot(y_train_lbl)
    X_fit, X_holdout, y_fit, y_holdout = train_test_split(
        X_train_all,
        y_train_all,
        test_size=config.validation_split,
        stratify=y_train_all.argmax(axis=1),
        random_state=config.seed,
    )
    logger.info(
        "Training rows %d -> fit %d / internal validation %d (%.0f%% stratified)",
        len(X_train_all),
        len(X_fit),
        len(X_holdout),
        config.validation_split * 100,
    )
    X_val, y_val_idx = transform(X_val_df), np.array([CLASS_TO_INDEX[v] for v in y_val_lbl])
    X_test, y_test_idx = transform(X_test_df), np.array([CLASS_TO_INDEX[v] for v in y_test_lbl])

    # --- model ---------------------------------------------------------------
    model = compile_model(build_model(config), config)
    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    logger.info("Model:\n%s", "\n".join(summary_lines))

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model_mlp_best.h5"
    callbacks = make_callbacks(config, checkpoint_path)
    epoch_logger = callbacks[-1]

    # --- training ------------------------------------------------------------
    t0 = time.perf_counter()
    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_holdout, y_holdout),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=0,
    )
    training_seconds = time.perf_counter() - t0
    epochs_run = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    logger.info(
        "Training finished: %d epochs in %.1f s (%.2f s/epoch); best val_loss %.4f at epoch %d",
        epochs_run,
        training_seconds,
        training_seconds / max(epochs_run, 1),
        float(np.min(history.history["val_loss"])),
        best_epoch,
    )

    # --- evaluation ----------------------------------------------------------
    keras_eval = {}
    for name, X, y_idx in (
        ("internal_val", X_holdout, y_holdout.argmax(1)),
        ("val", X_val, y_val_idx),
        ("test", X_test, y_test_idx),
    ):
        values = model.evaluate(X, one_hot([CLASS_NAMES[i] for i in y_idx]), verbose=0)
        keras_eval[name] = {k: float(v) for k, v in zip(model.metrics_names, values, strict=True)}
    metrics = {
        "internal_validation": evaluate_split(
            model, X_holdout, y_holdout.argmax(axis=1), "internal_validation"
        ),
        "validation": evaluate_split(model, X_val, y_val_idx, "validation"),
        "test": evaluate_split(model, X_test, y_test_idx, "test"),
    }

    # --- plots ---------------------------------------------------------------
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    hist["learning_rate"] = epoch_logger.learning_rates
    hist["epoch_seconds"] = epoch_logger.epoch_times
    plot_paths = plot_history(hist, output_dir)
    plot_paths["confusion_matrix"] = plot_confusion_matrix(
        metrics["test"]["confusion_matrix"]["matrix"], output_dir, "test"
    )

    # --- artefacts -----------------------------------------------------------
    h5_path = output_dir / "model_mlp.h5"
    model.save(h5_path, save_format="h5")
    saved_model_dir = output_dir / "model_mlp"
    model.save(saved_model_dir, save_format="tf")
    scaler_pkl = output_dir / "scaler_mlp.pkl"
    with scaler_pkl.open("wb") as fh:
        pickle.dump(scaler, fh)
    (output_dir / "history_mlp.json").write_text(json.dumps(hist, indent=2))
    (output_dir / "feature_order.json").write_text(json.dumps(list(MODEL_FEATURE_NAMES), indent=2))
    (output_dir / "class_mapping.json").write_text(
        json.dumps(
            {"index_to_class": dict(enumerate(CLASS_NAMES)), "one_hot": {"male": [1, 0], "female": [0, 1]}},
            indent=2,
        )
    )

    finished = datetime.now(UTC)
    metadata = {
        "model_name": "gender_mlp",
        "architecture": {
            "input_features": N_MODEL_FEATURES,
            "hidden_units": list(config.hidden_units),
            "activation": "relu",
            "dropout_rate": config.dropout_rate,
            "output": {"units": len(CLASS_NAMES), "activation": "softmax"},
            "parameters": int(model.count_params()),
        },
        "training": {
            "optimizer": "Nadam",
            "learning_rate": config.learning_rate,
            "beta_1": config.beta_1,
            "beta_2": config.beta_2,
            "loss": "categorical_crossentropy",
            "epochs_configured": config.epochs,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "batch_size": config.batch_size,
            "validation_split": config.validation_split,
            "early_stopping": {
                "monitor": "val_loss",
                "patience": config.early_stopping_patience,
                "restore_best_weights": True,
            },
            "reduce_lr_on_plateau": {
                "monitor": "val_loss",
                "patience": config.reduce_lr_patience,
                "factor": config.reduce_lr_factor,
                "min_lr": config.min_learning_rate,
            },
            "training_seconds": round(training_seconds, 2),
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
        },
        "data": {
            "data_dir": str(config.data_dir),
            "description": config.dataset_description,
            "statistics": dataset_stats,
            "scaler": {"type": "StandardScaler", "fitted_on": "train.csv only", "source": scaler_source},
        },
        "feature_order": list(MODEL_FEATURE_NAMES),
        "class_mapping": {
            "index_to_class": dict(enumerate(CLASS_NAMES)),
            "one_hot": {"male": [1, 0], "female": [0, 1]},
        },
        "metrics": metrics,
        "keras_evaluate": keras_eval,
        "artifacts": {
            "model_h5": str(h5_path),
            "saved_model_dir": str(saved_model_dir),
            "best_checkpoint": str(checkpoint_path),
            "scaler_pkl": str(scaler_pkl),
            "history": str(output_dir / "history_mlp.json"),
            "plots": {k: str(v) for k, v in plot_paths.items()},
            "log": str(output_dir / "training_log.txt"),
        },
        "reproducibility": {
            "seed": config.seed,
            "deterministic_ops": config.deterministic,
            "random_state_file": str(output_dir / "random_state.json"),
        },
        "git": get_git_info(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "voice_features": voice_features.__version__,
        },
        **config.extra_metadata,
    }
    (output_dir / "metrics_mlp.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    t = metrics["test"]
    logger.info(
        "FINAL TEST METRICS | accuracy %.4f | precision(macro) %.4f | recall(macro) %.4f "
        "| F1(macro) %.4f | ROC-AUC %.4f",
        t["accuracy"],
        t["precision_macro"],
        t["recall_macro"],
        t["f1_macro"],
        t["roc_auc"],
    )
    logger.info("Artifacts written to %s", output_dir)
    logger.info(
        "=== MLP training finished %s (total %.1f s) ===",
        finished.isoformat(),
        (finished - started).total_seconds(),
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the 20-feature gender MLP")
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory with train/val/test CSVs"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-deterministic", action="store_true", help="Disable TF op determinism (faster)")
    parser.add_argument(
        "--description", default="", help="Free-text dataset description stored in the metadata"
    )
    args = parser.parse_args(argv)

    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        deterministic=not args.no_deterministic,
        dataset_description=args.description,
    )
    train(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
