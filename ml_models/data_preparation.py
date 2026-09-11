"""Data preparation: audio validation, acoustic feature extraction, dataset assembly and splitting.

Usage (Python)::

    prepare = DataPreparation()
    prepare.load_audio_files("data/raw/audio")
    prepare.create_dataset("data/raw/audio", "data/raw/labels.csv")
    prepare.save_splits("data/processed")

Usage (CLI)::

    python data_preparation.py --audio-dir data/raw/audio --labels data/raw/labels.csv --output data/processed

The per-signal feature maths lives in the ``voice_features`` package (shared
with the inference API); this module only orchestrates it.
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from voice_features import (
    CANONICAL_SAMPLE_RATE,
    EXCLUDED_FROM_MODEL,
    MODEL_FEATURE_NAMES,
    FeatureExtractionError,
    SpecanConfig,
    compute_specan_features,
    extract_model_features,
    features_to_vector,
    load_audio,
)

ML_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ML_DIR / "data" / "processed"
DEFAULT_LOG_DIR = ML_DIR / "logs"

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".webm", ".m4a"}
LABEL_VALUES = ("male", "female")
FILENAME_COLUMNS = ("filename", "file", "path", "sound.files", "audio")
LABEL_COLUMNS = ("label", "gender", "sex")
METADATA_COLUMNS = ("filename", "native_sample_rate", "duration", "peakf")

logger = logging.getLogger("data_preparation")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_dir: Path | str = DEFAULT_LOG_DIR, level: int = logging.INFO) -> logging.Logger:
    """Console + rotating file logging (``<log_dir>/data_preparation.log``). Idempotent."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(level)
    if any(getattr(h, "_data_prep", False) for h in logger.handlers):
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console._data_prep = True  # type: ignore[attr-defined]
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "data_preparation.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler._data_prep = True  # type: ignore[attr-defined]
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Audio inventory
# ---------------------------------------------------------------------------


@dataclass
class AudioFileInfo:
    path: str
    relative_name: str
    native_sample_rate: int | None
    duration_seconds: float | None
    status: str  # "ok" or a skip reason: corrupt | empty | too_short | too_long

    @property
    def is_valid(self) -> bool:
        return self.status == "ok"


def _probe_audio(path: Path) -> tuple[int, float]:
    """Return (native_sample_rate, duration_seconds) without decoding when possible."""
    try:
        info = sf.info(str(path))
        return int(info.samplerate), float(info.duration)
    except Exception:
        # Formats soundfile cannot read (e.g. mp3 on some builds): fall back to a decode.
        import warnings

        import librosa

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # librosa warns about the audioread fallback
            duration = float(librosa.get_duration(path=str(path)))
            sr = int(librosa.get_samplerate(str(path)))
        return sr, duration


# ---------------------------------------------------------------------------
# DataPreparation
# ---------------------------------------------------------------------------


class DataPreparation:
    """End-to-end preparation of a voice-gender dataset from labelled audio files."""

    def __init__(
        self,
        sample_rate: int = CANONICAL_SAMPLE_RATE,
        min_duration: float = 1.0,
        max_duration: float = 10.0,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        log_dir: Path | str = DEFAULT_LOG_DIR,
        random_state: int = 42,
        config: SpecanConfig | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.config = config or SpecanConfig(sample_rate=sample_rate)
        if self.config.sample_rate != sample_rate:
            raise ValueError("config.sample_rate must equal sample_rate")

        self.audio_files: list[AudioFileInfo] = []
        self.audio_directory: Path | None = None
        self.full_dataset: pd.DataFrame | None = None  # metadata + 20 features + label
        self.dataset: pd.DataFrame | None = None  # 20 features + label (voice.csv layout)
        self.splits: dict[str, pd.DataFrame] | None = None
        self.scaler: StandardScaler | None = None
        self.skip_counter: Counter[str] = Counter()

        setup_logging(log_dir)

    # ----- 1. load and validate -------------------------------------------------

    def load_audio_files(self, directory_path: Path | str, recursive: bool = True) -> list[AudioFileInfo]:
        """Inventory and validate audio files. Invalid files are kept in the list with a skip status."""
        directory = Path(directory_path)
        if not directory.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {directory}")
        pattern = "**/*" if recursive else "*"
        candidates = sorted(
            p for p in directory.glob(pattern) if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )
        logger.info("Scanning %s: %d candidate audio files", directory, len(candidates))

        files: list[AudioFileInfo] = []
        for path in tqdm(candidates, desc="Validating audio", unit="file", leave=False):
            rel = path.relative_to(directory).as_posix()
            try:
                native_sr, duration = _probe_audio(path)
            except Exception as exc:  # noqa: BLE001 - any decoder failure means "corrupt"
                logger.warning("Corrupt or unreadable file skipped: %s (%s)", rel, exc.__class__.__name__)
                files.append(AudioFileInfo(str(path), rel, None, None, "corrupt"))
                continue

            if duration <= 0:
                status = "empty"
            elif duration < self.min_duration:
                status = "too_short"
            elif duration > self.max_duration:
                status = "too_long"
            else:
                status = "ok"
            if status != "ok":
                logger.warning("Skipped %s: %s (%.2f s)", rel, status, duration)
            files.append(AudioFileInfo(str(path), rel, native_sr, duration, status))

        self.audio_files = files
        self.audio_directory = directory
        self._log_inventory_statistics()
        return files

    def _log_inventory_statistics(self) -> None:
        valid = [f for f in self.audio_files if f.is_valid]
        self.skip_counter = Counter(f.status for f in self.audio_files if not f.is_valid)
        logger.info("Valid files: %d / %d", len(valid), len(self.audio_files))
        for reason, count in sorted(self.skip_counter.items()):
            logger.info("  skipped (%s): %d", reason, count)
        if valid:
            durations = np.array([f.duration_seconds for f in valid], dtype=float)
            logger.info(
                "Duration (s): mean %.2f, min %.2f, max %.2f, total %.1f",
                durations.mean(),
                durations.min(),
                durations.max(),
                durations.sum(),
            )
            rates = Counter(f.native_sample_rate for f in valid)
            logger.info("Native sample rates: %s", dict(rates))
            resampled = sum(c for sr, c in rates.items() if sr != self.sample_rate)
            if resampled:
                logger.info(
                    "%d files are not %d Hz and will be resampled on load", resampled, self.sample_rate
                )

    # ----- 2. feature extraction ------------------------------------------------

    def _load_signal(self, source: Any, sample_rate: int | None) -> tuple[np.ndarray, int]:
        if isinstance(source, (str, Path)):
            return load_audio(source, self.sample_rate)
        if sample_rate is None:
            raise ValueError("sample_rate is required when passing an audio array")
        return np.asarray(source), int(sample_rate)

    def extract_all_features(self, source: Any, sample_rate: int | None = None) -> dict[str, float]:
        """All 22 specan-style measurements (includes duration and peakf)."""
        y, sr = self._load_signal(source, sample_rate)
        return compute_specan_features(y, sr, self.config)

    def extract_features(
        self, source: Any, sample_rate: int | None = None, scaler: StandardScaler | None = None
    ) -> np.ndarray:
        """The 20 model features as an array of shape (1, 20).

        ``source`` is a file path or a mono array (then ``sample_rate`` is required).
        If ``scaler`` (or ``self.scaler`` after ``save_splits``) is available the
        vector is standardised; otherwise raw feature values are returned.
        """
        y, sr = self._load_signal(source, sample_rate)
        features = extract_model_features(y, sr, self.config, strict=True)
        vector = features_to_vector(features)
        scaler = scaler or self.scaler
        if scaler is not None:
            vector = scaler.transform(vector).astype(np.float32)
        if vector.shape != (1, len(MODEL_FEATURE_NAMES)):
            raise FeatureExtractionError(f"Unexpected feature shape {vector.shape}")
        return vector

    # ----- 4. dataset --------------------------------------------------------------

    @staticmethod
    def _load_labels(labels_csv: Path | str) -> dict[str, str]:
        df = pd.read_csv(labels_csv)
        columns = {c.lower(): c for c in df.columns}
        file_col = next((columns[c] for c in FILENAME_COLUMNS if c in columns), None)
        label_col = next((columns[c] for c in LABEL_COLUMNS if c in columns), None)
        if file_col is None or label_col is None:
            raise ValueError(
                f"labels CSV must have a filename column {FILENAME_COLUMNS} "
                f"and a label column {LABEL_COLUMNS}; found {list(df.columns)}"
            )
        labels: dict[str, str] = {}
        for name, label in zip(df[file_col].astype(str), df[label_col].astype(str), strict=True):
            norm = label.strip().lower()
            if norm not in LABEL_VALUES:
                logger.warning("Ignoring label %r for %s (expected one of %s)", label, name, LABEL_VALUES)
                continue
            key = Path(name.strip()).as_posix()
            labels[key] = norm
            labels.setdefault(Path(key).name, norm)  # allow matching by basename
        return labels

    def create_dataset(
        self,
        audio_directory: Path | str,
        labels_csv: Path | str,
        output_csv: str = "voice_features.csv",
        batch_size: int = 64,
    ) -> pd.DataFrame:
        """Extract features for every valid, labelled file and write ``voice_features.csv``.

        Rows are flushed to ``voice_features_full.csv`` every ``batch_size`` files so
        memory stays flat for large corpora. Returns the (n_samples, 21) dataset.
        """
        directory = Path(audio_directory)
        if not self.audio_files or self.audio_directory != directory:
            self.load_audio_files(directory)
        labels = self._load_labels(labels_csv)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        full_path = self.output_dir / "voice_features_full.csv"
        out_path = self.output_dir / output_csv
        if full_path.exists():
            full_path.unlink()

        columns = [*METADATA_COLUMNS, *MODEL_FEATURE_NAMES, "label"]
        buffer: list[dict[str, Any]] = []
        written = 0
        counts: Counter[str] = Counter()

        def flush() -> None:
            nonlocal written
            if not buffer:
                return
            pd.DataFrame(buffer, columns=columns).to_csv(
                full_path, mode="a", header=not full_path.exists(), index=False
            )
            written += len(buffer)
            buffer.clear()

        valid = [f for f in self.audio_files if f.is_valid]
        for info in tqdm(valid, desc="Extracting features", unit="file"):
            label = labels.get(info.relative_name) or labels.get(Path(info.relative_name).name)
            if label is None:
                logger.warning("No label for %s; skipped", info.relative_name)
                counts["no_label"] += 1
                continue
            try:
                feats = self.extract_all_features(info.path)
            except FeatureExtractionError as exc:
                logger.warning("Feature extraction failed for %s: %s", info.relative_name, exc)
                counts["feature_error"] += 1
                continue
            except Exception as exc:  # noqa: BLE001 - keep going on unexpected decoder errors
                logger.error("Unexpected error for %s: %s", info.relative_name, exc)
                counts["unexpected_error"] += 1
                continue
            bad = [n for n in MODEL_FEATURE_NAMES if not np.isfinite(feats[n])]
            if bad:
                logger.warning("Non-finite features %s for %s; skipped", bad, info.relative_name)
                counts["non_finite"] += 1
                continue
            row = {
                "filename": info.relative_name,
                "native_sample_rate": info.native_sample_rate,
                "duration": feats["duration"],
                "peakf": feats["peakf"],
                **{n: feats[n] for n in MODEL_FEATURE_NAMES},
                "label": label,
            }
            buffer.append(row)
            counts[label] += 1
            if len(buffer) >= batch_size:
                flush()
        flush()

        if written == 0:
            raise RuntimeError("No features were extracted; check the audio directory and labels CSV")

        self.full_dataset = pd.read_csv(full_path)
        self.dataset = self.full_dataset[[*MODEL_FEATURE_NAMES, "label"]]
        self.dataset.to_csv(out_path, index=False)
        logger.info(
            "Dataset written: %s with shape %s (male=%d, female=%d)",
            out_path,
            self.dataset.shape,
            counts["male"],
            counts["female"],
        )
        for reason in ("no_label", "feature_error", "non_finite", "unexpected_error"):
            if counts[reason]:
                logger.info("  skipped (%s): %d", reason, counts[reason])
        self.skip_counter.update({k: v for k, v in counts.items() if k not in LABEL_VALUES})
        self.compute_statistics(self.dataset)
        return self.dataset

    # ----- 5. splitting ------------------------------------------------------------

    def split_dataset(
        self,
        df: pd.DataFrame | None = None,
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> dict[str, pd.DataFrame]:
        """Stratified train / val / test split preserving the label distribution."""
        df = self.dataset if df is None else df
        if df is None:
            raise RuntimeError("No dataset; call create_dataset() first or pass a DataFrame")
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError("train_size + val_size + test_size must equal 1")

        holdout = val_size + test_size
        train_df, rest_df = train_test_split(
            df, test_size=holdout, stratify=df["label"], random_state=self.random_state
        )
        val_df, test_df = train_test_split(
            rest_df,
            test_size=test_size / holdout,
            stratify=rest_df["label"],
            random_state=self.random_state,
        )
        self.splits = {"train": train_df, "val": val_df, "test": test_df}
        for name, part in self.splits.items():
            dist = part["label"].value_counts().to_dict()
            logger.info("%s split: %d rows, distribution %s", name, len(part), dist)
        return self.splits

    def save_splits(self, output_dir: Path | str | None = None) -> dict[str, Path]:
        """Write train/val/test CSVs, the StandardScaler fitted on train, scaled copies and a summary."""
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.splits is None:
            self.split_dataset()
        assert self.splits is not None

        features = list(MODEL_FEATURE_NAMES)
        self.scaler = StandardScaler().fit(self.splits["train"][features].to_numpy(dtype=np.float64))

        paths: dict[str, Path] = {}
        for name, part in self.splits.items():
            path = self.output_dir / f"{name}.csv"
            part.to_csv(path, index=False)
            paths[name] = path

            scaled = part.copy()
            scaled[features] = self.scaler.transform(part[features].to_numpy(dtype=np.float64))
            scaled_path = self.output_dir / f"{name}_scaled.csv"
            scaled.to_csv(scaled_path, index=False)
            paths[f"{name}_scaled"] = scaled_path

        scaler_path = self.output_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        paths["scaler"] = scaler_path

        names_path = self.output_dir / "feature_names.json"
        names_path.write_text(
            json.dumps(
                {
                    "model_features": features,
                    "excluded_specan_features": list(EXCLUDED_FROM_MODEL),
                    "sample_rate": self.sample_rate,
                    "config": asdict(self.config),
                },
                indent=2,
            )
        )
        paths["feature_names"] = names_path

        summary = {
            name: {
                "rows": int(len(part)),
                "labels": {k: int(v) for k, v in part["label"].value_counts().items()},
            }
            for name, part in self.splits.items()
        }
        summary_path = self.output_dir / "split_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        paths["summary"] = summary_path

        logger.info(
            "Splits saved to %s (scaler fitted on %d training rows)",
            self.output_dir,
            len(self.splits["train"]),
        )
        return paths

    # ----- 6. statistics and validation ------------------------------------------

    def compute_statistics(self, df: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
        """Per-feature mean/std/min/max plus NaN and inf counts. Logs a summary table."""
        df = self.dataset if df is None else df
        if df is None:
            raise RuntimeError("No dataset to describe")
        features = df[list(MODEL_FEATURE_NAMES)]
        stats = pd.DataFrame(
            {
                "mean": features.mean(),
                "std": features.std(),
                "min": features.min(),
                "max": features.max(),
                "n_nan": features.isna().sum(),
                "n_inf": np.isinf(features.to_numpy(dtype=np.float64)).sum(axis=0),
            }
        )
        n_bad = int(stats["n_nan"].sum() + stats["n_inf"].sum())
        if n_bad:
            logger.warning("Dataset contains %d NaN/inf feature values", n_bad)
        else:
            logger.info("No NaN or infinite feature values")
        with pd.option_context("display.width", 140, "display.float_format", "{:.5f}".format):
            logger.info("Feature statistics (n=%d):\n%s", len(df), stats.to_string())
        if save:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            stats.to_csv(self.output_dir / "feature_statistics.csv")
        return stats

    def plot_feature_distributions(
        self, df: pd.DataFrame | None = None, filename: str = "feature_distributions.png"
    ) -> Path:
        """Histogram grid of every feature, coloured by label. Saved under output_dir."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = self.dataset if df is None else df
        if df is None:
            raise RuntimeError("No dataset to plot")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(4, 5, figsize=(20, 14))
        colours = {"male": "#2563eb", "female": "#db2777"}
        for ax, name in zip(axes.flat, MODEL_FEATURE_NAMES, strict=True):
            for label, colour in colours.items():
                values = df.loc[df["label"] == label, name].dropna()
                if len(values):
                    ax.hist(values, bins=30, alpha=0.55, color=colour, label=label)
            ax.set_title(name)
            ax.tick_params(labelsize=8)
        axes.flat[0].legend()
        fig.suptitle(f"Feature distributions (n={len(df)})")
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=110)
        plt.close(fig)
        logger.info("Saved %s", path)
        return path

    def summary(self) -> dict[str, Any]:
        return {
            "audio_files": len(self.audio_files),
            "valid_files": sum(f.is_valid for f in self.audio_files),
            "skipped": dict(self.skip_counter),
            "dataset_shape": None if self.dataset is None else list(self.dataset.shape),
            "splits": None if self.splits is None else {k: len(v) for k, v in self.splits.items()},
            "output_dir": str(self.output_dir),
        }


# ---------------------------------------------------------------------------
# Function-style API (as requested) and CLI
# ---------------------------------------------------------------------------


def load_audio_files(directory_path: Path | str, **kwargs: Any) -> list[AudioFileInfo]:
    return DataPreparation(**kwargs).load_audio_files(directory_path)


def extract_20_features(
    source: Any, sample_rate: int | None = None, scaler: StandardScaler | None = None, **kwargs: Any
) -> np.ndarray:
    """20 model features for a WAV path or an audio array, shape (1, 20)."""
    return DataPreparation(**kwargs).extract_features(source, sample_rate, scaler)


def create_dataset(audio_directory: Path | str, labels_csv: Path | str, **kwargs: Any) -> pd.DataFrame:
    return DataPreparation(**kwargs).create_dataset(audio_directory, labels_csv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract acoustic features and build train/val/test splits")
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio files")
    parser.add_argument("--labels", required=True, help="CSV with filename and label (male/female) columns")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    prepare = DataPreparation(
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        output_dir=args.output,
        random_state=args.seed,
    )
    prepare.load_audio_files(args.audio_dir)
    prepare.create_dataset(args.audio_dir, args.labels, batch_size=args.batch_size)
    prepare.save_splits(args.output)
    if not args.no_plots:
        prepare.plot_feature_distributions()
    logger.info("Summary: %s", json.dumps(prepare.summary()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
