"""Fetch the historical ``voice.csv`` benchmark and build train/val/test splits from it.

``voice.csv`` contains 3 168 rows of 20 acoustic features that were extracted in
R (``specan3`` / warbleR) from public voice recordings, plus a ``label`` column.
It is the dataset the original B.Tech project's Keras MLP was trained on and is
reference [10] of Buyukyilmaz & Cibikdiken (2016).

IMPORTANT: these are pre-extracted features. Training on them benchmarks the
model in the historical feature space only; it says nothing about the
correctness of the Python feature extractor on live audio (see
docs/dataset_provenance.md).

The splits are produced with the Phase 2A ``DataPreparation`` splitter so the
stratification, scaler fitting and file layout are identical to a real corpus.

Usage::

    python prepare_voice_csv.py                       # download + verify + split
    python prepare_voice_csv.py --csv path/to/voice.csv   # use a local copy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from data_preparation import DataPreparation, setup_logging
from voice_features import MODEL_FEATURE_NAMES

ML_DIR = Path(__file__).resolve().parent
VOICE_CSV_URL = "https://raw.githubusercontent.com/primaryobjects/voice-gender/master/voice.csv"
VOICE_CSV_REPO = "https://github.com/primaryobjects/voice-gender"
DEFAULT_CSV_PATH = ML_DIR / "data" / "raw" / "voice.csv"
DEFAULT_OUTPUT_DIR = ML_DIR / "data" / "processed" / "voice_csv"

EXPECTED_COLUMNS = [*MODEL_FEATURE_NAMES, "label"]
EXPECTED_ROWS = 3168
EXPECTED_LABELS = {"male": 1584, "female": 1584}

logger = logging.getLogger("data_preparation")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str = VOICE_CSV_URL, dest: Path = DEFAULT_CSV_PATH) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
        data = response.read()
    dest.write_bytes(data)
    logger.info("Downloaded %d bytes", len(data))
    return dest


def verify(csv_path: Path) -> dict:
    """Check the schema and label balance; raise if the file is not the expected dataset."""
    df = pd.read_csv(csv_path)
    problems: list[str] = []
    if list(df.columns) != EXPECTED_COLUMNS:
        problems.append(f"columns {list(df.columns)} != expected {EXPECTED_COLUMNS}")
    if len(df) != EXPECTED_ROWS:
        problems.append(f"{len(df)} rows, expected {EXPECTED_ROWS}")
    counts = df["label"].value_counts().to_dict() if "label" in df else {}
    if counts != EXPECTED_LABELS:
        problems.append(f"label counts {counts} != expected {EXPECTED_LABELS}")
    feature_block = df[[c for c in MODEL_FEATURE_NAMES if c in df.columns]]
    n_nan = int(feature_block.isna().sum().sum())
    if n_nan:
        problems.append(f"{n_nan} NaN feature values")
    if problems:
        raise ValueError("voice.csv verification failed: " + "; ".join(problems))

    stats = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "label_counts": {k: int(v) for k, v in counts.items()},
        "n_nan": n_nan,
        "feature_ranges": {
            c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in MODEL_FEATURE_NAMES
        },
        "centroid_equals_meanfreq": bool((df["centroid"] == df["meanfreq"]).all()),
    }
    logger.info("voice.csv verified: %d rows, labels %s, %d NaN", stats["rows"], counts, n_nan)
    return stats


def make_splits(csv_path: Path, output_dir: Path, seed: int = 42) -> dict:
    df = pd.read_csv(csv_path)
    df["label"] = df["label"].str.lower()
    prep = DataPreparation(output_dir=output_dir, random_state=seed)
    prep.dataset = df[EXPECTED_COLUMNS]
    prep.compute_statistics(prep.dataset)
    prep.split_dataset()
    paths = prep.save_splits(output_dir)
    prep.plot_feature_distributions()
    return {name: str(path) for name, path in paths.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download, verify and split the historical voice.csv")
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV_PATH, help="Local path (downloaded if missing)"
    )
    parser.add_argument("--url", default=VOICE_CSV_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args(argv)

    setup_logging()
    if args.force_download or not args.csv.exists():
        download(args.url, args.csv)
    else:
        logger.info("Using existing %s", args.csv)

    digest = sha256_of(args.csv)
    stats = verify(args.csv)
    paths = make_splits(args.csv, args.output, args.seed)

    provenance = {
        "dataset": "voice.csv (historical, pre-extracted acoustic features)",
        "source_url": args.url,
        "source_repository": VOICE_CSV_REPO,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "sha256": digest,
        "local_path": str(args.csv),
        "verification": stats,
        "split_seed": args.seed,
        "split_files": paths,
        "caveat": (
            "Features were extracted in R by the dataset author (specan3/warbleR). Models trained on "
            "them are benchmarked in that feature space only; this does not validate the Python "
            "feature extractor on live audio."
        ),
    }
    prov_path = args.output / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    logger.info("Provenance written to %s (sha256 %s)", prov_path, digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
