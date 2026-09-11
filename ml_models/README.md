# ml_models/

Acoustic feature extraction, dataset preparation, model training and evaluation.

- Primary approach: 20 acoustic features -> multilayer perceptron (neural network).
- Legacy experiment: MFCC -> Gaussian Mixture Models (kept for comparison only, later phase).

## Layout

```
ml_models/
├── voice_features/          Installable package: specan-compatible feature extraction
│   └── specan.py            The single definition of the 20 model features (shared with backend)
├── data_preparation.py      Audio validation -> features -> voice_features.csv -> stratified splits
├── tests/                   pytest suite (synthetic voiced signals; no real corpus required)
├── data/
│   ├── raw/                 Put audio files + labels.csv here (git-ignored)
│   └── processed/           Outputs: CSVs, scaler, statistics, plots (git-ignored)
├── logs/                    data_preparation.log (git-ignored)
├── requirements.txt         Pinned ML environment (Python 3.10/3.11)
└── pyproject.toml           Package metadata, pytest and ruff config
```

The feature definitions, ordering, units and every deviation from the historical `warbleR::specan` output are documented in [docs/acoustic_features.md](../docs/acoustic_features.md).

## Setup

```bash
cd ml_models
uv venv .venv --python 3.11          # or: py -3.11 -m venv .venv
uv pip install --python .venv -r requirements.txt
```

`requirements.txt` installs this directory as the editable package `voice-features`. The backend installs the same package (`-e ../ml_models` in `backend/requirements.txt`) so inference uses byte-for-byte the same code.

## Data preparation

Expected input: a directory of audio files (`.wav` preferred; 1-10 s each) and a CSV with a filename column (`filename`) and a label column (`label`, values `male` / `female`).

```bash
.venv\Scripts\activate
python data_preparation.py --audio-dir data/raw/audio --labels data/raw/labels.csv --output data/processed
```

or from Python:

```python
from data_preparation import DataPreparation

prepare = DataPreparation()
prepare.load_audio_files("data/raw/audio")          # validates: decodable, 1-10 s, logs stats
prepare.create_dataset("data/raw/audio", "data/raw/labels.csv")   # voice_features.csv (n, 21)
prepare.save_splits("data/processed")                # train/val/test (70/15/15, stratified) + scaler
prepare.plot_feature_distributions()                 # feature_distributions.png
```

Outputs in `data/processed/`:

| File | Content |
|---|---|
| `voice_features.csv` | 20 features + `label`, same layout as the historical `voice.csv` |
| `voice_features_full.csv` | adds `filename`, `native_sample_rate`, `duration`, `peakf` |
| `train.csv`, `val.csv`, `test.csv` | stratified splits (raw values) |
| `*_scaled.csv` | the same splits standardised with the scaler below |
| `scaler.joblib` | `StandardScaler` fitted on the training split only |
| `feature_names.json` | feature order and analysis parameters |
| `feature_statistics.csv` | mean/std/min/max, NaN and inf counts per feature |
| `split_summary.json` | row and label counts per split |
| `feature_distributions.png` | histogram grid per feature, coloured by label |

Files are skipped (and logged) when they are corrupt, empty, shorter than 1 s, longer than 10 s, have no label, or produce non-finite features (no voiced frames detected).

## Tests

```bash
pytest
ruff check . && ruff format --check .
```

The tests synthesise harmonic "voices" with known fundamentals, so they verify the fundamental tracker, the band-limited spectral statistics, the R index conventions and the full CSV/split pipeline without any dataset.

## Historical benchmark data (`voice.csv`)

No audio corpus is present in this repository. To train and benchmark the model, `prepare_voice_csv.py` downloads the historical `voice.csv` (3 168 rows of pre-extracted specan features, the dataset the original project used), verifies its schema and SHA-256, and builds stratified splits with the Phase 2A splitter:

```bash
python prepare_voice_csv.py        # -> data/raw/voice.csv, data/processed/voice_csv/{train,val,test}.csv, scaler.joblib, provenance.json
```

Provenance, verification results and the important caveat are recorded in [docs/dataset_provenance.md](../docs/dataset_provenance.md): these features were computed in R by the dataset author, so a model trained on them is benchmarked **in that feature space only**. This does not validate the Python live-audio extractor.

## Neural network training

`train_neural_network.py` trains the fixed MLP (20 -> 64 -> 256 -> 256 -> 64 -> 2, ReLU, Dropout 0.25, softmax; Nadam lr 1e-3; categorical cross-entropy; EarlyStopping patience 10 on val_loss with best-weights restore; ReduceLROnPlateau patience 5, factor 0.5, min 1e-6; up to 150 epochs, batch 32; a stratified 10 % of `train.csv` is held out as the internal validation set).

```bash
python train_neural_network.py --data-dir data/processed/voice_csv --output models
```

Artifacts in `models/` (git-ignored): `model_mlp.h5`, `model_mlp/` (SavedModel), `checkpoints/model_mlp_best.h5`, `scaler_mlp.pkl` (the train-only `StandardScaler`, identical to `scaler.joblib` from data preparation), `feature_order.json`, `class_mapping.json` (index 0 = male `[1,0]`, 1 = female `[0,1]`), `history_mlp.json`, `metrics_mlp.json` / `model_metadata.json` (hyperparameters, dataset statistics, all metrics, git state, environment), `random_state.json`, `training_log.txt`, `training_loss.png`, `training_accuracy.png`, `confusion_matrix_test.png`.

### Latest run on the `voice.csv` benchmark (2026-09-11, seed 42, deterministic ops)

Feature space: historical `voice.csv` (R-extracted). Splits 2 217 / 475 / 476. Training stopped at epoch 33 (best epoch 23), 15.9 s on CPU.

| Split | n | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC |
|---|---|---|---|---|---|---|
| validation (`val.csv`) | 475 | 0.9811 | 0.9811 | 0.9811 | 0.9811 | 0.9972 |
| test (`test.csv`) | 476 | 0.9811 | 0.9812 | 0.9811 | 0.9811 | 0.9977 |

Test confusion matrix (rows = true male, female; cols = predicted male, female): `[[235, 3], [6, 232]]`.

These are the numbers this codebase actually produced; they are not targets and they are not the historical 96.74 % figure, which belongs to the reference paper. They apply to the `voice.csv` feature space. End-to-end accuracy on live audio through the Python extractor has **not** been measured, because no labelled recordings are available.

## Scikit-learn comparison models

`train_sklearn_models.py` trains Random Forest, SVM (RBF), Gradient Boosting and a Decision Tree baseline with the hyperparameters fixed by the specification, each as `Pipeline(StandardScaler, model)` fitted on the same `train.csv` (the pipeline scaler is verified identical to the Phase 2A scaler), evaluated on the same `val.csv` / `test.csv` with the same metric code as the MLP (`evaluation.py`), plus 5-fold stratified CV on train, inference timing (1000 predictions, batch and single-call), pickle sizes and feature importances (native `feature_importances_`; permutation importance on `val.csv` for the SVM). `--tune` runs `GridSearchCV` with the specified grids.

```bash
python train_sklearn_models.py --data-dir data/processed/voice_csv --output models/sklearn --tune
```

Artifacts in `models/sklearn/` (git-ignored): `<model>_model.pkl` (full pipeline, scaler included), `<model>_scaler.pkl`, `<model>_tuned_model.pkl`, `feature_importance.json`, `tuning_results.json`, `comparison_report.json`, `comparison_plots.png` plus `accuracy_comparison.png`, `confusion_matrices.png`, `roc_curves.png`, `feature_importance_top15.png`, `feature_importance_<model>.png`, `training_log.txt`.

### Latest run on the `voice.csv` benchmark (2026-09-11, seed 42)

Same splits as the MLP (2 217 / 475 / 476). All figures measured by this codebase on the `voice.csv` feature space; the MLP row is the Phase 2B run.

| Model | Train acc | Val acc | Test acc | Test F1 (macro) | Test ROC-AUC | 5-fold CV (train) | Fit time | Single-call latency |
|---|---|---|---|---|---|---|---|---|
| Random Forest | 1.0000 | 0.9853 | 0.9811 | 0.9811 | 0.9983 | 0.9779 ± 0.0054 | 0.55 s | 50.7 ms* |
| SVM (RBF) | 0.9842 | 0.9853 | 0.9811 | 0.9811 | 0.9989 | 0.9774 ± 0.0062 | 0.35 s | 0.24 ms |
| Gradient Boosting | 1.0000 | 0.9874 | 0.9811 | 0.9811 | 0.9988 | 0.9765 ± 0.0034 | 14.7 s | 0.36 ms |
| Decision Tree (baseline) | 1.0000 | 0.9705 | 0.9727 | 0.9727 | 0.9727 | 0.9576 ± 0.0087 | 0.17 s | 0.15 ms |
| MLP (Phase 2B) | – | 0.9811 | 0.9811 | 0.9811 | 0.9977 | – | 15.9 s | – |

\* Random Forest single-row `predict_proba` with `n_jobs=-1` pays a thread-pool start-up cost per call; its 1000-row batch takes 61 ms in total. Set `n_jobs=1` for single-sample serving.

Test confusion matrices (rows true male/female, cols predicted): RF `[[236, 2], [7, 231]]`, SVM `[[235, 3], [6, 232]]`, GB `[[236, 2], [7, 231]]`, DT `[[233, 5], [8, 230]]`.

Grid search (5-fold CV on train): RF best `n_estimators=200` (as specified), SVM best `C=1` (as specified), Gradient Boosting best `learning_rate=0.2` (CV 0.9779, test 0.9853), Decision Tree best `max_depth=5` (CV 0.9612, test 0.9748). The saved `<model>_model.pkl` files are the specified configurations; tuned variants are saved separately.

`meanfun` and `IQR` are the two most important features for every model, consistent with the reference papers' observation that fundamental-frequency statistics carry most of the signal.

## Model comparison and selection

`model_comparison.py` loads every Phase 2B/2C model (base and tuned), evaluates all of them on the same untouched `test.csv` with the shared metric code, measures single-call latency and memory footprint in isolated processes, checks train/test gaps, applies a pre-declared selection policy (validation accuracy tie band of 1 pp, then neural-network objective, coarse latency/memory buckets, validation ROC-AUC/F1), writes `models/comparison/` (`comparison_report.md`, `comparison_report.json`, `comparison_metrics.csv`, `comparison_plots.png` and individual figures) and deploys the selected model with `model_config.json` (the single manifest: artifact hashes, feature order, class mapping, scaler, input dimension, metrics, validity flags) to `backend/app/ml/models/`.

```bash
python model_comparison.py --models-dir models --data-dir data/processed/voice_csv --output models/comparison --deploy-dir ../backend/app/ml/models
```

### Result of the 2026-09-11 run

Eight of nine candidates fall within 1 pp of the best validation accuracy (Gradient Boosting, 0.9874), so accuracy does not separate them; on the test split RF, SVM, GB and the MLP are exactly tied at 0.9811. **Selected production model: MLP** because it is tied on every accuracy metric and it is the only candidate that fulfils the project's neural-network objective; on engineering criteria alone the pick would be the tuned Gradient Boosting (0.29 ms per call, 10 MB, validation ROC-AUC 0.9995). Edge alternative: SVM (60 KB, 2.7 MB footprint, 0.2 ms). Backup: Gradient Boosting (best validation accuracy). Methodology flags, including the fact that test metrics had already been published before selection, are recorded in the report. None of the models shows a train/test gap above 10 pp. The historical 96.74 % is listed as a reference row only.

## Status

- Phase 2A: feature extraction and data preparation complete.
- Phase 2B: MLP training and evaluation complete on the `voice.csv` benchmark; artifacts for inference saved in `models/`.
- Phase 2C: scikit-learn comparison models trained, tuned and reported in `models/sklearn/`.
- Phase 2D: comparison report and production manifest produced; MLP deployed to `backend/app/ml/models/`.
