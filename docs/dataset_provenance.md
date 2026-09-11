# Dataset provenance

## Historical benchmark: `voice.csv`

| Item | Value |
|---|---|
| Source URL | `https://raw.githubusercontent.com/primaryobjects/voice-gender/master/voice.csv` |
| Source repository | `https://github.com/primaryobjects/voice-gender` (Kory Becker, "Identifying the Gender of a Voice using Machine Learning", 2016) |
| Cited by | Reference paper 2, Buyukyilmaz & Cibikdiken (2016), reference [10]; the original B.Tech project trained its Keras MLP on this file |
| Retrieved | 2026-09-11 (UTC), by `ml_models/prepare_voice_csv.py` |
| Size | 1 065 381 bytes |
| SHA-256 | `96303b30acb620724daea3f71cdf27d0ab5818342ab9825add0a7a4e0a898a22` |
| Local path | `ml_models/data/raw/voice.csv` (git-ignored; re-download with the script and compare the hash) |

### Schema verification (performed by `prepare_voice_csv.py`)

- 3 168 rows, 21 columns.
- Columns are exactly the 20 model features in the Phase 2A order, plus `label`: `meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp.ent, sfm, mode, centroid, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx, label`.
- Labels: 1 584 `male`, 1 584 `female`. No missing values.
- `centroid == meanfreq` holds on every row, consistent with the `specprop` formula documented in [acoustic_features.md](acoustic_features.md).

### How the data was produced (by the dataset author, not by this project)

The rows are acoustic measurements computed in **R** with the author's `specan3()` function (a modified `warbleR::specan` built on `seewave`) from public recordings (Harvard-Haskins Database of Regularly-Timed Speech, TIMIT, VoxForge, Festvox CMU_ARCTIC), analysed on the 0–280 Hz band. The audio files themselves are **not** part of this repository and were not available to this project.

### What training on it does and does not show

- **Does:** benchmark the MLP architecture in the historical 20-feature space, with real, verifiable data and reproducible splits. Metrics reported from this run apply to that feature space.
- **Does not:** validate the Python feature extractor (`ml_models/voice_features`) on live audio. The Python extractor reproduces the same formulas, but no recording exists in this repository for which both the R output and the Python output are known, so numeric parity is unverified. End-to-end live-audio inference validity must be established separately once labelled recordings are available (Phase 2A `data_preparation.py` is ready for that).

## Splits derived from `voice.csv`

Created by `prepare_voice_csv.py` using the Phase 2A `DataPreparation.split_dataset()` / `save_splits()` (stratified 70 / 15 / 15, seed 42) in `ml_models/data/processed/voice_csv/` (git-ignored, regenerable):

| Split | Rows | male | female |
|---|---|---|---|
| train | 2 217 | 1 109 | 1 108 |
| val | 475 | 237 | 238 |
| test | 476 | 238 | 238 |

`scaler.joblib` in that directory is a `StandardScaler` fitted on the training split only. `provenance.json` next to the splits records the URL, hash, retrieval time and verification results for the exact run.

## Live-audio corpus

None is present in the repository. When one is added, place audio under `ml_models/data/raw/` with a `labels.csv` and run `data_preparation.py`; record its provenance in this document.
