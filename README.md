# Gender Recognition Through Voice

A web application that predicts the gender of a speaker from a short voice recording using a neural network trained on acoustic features.

This is a modernised rebuild of the B.Tech final-year project *"Gender Recognition through Voice using Neural Networks"* (Gayatri Vidya Parishad College of Engineering for Women, Department of IT, Batch 10, 2020-21). The original work used R (`warbleR::specan`) for acoustic feature extraction and a Keras multilayer perceptron for classification. This repository keeps that methodology as its academic foundation and rebuilds it as a production-style, end-to-end system.

> **Project status:** Phase 1 (repository scaffold). Application code lands in later phases.

## Objectives

- Reproduce the original 20-acoustic-feature pipeline in pure Python (no R dependency) so that live recordings and the training dataset share one feature definition.
- Train a multilayer perceptron (MLP) neural network as the primary classifier, with honest, reproducible evaluation.
- Keep the MFCC + Gaussian Mixture Model approach from the original code as a documented legacy experiment, not as the primary model.
- Expose inference through a REST API and a browser UI with in-page recording.
- Package everything for containerised deployment.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js (React), browser audio recording |
| Backend API | FastAPI (Python) |
| Machine learning | TensorFlow / Keras MLP, NumPy, SciPy, librosa, scikit-learn |
| Database | PostgreSQL |
| Infrastructure | Docker, Docker Compose, AWS |

## Key features

- Record audio directly in the browser or upload a `.wav` file.
- Python implementation of the acoustic features used by the original R pipeline (mean frequency, spectral entropy, spectral flatness, fundamental-frequency statistics, dominant-frequency statistics, modulation index, and others).
- Neural-network classifier with prediction confidence.
- Training scripts that produce real evaluation metrics: accuracy, precision, recall, F1 score and confusion matrix.
- Prediction history persisted in PostgreSQL.
- Containerised services for local development and cloud deployment.

## Accuracy targets

These are **targets**, not results. Every metric reported by this project must come from actual training and evaluation runs performed on this codebase; see `ml_models/` once the training phase lands.

| Metric | Target (held-out test set) |
|---|---|
| Accuracy | >= 95 % |
| Precision / Recall (per class) | >= 94 % |
| F1 score (macro) | >= 94 % |

For context only, the original report and its reference papers cite historical accuracies of roughly 95-97 % on the same public voice dataset using a comparable MLP. Those figures belong to the historical work and are not claimed by this system.

## Quick start

The commands below describe the intended developer workflow. Individual steps become runnable as their phase is completed.

```bash
# 1. Clone
git clone <repository-url>
cd Gender_recognition_through_voice

# 2. Backend (FastAPI)  -- requires Python 3.10 or 3.11, see backend/README.md
cd backend
py -3.11 -m venv .venv          # or: uv venv .venv --python 3.11
.venv\Scripts\activate          # Windows   |   source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload   # http://localhost:8000/health

# 3. Frontend (Next.js)  -- requires Node.js 18.17+
cd ../frontend
npm install
cp .env.example .env.local
npm run dev                     # http://localhost:3000

# 4. ML training
cd ../ml_models
pip install -r requirements.txt
python train.py

# 5. Everything via Docker
cd ../infrastructure
docker compose up --build
```

Optional developer tooling:

```bash
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

## Architecture

```
 +-------------------+        HTTPS         +--------------------+
 |     Browser       | -------------------> |   Next.js (React)  |
 |  mic recording /  |                      |     frontend/      |
 |  .wav upload      | <------------------- |                    |
 +-------------------+                      +---------+----------+
                                                      |
                                                      |  REST (JSON / multipart audio)
                                                      v
                                            +---------+----------+
                                            |   FastAPI backend  |
                                            |     backend/       |
                                            |                    |
                                            |  /predict          |
                                            |  /history          |
                                            |  /health           |
                                            +----+----------+----+
                                                 |          |
                        audio -> features -> NN  |          |  predictions, metadata
                                                 v          v
                                     +-----------+---+  +---+-------------+
                                     |  Inference    |  |   PostgreSQL    |
                                     |  service      |  |                 |
                                     |  (TensorFlow  |  +-----------------+
                                     |   MLP model)  |
                                     +-------+-------+
                                             ^
                                             |  trained model + scaler artifacts
                                             |
                                     +-------+-------+
                                     |  ml_models/   |
                                     |  feature      |
                                     |  extraction,  |
                                     |  training,    |
                                     |  evaluation   |
                                     +---------------+

 infrastructure/  ->  Dockerfiles, docker-compose, AWS deployment configuration
 docs/            ->  design notes, API reference, contribution guide
```

## Repository layout

```
.
├── backend/          FastAPI application (API, inference service, persistence)
├── frontend/         Next.js React application (recording UI, results)
├── ml_models/        Feature extraction, training, evaluation, model artifacts
├── infrastructure/   Docker and deployment configuration
├── docs/             Documentation
├── .gitattributes    Line-ending normalisation and Git LFS rules
├── .gitignore
├── .gitmessage       Conventional Commits template
├── .pre-commit-config.yaml
└── README.md
```

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for branch naming, commit conventions and the pull-request checklist.

## Acknowledgements

Original project team: P. Sai Prathyusha, Koyya Sravani, V. Sharmila Rani, Dokkari Niharika, under the guidance of G. Tirupati, Assistant Professor, Department of IT, GVPCEW.

Reference works:

- Buyukyilmaz, M. and Cibikdiken, A. O., "Voice Gender Recognition Using Deep Learning", MSOTA 2016.
- Nair, R. R. and Vijayan, B., "Voice based Gender Recognition", IRJET, vol. 6, issue 5, 2019.
- Becker, K., "Identifying the Gender of a Voice using Machine Learning" and the accompanying `voice.csv` dataset.
