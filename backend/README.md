# backend/

FastAPI application: REST API, inference service wrapper, and PostgreSQL persistence.

## Layout

```
backend/
├── app/
│   ├── main.py            FastAPI app, CORS + logging middleware, lifespan, GET /health
│   ├── config.py          Pydantic Settings (env / .env)
│   ├── api/               Routers: auth.py, predictions.py  (mounted under /api/v1)
│   ├── services/          Business logic: auth.py (hashing, JWT), prediction.py (pipeline)
│   ├── models/            SQLAlchemy engine, session and ORM models (database.py)
│   ├── schemas/           Pydantic schemas: user.py, prediction.py, health.py
│   ├── ml/                inference.py (model loading / predict), features.py (feature contract)
│   └── utils/             audio.py (upload validation, decoding), logger.py
├── tests/                 pytest suite (httpx TestClient)
├── logs/                  Rotating application log (git-ignored)
├── requirements.txt       Exact pinned dependencies
├── pyproject.toml         pytest and ruff configuration
└── .env.example           Configuration template
```

## Python version

The pinned stack (TensorFlow 2.14, NumPy 1.24) requires **Python 3.10 or 3.11**. If your default interpreter is newer, create the virtualenv with a 3.11 toolchain. The quickest route on any OS is [uv](https://docs.astral.sh/uv/):

```bash
pip install uv
uv python install 3.11
```

## Setup

```bash
cd backend
uv venv .venv --python 3.11            # or: py -3.11 -m venv .venv
uv pip install -r requirements.txt --python .venv   # or: .venv\Scripts\pip install -r requirements.txt
cp .env.example .env                   # then edit values as needed
```

## Run

```bash
.venv\Scripts\activate                 # Windows;  source .venv/bin/activate on Linux/macOS
uvicorn app.main:app --reload --port 8000
```

- Health check: <http://localhost:8000/health>
- Interactive docs: <http://localhost:8000/docs>

The API starts even when PostgreSQL or the trained model is missing; `/health` then reports `"status": "degraded"` with `database` / `model_loaded` telling you which is absent.

## Application setup (`app/main.py`)

- Metadata: title "Gender Voice Recognition API", version 1.0.0, Swagger UI at `/docs`, ReDoc at `/redoc`, tagged OpenAPI groups (system, authentication, predictions, analytics); every endpoint carries a summary/docstring and the main schemas ship examples.
- Middleware order (outermost first): CORS → request logging → exception handling → routes. CORS allows the configured origins with `GET, POST, DELETE, OPTIONS`, the `Content-Type` and `Authorization` headers, credentials, and a 3600 s preflight cache; because it is outermost, error responses carry CORS headers too.
- Request logging records method, path, status and time for every request; `/auth/login`, `/auth/register` and `/auth/refresh` are logged without their query string and marked `[sensitive]`; bodies are never logged.
- Errors always have the shape `{"detail": "message"}`: `HTTPException` (4xx/5xx), validation errors (422 with a readable `detail` plus `errors`; 400 on auth routes) and unhandled exceptions (500 "Internal server error", full traceback in the log).
- Lifecycle (`lifespan`): logs the configuration with secrets masked, probes PostgreSQL and applies migrations, verifies the model artifacts exist, loads the MLP once and warms up feature extraction; on shutdown it releases the model, disposes the connection pool and removes leftover upload temp files.
- Health: `GET /health` (no auth) returns `status` (`ok` when database and model are available, otherwise `degraded`), `version` and `timestamp`; `GET /` returns API information.
- Entry points: `uvicorn app.main:app --port 8000` or `python -m app.main` (binds 0.0.0.0:8000, no reload).

## Tests and lint

```bash
pytest                                   # from backend/
pytest --cov=app --cov-report=term-missing
ruff check . && ruff format --check .
```

From the repository root the same suite runs as `pytest backend/tests/` and `pytest --cov=app backend/tests/` (use the backend virtualenv's interpreter).

- The API tests run against the FastAPI app with the database dependency redirected to an in-memory SQLite database; PostgreSQL is not required (only `test_migrations.py` needs it and skips otherwise). The application lifespan runs once per session, so the deployed model artifacts in `app/ml/models/` are loaded exactly as in production.
- Shared fixtures (`tests/conftest.py`): `client` / `test_client`, `test_engine`, `db_session`, `test_user` (registered through the real endpoint), `valid_token`, `auth_headers_for_user`, `test_audio_file`.
- `tests/fixtures/sample_audio.wav` is the committed 3-second reference recording (16 kHz, mono, 16-bit PCM; synthetic voiced signal with a 120 Hz fundamental). Regenerate it with `python tests/fixtures/generate_sample_audio.py`; `test_sample_audio_fixture` checks the file matches the generator.
- `tests/test_predictions.py` holds the Phase 5 specification suite (feature extraction, model inference, prediction endpoint, authentication, analytics, database, performance budgets: inference < 500 ms, end-to-end prediction < 2 s, history/statistics queries < 100 ms). Coverage is configured in `pyproject.toml`; the suite currently measures about 91 % of `app/` (target > 80 %).

## Docker environment (local development)

The full multi-service `docker compose` stack is delivered with `infrastructure/`. For backend development you only need a PostgreSQL container that matches the default `DATABASE_URL`:

```bash
docker run -d --name voice-gender-db \
  -e POSTGRES_USER=voice \
  -e POSTGRES_PASSWORD=voice \
  -e POSTGRES_DB=voice_gender \
  -p 5432:5432 \
  -v voice-gender-pgdata:/var/lib/postgresql/data \
  postgres:16-alpine
```

Useful commands:

```bash
docker logs -f voice-gender-db          # follow database logs
docker exec -it voice-gender-db psql -U voice -d voice_gender   # open a psql shell
docker stop voice-gender-db && docker rm voice-gender-db        # tear down (data volume is kept)
docker volume rm voice-gender-pgdata    # remove the data volume too
```

With the container running, `/health` reports `"database": "connected"` and the application applies the Alembic migrations at startup (`init_db()` runs `alembic upgrade head`).

## Database schema and migrations

ORM models live in `app/models/database.py`:

| Table | Purpose | Notes |
|---|---|---|
| `users` | accounts | `username` and `email` unique + indexed; bcrypt `password_hash`; `set_password()` / `check_password()` |
| `predictions` | one row per inference | `user_id` (nullable, cascade delete, indexed), `predicted_gender` (`male`/`female`, CHECK), `confidence_score` (0..1, CHECK), `probabilities` JSON, `model_used`, `model_version`, `inference_time_ms`, `audio_file_path`, `audio_duration`, `created_at` indexed |
| `acoustic_features` | the 22 specan measurements of a prediction | snake_case columns (`q25`, `sp_ent`, ...); `to_dict()` returns canonical names; cascade-deleted with the prediction |
| `model_metrics` | benchmark metrics per trained model | `model_name` indexed |

`DATABASE_URL` accepts `postgresql://user:password@host:5432/dbname` or the explicit `postgresql+psycopg2://...` form.

Migrations use Alembic (`alembic.ini`, `alembic/env.py`, `alembic/versions/`). The URL comes from the application settings, so nothing needs editing in `alembic.ini`.

```bash
.venv\Scripts\activate
alembic upgrade head                                   # apply all migrations
alembic current                                        # show the applied revision
alembic revision --autogenerate -m "describe change"   # after editing the ORM models
alembic downgrade -1                                   # roll back one revision
alembic upgrade head --sql                             # print SQL without applying it
```

Autogenerate needs a database to diff against; it works against PostgreSQL or a scratch SQLite URL (`set DATABASE_URL=sqlite:///scratch.db`). Review generated files before committing: check `server_default`, JSON and constraint details for the target dialect.

`init_db()` (called in the FastAPI lifespan) prefers migrations and only falls back to `Base.metadata.create_all()` when Alembic is unavailable, so a database created by the app is always stamped with the current revision.

## Authentication

JWT bearer tokens (HS256). Access tokens live `ACCESS_TOKEN_EXPIRE_DAYS` (30) days, refresh tokens `REFRESH_TOKEN_EXPIRE_DAYS` (90). Payload: `{"sub": "<user id>", "exp": ..., "iat": ..., "type": "access" | "refresh"}`. `JWT_SECRET_KEY` must be at least 32 characters (the settings refuse shorter values).

| Method | Path | Body | Success | Errors |
|---|---|---|---|---|
| POST | `/api/v1/auth/register` | `username` (3-50 alphanumeric), `email`, `password` (8+), `password_confirm` | 201 `{message, user}` | 400 validation, 409 username/email taken |
| POST | `/api/v1/auth/login` | `email`, `password` | 200 `{access_token, token_type, expires_in, refresh_token}` | 401 invalid email or password (same message for both, no account enumeration) |
| POST | `/api/v1/auth/refresh` | `refresh_token` | 200 new token pair | 401 invalid/expired refresh token |
| POST | `/api/v1/auth/logout` | bearer token | 200 `{message}` | 401 |
| GET | `/api/v1/auth/me` | bearer token | 200 user | 401 missing/invalid/expired token, 404 user deleted |

Protect a route with the dependency:

```python
from app.api.auth import get_current_user

@router.get("/secret")
def secret(user: User = Depends(get_current_user)): ...
```

`get_optional_user` returns `None` instead of 401 when no token is sent (anonymous access). Validation errors on auth routes are returned as 400 with a readable `{"detail": "..."}`; other routes keep FastAPI's 422. Logout is stateless (no server-side token blacklist); clients discard their tokens. Successful logins (user id), failed logins (email) and registration attempts are logged under the `app.auth` logger.

## Predictions

| Method | Path | Body | Success | Errors |
|---|---|---|---|---|
| POST | `/api/v1/predictions` | multipart field `audio` (`.wav` or `.mp3`; `file` accepted as an alias), max 10 MB, 1-10 s | 200 prediction | 400 no file / too large / invalid format / bad duration / corrupted, 401, 422 no voiced speech, 500 model or database failure, 503 model not loaded |
| POST | `/api/v1/predictions/batch` | multipart field `files` (up to `MAX_BATCH_FILES`) | 200 per-file results, one batched model call | 400, 401 |
| GET | `/api/v1/predictions` | bearer token | history (later phase) | 501 |

Pipeline per file: stream to a temp file (size limit enforced while reading) → mp3 to wav conversion (libsndfile, no ffmpeg) → librosa load, mono, 16 kHz, duration check → 22 specan measurements via the shared `voice_features` package → scaler + MLP (loaded once at startup, warmed up, manifest hashes verified) → `Prediction` + `AcousticFeatures` rows → optional S3 upload (`S3_BUCKET_NAME`) → temp files removed.

Response fields: `prediction_id`, `gender` (`male`/`female`, decided by P(male) > 0.5), `confidence`, `confidence_percentage`, `probabilities`, `inference_time_ms`, `feature_extraction_time_ms`, `total_time_ms`, `audio_duration`, `sample_rate`, `audio_features` (the 20 model inputs), `model_used`, `model_version`, plus the aliases `predicted_gender`, `features`, `duration_seconds` used by the frontend.

Authentication is required (`PREDICTIONS_REQUIRE_AUTH=true`); set it to `false` for anonymous demo use. When the database is unreachable the prediction is still returned with `prediction_id: null`. Every prediction is logged with user id, gender, confidence and timings; running statistics are logged every 25 predictions.

## Analytics and history

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/v1/me/predictions?limit=20&offset=0&sort_by=-date&gender_filter=all` | yes | Paginated history (`limit` 1-100; sort keys `date`, `confidence`, `duration`, prefix `-` for descending; 400 on bad parameters) |
| GET | `/api/v1/me/predictions/{id}` | yes | One prediction with all 22 stored acoustic measurements (403 if owned by someone else, 404 if missing) |
| DELETE | `/api/v1/me/predictions/{id}` | yes | Delete a prediction, its features and its S3 object (500 if the S3 delete fails) |
| GET | `/api/v1/me/statistics` | yes | Totals, gender distribution, confidence by gender, duration range, daily counts for the last 30 days |
| GET | `/api/v1/models/performance` | no | Benchmark metrics of the trained models and the current production model |
| GET | `/api/v1/predictions` | yes | Legacy listing in the `{items, total}` shape used by the current frontend |

Statistics are cached per user under `user:{id}:statistics` for `CACHE_TTL_SECONDS` (1 hour) and invalidated when the user creates or deletes a prediction. Redis is used when `REDIS_URL` is set and reachable, otherwise a process-local in-memory cache; hits, misses and the running hit rate are logged under `app.cache`. Model metrics come from the `model_metrics` table, which is seeded at startup from `model_config.json` and, when present, the Phase 2D `comparison_report.json`; they are benchmark figures on the historical `voice.csv` feature space.

## Configuration

All variables are documented in [.env.example](.env.example). Highlights:

| Variable | Purpose | Default |
|---|---|---|
| `ENVIRONMENT` | `development` / `staging` / `production` | `development` |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` |
| `CORS_ORIGINS` | Comma-separated allowed origins | `http://localhost:3000` |
| `DATABASE_URL` | SQLAlchemy PostgreSQL URL | local docker container |
| `JWT_SECRET_KEY` | Token signing secret (required in production) | insecure dev value |
| `AWS_*`, `S3_BUCKET_NAME` | Optional S3 storage / deployment | unset |
| `MODEL_PATH`, `SCALER_PATH` | Trained artifacts exported by `ml_models/` | `../ml_models/models/...` |
