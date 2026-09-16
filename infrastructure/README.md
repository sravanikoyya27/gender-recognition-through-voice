# infrastructure/

Dockerfiles and Docker Compose stack for Gender Recognition Through Voice. AWS deployment
configuration follows in the deployment phase.

```
infrastructure/
├── Dockerfile.backend            python:3.11-slim, FastAPI + TensorFlow MLP, port 8000
├── Dockerfile.frontend           node:18-alpine multi-stage, Next.js standalone server, port 3000
├── docker-compose.yml            postgres, redis, backend, frontend (production-like base)
├── docker-compose.override.yml   development additions: backend source mount + uvicorn --reload
└── .env.example                  template for the stack's .env
```

The build context of both images is the **repository root** (`context: ..`): the backend
installs the shared feature package from `ml_models/` (`-e ../ml_models` in
`backend/requirements.txt`), so a `./backend`-only context cannot build it. The root
`.dockerignore` keeps virtualenvs, `node_modules`, `.env` files, tests and training-only
artifacts out of the images.

## Local development

```bash
cd infrastructure
cp .env.example .env
# Edit .env with your values (POSTGRES_PASSWORD and JWT_SECRET_KEY at least)
docker compose up -d

# View logs:
docker compose logs -f backend
docker compose logs -f frontend

# Stop services:
docker compose down
```

`docker-compose` (the v1 spelling) works the same with Docker Desktop's compatibility shim.
The first build downloads TensorFlow (about 500 MB) and takes several minutes; later builds
reuse the dependency layers.

Then open http://localhost:3000 (frontend), http://localhost:8000/docs (API) and
http://localhost:8000/health.

### Before the first `up`

- The backend image copies the deployed model artifacts from `backend/app/ml/models/`
  (`model_mlp.h5`, `scaler_mlp.pkl`, `model_config.json`). They are produced by
  `ml_models/model_comparison.py` and, apart from the manifest, are not tracked by git. Without
  them the API starts in degraded mode and predictions return 503.
- Host port 5432 can only be published by one container. If the standalone development
  database is running (`voice-gender-db`), either stop it (`docker stop voice-gender-db`) or set
  `POSTGRES_PORT=5433` in `.env` before `docker compose up`. A failed port binding leaves the
  `postgres` container off the `voice-app` network, and the backend then logs
  `Database unavailable: OperationalError` and stays unhealthy; fix it with
  `docker compose up -d --force-recreate postgres && docker compose restart backend`.
- `NEXT_PUBLIC_API_URL` is inlined into the browser bundle when the frontend image is built.
  After changing it run `docker compose build frontend`.

## Services and network

All services join the single bridge network `voice-app` and address each other by service
name:

| Service | Image | Reached inside the network as | Published port |
|---|---|---|---|
| postgres | postgres:15-alpine | `postgres:5432` | 5432 |
| redis | redis:7-alpine | `redis:6379` | 6379 |
| backend | built from Dockerfile.backend | `backend:8000` | 8000 |
| frontend | built from Dockerfile.frontend | `frontend:3000` | 3000 |

The backend receives `DATABASE_URL=postgresql://<user>:<password>@postgres:5432/<db>` (derived
from the `POSTGRES_*` values unless `DATABASE_URL` is set) and `REDIS_URL=redis://redis:6379/0`.
The frontend container can reach the API at `http://backend:8000` (`API_INTERNAL_URL`), but the
browser cannot resolve service names, so the public URL stays `http://localhost:8000`.

Start-up order is enforced with health checks: `backend` waits for `postgres` and `redis` to be
healthy, `frontend` waits for `backend`'s `/health` (the model takes tens of seconds to load;
the backend health check allows a 120 s start period).

## Volumes

| Volume / mount | Purpose |
|---|---|
| `postgres_data` (named volume) | Persists the database across `docker compose down`. `docker compose down -v` deletes it. |
| `../backend/app:/app/app`, `../backend/alembic:/app/alembic` (override file only) | Development hot-reload: uvicorn restarts on edits under `backend/app`. |

Production runs without source mounts: `docker compose -f docker-compose.yml up -d` uses only
the base file, so the containers run the code baked into the images with no `--reload`.

## CI/CD (GitHub Actions)

`.github/workflows/deploy.yml` runs on pull requests to `main`, pushes to `main` and manually
(`workflow_dispatch`).

| Job | When | What |
|---|---|---|
| Backend tests | every run | Python 3.11, `pip install -r backend/requirements.txt`, PostgreSQL service for the migration tests, `pytest backend/tests/ --cov=app --cov-fail-under=80`, optional Codecov upload |
| Frontend tests | every run | Node 18, `npm ci`, ESLint, strict `tsc`, `next build`, `npm test --if-present` |
| Build and push images | `main` only | ECR login, build both images (repository-root context), tag `<account>.dkr.ecr.<region>.amazonaws.com/voice-backend:<commit-sha>` (and `latest`), same for `voice-frontend` |
| Deploy to EC2 | `main` only, after a successful build | copies `docker-compose.yml` + `deploy.sh` to the instance over SSH, then runs `deploy.sh deploy <sha>` |

The model artifacts are not in git, so the test job restores them from the Actions cache
(keyed on the `ml_models/` sources) and, on a cache miss, regenerates them with the real Phase 2
pipeline (`prepare_voice_csv.py` → `train_neural_network.py` → `train_sklearn_models.py` →
`model_comparison.py`). The build job downloads them as a workflow artifact.

### Secrets and variables

Secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `EC2_HOST`, `EC2_USER`,
`EC2_PRIVATE_KEY`, optional `DEPLOY_USER` (SSH user override), `SLACK_WEBHOOK_URL` (notifications;
skipped when unset), `CODECOV_TOKEN` (coverage upload; skipped when unset).
Variables: `NEXT_PUBLIC_API_URL` (public API URL baked into the frontend image), `DEPLOY_PATH`
(default `/opt/voice-gender`), `ECR_BACKEND_REPO` / `ECR_FRONTEND_REPO` (defaults `voice-backend` /
`voice-frontend`). The `production` environment can carry approval rules for the deploy job.

### The EC2 instance

One-time preparation: Docker Engine with the Compose plugin, the AWS CLI, `curl`; the deploy
user in the `docker` group; `DEPLOY_PATH` containing a filled-in `.env` (from `.env.example`,
with production values and `ENVIRONMENT=production`); two ECR repositories; and either an
instance role allowed to pull from ECR or the same AWS keys as the workflow.

`deploy.sh deploy <tag>` logs in to ECR, pulls both images, writes `BACKEND_IMAGE` /
`FRONTEND_IMAGE` into `.env` (which `docker-compose.yml` reads), starts postgres and redis,
runs `docker compose run --rm backend alembic upgrade head`, then `docker compose down` and
`up -d` (base file only: no source mounts), and polls `/health` five times, 15 s apart.

### Rollback

Old images stay on the instance (nothing is pruned), and the last two deployed tags are recorded
in `.deploy/current_tag` and `.deploy/previous_tag`.

- Automatic: if the health check fails, `deploy.sh` re-activates the previous tag and the
  workflow fails (Slack is told).
- Manual: run the workflow with the `image_tag` input set to any existing ECR tag (tests and
  build are skipped), or on the instance run `./deploy.sh rollback` (previous tag) or
  `./deploy.sh deploy <tag>`; `./deploy.sh status` shows the tags and container health.

## Useful commands

```bash
docker compose ps                       # status and health of each service
docker compose build backend            # rebuild after changing requirements.txt
docker compose exec backend alembic current
docker compose exec postgres psql -U postgres -d voicedb
docker compose down -v                  # stop and delete the database volume
```
