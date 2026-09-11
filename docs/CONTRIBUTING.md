# Contributing

## Branches

- `main` is always deployable.
- Work on short-lived branches named `<type>/<short-description>`, for example `feat/python-specan-features` or `fix/upload-size-limit`.

## Commits

This repository uses [Conventional Commits](https://www.conventionalcommits.org). A commit template is configured in `.gitmessage`; run `git commit` without `-m` to see it.

```
<type>(<scope>): <summary>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
Scopes: `backend`, `frontend`, `ml`, `infra`, `docs`, `repo`.

## Pre-commit hooks (optional but recommended)

```bash
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type commit-msg
pre-commit run --all-files
```

Hooks run `ruff` on Python, `prettier` on frontend files, basic hygiene checks, and validate the commit message format.

## Large files

Trained models (`*.h5`, `*.keras`, `*.pkl`, `*.joblib`) and datasets are ignored by default. If a model must be versioned, place it under an explicitly whitelisted path and it will be stored through Git LFS as configured in `.gitattributes`.

## Reporting results

Any accuracy, precision, recall, F1 or confusion-matrix figure committed to this repository must be produced by a script in `ml_models/` on this codebase. Do not copy numbers from the original report or reference papers.

## Pull request checklist

- [ ] Branch is rebased on `main`.
- [ ] Commit messages follow Conventional Commits.
- [ ] Tests and lint pass locally.
- [ ] Documentation updated where behaviour changed.
- [ ] No secrets, datasets or model binaries added outside the LFS-tracked paths.
