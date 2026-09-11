"""Logging configuration: console + rotating file handler under `logs/`."""

import logging
import logging.handlers
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure the root logger once. Safe to call multiple times."""
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on reload (uvicorn --reload re-imports the app).
    for handler in list(root.handlers):
        if getattr(handler, "_voice_gender_handler", False):
            root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console._voice_gender_handler = True  # type: ignore[attr-defined]
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler._voice_gender_handler = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)

    # Quieten noisy third-party loggers.
    for name in ("uvicorn.access", "sqlalchemy.engine", "botocore", "numba"):
        logging.getLogger(name).setLevel(max(logging.WARNING, root.level))
