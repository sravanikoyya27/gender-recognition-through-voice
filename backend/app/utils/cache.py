"""Small caching layer: Redis when ``REDIS_URL`` is configured and reachable, in-memory otherwise.

Values are JSON documents. Hit/miss counters are kept so the hit rate can be logged.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from functools import lru_cache
from typing import Any, Protocol

from app.config import get_settings

logger = logging.getLogger("app.cache")


class CacheBackend(Protocol):
    name: str

    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl: int) -> None: ...
    def delete(self, key: str) -> None: ...


class InMemoryCache:
    """Thread-safe dict with per-key expiry (process-local; fine for one worker or tests)."""

    name = "memory"

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at < time.monotonic():
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: str, ttl: int) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + ttl, value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class RedisCache:
    name = "redis"

    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(
            url, socket_connect_timeout=2, socket_timeout=2, decode_responses=True
        )
        self._client.ping()  # fail fast so the caller can fall back

    def get(self, key: str) -> str | None:
        return self._client.get(key)

    def set(self, key: str, value: str, ttl: int) -> None:
        self._client.set(key, value, ex=ttl)

    def delete(self, key: str) -> None:
        self._client.delete(key)


class CacheService:
    """JSON get/set/delete with hit-rate accounting on top of a backend."""

    def __init__(self, backend: CacheBackend, default_ttl: int) -> None:
        self.backend = backend
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend.name,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 3),
        }

    def get_json(self, key: str) -> Any | None:
        try:
            raw = self.backend.get(key)
        except Exception:  # noqa: BLE001 - a cache outage must never break a request
            logger.warning("Cache get failed for %s", key, exc_info=True)
            raw = None
        with self._lock:
            if raw is None:
                self.misses += 1
            else:
                self.hits += 1
        logger.info("cache %s key=%s hit_rate=%.2f", "hit" if raw is not None else "miss", key, self.hit_rate)
        return json.loads(raw) if raw is not None else None

    def set_json(self, key: str, value: Any, ttl: int | None = None) -> None:
        try:
            self.backend.set(key, json.dumps(value, default=str), ttl or self.default_ttl)
        except Exception:  # noqa: BLE001
            logger.warning("Cache set failed for %s", key, exc_info=True)

    def delete(self, key: str) -> None:
        try:
            self.backend.delete(key)
            logger.debug("cache invalidate key=%s", key)
        except Exception:  # noqa: BLE001
            logger.warning("Cache delete failed for %s", key, exc_info=True)


def statistics_key(user_id: int) -> str:
    return f"user:{user_id}:statistics"


@lru_cache
def get_cache() -> CacheService:
    settings = get_settings()
    backend: CacheBackend
    if settings.REDIS_URL:
        try:
            backend = RedisCache(settings.REDIS_URL)
            logger.info("Cache backend: redis (%s)", settings.REDIS_URL)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis unavailable (%s); using in-memory cache", exc.__class__.__name__)
            backend = InMemoryCache()
    else:
        backend = InMemoryCache()
        logger.info("Cache backend: in-memory (REDIS_URL not set)")
    return CacheService(backend, settings.CACHE_TTL_SECONDS)
