"""Redis cache helpers for the Echo Chamber Detector API.

Provides a lazily-initialized Redis client and typed get/set wrappers used
by API route handlers.  All keys follow the conventions defined in
``graph/redis_keys.py``.

TTL
---
All cache entries use :data:`DEFAULT_TTL_SECONDS` (24 h = 86 400 s) unless
the caller passes an explicit *ttl* argument.

Environment variables
---------------------
``REDIS_URL``
    Redis connection URL (default: ``redis://localhost:6379/0``).
    Set to an empty string or ``"disabled"`` to disable caching entirely
    (useful for tests that do not want a real Redis instance).

References: Requirements 7.7, design.md Redis key conventions
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Default cache TTL in seconds (24 hours), matching the spec convention.
DEFAULT_TTL_SECONDS: int = 86_400

#: Sentinel that disables caching without raising errors.
_CACHE_DISABLED = object()

_redis_client: Any = None  # redis.Redis | _CACHE_DISABLED | None


def _get_redis() -> Any:
    """Return a lazily-initialized Redis client, or the disabled sentinel.

    If ``REDIS_URL`` is empty or ``"disabled"``, caching is skipped silently.
    Connection errors on the first call also disable caching for the process
    lifetime (avoids log spam on every request when Redis is unavailable).
    """
    global _redis_client  # noqa: PLW0603

    if _redis_client is not None:
        return _redis_client

    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0").strip()
    if not url or url.lower() == "disabled":
        logger.info("Redis caching disabled (REDIS_URL=%r)", url)
        _redis_client = _CACHE_DISABLED
        return _redis_client

    try:
        import redis  # noqa: PLC0415

        client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
        # Ping to verify connectivity eagerly
        client.ping()
        _redis_client = client
        logger.info("Redis cache connected: %s", url)
    except Exception as exc:  # pragma: no cover
        logger.warning("Redis unavailable (%s) — caching disabled", exc)
        _redis_client = _CACHE_DISABLED

    return _redis_client


def cache_get(key: str) -> Optional[str]:
    """Return the cached string for *key*, or ``None`` on miss / error.

    Args:
        key: Redis key to fetch.

    Returns:
        Cached string value, or ``None`` if the key is absent or Redis is
        unavailable.
    """
    client = _get_redis()
    if client is _CACHE_DISABLED:
        return None
    try:
        return client.get(key)  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover
        logger.warning("Redis GET failed for key %r: %s", key, exc)
        return None


def cache_set(key: str, value: str, ttl: int = DEFAULT_TTL_SECONDS) -> None:
    """Store *value* under *key* with an expiry of *ttl* seconds.

    Failures are logged and silently swallowed so that a Redis outage never
    breaks a successful API response.

    Args:
        key:   Redis key.
        value: String value to store (typically JSON-serialized).
        ttl:   Expiry in seconds (default 24 h).
    """
    client = _get_redis()
    if client is _CACHE_DISABLED:
        return
    try:
        client.setex(key, ttl, value)  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover
        logger.warning("Redis SET failed for key %r: %s", key, exc)


def cache_get_json(key: str) -> Optional[Any]:
    """Return a deserialized JSON object from *key*, or ``None`` on miss."""
    raw = cache_get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover
        logger.warning("Failed to decode cached JSON for key %r: %s", key, exc)
        return None


def cache_set_json(key: str, value: Any, ttl: int = DEFAULT_TTL_SECONDS) -> None:
    """Serialize *value* to JSON and store it under *key* with *ttl*."""
    try:
        cache_set(key, json.dumps(value, default=str), ttl)
    except (TypeError, ValueError) as exc:  # pragma: no cover
        logger.warning("Failed to serialize value for cache key %r: %s", key, exc)


def reset_client() -> None:
    """Force re-initialization of the Redis client on the next access.

    Intended for tests that need to swap out the Redis URL between cases.
    """
    global _redis_client  # noqa: PLW0603
    _redis_client = None
