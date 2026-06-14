"""Sliding-window rate limiter for the Echo Chamber Detector API.

Each API key (or JWT ``sub`` identity) is allowed at most
:data:`RATE_LIMIT_REQUESTS` requests per :data:`RATE_LIMIT_WINDOW_SECONDS`
(default: 100 req / 60 s).

Implementation
--------------
When Redis is available the limiter uses a sorted-set sliding window —
a standard, accurate approach:

1. Remove all members whose score (timestamp in milliseconds) is older than
   the window.
2. Count remaining members.  If the count >= limit, reject with HTTP 429.
3. Add the current timestamp as a new member and set the key TTL to the
   window duration.

When Redis is unavailable (or ``REDIS_URL`` is ``"disabled"``), the limiter
falls back to an in-process ``dict`` + ``collections.deque`` sliding window.
This is accurate within a single process but does **not** share state across
multiple workers — acceptable for development / unit tests.

Environment variables
---------------------
``RATE_LIMIT_REQUESTS``
    Maximum requests per window (default ``100``).
``RATE_LIMIT_WINDOW_SECONDS``
    Rolling window duration in seconds (default ``60``).

References: Requirements 7.10
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Any

from fastapi import Depends, HTTPException, status

from api.auth import get_current_user

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RATE_LIMIT_REQUESTS: int = int(os.environ.get("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW_SECONDS: int = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

# ---------------------------------------------------------------------------
# In-process fallback store  { identity: deque[float] }
# ---------------------------------------------------------------------------

_local_windows: dict[str, deque] = {}


def _check_local(identity: str) -> bool:
    """Return ``True`` if the request is within the limit using the local store.

    Args:
        identity: API key or JWT subject string.

    Returns:
        ``True`` if allowed, ``False`` if the rate limit is exceeded.
    """
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    dq = _local_windows.setdefault(identity, deque())
    # Evict timestamps outside the current window
    while dq and dq[0] < window_start:
        dq.popleft()

    if len(dq) >= RATE_LIMIT_REQUESTS:
        return False

    dq.append(now)
    return True


def _check_redis(identity: str, client: Any) -> bool:
    """Return ``True`` if the request is within the limit using Redis sorted sets.

    Uses the standard sorted-set sliding-window pattern.  Each member is the
    current timestamp in milliseconds (stringified) with the timestamp as
    score so that the set is deduplicated by score on collision.

    Args:
        identity: API key or JWT subject string.
        client:   A connected ``redis.Redis`` instance.

    Returns:
        ``True`` if allowed, ``False`` if the rate limit is exceeded.
    """
    key = f"ratelimit:{identity}"
    now_ms = int(time.time() * 1_000)
    window_start_ms = now_ms - RATE_LIMIT_WINDOW_SECONDS * 1_000

    pipe = client.pipeline()
    # Remove stale entries
    pipe.zremrangebyscore(key, "-inf", window_start_ms)
    # Count remaining entries in window
    pipe.zcard(key)
    # Add current request — member is "ts_ms:random" to allow duplicates at
    # the same millisecond (use the value itself as a unique label)
    member = f"{now_ms}:{id(pipe)}"
    pipe.zadd(key, {member: now_ms})
    # Reset TTL
    pipe.expire(key, RATE_LIMIT_WINDOW_SECONDS + 1)
    results = pipe.execute()

    count_after_eviction: int = results[1]
    return count_after_eviction < RATE_LIMIT_REQUESTS


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def rate_limit(identity: str = Depends(get_current_user)) -> str:
    """FastAPI dependency that enforces per-identity rate limiting.

    Injects the authenticated identity from :func:`~api.auth.get_current_user`
    and checks it against the sliding-window rate limit.

    Returns:
        The *identity* string (pass-through, so downstream handlers can still
        use ``Depends(rate_limit)`` and receive the user ID).

    Raises:
        HTTPException(429): When the caller has exceeded the rate limit.
    """
    # Try Redis first; fall back to local store on any error
    allowed = False
    try:
        from api.cache import _get_redis, _CACHE_DISABLED  # noqa: PLC0415

        client = _get_redis()
        if client is _CACHE_DISABLED:
            allowed = _check_local(identity)
        else:
            allowed = _check_redis(identity, client)
    except Exception as exc:  # pragma: no cover
        logger.warning("Rate limit Redis check failed (%s); using local fallback", exc)
        allowed = _check_local(identity)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: max {RATE_LIMIT_REQUESTS} requests "
                f"per {RATE_LIMIT_WINDOW_SECONDS} seconds"
            ),
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
        )

    return identity
