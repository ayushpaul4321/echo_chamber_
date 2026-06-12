"""Redis key-building helpers for the Echo Chamber Detector.

Key conventions
---------------
All cached values use a 24-hour (86 400 second) TTL.  Callers are responsible
for setting the TTL when writing to Redis, for example::

    redis_client.setex(polarization_key(snap_id), 86400, serialized_value)

Key patterns
~~~~~~~~~~~~
``metrics:{snapshot_id}:polarization``
    Serialized ``PolarizationMetrics`` for the given snapshot.

``user:{user_id}:recommendations``
    JSON list of ``Recommendation`` objects for the given user.

``graph:{snapshot_id}:page:{n}``
    Paginated adjacency-list chunk for cursor-based graph API responses.
    ``n`` is the zero-based page index.
"""

from __future__ import annotations

#: Default TTL for all cached entries, in seconds (24 hours).
DEFAULT_TTL_SECONDS: int = 86_400


def polarization_key(snapshot_id: str) -> str:
    """Return the Redis key for a snapshot's polarization metrics cache.

    Pattern: ``metrics:{snapshot_id}:polarization``

    Args:
        snapshot_id: UUID string identifying the snapshot.

    Returns:
        Redis key string, e.g. ``"metrics:snap-abc123:polarization"``.
    """
    return f"metrics:{snapshot_id}:polarization"


def recommendations_key(user_id: str) -> str:
    """Return the Redis key for a user's recommendation list cache.

    Pattern: ``user:{user_id}:recommendations``

    Args:
        user_id: String identifier for the target user.

    Returns:
        Redis key string, e.g. ``"user:user_42:recommendations"``.
    """
    return f"user:{user_id}:recommendations"


def graph_page_key(snapshot_id: str, page: int) -> str:
    """Return the Redis key for a paginated graph chunk.

    Pattern: ``graph:{snapshot_id}:page:{page}``

    Args:
        snapshot_id: UUID string identifying the snapshot.
        page: Zero-based page index.

    Returns:
        Redis key string, e.g. ``"graph:snap-abc123:page:3"``.
    """
    return f"graph:{snapshot_id}:page:{page}"
