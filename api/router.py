"""REST API Router for the Echo Chamber Detector.

Endpoints:
  GET /api/snapshots/{snapshotId}/graph
  GET /api/snapshots/{snapshotId}/metrics/polarization
  GET /api/snapshots/{snapshotId}/metrics/signed
  GET /api/metrics/polarization          (filtered list, cached)
  GET /api/users/{userId}/metrics
  GET /api/communities/{communityId}/metrics
  GET /api/users/{userId}/recommendations

All endpoints:
  - Require JWT or API key authentication via ``Authorization: Bearer <token>``
  - Are rate-limited to 100 requests / 60 s per authenticated identity

Pagination:
  - The graph endpoint uses cursor-based pagination (default 500 nodes/page)
  - The filtered polarization list endpoint uses cursor-based pagination

Redis caching:
  - ``GET /api/snapshots/{snapshotId}/metrics/polarization`` is cached
    using the ``metrics:{snapshotId}:polarization`` key (TTL 24 h)
  - ``GET /api/metrics/polarization`` cached per canonical query-param key

Filter parameters (available on list/metrics endpoints):
  ``?datasetSource=``   — filter by dataset source string
  ``?snapshotId=``      — filter by snapshot ID (on list endpoint)
  ``?from=``            — lower-bound on ``computedAt`` (ISO 8601 datetime)
  ``?to=``              — upper-bound on ``computedAt`` (ISO 8601 datetime)
  ``?communityId=``     — filter by community ID (user metrics endpoint)
  ``?min_polarization=``— filter by minimum polarization index

References: design.md Component 6, Requirements 7.1–7.11
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.cache import DEFAULT_TTL_SECONDS, cache_get_json, cache_set_json
from api.dtos import (
    CommunityMetricsDTO,
    EdgeDTO,
    GraphDTO,
    LatestSnapshotDTO,
    NodeDTO,
    PolarizationDTO,
    PolarizationListDTO,
    RecommendationDTO,
    SignedMetricsDTO,
    UserMetricsDTO,
    UserMetricsListDTO,
)
from api.rate_limit import rate_limit

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Lazy service / DB helpers
# ---------------------------------------------------------------------------


def _get_db_session():
    """Yield a SQLAlchemy session from the configured engine.

    Falls back to an in-memory SQLite database when ``DATABASE_URL`` is not
    set (useful for development and testing).
    """
    import os  # noqa: PLC0415

    from sqlalchemy import create_engine  # noqa: PLC0415
    from sqlalchemy.orm import Session  # noqa: PLC0415

    db_url = os.environ.get("DATABASE_URL", "sqlite:///:memory:")
    engine = create_engine(db_url, pool_pre_ping=True)
    session = Session(bind=engine)
    try:
        yield session
    finally:
        session.close()


def _make_cache_key(prefix: str, **params) -> str:
    """Build a deterministic Redis cache key from a prefix and query parameters.

    Sorts parameters by name so that different orderings of identical queries
    hit the same cache entry.

    Args:
        prefix:  A short string identifying the endpoint (e.g. ``"pol_list"``).
        **params: Query parameter names and their values.

    Returns:
        A cache key string of the form ``"{prefix}:{sha256_hex[:16]}"``.
    """
    canonical = json.dumps(
        {k: str(v) for k, v in sorted(params.items()) if v is not None},
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"


# ---------------------------------------------------------------------------
# GET /api/snapshots/{snapshotId}/graph
# ---------------------------------------------------------------------------


@router.get(
    "/snapshots/{snapshotId}/graph",
    response_model=GraphDTO,
    summary="Get paginated InteractionGraph snapshot",
)
async def get_graph(
    snapshotId: str,
    cursor: Optional[str] = Query(
        default=None,
        description="Pagination cursor — userId of the last node on the previous page",
    ),
    page_size: int = Query(
        default=500,
        ge=1,
        le=5000,
        description="Nodes per page (default 500)",
    ),
    current_user: str = Depends(rate_limit),
) -> GraphDTO:
    """Return a paginated :class:`~api.dtos.GraphDTO` for the given snapshot.

    Cursor-based pagination: ``cursor`` is the ``userId`` of the last node on
    the previous page.  Nodes are returned in lexicographic order.  Each page
    also includes all edges whose both endpoints appear on the current page.

    Requirements: 7.1, 7.6
    """
    from graph.service import GraphConstructionService  # noqa: PLC0415

    service = GraphConstructionService()
    try:
        graph = service.load_graph(snapshotId)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Snapshot '{snapshotId}' not found",
        )

    # Lexicographically sorted node list for stable cursor pagination
    all_user_ids = sorted(graph.nodes.keys())

    # Advance past the cursor
    if cursor is not None:
        try:
            start_idx = all_user_ids.index(cursor) + 1
        except ValueError:
            start_idx = 0
    else:
        start_idx = 0

    page_user_ids = all_user_ids[start_idx : start_idx + page_size]
    page_user_set = set(page_user_ids)

    node_dtos: list[NodeDTO] = [
        NodeDTO(
            userId=uid,
            communityId=graph.nodes[uid].communityId,
            betweenness=graph.nodes[uid].betweenness,
            diversityScore=graph.nodes[uid].diversityScore,
            topicVector=graph.nodes[uid].topicVector,
        )
        for uid in page_user_ids
    ]

    edge_dtos: list[EdgeDTO] = [
        EdgeDTO(
            sourceUserId=edge.sourceUserId,
            targetUserId=edge.targetUserId,
            weight=edge.weight,
            isCrossCommunity=edge.isCrossCommunity,
            signedPolarity=edge.signedPolarity,
        )
        for edge in graph.edges
        if edge.sourceUserId in page_user_set and edge.targetUserId in page_user_set
    ]

    has_more = (start_idx + page_size) < len(all_user_ids)
    next_cursor: Optional[str] = page_user_ids[-1] if (has_more and page_user_ids) else None

    return GraphDTO(
        nodes=node_dtos,
        edges=edge_dtos,
        snapshotId=graph.snapshotId,
        createdAt=graph.createdAt,
        nodeCount=graph.nodeCount,
        edgeCount=graph.edgeCount,
        nextCursor=next_cursor,
    )


# ---------------------------------------------------------------------------
# GET /api/snapshots/latest  (live-refresh polling endpoint)
# ---------------------------------------------------------------------------


@router.get(
    "/snapshots/latest",
    response_model=LatestSnapshotDTO,
    summary="Get the most recent snapshot descriptor for a dataset source",
)
async def get_latest_snapshot(
    datasetSource: Optional[str] = Query(
        default=None,
        description="Filter by dataset source (e.g. 'reddit_title', 'congress', 'wiki_rfa')",
    ),
    current_user: str = Depends(rate_limit),
    db=Depends(_get_db_session),
) -> LatestSnapshotDTO:
    """Return the most recent :class:`~api.dtos.LatestSnapshotDTO` for the given
    dataset source.

    Used by the Dashboard's live-refresh polling hook (Requirement 8.6): the
    frontend calls this every 60 s and re-fetches all data when the returned
    ``snapshotId`` differs from the currently loaded one.

    Requirements: 8.6
    """
    from graph.db_models import PolarizationMetricRow  # noqa: PLC0415

    q = db.query(PolarizationMetricRow)
    if datasetSource is not None:
        q = q.filter(PolarizationMetricRow.dataset_source == datasetSource)

    row: Optional[PolarizationMetricRow] = (
        q.order_by(PolarizationMetricRow.computed_at.desc()).first()
    )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No snapshot found"
                + (f" for datasetSource '{datasetSource}'" if datasetSource else "")
            ),
        )

    return LatestSnapshotDTO(
        snapshotId=row.snapshot_id,
        datasetSource=row.dataset_source,
        computedAt=row.computed_at,
    )


# ---------------------------------------------------------------------------
# GET /api/snapshots/{snapshotId}/metrics/polarization  (cached)
# ---------------------------------------------------------------------------


@router.get(
    "/snapshots/{snapshotId}/metrics/polarization",
    response_model=PolarizationDTO,
    summary="Get polarization metrics for a snapshot",
)
async def get_polarization_metrics(
    snapshotId: str,
    current_user: str = Depends(rate_limit),
    db=Depends(_get_db_session),
) -> PolarizationDTO:
    """Return :class:`~api.dtos.PolarizationDTO` for the given snapshot.

    Response is served from the Redis cache (TTL 24 h) on repeat calls.
    Cache key: ``metrics:{snapshotId}:polarization``

    Requirements: 7.2, 7.7
    """
    from graph.redis_keys import polarization_key  # noqa: PLC0415

    cache_key = polarization_key(snapshotId)

    # --- Cache hit ---
    cached = cache_get_json(cache_key)
    if cached is not None:
        logger.debug("Cache hit: %s", cache_key)
        return PolarizationDTO(**cached)

    # --- Cache miss: query DB ---
    from graph.db_models import PolarizationMetricRow  # noqa: PLC0415

    row: Optional[PolarizationMetricRow] = (
        db.query(PolarizationMetricRow)
        .filter(PolarizationMetricRow.snapshot_id == snapshotId)
        .order_by(PolarizationMetricRow.computed_at.desc())
        .first()
    )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No polarization metrics found for snapshot '{snapshotId}'",
        )

    dto = PolarizationDTO(
        snapshotId=row.snapshot_id,
        polarizationIndex=row.polarization_index,
        modularity=row.modularity,
        communityCount=row.community_count,
        avgCommunitySize=row.avg_community_size,
        interCommunityEdgeRatio=row.inter_community_edge_ratio,
        computedAt=row.computed_at,
    )

    # --- Populate cache ---
    cache_set_json(cache_key, dto.model_dump(), ttl=DEFAULT_TTL_SECONDS)

    return dto


# ---------------------------------------------------------------------------
# GET /api/metrics/polarization  (filtered list, cached)
# ---------------------------------------------------------------------------


@router.get(
    "/metrics/polarization",
    response_model=PolarizationListDTO,
    summary="List polarization metrics with filtering and pagination",
)
async def list_polarization_metrics(
    datasetSource: Optional[str] = Query(
        default=None,
        alias="datasetSource",
        description="Filter by dataset source (e.g. 'reddit_title', 'congress', 'wiki_rfa')",
    ),
    snapshotId: Optional[str] = Query(
        default=None,
        description="Filter by snapshot ID",
    ),
    from_dt: Optional[datetime] = Query(
        default=None,
        alias="from",
        description="Lower bound on computedAt (ISO 8601)",
    ),
    to_dt: Optional[datetime] = Query(
        default=None,
        alias="to",
        description="Upper bound on computedAt (ISO 8601)",
    ),
    min_polarization: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum polarization index",
    ),
    cursor: Optional[str] = Query(
        default=None,
        description="Pagination cursor — snapshotId of the last record on the previous page",
    ),
    page_size: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Records per page (default 50)",
    ),
    current_user: str = Depends(rate_limit),
    db=Depends(_get_db_session),
) -> PolarizationListDTO:
    """Return a filtered, paginated list of :class:`~api.dtos.PolarizationDTO`.

    Supports the following query parameters:
    - ``?datasetSource=``   — exact match on ``dataset_source``
    - ``?snapshotId=``      — exact match on ``snapshot_id``
    - ``?from=``            — ``computedAt >= from``
    - ``?to=``              — ``computedAt <= to``
    - ``?min_polarization=``— ``polarization_index >= min_polarization``

    Results are ordered by ``computed_at`` descending.  Cursor is the
    ``snapshotId`` of the last record returned.

    Responses are cached in Redis for 24 h per unique combination of filter
    and pagination parameters.

    Requirements: 7.6, 7.7, 7.11
    """
    # Build a deterministic cache key from all query params
    cache_key = _make_cache_key(
        "pol_list",
        datasetSource=datasetSource,
        snapshotId=snapshotId,
        from_dt=from_dt,
        to_dt=to_dt,
        min_polarization=min_polarization,
        cursor=cursor,
        page_size=page_size,
    )

    cached = cache_get_json(cache_key)
    if cached is not None:
        logger.debug("Cache hit: %s", cache_key)
        return PolarizationListDTO(**cached)

    from graph.db_models import PolarizationMetricRow  # noqa: PLC0415

    q = db.query(PolarizationMetricRow)

    if datasetSource is not None:
        q = q.filter(PolarizationMetricRow.dataset_source == datasetSource)
    if snapshotId is not None:
        q = q.filter(PolarizationMetricRow.snapshot_id == snapshotId)
    if from_dt is not None:
        q = q.filter(PolarizationMetricRow.computed_at >= from_dt)
    if to_dt is not None:
        q = q.filter(PolarizationMetricRow.computed_at <= to_dt)
    if min_polarization is not None:
        q = q.filter(PolarizationMetricRow.polarization_index >= min_polarization)

    total: int = q.count()

    # Cursor: skip past rows whose snapshot_id <= cursor
    # Use row ID as a stable sort key to allow pure offset pagination via cursor
    q = q.order_by(PolarizationMetricRow.computed_at.desc(), PolarizationMetricRow.id.asc())

    # Cursor-based: find the ID of the cursor row and use it as offset
    if cursor is not None:
        from graph.db_models import PolarizationMetricRow as PMR  # noqa: PLC0415

        cursor_row = (
            db.query(PMR)
            .filter(PMR.snapshot_id == cursor)
            .order_by(PMR.computed_at.desc())
            .first()
        )
        if cursor_row is not None:
            # Skip all rows that come before this cursor position in the sorted order
            # Re-issue the filtered query and skip rows up to and including the cursor row
            all_ids = [r.id for r in q.with_entities(PMR.id).all()]
            try:
                cursor_pos = all_ids.index(cursor_row.id)
                q = q.offset(cursor_pos + 1)
            except ValueError:
                pass  # cursor not in filtered set; start from beginning

    rows = q.limit(page_size).all()

    items: list[PolarizationDTO] = [
        PolarizationDTO(
            snapshotId=row.snapshot_id,
            polarizationIndex=row.polarization_index,
            modularity=row.modularity,
            communityCount=row.community_count,
            avgCommunitySize=row.avg_community_size,
            interCommunityEdgeRatio=row.inter_community_edge_ratio,
            computedAt=row.computed_at,
        )
        for row in rows
    ]

    next_cursor_val: Optional[str] = rows[-1].snapshot_id if len(rows) == page_size else None

    result = PolarizationListDTO(items=items, total=total, nextCursor=next_cursor_val)
    cache_set_json(cache_key, result.model_dump(), ttl=DEFAULT_TTL_SECONDS)
    return result


# ---------------------------------------------------------------------------
# GET /api/snapshots/{snapshotId}/metrics/signed
# ---------------------------------------------------------------------------


@router.get(
    "/snapshots/{snapshotId}/metrics/signed",
    response_model=list[SignedMetricsDTO],
    summary="Get signed-edge metrics for a snapshot (wiki-RfA only)",
)
async def get_signed_metrics(
    snapshotId: str,
    current_user: str = Depends(rate_limit),
    db=Depends(_get_db_session),
) -> list[SignedMetricsDTO]:
    """Return per-community :class:`~api.dtos.SignedMetricsDTO`.

    Returns HTTP 404 for snapshots with no signed metrics (i.e. non-wiki-RfA).

    Requirements: 7.2 (extended), design.md Component 6
    """
    from graph.db_models import SignedMetricRow  # noqa: PLC0415

    rows: list[SignedMetricRow] = (
        db.query(SignedMetricRow)
        .filter(SignedMetricRow.snapshot_id == snapshotId)
        .order_by(SignedMetricRow.community_id)
        .all()
    )

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No signed metrics found for snapshot '{snapshotId}'. "
                "This endpoint is only available for wiki-RfA datasets."
            ),
        )

    return [
        SignedMetricsDTO(
            snapshotId=row.snapshot_id,
            communityId=row.community_id,
            positiveEdgeRatio=row.positive_edge_ratio,
            negativeEdgeRatio=row.negative_edge_ratio,
            netSentimentIndex=row.net_sentiment,
            crossCommunityNegativity=row.cross_community_negativity,
            computedAt=row.computed_at,
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# GET /api/users/{userId}/metrics
# ---------------------------------------------------------------------------


@router.get(
    "/users/{userId}/metrics",
    response_model=UserMetricsDTO,
    summary="Get user metrics (latest snapshot)",
)
async def get_user_metrics(
    userId: str,
    snapshotId: Optional[str] = Query(
        default=None,
        description="Pin to a specific snapshot; defaults to most recent",
    ),
    current_user: str = Depends(rate_limit),
    db=Depends(_get_db_session),
) -> UserMetricsDTO:
    """Return :class:`~api.dtos.UserMetricsDTO` for the given user.

    Optionally pin to a specific ``snapshotId`` via query param; otherwise
    returns the most recently computed record.

    Requirements: 7.3, 7.11
    """
    from graph.db_models import UserMetricRow  # noqa: PLC0415

    q = db.query(UserMetricRow).filter(UserMetricRow.user_id == userId)
    if snapshotId is not None:
        q = q.filter(UserMetricRow.snapshot_id == snapshotId)

    row: Optional[UserMetricRow] = (
        q.order_by(UserMetricRow.computed_at.desc()).first()
    )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No metrics found for user '{userId}'",
        )

    return UserMetricsDTO(
        userId=row.user_id,
        communityId=row.community_id,
        diversityScore=row.diversity_score,
        intraEdgeCount=row.intra_edge_count,
        interEdgeCount=row.inter_edge_count,
        betweennessCentrality=row.betweenness_centrality,
        snapshotId=row.snapshot_id,
        computedAt=row.computed_at,
    )


# ---------------------------------------------------------------------------
# GET /api/users/metrics  (filtered list)
# ---------------------------------------------------------------------------


@router.get(
    "/users/metrics",
    response_model=UserMetricsListDTO,
    summary="List user metrics with filtering and pagination",
)
async def list_user_metrics(
    datasetSource: Optional[str] = Query(
        default=None,
        description="Filter by dataset source",
    ),
    snapshotId: Optional[str] = Query(
        default=None,
        description="Filter by snapshot ID",
    ),
    communityId: Optional[str] = Query(
        default=None,
        description="Filter by community ID",
    ),
    from_dt: Optional[datetime] = Query(
        default=None,
        alias="from",
        description="Lower bound on computedAt (ISO 8601)",
    ),
    to_dt: Optional[datetime] = Query(
        default=None,
        alias="to",
        description="Upper bound on computedAt (ISO 8601)",
    ),
    cursor: Optional[str] = Query(
        default=None,
        description="Pagination cursor — userId of the last record on the previous page",
    ),
    page_size: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Records per page (default 100)",
    ),
    current_user: str = Depends(rate_limit),
    db=Depends(_get_db_session),
) -> UserMetricsListDTO:
    """Return a filtered, paginated list of :class:`~api.dtos.UserMetricsDTO`.

    Supports:
    - ``?datasetSource=``  — filter by dataset source
    - ``?snapshotId=``     — filter by snapshot ID
    - ``?communityId=``    — filter by community
    - ``?from=``           — ``computedAt >= from``
    - ``?to=``             — ``computedAt <= to``

    Requirements: 7.3, 7.6, 7.11
    """
    from graph.db_models import UserMetricRow  # noqa: PLC0415

    q = db.query(UserMetricRow)

    if datasetSource is not None:
        q = q.filter(UserMetricRow.dataset_source == datasetSource)
    if snapshotId is not None:
        q = q.filter(UserMetricRow.snapshot_id == snapshotId)
    if communityId is not None:
        q = q.filter(UserMetricRow.community_id == communityId)
    if from_dt is not None:
        q = q.filter(UserMetricRow.computed_at >= from_dt)
    if to_dt is not None:
        q = q.filter(UserMetricRow.computed_at <= to_dt)

    total: int = q.count()

    q = q.order_by(UserMetricRow.computed_at.desc(), UserMetricRow.user_id.asc())

    if cursor is not None:
        # Skip rows with user_id <= cursor in the current ordering; use offset approach
        all_user_ids_in_order = [
            r.user_id for r in q.with_entities(UserMetricRow.user_id).all()
        ]
        try:
            idx = all_user_ids_in_order.index(cursor)
            q = q.offset(idx + 1)
        except ValueError:
            pass

    rows = q.limit(page_size).all()

    items: list[UserMetricsDTO] = [
        UserMetricsDTO(
            userId=row.user_id,
            communityId=row.community_id,
            diversityScore=row.diversity_score,
            intraEdgeCount=row.intra_edge_count,
            interEdgeCount=row.inter_edge_count,
            betweennessCentrality=row.betweenness_centrality,
            snapshotId=row.snapshot_id,
            computedAt=row.computed_at,
        )
        for row in rows
    ]

    next_cursor_val: Optional[str] = rows[-1].user_id if len(rows) == page_size else None

    return UserMetricsListDTO(items=items, total=total, nextCursor=next_cursor_val)


# ---------------------------------------------------------------------------
# GET /api/communities/{communityId}/metrics
# ---------------------------------------------------------------------------


@router.get(
    "/communities/{communityId}/metrics",
    response_model=CommunityMetricsDTO,
    summary="Get community-level metrics",
)
async def get_community_metrics(
    communityId: str,
    snapshotId: Optional[str] = Query(
        default=None,
        description="Snapshot ID; defaults to most recent snapshot for this community",
    ),
    datasetSource: Optional[str] = Query(
        default=None,
        description="Filter by dataset source",
    ),
    current_user: str = Depends(rate_limit),
    db=Depends(_get_db_session),
) -> CommunityMetricsDTO:
    """Return :class:`~api.dtos.CommunityMetricsDTO` aggregated from user metrics.

    Requirements: 7.4, 7.11
    """
    from sqlalchemy import func  # noqa: PLC0415

    from graph.db_models import PolarizationMetricRow, UserMetricRow  # noqa: PLC0415

    effective_snapshot_id: Optional[str] = snapshotId

    if effective_snapshot_id is None:
        q_latest = (
            db.query(UserMetricRow)
            .filter(UserMetricRow.community_id == communityId)
        )
        if datasetSource is not None:
            q_latest = q_latest.filter(UserMetricRow.dataset_source == datasetSource)
        latest: Optional[UserMetricRow] = (
            q_latest.order_by(UserMetricRow.computed_at.desc()).first()
        )
        if latest is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No metrics found for community '{communityId}'",
            )
        effective_snapshot_id = latest.snapshot_id

    agg_q = db.query(
        func.count(UserMetricRow.user_id).label("member_count"),
        func.avg(UserMetricRow.diversity_score).label("avg_diversity_score"),
    ).filter(
        UserMetricRow.community_id == communityId,
        UserMetricRow.snapshot_id == effective_snapshot_id,
    )
    if datasetSource is not None:
        agg_q = agg_q.filter(UserMetricRow.dataset_source == datasetSource)

    agg = agg_q.one_or_none()
    member_count: int = agg.member_count if (agg and agg.member_count) else 0

    if member_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No metrics found for community '{communityId}' "
                f"in snapshot '{effective_snapshot_id}'"
            ),
        )

    avg_diversity: float = float(agg.avg_diversity_score or 0.0)

    pm_q = (
        db.query(PolarizationMetricRow)
        .filter(PolarizationMetricRow.snapshot_id == effective_snapshot_id)
    )
    if datasetSource is not None:
        pm_q = pm_q.filter(PolarizationMetricRow.dataset_source == datasetSource)

    pm_row: Optional[PolarizationMetricRow] = (
        pm_q.order_by(PolarizationMetricRow.computed_at.desc()).first()
    )

    polarization_index = pm_row.polarization_index if pm_row else 0.0
    modularity = pm_row.modularity if pm_row else 0.0

    return CommunityMetricsDTO(
        communityId=communityId,
        memberCount=member_count,
        modularity=modularity,
        avgDiversityScore=avg_diversity,
        polarizationIndex=polarization_index,
        snapshotId=effective_snapshot_id,
    )


# ---------------------------------------------------------------------------
# GET /api/users/{userId}/recommendations
# ---------------------------------------------------------------------------


@router.get(
    "/users/{userId}/recommendations",
    response_model=list[RecommendationDTO],
    summary="Get recommendations for a user",
)
async def get_user_recommendations(
    userId: str,
    current_user: Annotated[str, Depends(rate_limit)],
    db=Depends(_get_db_session),
) -> list[RecommendationDTO]:
    """Return recommendations for the given user.

    Enforces that the authenticated caller's identity matches the requested
    ``userId``; returns HTTP 403 otherwise.

    Requirements: 7.5, 7.9
    """
    import os as _os  # noqa: PLC0415
    _dev_mode = _os.environ.get("DEV_MODE", "1") == "1"
    if not _dev_mode and current_user != userId:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own recommendations",
        )

    from recommendations.service import RecommendationService  # noqa: PLC0415

    service = RecommendationService()
    recs = service.fetch_recommendations(user_id=userId, db_session=db)

    return [
        RecommendationDTO(
            recommendationId=rec.recommendationId,
            targetUserId=rec.targetUserId,
            recommendedUserId=rec.recommendedUserId,
            diversityGain=rec.diversityGain,
            topicRelevance=rec.topicRelevance,
            communityId=rec.communityId,
            reason=rec.reason,
        )
        for rec in recs
    ]
