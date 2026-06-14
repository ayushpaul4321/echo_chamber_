"""Data Transfer Objects (DTOs) for the Echo Chamber Detector REST API.

Pydantic response models serialized by FastAPI endpoints.

References: design.md Component 6, Requirements 7.1–7.5, 7.6, 7.7, 7.10, 7.11
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Graph DTOs
# ---------------------------------------------------------------------------


class NodeDTO(BaseModel):
    """Serialized representation of a graph Node."""

    userId: str
    communityId: Optional[str] = None
    betweenness: float = 0.0
    diversityScore: float = 0.0
    topicVector: list[float] = Field(default_factory=list)


class EdgeDTO(BaseModel):
    """Serialized representation of a graph Edge."""

    sourceUserId: str
    targetUserId: str
    weight: float
    isCrossCommunity: bool = False
    signedPolarity: Optional[int] = None


class GraphDTO(BaseModel):
    """Paginated graph snapshot response.

    ``nextCursor`` is set when there are more pages; ``None`` means last page.
    """

    nodes: list[NodeDTO]
    edges: list[EdgeDTO]
    snapshotId: str
    createdAt: datetime
    nodeCount: int
    edgeCount: int
    nextCursor: Optional[str] = None


# ---------------------------------------------------------------------------
# Metrics DTOs
# ---------------------------------------------------------------------------


class PolarizationDTO(BaseModel):
    """Graph-level polarization metrics for a snapshot."""

    snapshotId: str
    polarizationIndex: float
    modularity: float
    communityCount: int
    avgCommunitySize: float
    interCommunityEdgeRatio: float
    computedAt: datetime


class SignedMetricsDTO(BaseModel):
    """Per-community signed-edge sentiment metrics (wiki-RfA only)."""

    snapshotId: str
    communityId: str
    positiveEdgeRatio: float
    negativeEdgeRatio: float
    netSentimentIndex: float
    crossCommunityNegativity: float
    computedAt: datetime


class UserMetricsDTO(BaseModel):
    """Per-user diversity and centrality metrics."""

    userId: str
    communityId: str
    diversityScore: float
    intraEdgeCount: int
    interEdgeCount: int
    betweennessCentrality: float
    snapshotId: str
    computedAt: datetime


class CommunityMetricsDTO(BaseModel):
    """Aggregated community-level metrics."""

    communityId: str
    memberCount: int
    modularity: float
    avgDiversityScore: float
    polarizationIndex: float
    snapshotId: str


# ---------------------------------------------------------------------------
# Filtered / paginated list DTOs
# ---------------------------------------------------------------------------


class PolarizationListDTO(BaseModel):
    """Paginated list of polarization metric records matching a filter query.

    Used by ``GET /api/metrics/polarization`` with filter parameters.
    ``nextCursor`` is the ``snapshotId`` of the last record in the current
    page; ``None`` means this is the final page.
    """

    items: list[PolarizationDTO]
    total: int
    nextCursor: Optional[str] = None


class UserMetricsListDTO(BaseModel):
    """Paginated list of user metric records matching a filter query."""

    items: list[UserMetricsDTO]
    total: int
    nextCursor: Optional[str] = None


class LatestSnapshotDTO(BaseModel):
    """Lightweight descriptor of the most recent snapshot for a dataset source.

    Used by the Dashboard's live-refresh polling endpoint (Requirement 8.6).
    """

    snapshotId: str
    datasetSource: str
    computedAt: datetime


# ---------------------------------------------------------------------------
# Recommendation DTO
# ---------------------------------------------------------------------------


class RecommendationDTO(BaseModel):
    """Cross-community recommendation for a user."""

    recommendationId: str
    targetUserId: str
    recommendedUserId: str
    diversityGain: float
    topicRelevance: float
    communityId: str
    reason: str
