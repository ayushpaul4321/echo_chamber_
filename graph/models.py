"""Core data models for the Echo Chamber Detector pipeline.

Supports all four datasets:
- Reddit Title (soc-redditHyperlinks-title.tsv)
- Reddit Body (soc-redditHyperlinks-body.tsv)
- Congress Network (congress.edgelist + congress_network_data.json)
- Wiki-RfA (wiki-RfA.txt.gz)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class InteractionType(Enum):
    """Canonical interaction types supported across all four datasets."""

    HYPERLINK = "HYPERLINK"   # Reddit subreddit-to-subreddit hyperlink
    RETWEET = "RETWEET"       # Congress Twitter retweet / influence edge
    VOTE = "VOTE"             # Wiki-RfA adminship vote


# ---------------------------------------------------------------------------
# InteractionRecord
# ---------------------------------------------------------------------------


@dataclass
class InteractionRecord:
    """Normalized, deduplicated record of a single user-to-user interaction.

    Covers all four dataset schemas:
    - Reddit Title/Body: HYPERLINK with sentimentScore and optional bodyText
    - Congress Network: RETWEET with no timestamp; pre-normalized weight
    - Wiki-RfA: VOTE with votePolarity (+1/-1), voteResult (0/1), bodyText
    """

    id: str                              # UUID
    sourceUserId: str
    targetUserId: str
    interactionType: InteractionType
    datasetSource: str
    timestamp: Optional[datetime] = None       # Absent in Congress dataset
    contentId: Optional[str] = None
    topicTags: list[str] = field(default_factory=list)
    sentimentScore: Optional[float] = None     # LINK_SENTIMENT (Reddit) or VOT float (wiki-RfA)
    votePolarity: Optional[int] = None         # wiki-RfA only: +1 or -1
    bodyText: Optional[str] = None             # wiki-RfA TXT field; Reddit body PROPERTIES
    voteResult: Optional[int] = None           # wiki-RfA RES field: 0 or 1

    def __post_init__(self) -> None:
        # --- userId validations ---
        if not self.sourceUserId:
            raise ValueError("sourceUserId must be non-empty")
        if not self.targetUserId:
            raise ValueError("targetUserId must be non-empty")
        if self.sourceUserId == self.targetUserId:
            raise ValueError(
                f"sourceUserId must not equal targetUserId (got '{self.sourceUserId}')"
            )

        # --- timestamp must be in the past if present ---
        if self.timestamp is not None:
            # Make comparison timezone-aware
            now = datetime.now(timezone.utc)
            ts = self.timestamp
            if ts.tzinfo is None:
                # Treat naive datetimes as UTC for comparison
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= now:
                raise ValueError(
                    f"timestamp must be a past datetime (got {self.timestamp!r})"
                )

        # --- votePolarity must be +1 or -1 if present ---
        if self.votePolarity is not None and self.votePolarity not in (1, -1):
            raise ValueError(
                f"votePolarity must be +1 or -1 if present (got {self.votePolarity!r})"
            )

        # --- voteResult must be 0 or 1 if present ---
        if self.voteResult is not None and self.voteResult not in (0, 1):
            raise ValueError(
                f"voteResult must be 0 or 1 if present (got {self.voteResult!r})"
            )


# ---------------------------------------------------------------------------
# Graph structural models
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """A user node in the InteractionGraph.

    communityId is None before community detection runs.
    """

    userId: str
    communityId: Optional[str] = None          # None before detection
    betweenness: float = 0.0
    diversityScore: float = 0.0
    topicVector: list[float] = field(default_factory=list)


@dataclass
class Edge:
    """A directed, weighted edge between two user nodes.

    signedPolarity is wiki-RfA specific (+1 for support vote, -1 for oppose).
    """

    sourceUserId: str
    targetUserId: str
    weight: float
    isCrossCommunity: bool = False
    signedPolarity: Optional[int] = None       # wiki-RfA only: +1 or -1

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(
                f"Edge weight must be >= 0 (got {self.weight!r})"
            )


@dataclass
class InteractionGraph:
    """Weighted directed graph where nodes are users and edges represent interactions.

    snapshotId and createdAt identify a versioned, persisted graph state.

    rawEdgeCounts is an optional dict mapping (sourceUserId, targetUserId) → raw
    interaction count (integer for Reddit-style count-aggregated graphs, or float
    for pre-normalized graphs).  It is populated by GraphConstructionService so
    that updateGraph can merge incremental records without reverse-normalizing.
    For pre-normalized datasets (congress, wiki_rfa) it is not needed, but may
    still be stored for uniformity.
    """

    nodes: dict[str, Node]
    edges: list[Edge]
    snapshotId: str
    createdAt: datetime
    datasetSource: str = ""
    rawEdgeCounts: Optional[dict[tuple[str, str], float]] = field(
        default=None, repr=False
    )

    @property
    def nodeCount(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    @property
    def edgeCount(self) -> int:
        """Number of edges in the graph."""
        return len(self.edges)


# ---------------------------------------------------------------------------
# Community and metrics models
# ---------------------------------------------------------------------------


@dataclass
class CommunityPartition:
    """Assignment of user nodes to a single community after detection."""

    communityId: str
    memberIds: set[str]
    modularity: float
    intraEdges: int
    interEdges: int
    centroidNode: Optional[str] = None         # Highest-degree node (hub)
    isApproximate: bool = False                # True if iteration cap was hit
    girvan_newman_partition: Optional[list[set[str]]] = None  # Secondary validation partition (list of community member sets)


@dataclass
class PolarizationMetrics:
    """Graph-level polarization metrics persisted per snapshot."""

    snapshotId: str
    polarizationIndex: float                   # [0, 1]; 1 = fully polarized
    modularity: float                          # Graph modularity Q
    communityCount: int
    avgCommunitySize: float
    interCommunityEdgeRatio: float             # inter / (intra + inter)
    computedAt: datetime
    datasetSource: str = ""


@dataclass
class UserMetrics:
    """Per-user diversity and centrality metrics tied to a specific snapshot."""

    userId: str
    communityId: str
    diversityScore: float                      # [0, 1]; 1 = fully diverse
    intraEdgeCount: int
    interEdgeCount: int
    betweennessCentrality: float
    snapshotId: str
    computedAt: datetime


@dataclass
class SignedMetrics:
    """Per-community signed-edge sentiment metrics for wiki-RfA graphs."""

    snapshotId: str
    communityId: str
    positiveEdgeRatio: float          # [0, 1]
    negativeEdgeRatio: float          # [0, 1]; = 1 - positiveEdgeRatio
    netSentimentIndex: float          # mean votePolarity of intra-community edges
    crossCommunityNegativity: float   # ratio of negative cross-community edges / total negative edges
    computedAt: datetime
    datasetSource: str = "wiki_rfa"


@dataclass
class Recommendation:
    """A cross-community content recommendation for a low-diversity user."""

    recommendationId: str
    targetUserId: str
    recommendedUserId: str
    diversityGain: float
    topicRelevance: float                      # [0, 1] topic overlap score
    communityId: str                           # Community the rec comes from
    reason: str                                # Human-readable explanation
    contentId: Optional[str] = None            # Specific post/thread (optional)
