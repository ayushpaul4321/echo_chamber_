"""SQLAlchemy 2.0 ORM table definitions for the Echo Chamber Detector.

These mapped classes define the PostgreSQL schema.  Schema migrations are
managed exclusively via Alembic — do NOT call ``Base.metadata.create_all()``
in application code.

Tables
------
snapshots             — versioned graph snapshots (anchor table)
polarization_metrics  — graph-level polarization metrics per snapshot
user_metrics          — per-user diversity / centrality metrics per snapshot
recommendations       — cross-community recommendations per snapshot
signed_metrics        — wiki-RfA signed-edge sentiment metrics per snapshot
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DOUBLE_PRECISION,
    TEXT,
    TIMESTAMP,
    VARCHAR,
    Column,
    ForeignKey,
    Index,
    Integer,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base; its metadata is imported by Alembic env.py."""

    pass


metadata = Base.metadata


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------


class Snapshot(Base):
    """Versioned graph snapshot — anchor table for all FK references."""

    __tablename__ = "snapshots"

    snapshot_id: str = Column(VARCHAR(36), primary_key=True)
    dataset_source: str = Column(VARCHAR(64), nullable=False)
    created_at: datetime = Column(TIMESTAMP, nullable=False)
    node_count: int = Column(Integer, nullable=False, server_default="0")
    edge_count: int = Column(Integer, nullable=False, server_default="0")
    graphml_path: str | None = Column(TEXT, nullable=True)

    __table_args__ = (
        Index("ix_snapshots_dataset_source", "dataset_source"),
    )


# ---------------------------------------------------------------------------
# polarization_metrics
# ---------------------------------------------------------------------------


class PolarizationMetricRow(Base):
    """Graph-level polarization metrics record — one row per snapshot per run."""

    __tablename__ = "polarization_metrics"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: str = Column(
        VARCHAR(36),
        ForeignKey("snapshots.snapshot_id", ondelete="CASCADE"),
        nullable=False,
    )
    dataset_source: str = Column(VARCHAR(64), nullable=False)
    polarization_index: float = Column(DOUBLE_PRECISION, nullable=False)
    modularity: float = Column(DOUBLE_PRECISION, nullable=False)
    community_count: int = Column(Integer, nullable=False)
    avg_community_size: float = Column(DOUBLE_PRECISION, nullable=False)
    inter_community_edge_ratio: float = Column(DOUBLE_PRECISION, nullable=False)
    computed_at: datetime = Column(TIMESTAMP, nullable=False)

    __table_args__ = (
        Index("ix_polarization_metrics_snapshot_source", "snapshot_id", "dataset_source"),
    )


# ---------------------------------------------------------------------------
# user_metrics
# ---------------------------------------------------------------------------


class UserMetricRow(Base):
    """Per-user diversity and centrality metrics tied to a snapshot."""

    __tablename__ = "user_metrics"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: str = Column(
        VARCHAR(36),
        ForeignKey("snapshots.snapshot_id", ondelete="CASCADE"),
        nullable=False,
    )
    dataset_source: str = Column(VARCHAR(64), nullable=False)
    user_id: str = Column(VARCHAR(256), nullable=False)
    community_id: str = Column(VARCHAR(64), nullable=False)
    diversity_score: float = Column(DOUBLE_PRECISION, nullable=False)
    intra_edge_count: int = Column(Integer, nullable=False, server_default="0")
    inter_edge_count: int = Column(Integer, nullable=False, server_default="0")
    betweenness_centrality: float = Column(DOUBLE_PRECISION, nullable=False)
    computed_at: datetime = Column(TIMESTAMP, nullable=False)

    __table_args__ = (
        Index("ix_user_metrics_snapshot_user", "snapshot_id", "user_id"),
        Index("ix_user_metrics_snapshot_source", "snapshot_id", "dataset_source"),
    )


# ---------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------


class RecommendationRow(Base):
    """Cross-community recommendation persisted per snapshot."""

    __tablename__ = "recommendations"

    recommendation_id: str = Column(VARCHAR(36), primary_key=True)
    snapshot_id: str = Column(
        VARCHAR(36),
        ForeignKey("snapshots.snapshot_id", ondelete="CASCADE"),
        nullable=False,
    )
    dataset_source: str = Column(VARCHAR(64), nullable=False)
    target_user_id: str = Column(VARCHAR(256), nullable=False)
    recommended_user_id: str = Column(VARCHAR(256), nullable=False)
    diversity_gain: float = Column(DOUBLE_PRECISION, nullable=False)
    topic_relevance: float = Column(DOUBLE_PRECISION, nullable=False)
    community_id: str = Column(VARCHAR(64), nullable=False)
    reason: str = Column(TEXT, nullable=False)
    content_id: str | None = Column(TEXT, nullable=True)
    created_at: datetime = Column(
        TIMESTAMP, nullable=False, server_default="NOW()"
    )

    __table_args__ = (
        Index("ix_recommendations_target_snapshot", "target_user_id", "snapshot_id"),
        Index("ix_recommendations_snapshot_source", "snapshot_id", "dataset_source"),
    )


# ---------------------------------------------------------------------------
# signed_metrics  (wiki-RfA specific)
# ---------------------------------------------------------------------------


class SignedMetricRow(Base):
    """Per-community signed-edge sentiment metrics for wiki-RfA snapshots."""

    __tablename__ = "signed_metrics"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: str = Column(
        VARCHAR(36),
        ForeignKey("snapshots.snapshot_id", ondelete="CASCADE"),
        nullable=False,
    )
    dataset_source: str = Column(VARCHAR(64), nullable=False, server_default="wiki_rfa")
    community_id: str = Column(VARCHAR(64), nullable=False)
    # Fraction of positive edges within this community [0, 1]
    positive_edge_ratio: float = Column(DOUBLE_PRECISION, nullable=False)
    # Fraction of negative edges within this community (= 1 - positive_edge_ratio)
    negative_edge_ratio: float = Column(DOUBLE_PRECISION, nullable=False)
    # Mean votePolarity of intra-community edges
    net_sentiment: float = Column(DOUBLE_PRECISION, nullable=False)
    # Ratio of negative cross-community edges vs total negative edges
    cross_community_negativity: float = Column(
        DOUBLE_PRECISION, nullable=False, server_default="0.0"
    )
    computed_at: datetime = Column(TIMESTAMP, nullable=False)

    __table_args__ = (
        Index("ix_signed_metrics_snapshot_community", "snapshot_id", "community_id"),
        UniqueConstraint(
            "snapshot_id",
            "community_id",
            name="uq_signed_metrics_snapshot_community",
        ),
    )
