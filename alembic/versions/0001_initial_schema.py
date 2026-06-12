"""Initial schema: snapshots, polarization_metrics, user_metrics, recommendations, signed_metrics.

Revision ID: 0001
Revises: None
Create Date: 2024-01-01 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # snapshots — anchor table; all others FK to this
    # ------------------------------------------------------------------
    op.create_table(
        "snapshots",
        sa.Column("snapshot_id", sa.VARCHAR(36), nullable=False),
        sa.Column("dataset_source", sa.VARCHAR(64), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False),
        sa.Column("node_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("edge_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("graphml_path", sa.TEXT(), nullable=True),
        sa.PrimaryKeyConstraint("snapshot_id"),
    )
    op.create_index("ix_snapshots_dataset_source", "snapshots", ["dataset_source"])

    # ------------------------------------------------------------------
    # polarization_metrics
    # ------------------------------------------------------------------
    op.create_table(
        "polarization_metrics",
        sa.Column(
            "id",
            sa.Integer(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("snapshot_id", sa.VARCHAR(36), nullable=False),
        sa.Column("dataset_source", sa.VARCHAR(64), nullable=False),
        sa.Column("polarization_index", sa.DOUBLE_PRECISION(), nullable=False),
        sa.Column("modularity", sa.DOUBLE_PRECISION(), nullable=False),
        sa.Column("community_count", sa.Integer(), nullable=False),
        sa.Column("avg_community_size", sa.DOUBLE_PRECISION(), nullable=False),
        sa.Column("inter_community_edge_ratio", sa.DOUBLE_PRECISION(), nullable=False),
        sa.Column("computed_at", sa.TIMESTAMP(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["snapshots.snapshot_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_polarization_metrics_snapshot_source",
        "polarization_metrics",
        ["snapshot_id", "dataset_source"],
    )

    # ------------------------------------------------------------------
    # user_metrics
    # ------------------------------------------------------------------
    op.create_table(
        "user_metrics",
        sa.Column(
            "id",
            sa.Integer(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("snapshot_id", sa.VARCHAR(36), nullable=False),
        sa.Column("dataset_source", sa.VARCHAR(64), nullable=False),
        sa.Column("user_id", sa.VARCHAR(256), nullable=False),
        sa.Column("community_id", sa.VARCHAR(64), nullable=False),
        sa.Column("diversity_score", sa.DOUBLE_PRECISION(), nullable=False),
        sa.Column("intra_edge_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("inter_edge_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("betweenness_centrality", sa.DOUBLE_PRECISION(), nullable=False),
        sa.Column("computed_at", sa.TIMESTAMP(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["snapshots.snapshot_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_user_metrics_snapshot_user",
        "user_metrics",
        ["snapshot_id", "user_id"],
    )
    op.create_index(
        "ix_user_metrics_snapshot_source",
        "user_metrics",
        ["snapshot_id", "dataset_source"],
    )

    # ------------------------------------------------------------------
    # recommendations
    # ------------------------------------------------------------------
    op.create_table(
        "recommendations",
        sa.Column("recommendation_id", sa.VARCHAR(36), nullable=False),
        sa.Column("snapshot_id", sa.VARCHAR(36), nullable=False),
        sa.Column("dataset_source", sa.VARCHAR(64), nullable=False),
        sa.Column("target_user_id", sa.VARCHAR(256), nullable=False),
        sa.Column("recommended_user_id", sa.VARCHAR(256), nullable=False),
        sa.Column("diversity_gain", sa.DOUBLE_PRECISION(), nullable=False),
        sa.Column("topic_relevance", sa.DOUBLE_PRECISION(), nullable=False),
        sa.Column("community_id", sa.VARCHAR(64), nullable=False),
        sa.Column("reason", sa.TEXT(), nullable=False),
        sa.Column("content_id", sa.TEXT(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(),
            nullable=False,
            server_default="NOW()",
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["snapshots.snapshot_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("recommendation_id"),
    )
    op.create_index(
        "ix_recommendations_target_snapshot",
        "recommendations",
        ["target_user_id", "snapshot_id"],
    )
    op.create_index(
        "ix_recommendations_snapshot_source",
        "recommendations",
        ["snapshot_id", "dataset_source"],
    )

    # ------------------------------------------------------------------
    # signed_metrics  (wiki-RfA specific)
    # ------------------------------------------------------------------
    op.create_table(
        "signed_metrics",
        sa.Column(
            "id",
            sa.Integer(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("snapshot_id", sa.VARCHAR(36), nullable=False),
        sa.Column(
            "dataset_source",
            sa.VARCHAR(64),
            nullable=False,
            server_default="wiki_rfa",
        ),
        sa.Column("community_id", sa.VARCHAR(64), nullable=False),
        sa.Column("positive_edge_ratio", sa.DOUBLE_PRECISION(), nullable=False),
        sa.Column("negative_edge_ratio", sa.DOUBLE_PRECISION(), nullable=False),
        sa.Column("net_sentiment", sa.DOUBLE_PRECISION(), nullable=False),
        sa.Column(
            "cross_community_negativity",
            sa.DOUBLE_PRECISION(),
            nullable=False,
            server_default="0.0",
        ),
        sa.Column("computed_at", sa.TIMESTAMP(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["snapshots.snapshot_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "snapshot_id",
            "community_id",
            name="uq_signed_metrics_snapshot_community",
        ),
    )
    op.create_index(
        "ix_signed_metrics_snapshot_community",
        "signed_metrics",
        ["snapshot_id", "community_id"],
    )


def downgrade() -> None:
    # Drop in reverse creation order (signed_metrics → recommendations →
    # user_metrics → polarization_metrics → snapshots)
    op.drop_index("ix_signed_metrics_snapshot_community", table_name="signed_metrics")
    op.drop_table("signed_metrics")

    op.drop_index("ix_recommendations_snapshot_source", table_name="recommendations")
    op.drop_index("ix_recommendations_target_snapshot", table_name="recommendations")
    op.drop_table("recommendations")

    op.drop_index("ix_user_metrics_snapshot_source", table_name="user_metrics")
    op.drop_index("ix_user_metrics_snapshot_user", table_name="user_metrics")
    op.drop_table("user_metrics")

    op.drop_index(
        "ix_polarization_metrics_snapshot_source",
        table_name="polarization_metrics",
    )
    op.drop_table("polarization_metrics")

    op.drop_index("ix_snapshots_dataset_source", table_name="snapshots")
    op.drop_table("snapshots")
