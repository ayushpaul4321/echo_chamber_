"""Tests for graph/db_models.py and graph/redis_keys.py.

Validates:
- All ORM models import without error
- Each table exposes the expected columns via SQLAlchemy metadata (no live DB required)
- Redis key helper functions return correctly formatted strings
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# ORM import smoke-test
# ---------------------------------------------------------------------------


def test_orm_imports_without_error() -> None:
    """All five ORM classes and the shared metadata must be importable."""
    from graph.db_models import (  # noqa: F401
        Base,
        PolarizationMetricRow,
        RecommendationRow,
        SignedMetricRow,
        Snapshot,
        UserMetricRow,
        metadata,
    )


# ---------------------------------------------------------------------------
# Helper: fetch column names for a mapped class
# ---------------------------------------------------------------------------


def _column_names(mapped_class) -> set[str]:
    """Return the set of column names registered in the table's metadata."""
    return {col.name for col in mapped_class.__table__.columns}


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------


def test_snapshots_has_expected_columns() -> None:
    from graph.db_models import Snapshot

    cols = _column_names(Snapshot)
    assert "snapshot_id" in cols
    assert "dataset_source" in cols
    assert "created_at" in cols
    assert "node_count" in cols
    assert "edge_count" in cols
    assert "graphml_path" in cols


def test_snapshots_primary_key_is_snapshot_id() -> None:
    from graph.db_models import Snapshot

    pk_cols = {col.name for col in Snapshot.__table__.primary_key}
    assert "snapshot_id" in pk_cols


# ---------------------------------------------------------------------------
# polarization_metrics
# ---------------------------------------------------------------------------


def test_polarization_metrics_has_expected_columns() -> None:
    from graph.db_models import PolarizationMetricRow

    cols = _column_names(PolarizationMetricRow)
    assert "polarization_index" in cols
    assert "inter_community_edge_ratio" in cols
    assert "dataset_source" in cols
    assert "snapshot_id" in cols
    assert "modularity" in cols
    assert "community_count" in cols
    assert "avg_community_size" in cols
    assert "computed_at" in cols


def test_polarization_metrics_fk_to_snapshots() -> None:
    from graph.db_models import PolarizationMetricRow

    fk_targets = {
        fk.target_fullname
        for fk in PolarizationMetricRow.__table__.foreign_keys
    }
    assert "snapshots.snapshot_id" in fk_targets


# ---------------------------------------------------------------------------
# user_metrics
# ---------------------------------------------------------------------------


def test_user_metrics_has_expected_columns() -> None:
    from graph.db_models import UserMetricRow

    cols = _column_names(UserMetricRow)
    assert "diversity_score" in cols
    assert "betweenness_centrality" in cols
    assert "user_id" in cols
    assert "snapshot_id" in cols
    assert "dataset_source" in cols
    assert "community_id" in cols
    assert "intra_edge_count" in cols
    assert "inter_edge_count" in cols
    assert "computed_at" in cols


def test_user_metrics_fk_to_snapshots() -> None:
    from graph.db_models import UserMetricRow

    fk_targets = {fk.target_fullname for fk in UserMetricRow.__table__.foreign_keys}
    assert "snapshots.snapshot_id" in fk_targets


# ---------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------


def test_recommendations_has_expected_columns() -> None:
    from graph.db_models import RecommendationRow

    cols = _column_names(RecommendationRow)
    assert "recommendation_id" in cols
    assert "target_user_id" in cols
    assert "diversity_gain" in cols
    assert "topic_relevance" in cols
    assert "snapshot_id" in cols
    assert "dataset_source" in cols
    assert "recommended_user_id" in cols
    assert "community_id" in cols
    assert "reason" in cols
    assert "content_id" in cols
    assert "created_at" in cols


def test_recommendations_primary_key_is_recommendation_id() -> None:
    from graph.db_models import RecommendationRow

    pk_cols = {col.name for col in RecommendationRow.__table__.primary_key}
    assert "recommendation_id" in pk_cols


def test_recommendations_fk_to_snapshots() -> None:
    from graph.db_models import RecommendationRow

    fk_targets = {fk.target_fullname for fk in RecommendationRow.__table__.foreign_keys}
    assert "snapshots.snapshot_id" in fk_targets


# ---------------------------------------------------------------------------
# signed_metrics
# ---------------------------------------------------------------------------


def test_signed_metrics_has_expected_columns() -> None:
    from graph.db_models import SignedMetricRow

    cols = _column_names(SignedMetricRow)
    assert "positive_edge_ratio" in cols
    assert "negative_edge_ratio" in cols
    assert "net_sentiment" in cols
    assert "cross_community_negativity" in cols
    assert "community_id" in cols
    assert "snapshot_id" in cols
    assert "dataset_source" in cols
    assert "computed_at" in cols


def test_signed_metrics_fk_to_snapshots() -> None:
    from graph.db_models import SignedMetricRow

    fk_targets = {fk.target_fullname for fk in SignedMetricRow.__table__.foreign_keys}
    assert "snapshots.snapshot_id" in fk_targets


def test_signed_metrics_unique_constraint() -> None:
    """One row per (snapshot_id, community_id) — enforced by UniqueConstraint."""
    from graph.db_models import SignedMetricRow

    constraint_names = {
        c.name
        for c in SignedMetricRow.__table__.constraints
        if hasattr(c, "columns")
    }
    assert "uq_signed_metrics_snapshot_community" in constraint_names


# ---------------------------------------------------------------------------
# metadata object
# ---------------------------------------------------------------------------


def test_metadata_contains_all_five_tables() -> None:
    from graph.db_models import metadata

    table_names = set(metadata.tables.keys())
    assert "snapshots" in table_names
    assert "polarization_metrics" in table_names
    assert "user_metrics" in table_names
    assert "recommendations" in table_names
    assert "signed_metrics" in table_names


# ---------------------------------------------------------------------------
# Redis key helpers
# ---------------------------------------------------------------------------


def test_polarization_key_format() -> None:
    from graph.redis_keys import polarization_key

    assert polarization_key("snap-123") == "metrics:snap-123:polarization"


def test_recommendations_key_format() -> None:
    from graph.redis_keys import recommendations_key

    assert recommendations_key("user_42") == "user:user_42:recommendations"


def test_graph_page_key_format() -> None:
    from graph.redis_keys import graph_page_key

    assert graph_page_key("snap-123", 3) == "graph:snap-123:page:3"


def test_graph_page_key_page_zero() -> None:
    from graph.redis_keys import graph_page_key

    assert graph_page_key("snap-001", 0) == "graph:snap-001:page:0"


def test_polarization_key_with_uuid() -> None:
    from graph.redis_keys import polarization_key

    snap = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    assert polarization_key(snap) == f"metrics:{snap}:polarization"


def test_recommendations_key_special_chars() -> None:
    """user_id may contain colons or slashes — key must embed them as-is."""
    from graph.redis_keys import recommendations_key

    assert recommendations_key("u:99") == "user:u:99:recommendations"


def test_default_ttl_constant() -> None:
    from graph.redis_keys import DEFAULT_TTL_SECONDS

    assert DEFAULT_TTL_SECONDS == 86_400
