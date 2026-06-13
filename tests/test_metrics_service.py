"""Tests for metrics/service.py — MetricsService.compute_polarization_index.

Covers Requirements 4.1–4.5:
  4.1  polarizationIndex is in [0, 1]
  4.2  All-intra edges → polarizationIndex = 1.0
  4.3  All-inter edges → polarizationIndex = 0.0
  4.4  polarizationIndex + interCommunityEdgeRatio = 1.0
  4.5  totalEdgeWeight = 0 → polarizationIndex = 0.0, interCommunityEdgeRatio = 0.0

Also covers:
  - edge.isCrossCommunity side-effect
  - Both partition input forms (list[CommunityPartition] and dict[str, str])
  - Empty graph
  - Modularity and community statistics in output

References: design.md Algorithm 3, Requirements 4.1–4.5
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import pytest

from graph.models import (
    CommunityPartition,
    Edge,
    InteractionGraph,
    Node,
    PolarizationMetrics,
)
from metrics.service import MetricsService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAST = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _make_graph(
    edges: list[tuple[str, str, float]],
    *,
    dataset_source: str = "reddit_title",
    snapshot_id: Optional[str] = None,
) -> InteractionGraph:
    """Build an InteractionGraph from (src, tgt, weight) triples."""
    nodes: dict[str, Node] = {}
    edge_list: list[Edge] = []

    for src, tgt, weight in edges:
        if src not in nodes:
            nodes[src] = Node(userId=src)
        if tgt not in nodes:
            nodes[tgt] = Node(userId=tgt)
        edge_list.append(Edge(sourceUserId=src, targetUserId=tgt, weight=weight))

    return InteractionGraph(
        nodes=nodes,
        edges=edge_list,
        snapshotId=snapshot_id or str(uuid.uuid4()),
        createdAt=_PAST,
        datasetSource=dataset_source,
    )


def _empty_graph(snapshot_id: Optional[str] = None) -> InteractionGraph:
    """Build an empty InteractionGraph (no nodes, no edges)."""
    return InteractionGraph(
        nodes={},
        edges=[],
        snapshotId=snapshot_id or str(uuid.uuid4()),
        createdAt=_PAST,
        datasetSource="reddit_title",
    )


def _svc() -> MetricsService:
    return MetricsService()


# ---------------------------------------------------------------------------
# Requirement 4.2 — all-intra edges → PI = 1.0
# ---------------------------------------------------------------------------


class TestAllIntraCommunity:
    """When every edge is intra-community, PI must equal 1.0 (Req 4.2)."""

    def test_all_intra_pi_equals_one_dict_partition(self) -> None:
        """All edges intra-community → polarizationIndex = 1.0 (dict partition)."""
        graph = _make_graph([
            ("A", "B", 1.0),
            ("B", "C", 0.5),
            ("A", "C", 0.8),
        ])
        partition = {"A": "c1", "B": "c1", "C": "c1"}

        metrics = _svc().compute_polarization_index(graph, partition)

        assert metrics.polarizationIndex == pytest.approx(1.0), (
            f"Expected PI=1.0 for all-intra graph, got {metrics.polarizationIndex}"
        )

    def test_all_intra_pi_equals_one_list_partition(self) -> None:
        """All edges intra-community → polarizationIndex = 1.0 (list partition)."""
        graph = _make_graph([
            ("X", "Y", 0.7),
            ("Y", "Z", 0.3),
        ])
        partition = [
            CommunityPartition(
                communityId="0",
                memberIds={"X", "Y", "Z"},
                modularity=0.0,
                intraEdges=2,
                interEdges=0,
            )
        ]

        metrics = _svc().compute_polarization_index(graph, partition)

        assert metrics.polarizationIndex == pytest.approx(1.0)

    def test_all_intra_inter_ratio_is_zero(self) -> None:
        """All edges intra-community → interCommunityEdgeRatio = 0.0."""
        graph = _make_graph([("A", "B", 0.5)])
        partition = {"A": "comm", "B": "comm"}

        metrics = _svc().compute_polarization_index(graph, partition)

        assert metrics.interCommunityEdgeRatio == pytest.approx(0.0)

    def test_all_intra_isCrossCommunity_flag_false(self) -> None:
        """All edges intra-community → every edge.isCrossCommunity == False."""
        graph = _make_graph([
            ("A", "B", 1.0),
            ("B", "C", 1.0),
        ])
        partition = {"A": "c0", "B": "c0", "C": "c0"}

        _svc().compute_polarization_index(graph, partition)

        for edge in graph.edges:
            assert edge.isCrossCommunity is False, (
                f"Edge ({edge.sourceUserId}->{edge.targetUserId}) should be "
                "isCrossCommunity=False"
            )


# ---------------------------------------------------------------------------
# Requirement 4.3 — all-inter edges → PI = 0.0
# ---------------------------------------------------------------------------


class TestAllInterCommunity:
    """When every edge is inter-community, PI must equal 0.0 (Req 4.3)."""

    def test_all_inter_pi_equals_zero_dict_partition(self) -> None:
        """All edges inter-community → polarizationIndex = 0.0."""
        graph = _make_graph([
            ("A", "B", 1.0),
            ("C", "D", 0.5),
        ])
        # Each node in its own community
        partition = {"A": "c1", "B": "c2", "C": "c3", "D": "c4"}

        metrics = _svc().compute_polarization_index(graph, partition)

        assert metrics.polarizationIndex == pytest.approx(0.0), (
            f"Expected PI=0.0 for all-inter graph, got {metrics.polarizationIndex}"
        )

    def test_all_inter_ratio_equals_one(self) -> None:
        """All edges inter-community → interCommunityEdgeRatio = 1.0."""
        graph = _make_graph([("A", "B", 0.9)])
        partition = {"A": "c1", "B": "c2"}

        metrics = _svc().compute_polarization_index(graph, partition)

        assert metrics.interCommunityEdgeRatio == pytest.approx(1.0)

    def test_all_inter_isCrossCommunity_flag_true(self) -> None:
        """All edges inter-community → every edge.isCrossCommunity == True."""
        graph = _make_graph([
            ("A", "B", 1.0),
            ("C", "D", 0.5),
        ])
        partition = {"A": "c1", "B": "c2", "C": "c3", "D": "c4"}

        _svc().compute_polarization_index(graph, partition)

        for edge in graph.edges:
            assert edge.isCrossCommunity is True, (
                f"Edge ({edge.sourceUserId}->{edge.targetUserId}) should be "
                "isCrossCommunity=True"
            )


# ---------------------------------------------------------------------------
# Requirement 4.4 — PI + interCommunityEdgeRatio = 1.0
# ---------------------------------------------------------------------------


class TestPIplusInterRatioEqualsOne:
    """PI + interCommunityEdgeRatio must equal 1.0 (Req 4.4)."""

    def test_pi_plus_inter_ratio_mixed_graph(self) -> None:
        """Mixed intra/inter edges → PI + interCommunityEdgeRatio = 1.0."""
        graph = _make_graph([
            ("A", "B", 1.0),   # intra (c1)
            ("C", "D", 1.0),   # intra (c2)
            ("A", "C", 1.0),   # inter
        ])
        partition = {"A": "c1", "B": "c1", "C": "c2", "D": "c2"}

        metrics = _svc().compute_polarization_index(graph, partition)

        assert metrics.polarizationIndex + metrics.interCommunityEdgeRatio == pytest.approx(
            1.0
        ), (
            f"PI={metrics.polarizationIndex} + ICER={metrics.interCommunityEdgeRatio} "
            f"should equal 1.0"
        )

    def test_pi_plus_inter_ratio_all_intra(self) -> None:
        graph = _make_graph([("A", "B", 0.5)])
        partition = {"A": "c0", "B": "c0"}
        metrics = _svc().compute_polarization_index(graph, partition)
        assert metrics.polarizationIndex + metrics.interCommunityEdgeRatio == pytest.approx(1.0)

    def test_pi_plus_inter_ratio_all_inter(self) -> None:
        graph = _make_graph([("A", "B", 0.5)])
        partition = {"A": "c1", "B": "c2"}
        metrics = _svc().compute_polarization_index(graph, partition)
        assert metrics.polarizationIndex + metrics.interCommunityEdgeRatio == pytest.approx(1.0)

    def test_pi_plus_inter_ratio_weighted_edges(self) -> None:
        """PI + ICER = 1.0 holds with heterogeneous edge weights."""
        graph = _make_graph([
            ("A", "B", 0.2),   # intra (c1)
            ("A", "C", 0.7),   # inter
            ("B", "D", 0.1),   # inter
        ])
        partition = {"A": "c1", "B": "c1", "C": "c2", "D": "c3"}

        metrics = _svc().compute_polarization_index(graph, partition)

        assert metrics.polarizationIndex + metrics.interCommunityEdgeRatio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Requirement 4.5 — empty/zero-weight graph → PI = 0.0, ICER = 0.0
# ---------------------------------------------------------------------------


class TestEmptyAndZeroWeightGraph:
    """Requirement 4.5: zero total weight → PI = 0.0, ICER = 0.0."""

    def test_empty_graph_no_edges(self) -> None:
        """Empty graph (no edges) → PI = 0.0, ICER = 0.0."""
        graph = _empty_graph()
        partition: dict[str, str] = {}

        metrics = _svc().compute_polarization_index(graph, partition)

        assert metrics.polarizationIndex == pytest.approx(0.0)
        assert metrics.interCommunityEdgeRatio == pytest.approx(0.0)

    def test_graph_all_zero_weight_edges(self) -> None:
        """Graph with only zero-weight edges → PI = 0.0, ICER = 0.0."""
        graph = _make_graph([
            ("A", "B", 0.0),
            ("C", "D", 0.0),
        ])
        partition = {"A": "c1", "B": "c1", "C": "c2", "D": "c2"}

        metrics = _svc().compute_polarization_index(graph, partition)

        assert metrics.polarizationIndex == pytest.approx(0.0)
        assert metrics.interCommunityEdgeRatio == pytest.approx(0.0)

    def test_empty_graph_empty_list_partition(self) -> None:
        """Empty graph with empty list partition → PI = 0.0."""
        graph = _empty_graph()

        metrics = _svc().compute_polarization_index(graph, [])

        assert metrics.polarizationIndex == pytest.approx(0.0)
        assert metrics.interCommunityEdgeRatio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Requirement 4.1 — PI is in [0, 1]
# ---------------------------------------------------------------------------


class TestPIRange:
    """Requirement 4.1: polarizationIndex must be in [0, 1]."""

    def test_pi_in_range_all_intra(self) -> None:
        graph = _make_graph([("A", "B", 1.0)])
        partition = {"A": "c0", "B": "c0"}
        metrics = _svc().compute_polarization_index(graph, partition)
        assert 0.0 <= metrics.polarizationIndex <= 1.0

    def test_pi_in_range_all_inter(self) -> None:
        graph = _make_graph([("A", "B", 1.0)])
        partition = {"A": "c1", "B": "c2"}
        metrics = _svc().compute_polarization_index(graph, partition)
        assert 0.0 <= metrics.polarizationIndex <= 1.0

    def test_pi_in_range_mixed(self) -> None:
        graph = _make_graph([
            ("A", "B", 0.6),
            ("A", "C", 0.4),
        ])
        partition = {"A": "c1", "B": "c1", "C": "c2"}
        metrics = _svc().compute_polarization_index(graph, partition)
        assert 0.0 <= metrics.polarizationIndex <= 1.0

    def test_icer_in_range(self) -> None:
        """interCommunityEdgeRatio is also in [0, 1]."""
        graph = _make_graph([
            ("A", "B", 0.3),
            ("B", "C", 0.7),
        ])
        partition = {"A": "c1", "B": "c1", "C": "c2"}
        metrics = _svc().compute_polarization_index(graph, partition)
        assert 0.0 <= metrics.interCommunityEdgeRatio <= 1.0


# ---------------------------------------------------------------------------
# isCrossCommunity side-effect
# ---------------------------------------------------------------------------


class TestIsCrossCommunityFlag:
    """edge.isCrossCommunity must be correctly set as a side effect."""

    def test_mixed_edges_flags_set_correctly(self) -> None:
        """In a mixed graph, each edge gets the correct isCrossCommunity value."""
        graph = _make_graph([
            ("A", "B", 1.0),   # intra (c1)
            ("C", "D", 1.0),   # intra (c2)
            ("A", "C", 1.0),   # inter
            ("B", "D", 1.0),   # inter
        ])
        partition = {"A": "c1", "B": "c1", "C": "c2", "D": "c2"}

        _svc().compute_polarization_index(graph, partition)

        edge_flags = {
            (e.sourceUserId, e.targetUserId): e.isCrossCommunity
            for e in graph.edges
        }
        assert edge_flags[("A", "B")] is False
        assert edge_flags[("C", "D")] is False
        assert edge_flags[("A", "C")] is True
        assert edge_flags[("B", "D")] is True

    def test_flag_reflects_list_partition(self) -> None:
        """isCrossCommunity is set correctly when using list[CommunityPartition]."""
        graph = _make_graph([
            ("u1", "u2", 0.5),  # intra
            ("u1", "v1", 0.5),  # inter
        ])
        partition = [
            CommunityPartition(
                communityId="0",
                memberIds={"u1", "u2"},
                modularity=0.0,
                intraEdges=1,
                interEdges=1,
            ),
            CommunityPartition(
                communityId="1",
                memberIds={"v1"},
                modularity=0.0,
                intraEdges=0,
                interEdges=1,
            ),
        ]

        _svc().compute_polarization_index(graph, partition)

        for edge in graph.edges:
            if edge.targetUserId == "u2":
                assert edge.isCrossCommunity is False
            elif edge.targetUserId == "v1":
                assert edge.isCrossCommunity is True


# ---------------------------------------------------------------------------
# Output metadata
# ---------------------------------------------------------------------------


class TestOutputMetadata:
    """PolarizationMetrics carries correct metadata fields."""

    def test_snapshot_id_copied(self) -> None:
        snap_id = "my-snap-001"
        graph = _make_graph([("A", "B", 0.5)], snapshot_id=snap_id)
        metrics = _svc().compute_polarization_index(graph, {"A": "c0", "B": "c0"})
        assert metrics.snapshotId == snap_id

    def test_dataset_source_copied(self) -> None:
        graph = _make_graph([("A", "B", 0.5)], dataset_source="congress")
        metrics = _svc().compute_polarization_index(graph, {"A": "c0", "B": "c0"})
        assert metrics.datasetSource == "congress"

    def test_computed_at_is_recent(self) -> None:
        graph = _make_graph([("A", "B", 0.5)])
        metrics = _svc().compute_polarization_index(graph, {"A": "c0", "B": "c0"})
        now = datetime.now(timezone.utc)
        diff = (now - metrics.computedAt).total_seconds()
        assert 0 <= diff < 5, f"computedAt too far from now: {diff}s"

    def test_community_count(self) -> None:
        """communityCount reflects the number of distinct communities."""
        graph = _make_graph([
            ("A", "B", 1.0),
            ("C", "D", 1.0),
        ])
        partition = {"A": "c1", "B": "c1", "C": "c2", "D": "c2"}
        metrics = _svc().compute_polarization_index(graph, partition)
        assert metrics.communityCount == 2

    def test_avg_community_size(self) -> None:
        """avgCommunitySize = total_members / community_count."""
        graph = _make_graph([
            ("A", "B", 1.0),
            ("C", "D", 1.0),
        ])
        # 2 communities of 2 each → avg = 2.0
        partition = {"A": "c1", "B": "c1", "C": "c2", "D": "c2"}
        metrics = _svc().compute_polarization_index(graph, partition)
        assert metrics.avgCommunitySize == pytest.approx(2.0)

    def test_returns_polarization_metrics_instance(self) -> None:
        graph = _make_graph([("A", "B", 1.0)])
        metrics = _svc().compute_polarization_index(graph, {"A": "c0", "B": "c0"})
        assert isinstance(metrics, PolarizationMetrics)


# ---------------------------------------------------------------------------
# Numeric correctness (hand-computed values)
# ---------------------------------------------------------------------------


class TestNumericCorrectness:
    """Verify exact PI values against hand-computed expectations."""

    def test_one_third_inter_edges(self) -> None:
        """2 intra-edges (weight=1) + 1 inter-edge (weight=1) → PI = 2/3 ≈ 0.667."""
        graph = _make_graph([
            ("A", "B", 1.0),   # intra
            ("C", "D", 1.0),   # intra
            ("A", "C", 1.0),   # inter
        ])
        partition = {"A": "c1", "B": "c1", "C": "c2", "D": "c2"}

        metrics = _svc().compute_polarization_index(graph, partition)

        # interRatio = 1/3; PI = 1 - 1/3 = 2/3
        assert metrics.polarizationIndex == pytest.approx(2 / 3, rel=1e-6)
        assert metrics.interCommunityEdgeRatio == pytest.approx(1 / 3, rel=1e-6)

    def test_weighted_pi_computation(self) -> None:
        """PI computed correctly for non-uniform edge weights."""
        # intra: 0.8 + 0.6 = 1.4; inter: 0.4 + 0.1 = 0.5; total = 1.9
        # interRatio = 0.5 / 1.9 ≈ 0.2632; PI ≈ 0.7368
        graph = _make_graph([
            ("A", "B", 0.8),   # intra (c1)
            ("C", "D", 0.6),   # intra (c2)
            ("A", "C", 0.4),   # inter
            ("B", "D", 0.1),   # inter
        ])
        partition = {"A": "c1", "B": "c1", "C": "c2", "D": "c2"}

        metrics = _svc().compute_polarization_index(graph, partition)

        expected_inter_ratio = 0.5 / 1.9
        expected_pi = 1.0 - expected_inter_ratio
        assert metrics.polarizationIndex == pytest.approx(expected_pi, rel=1e-6)
        assert metrics.interCommunityEdgeRatio == pytest.approx(expected_inter_ratio, rel=1e-6)


# ---------------------------------------------------------------------------
# Requirement 5.x — Diversity Score (per-user and per-community)
# ---------------------------------------------------------------------------


class TestComputeDiversityScoreUser:
    """Tests for MetricsService.compute_diversity_score (Requirements 5.1–5.4)."""

    def test_all_intra_community_returns_zero(self) -> None:
        """User with all outgoing edges intra-community → diversityScore = 0.0 (Req 5.2)."""
        graph = _make_graph([
            ("A", "B", 1.0),
            ("A", "C", 0.5),
        ])
        partition = {"A": "c1", "B": "c1", "C": "c1"}

        score = _svc().compute_diversity_score("A", graph, partition)

        assert score == pytest.approx(0.0), (
            f"Expected DS=0.0 for all-intra outgoing edges, got {score}"
        )

    def test_all_inter_community_returns_one(self) -> None:
        """User with all outgoing edges inter-community → diversityScore = 1.0 (Req 5.3)."""
        graph = _make_graph([
            ("A", "B", 0.8),
            ("A", "C", 0.2),
        ])
        partition = {"A": "c1", "B": "c2", "C": "c3"}

        score = _svc().compute_diversity_score("A", graph, partition)

        assert score == pytest.approx(1.0), (
            f"Expected DS=1.0 for all-inter outgoing edges, got {score}"
        )

    def test_no_outgoing_edges_returns_zero(self) -> None:
        """User with no outgoing edges → diversityScore = 0.0 (Req 5.4)."""
        # "A" is only a target, never a source
        graph = _make_graph([("B", "A", 1.0)])
        partition = {"A": "c1", "B": "c1"}

        score = _svc().compute_diversity_score("A", graph, partition)

        assert score == pytest.approx(0.0), (
            f"Expected DS=0.0 for user with no outgoing edges, got {score}"
        )

    def test_no_outgoing_edges_isolated_user(self) -> None:
        """Isolated user not appearing in any edge → diversityScore = 0.0 (Req 5.4)."""
        graph = _make_graph([("B", "C", 1.0)])
        # Add isolated node manually
        from graph.models import Node
        graph.nodes["iso"] = Node(userId="iso")
        partition = {"B": "c1", "C": "c1", "iso": "c1"}

        score = _svc().compute_diversity_score("iso", graph, partition)

        assert score == pytest.approx(0.0)

    def test_mixed_edges_fractional_score(self) -> None:
        """User with mixed intra and inter edges → correct fractional score."""
        # A→B (intra, weight=1.0), A→C (inter, weight=1.0), A→D (inter, weight=2.0)
        # crossWeight = 3.0, totalWeight = 4.0 → score = 0.75
        graph = _make_graph([
            ("A", "B", 1.0),   # intra c1
            ("A", "C", 1.0),   # inter c2
            ("A", "D", 2.0),   # inter c3
        ])
        partition = {"A": "c1", "B": "c1", "C": "c2", "D": "c3"}

        score = _svc().compute_diversity_score("A", graph, partition)

        assert score == pytest.approx(0.75, rel=1e-6)

    def test_mixed_edges_weighted_correctly(self) -> None:
        """Weighted mixed edges produce correct ratio."""
        # A→B (intra, weight=3.0), A→C (inter, weight=1.0)
        # crossWeight = 1.0, totalWeight = 4.0 → score = 0.25
        graph = _make_graph([
            ("A", "B", 3.0),
            ("A", "C", 1.0),
        ])
        partition = {"A": "c1", "B": "c1", "C": "c2"}

        score = _svc().compute_diversity_score("A", graph, partition)

        assert score == pytest.approx(0.25, rel=1e-6)

    def test_result_in_range_zero_to_one(self) -> None:
        """Result is always in [0, 1] (Req 5.1)."""
        graph = _make_graph([
            ("A", "B", 0.6),
            ("A", "C", 0.4),
        ])
        partition = {"A": "c1", "B": "c1", "C": "c2"}

        score = _svc().compute_diversity_score("A", graph, partition)

        assert 0.0 <= score <= 1.0

    def test_side_effect_node_diversity_score_updated(self) -> None:
        """compute_diversity_score stores result in graph.nodes[userId].diversityScore."""
        graph = _make_graph([
            ("A", "B", 1.0),
            ("A", "C", 1.0),
        ])
        partition = {"A": "c1", "B": "c2", "C": "c2"}

        score = _svc().compute_diversity_score("A", graph, partition)

        assert graph.nodes["A"].diversityScore == pytest.approx(score)
        assert graph.nodes["A"].diversityScore == pytest.approx(1.0)

    def test_side_effect_no_outgoing_updates_node(self) -> None:
        """Side effect still updates the node even when score = 0.0 (no outgoing)."""
        graph = _make_graph([("B", "A", 1.0)])
        partition = {"A": "c1", "B": "c1"}

        _svc().compute_diversity_score("A", graph, partition)

        assert graph.nodes["A"].diversityScore == pytest.approx(0.0)

    def test_works_with_dict_partition(self) -> None:
        """compute_diversity_score accepts dict partition."""
        graph = _make_graph([("A", "B", 1.0)])
        partition = {"A": "c1", "B": "c2"}

        score = _svc().compute_diversity_score("A", graph, partition)

        assert score == pytest.approx(1.0)

    def test_works_with_list_partition(self) -> None:
        """compute_diversity_score accepts list[CommunityPartition] partition."""
        graph = _make_graph([("A", "B", 1.0)])
        partition = [
            CommunityPartition(
                communityId="c1",
                memberIds={"A"},
                modularity=0.0,
                intraEdges=0,
                interEdges=1,
            ),
            CommunityPartition(
                communityId="c2",
                memberIds={"B"},
                modularity=0.0,
                intraEdges=0,
                interEdges=1,
            ),
        ]

        score = _svc().compute_diversity_score("A", graph, partition)

        assert score == pytest.approx(1.0)

    def test_only_outgoing_edges_counted(self) -> None:
        """Incoming edges to userId are NOT counted in diversity score."""
        # X→A (incoming), A→B (outgoing intra)
        # Only A→B matters → all intra → score = 0.0
        graph = _make_graph([
            ("X", "A", 5.0),  # incoming to A
            ("A", "B", 1.0),  # outgoing intra
        ])
        partition = {"X": "c2", "A": "c1", "B": "c1"}

        score = _svc().compute_diversity_score("A", graph, partition)

        assert score == pytest.approx(0.0)


class TestComputeCommunityDiversityScore:
    """Tests for MetricsService.compute_community_diversity_score (Requirement 5.5)."""

    def test_all_members_score_zero(self) -> None:
        """Community where all members have diversityScore=0 → community score = 0.0."""
        # A and B only send to each other, both in c1
        graph = _make_graph([
            ("A", "B", 1.0),
            ("B", "A", 1.0),
        ])
        partition = {"A": "c1", "B": "c1"}

        score = _svc().compute_community_diversity_score("c1", graph, partition)

        assert score == pytest.approx(0.0)

    def test_all_members_score_one(self) -> None:
        """Community where all members have diversityScore=1.0 → community score = 1.0."""
        # A→C (inter), B→D (inter); A and B in c1, C and D in c2
        graph = _make_graph([
            ("A", "C", 1.0),
            ("B", "D", 1.0),
        ])
        partition = {"A": "c1", "B": "c1", "C": "c2", "D": "c2"}

        score = _svc().compute_community_diversity_score("c1", graph, partition)

        assert score == pytest.approx(1.0)

    def test_mixed_member_scores_arithmetic_mean(self) -> None:
        """Mixed member diversity scores → arithmetic mean (Req 5.5)."""
        # c1 members: A, B, C (D is in c2)
        # A→B (intra): A score = 0.0
        # B has no outgoing edges: B score = 0.0
        # C→D (inter): C score = 1.0
        # Mean = (0.0 + 0.0 + 1.0) / 3 ≈ 0.333
        graph = _make_graph([
            ("A", "B", 1.0),   # A→B intra c1
            ("C", "D", 1.0),   # C→D inter (C in c1, D in c2)
        ])
        partition = {"A": "c1", "B": "c1", "C": "c1", "D": "c2"}

        score = _svc().compute_community_diversity_score("c1", graph, partition)

        assert score == pytest.approx(1.0 / 3.0, rel=1e-6)

    def test_mixed_member_scores_weighted_mean(self) -> None:
        """Four members with different scores → correct arithmetic mean."""
        # c1 members: A, B, C, Y
        # A→X (inter, X in c2): A score = 1.0
        # B→Y (intra, Y in c1): B score = 0.0
        # C→Z (inter, Z in c2), C→Y (intra, Y in c1): C score = 0.5
        # Y: no outgoing edges → Y score = 0.0
        # Mean = (1.0 + 0.0 + 0.5 + 0.0) / 4 = 0.375
        graph = _make_graph([
            ("A", "X", 1.0),   # A→X inter (X in c2)
            ("B", "Y", 1.0),   # B→Y intra (Y in c1)
            ("C", "Z", 1.0),   # C→Z inter (Z in c2)
            ("C", "Y", 1.0),   # C→Y intra
        ])
        partition = {"A": "c1", "B": "c1", "C": "c1", "X": "c2", "Y": "c1", "Z": "c2"}

        score = _svc().compute_community_diversity_score("c1", graph, partition)

        # A=1.0, B=0.0, C=0.5, Y=0.0 → mean = 0.375
        assert score == pytest.approx(0.375, rel=1e-6)

    def test_empty_community_returns_zero(self) -> None:
        """Community not in partition → return 0.0."""
        graph = _make_graph([("A", "B", 1.0)])
        partition = {"A": "c1", "B": "c1"}

        score = _svc().compute_community_diversity_score("nonexistent", graph, partition)

        assert score == pytest.approx(0.0)

    def test_missing_community_empty_partition(self) -> None:
        """Empty partition → return 0.0."""
        graph = _make_graph([("A", "B", 1.0)])

        score = _svc().compute_community_diversity_score("c1", graph, {})

        assert score == pytest.approx(0.0)

    def test_community_score_equals_mean_of_member_scores(self) -> None:
        """Community score equals the mean of individually computed member scores (Req 5.5)."""
        graph = _make_graph([
            ("u1", "u2", 0.5),   # intra c1
            ("u1", "v1", 0.5),   # inter (v1 in c2)
            ("u2", "v1", 1.0),   # inter
        ])
        partition = {"u1": "c1", "u2": "c1", "v1": "c2"}

        svc = _svc()

        # Compute individual scores using the same service
        s_u1 = svc.compute_diversity_score("u1", graph, partition)
        s_u2 = svc.compute_diversity_score("u2", graph, partition)
        expected_mean = (s_u1 + s_u2) / 2

        # Reset diversityScore side effects to ensure community call recomputes
        graph.nodes["u1"].diversityScore = 0.0
        graph.nodes["u2"].diversityScore = 0.0

        community_score = svc.compute_community_diversity_score("c1", graph, partition)

        assert community_score == pytest.approx(expected_mean, rel=1e-6)

    def test_community_score_with_list_partition(self) -> None:
        """compute_community_diversity_score accepts list[CommunityPartition]."""
        graph = _make_graph([
            ("A", "C", 1.0),   # A→C inter
            ("B", "C", 1.0),   # B→C inter
        ])
        partition = [
            CommunityPartition(
                communityId="c1",
                memberIds={"A", "B"},
                modularity=0.0,
                intraEdges=0,
                interEdges=2,
            ),
            CommunityPartition(
                communityId="c2",
                memberIds={"C"},
                modularity=0.0,
                intraEdges=0,
                interEdges=2,
            ),
        ]

        score = _svc().compute_community_diversity_score("c1", graph, partition)

        assert score == pytest.approx(1.0)
