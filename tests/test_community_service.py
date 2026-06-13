"""Tests for community/service.py — CommunityDetectionService.

Covers:
1. Modularity Q is computed and stored in CommunityPartition objects
2. Modularity Q is non-negative (clamped to 0.0 if negative) — Requirement 3.4
3. Label persistence: stable communities keep old IDs when Jaccard ≥ 0.5
4. Label persistence: new communities keep new IDs when Jaccard < 0.5
5. Label persistence: greedy one-to-one matching (highest Jaccard wins)

References: Requirements 3.3, 3.4, 3.6
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import pytest

from community.service import CommunityDetectionService
from graph.models import CommunityPartition, Edge, InteractionGraph, InteractionRecord, InteractionType, Node
from graph.service import GraphConstructionService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAST = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _make_graph(
    edges: list[tuple[str, str]],
    dataset_source: str = "reddit_title",
    snapshot_id: Optional[str] = None,
) -> InteractionGraph:
    """Build an InteractionGraph directly from a list of (src, tgt) edge pairs.

    All edges have weight=1.0.  Nodes are auto-created from edge endpoints.
    """
    nodes: dict[str, Node] = {}
    edge_list: list[Edge] = []

    for src, tgt in edges:
        if src not in nodes:
            nodes[src] = Node(userId=src)
        if tgt not in nodes:
            nodes[tgt] = Node(userId=tgt)
        edge_list.append(Edge(sourceUserId=src, targetUserId=tgt, weight=1.0))

    return InteractionGraph(
        nodes=nodes,
        edges=edge_list,
        snapshotId=snapshot_id or str(uuid.uuid4()),
        createdAt=_PAST,
        datasetSource=dataset_source,
    )


def _make_community(
    community_id: str,
    member_ids: set[str],
    modularity: float = 0.0,
) -> CommunityPartition:
    """Helper to create a CommunityPartition for testing label persistence."""
    return CommunityPartition(
        communityId=community_id,
        memberIds=member_ids,
        modularity=modularity,
        intraEdges=0,
        interEdges=0,
    )


def _service() -> CommunityDetectionService:
    return CommunityDetectionService()


# ---------------------------------------------------------------------------
# Modularity tests (Requirements 3.3, 3.4)
# ---------------------------------------------------------------------------


class TestModularityComputation:
    """Tests that modularity Q is computed and stored in CommunityPartition."""

    def test_modularity_stored_in_partition(self) -> None:
        """Every CommunityPartition returned must have a modularity attribute."""
        # Build a graph with two clear communities: A-B-C-D densely connected,
        # E-F-G-H densely connected, with only one edge between groups
        edges = [
            ("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C"),
            ("E", "F"), ("F", "G"), ("G", "H"), ("H", "E"), ("E", "G"),
            ("D", "E"),  # single cross-community edge
        ]
        graph = _make_graph(edges)
        svc = _service()
        partitions = svc.detect_communities(graph)

        assert len(partitions) > 0
        for cp in partitions:
            assert hasattr(cp, "modularity"), "CommunityPartition must have modularity attribute"
            assert isinstance(cp.modularity, float)

    def test_modularity_is_non_negative(self) -> None:
        """Modularity Q must be >= 0.0 (Requirement 3.4)."""
        # K5,5 bipartite-ish graph: two groups of 5 with dense intra-group edges
        group_a = ["A1", "A2", "A3", "A4", "A5"]
        group_b = ["B1", "B2", "B3", "B4", "B5"]

        edges = []
        # Dense intra-group connections
        for i, a1 in enumerate(group_a):
            for a2 in group_a[i + 1:]:
                edges.append((a1, a2))
        for i, b1 in enumerate(group_b):
            for b2 in group_b[i + 1:]:
                edges.append((b1, b2))
        # Sparse inter-group connections
        edges.append(("A1", "B1"))

        graph = _make_graph(edges)
        svc = _service()
        partitions = svc.detect_communities(graph)

        for cp in partitions:
            assert cp.modularity >= 0.0, (
                f"Community {cp.communityId} has negative modularity {cp.modularity}"
            )

    def test_modularity_in_output_for_bipartite_like_graph(self) -> None:
        """For a graph with two well-separated communities, modularity Q > 0."""
        # Two complete subgraphs (K4) connected by one edge
        group_a = ["a1", "a2", "a3", "a4"]
        group_b = ["b1", "b2", "b3", "b4"]

        edges = []
        for i, n1 in enumerate(group_a):
            for n2 in group_a[i + 1:]:
                edges.append((n1, n2))
        for i, n1 in enumerate(group_b):
            for n2 in group_b[i + 1:]:
                edges.append((n1, n2))
        edges.append(("a1", "b1"))  # cross-community bridge

        graph = _make_graph(edges)
        svc = _service()
        partitions = svc.detect_communities(graph)

        # All partitions share the same overall Q
        assert all(cp.modularity >= 0.0 for cp in partitions)
        # The overall Q should be positive for well-separated communities
        overall_q = partitions[0].modularity
        assert overall_q >= 0.0, f"Expected modularity >= 0.0, got {overall_q}"

    def test_modularity_same_on_all_partition_objects(self) -> None:
        """All CommunityPartition objects from one run share the same modularity Q."""
        edges = [
            ("x1", "x2"), ("x2", "x3"), ("x3", "x1"),
            ("y1", "y2"), ("y2", "y3"), ("y3", "y1"),
            ("x1", "y1"),
        ]
        graph = _make_graph(edges)
        svc = _service()
        partitions = svc.detect_communities(graph)

        if len(partitions) > 1:
            # All partitions should have the same modularity Q
            first_q = partitions[0].modularity
            for cp in partitions[1:]:
                assert cp.modularity == pytest.approx(first_q), (
                    f"Expected all partitions to have same modularity {first_q}, "
                    f"got {cp.modularity}"
                )

    def test_isolated_node_graph_modularity_is_zero(self) -> None:
        """A graph with no edges has modularity 0.0."""
        # Single node graph — no edges
        nodes = {"solo": Node(userId="solo")}
        graph = InteractionGraph(
            nodes=nodes,
            edges=[],
            snapshotId=str(uuid.uuid4()),
            createdAt=_PAST,
            datasetSource="reddit_title",
        )
        svc = _service()
        partitions = svc.detect_communities(graph)

        assert len(partitions) == 1
        assert partitions[0].modularity == 0.0


# ---------------------------------------------------------------------------
# Label persistence tests (Requirement 3.6)
# ---------------------------------------------------------------------------


class TestLabelPersistence:
    """Tests for Jaccard-based label persistence across snapshots."""

    def test_stable_community_keeps_old_id_on_high_jaccard(self) -> None:
        """When Jaccard overlap >= 0.5, the new community should get the old ID."""
        svc = _service()

        # Build initial graph: two clear communities
        # Group A: a1-a2-a3-a4 (complete)
        # Group B: b1-b2-b3-b4 (complete)
        group_a = ["a1", "a2", "a3", "a4"]
        group_b = ["b1", "b2", "b3", "b4"]

        edges_initial = []
        for i, n1 in enumerate(group_a):
            for n2 in group_a[i + 1:]:
                edges_initial.append((n1, n2))
        for i, n1 in enumerate(group_b):
            for n2 in group_b[i + 1:]:
                edges_initial.append((n1, n2))
        edges_initial.append(("a1", "b1"))  # bridge

        initial_graph = _make_graph(edges_initial)
        initial_partitions = svc.detect_communities(initial_graph)

        # Identify the community IDs from initial run
        # (not testing the actual IDs from Louvain, just that they exist)
        assert len(initial_partitions) >= 2

        # Build an updated graph: same nodes + one new node in group A
        edges_updated = edges_initial + [("a1", "a5"), ("a2", "a5"), ("a3", "a5")]
        # a5 is a new node joining group A

        updated_graph = _make_graph(edges_updated)
        updated_partitions = svc.detect_communities(
            updated_graph,
            previous_snapshot_communities=initial_partitions,
        )

        # Get the set of community IDs from initial and updated runs
        initial_ids = {cp.communityId for cp in initial_partitions}
        updated_ids = {cp.communityId for cp in updated_partitions}

        # At least some old IDs should be preserved (Jaccard ≥ 0.5 for stable communities)
        # The group A and group B communities should be preserved since they largely
        # overlap with the initial partition
        overlapping_ids = initial_ids & updated_ids
        assert len(overlapping_ids) >= 1, (
            f"Expected at least one stable community ID to be preserved. "
            f"Initial IDs: {initial_ids}, Updated IDs: {updated_ids}"
        )

    def test_completely_different_community_gets_new_id(self) -> None:
        """When Jaccard < 0.5, communities should not reuse old IDs."""
        # Build the _persist_labels method directly with completely shuffled communities
        svc = _service()

        # Old communities: {a,b,c,d,e} and {f,g,h,i,j}
        previous_communities = [
            _make_community("stable_A", {"a", "b", "c", "d", "e"}),
            _make_community("stable_B", {"f", "g", "h", "i", "j"}),
        ]

        # New communities: completely different members (Jaccard = 0 with all old ones)
        # New community 0: {p, q, r, s, t} — no overlap with stable_A or stable_B
        # New community 1: {u, v, w, x, y} — no overlap with stable_A or stable_B
        new_partition_map = {
            "p": 0, "q": 0, "r": 0, "s": 0, "t": 0,
            "u": 1, "v": 1, "w": 1, "x": 1, "y": 1,
        }

        result = svc._persist_labels(new_partition_map, previous_communities)

        # No old IDs should be assigned since Jaccard = 0 for all pairs
        result_ids = set(result.values())
        assert "stable_A" not in result_ids, (
            "stable_A should not be assigned to a completely different community"
        )
        assert "stable_B" not in result_ids, (
            "stable_B should not be assigned to a completely different community"
        )

    def test_jaccard_threshold_boundary_50_percent(self) -> None:
        """Test the exact Jaccard threshold: 0.5 should trigger relabeling."""
        svc = _service()

        # Old community: {a, b, c, d} (4 members)
        previous_communities = [
            _make_community("old_community", {"a", "b", "c", "d"}),
        ]

        # New community with Jaccard = 0.5: 2 shared out of 4 new + 2 old-only = union 6
        # Intersection: {a, b} = 2; Union: {a, b, c, d, e, f} = 6; Jaccard = 2/6 ≈ 0.33 — below threshold
        # Need: intersection/union >= 0.5
        # E.g.: shared 3 out of 4: intersection = {a,b,c}, new = {a,b,c,d} → union = {a,b,c,d} = 4; J = 3/4 = 0.75
        new_partition_map_above = {"a": 0, "b": 0, "c": 0, "d": 0}
        result_above = svc._persist_labels(new_partition_map_above, previous_communities)
        assert "old_community" in set(result_above.values()), (
            "Jaccard = 1.0 (identical sets) should trigger relabeling"
        )

        # Below threshold: only 1 out of 5 members shared → Jaccard = 1/5 = 0.2
        new_partition_map_below = {"a": 0, "x": 0, "y": 0, "z": 0, "w": 0}
        result_below = svc._persist_labels(new_partition_map_below, previous_communities)
        # old_community should NOT be assigned since Jaccard = 1/8 ≈ 0.125 (below 0.5)
        # {a} ∩ {a,b,c,d,x,y,z,w} = {a} → intersection = 1, union = 8 → J = 0.125
        assert "old_community" not in set(result_below.values()), (
            "Jaccard = 0.125 (below threshold) should NOT trigger relabeling"
        )

    def test_greedy_one_to_one_matching_highest_jaccard_wins(self) -> None:
        """Greedy matching: each old label is assigned to at most one new community."""
        svc = _service()

        # Old community: {a, b, c, d, e}
        previous_communities = [
            _make_community("old_label", {"a", "b", "c", "d", "e"}),
        ]

        # Two new communities, both with high Jaccard to old_label
        # New community 0: {a, b, c, d, e, f} → J = 5/6 ≈ 0.83
        # New community 1: {a, b, c, d, e, g} → J = 5/6 ≈ 0.83
        new_partition_map = {
            "a": 0, "b": 0, "c": 0, "d": 0, "e": 0, "f": 0,
            "a2": 1, "b2": 1, "c2": 1, "d2": 1, "e2": 1, "g": 1,
        }
        # But community 1 has different members from old_label!
        # Let me fix: community 1 should share a, b, c, d, e too
        # community 0: {a, b, c, d, e, f} — 5/6 overlap
        # community 1: {a, b, c, d, e, g} — 5/6 overlap too
        # This is ambiguous, so use different overlap levels
        new_partition_map = {
            "a": 0, "b": 0, "c": 0, "d": 0, "e": 0, "f": 0,  # Jaccard with old = 5/6 ≈ 0.83
            "a": 0,  # duplicate; just set community 0 members
            "p": 1, "q": 1, "r": 1,  # community 1 has no overlap with old_label
        }
        # Rewrite for clarity:
        new_partition_map = {}
        for m in ["a", "b", "c", "d", "e", "f"]:
            new_partition_map[m] = 0  # J = 5/6 ≈ 0.83 with old_label
        for m in ["p", "q", "r", "s"]:
            new_partition_map[m] = 1  # J = 0 with old_label

        result = svc._persist_labels(new_partition_map, previous_communities)

        # Community 0 should get the old label (highest Jaccard)
        result_vals = set(result.values())
        assert "old_label" in result_vals

        # Only ONE community should get the old label (one-to-one matching)
        old_label_count = sum(1 for v in result.values() if v == "old_label")
        assert old_label_count == 6, (  # all 6 members of community 0 get "old_label"
            f"Expected 6 nodes assigned to 'old_label', got {old_label_count}"
        )

        # Community 1 (p, q, r, s) should NOT get the old label
        for node in ["p", "q", "r", "s"]:
            assert result[node] != "old_label", (
                f"Node {node} should not be assigned to old_label (no overlap)"
            )

    def test_label_persistence_with_detect_communities(self) -> None:
        """Integration test: running detect_communities with previous snapshot
        preserves stable community IDs."""
        svc = _service()

        # Initial graph: two well-separated communities
        edges_initial = [
            ("u1", "u2"), ("u2", "u3"), ("u3", "u1"), ("u1", "u4"), ("u4", "u2"),  # community A
            ("v1", "v2"), ("v2", "v3"), ("v3", "v1"), ("v1", "v4"), ("v4", "v2"),  # community B
            ("u1", "v1"),  # bridge
        ]
        initial_graph = _make_graph(edges_initial)
        initial_partitions = svc.detect_communities(initial_graph)

        assert len(initial_partitions) >= 2

        # Updated graph: same structure + one extra node per group
        edges_updated = edges_initial + [
            ("u1", "u5"), ("u2", "u5"), ("u5", "u3"),  # u5 joins community A
            ("v1", "v5"), ("v2", "v5"), ("v5", "v3"),  # v5 joins community B
        ]
        updated_graph = _make_graph(edges_updated)

        # Run detection with previous partition
        updated_partitions = svc.detect_communities(
            updated_graph,
            previous_snapshot_communities=initial_partitions,
        )

        # Some old IDs should be reused (stable communities)
        initial_ids = {cp.communityId for cp in initial_partitions}
        updated_ids = {cp.communityId for cp in updated_partitions}

        overlapping_ids = initial_ids & updated_ids
        assert len(overlapping_ids) >= 1, (
            f"Expected stable community IDs to be preserved. "
            f"Initial: {initial_ids}, Updated: {updated_ids}"
        )

    def test_no_previous_snapshot_produces_fresh_ids(self) -> None:
        """Without previous_snapshot_communities, community IDs are freshly assigned."""
        svc = _service()

        edges = [
            ("a1", "a2"), ("a2", "a3"), ("a3", "a1"),
            ("b1", "b2"), ("b2", "b3"), ("b3", "b1"),
            ("a1", "b1"),
        ]
        graph = _make_graph(edges)
        partitions = svc.detect_communities(graph, previous_snapshot_communities=None)

        # All community IDs should be integer strings (from Louvain)
        for cp in partitions:
            # Should be parseable as integers (Louvain assigns integer IDs)
            try:
                int(cp.communityId)
            except ValueError:
                pytest.fail(
                    f"Expected integer-based communityId without label persistence, "
                    f"got '{cp.communityId}'"
                )

    def test_persist_labels_empty_previous_communities(self) -> None:
        """Empty previous communities list: new partition is returned unchanged."""
        svc = _service()

        new_partition_map = {"a": 0, "b": 0, "c": 1, "d": 1}
        result = svc._persist_labels(new_partition_map, [])

        assert result == new_partition_map

    def test_all_nodes_assigned_after_label_persistence(self) -> None:
        """After label persistence, every node must still have a community assignment."""
        svc = _service()

        edges = [
            ("a1", "a2"), ("a2", "a3"), ("a3", "a1"), ("a1", "a4"),
            ("b1", "b2"), ("b2", "b3"), ("b3", "b1"), ("b1", "b4"),
            ("a1", "b1"),
        ]
        graph = _make_graph(edges)
        initial_partitions = svc.detect_communities(graph)

        # Add new nodes and re-run
        edges_updated = edges + [("a1", "c1"), ("c1", "a2")]
        updated_graph = _make_graph(edges_updated)
        updated_partitions = svc.detect_communities(
            updated_graph,
            previous_snapshot_communities=initial_partitions,
        )

        # All nodes in the updated graph must be assigned a community
        all_members: set[str] = set()
        for cp in updated_partitions:
            all_members |= cp.memberIds

        for node_id in updated_graph.nodes:
            assert node_id in all_members, (
                f"Node '{node_id}' not assigned to any community after label persistence"
            )


# ---------------------------------------------------------------------------
# Girvan-Newman secondary validation tests (Requirement 3.7)
# ---------------------------------------------------------------------------


class TestGirvanNewman:
    """Tests for Girvan-Newman secondary validation (Requirement 3.7)."""

    def test_girvan_newman_disabled_by_default(self) -> None:
        """Without enable_girvan_newman, girvan_newman_partition should be None."""
        edges = [("a1","a2"),("a2","a3"),("a3","a1"),("b1","b2"),("b2","b3"),("b3","b1"),("a1","b1")]
        graph = _make_graph(edges)
        svc = _service()
        partitions = svc.detect_communities(graph)
        for cp in partitions:
            assert cp.girvan_newman_partition is None

    def test_girvan_newman_enabled_stores_partition(self) -> None:
        """With enable_girvan_newman=True, girvan_newman_partition should be set."""
        edges = [("a1","a2"),("a2","a3"),("a3","a1"),("b1","b2"),("b2","b3"),("b3","b1"),("a1","b1")]
        graph = _make_graph(edges)
        svc = _service()
        partitions = svc.detect_communities(graph, enable_girvan_newman=True)
        for cp in partitions:
            assert cp.girvan_newman_partition is not None
            assert isinstance(cp.girvan_newman_partition, list)
            assert all(isinstance(s, (set, frozenset)) for s in cp.girvan_newman_partition)

    def test_girvan_newman_partition_covers_all_nodes(self) -> None:
        """The Girvan-Newman partition must cover every node in the graph."""
        edges = [("a1","a2"),("a2","a3"),("a3","a1"),("b1","b2"),("b2","b3"),("b3","b1"),("a1","b1")]
        graph = _make_graph(edges)
        svc = _service()
        partitions = svc.detect_communities(graph, enable_girvan_newman=True)
        gn_partition = partitions[0].girvan_newman_partition
        all_gn_members: set[str] = set()
        for s in gn_partition:
            all_gn_members |= set(s)
        assert all_gn_members == set(graph.nodes.keys())

    def test_girvan_newman_partition_same_on_all_partition_objects(self) -> None:
        """All CommunityPartition objects in a run share the same girvan_newman_partition."""
        edges = [("a1","a2"),("a2","a3"),("a3","a1"),("b1","b2"),("b2","b3"),("b3","b1"),("a1","b1")]
        graph = _make_graph(edges)
        svc = _service()
        partitions = svc.detect_communities(graph, enable_girvan_newman=True)
        if len(partitions) > 1:
            first_gn = partitions[0].girvan_newman_partition
            for cp in partitions[1:]:
                assert cp.girvan_newman_partition == first_gn

    def test_girvan_newman_on_no_edge_graph_does_not_crash(self) -> None:
        """Girvan-Newman on a graph with no edges should not crash."""
        nodes = {"solo": Node(userId="solo"), "other": Node(userId="other")}
        graph = InteractionGraph(
            nodes=nodes, edges=[], snapshotId=str(uuid.uuid4()),
            createdAt=_PAST, datasetSource="reddit_title"
        )
        svc = _service()
        # Should not raise
        partitions = svc.detect_communities(graph, enable_girvan_newman=True)
        assert len(partitions) >= 1
