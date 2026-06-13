"""Tests for recommendations/bridge_nodes.py — estimate_diversity_gain and
generate_recommendations (Algorithm 6, task 6.3).

Unit tests:
  - estimate_diversity_gain: same community → 0.0
  - estimate_diversity_gain: different community, no prior edges → gain > 0
  - estimate_diversity_gain: different community, all intra-edges → gain > 0
  - estimate_diversity_gain: clamp to [0, 1 - current_score]
  - estimate_diversity_gain: user not in graph → 0.0
  - estimate_diversity_gain: candidate not in graph → 0.0
  - generate_recommendations: topK=0 → empty list
  - generate_recommendations: no bridge candidates → empty list
  - generate_recommendations: returns ≤ topK items
  - generate_recommendations: all recs from different community
  - generate_recommendations: sorted descending by diversityGain
  - generate_recommendations: reason string is a non-empty string
  - generate_recommendations: wiki-RfA hostile community excluded

Property-based tests (Hypothesis):
  - Property 17: all recs from different community than user
  - Property 18: all recs have betweenness > threshold and topicRelevance >= threshold
  - Property 19: recs sorted descending by diversityGain
  - Property 20: len(result) <= topK

**Validates: Requirements 6.4, 6.5, 6.8**
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from graph.models import (
    Edge,
    InteractionGraph,
    Node,
    Recommendation,
    SignedMetrics,
    UserMetrics,
)
from recommendations.bridge_nodes import (
    BRIDGE_CENTRALITY_THRESHOLD,
    MIN_TOPIC_RELEVANCE_THRESHOLD,
    estimate_diversity_gain,
    generate_recommendations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics(
    user_id: str,
    community_id: str,
    diversity_score: float = 0.0,
) -> UserMetrics:
    return UserMetrics(
        userId=user_id,
        communityId=community_id,
        diversityScore=diversity_score,
        intraEdgeCount=0,
        interEdgeCount=0,
        betweennessCentrality=0.0,
        snapshotId="test-snapshot",
        computedAt=datetime.now(timezone.utc),
    )


def _make_graph(
    nodes: dict[str, Optional[str]],  # userId → communityId
    edges: Optional[list[tuple[str, str, float]]] = None,
    topic_vectors: Optional[dict[str, list[float]]] = None,
    betweenness: Optional[dict[str, float]] = None,
    dataset_source: str = "reddit_title",
) -> InteractionGraph:
    """Build a minimal InteractionGraph for testing."""
    node_objs: dict[str, Node] = {}
    for uid, cid in nodes.items():
        tv = (topic_vectors or {}).get(uid, [])
        bw = (betweenness or {}).get(uid, 0.0)
        node_objs[uid] = Node(
            userId=uid,
            communityId=cid,
            betweenness=bw,
            diversityScore=0.0,
            topicVector=tv,
        )

    edge_objs: list[Edge] = []
    if edges:
        for src, tgt, w in edges:
            edge_objs.append(Edge(sourceUserId=src, targetUserId=tgt, weight=w))

    return InteractionGraph(
        nodes=node_objs,
        edges=edge_objs,
        snapshotId=str(uuid.uuid4()),
        createdAt=datetime.now(timezone.utc),
        datasetSource=dataset_source,
    )


# ===========================================================================
# estimate_diversity_gain — unit tests
# ===========================================================================


class TestEstimateDiversityGain:
    """estimate_diversity_gain behaves correctly in all edge cases."""

    def test_same_community_returns_zero(self) -> None:
        """Candidate in same community as user → gain = 0.0."""
        graph = _make_graph({"alice": "A", "bob": "A"})
        metrics = _make_metrics("alice", "A", diversity_score=0.2)
        gain = estimate_diversity_gain("alice", "bob", graph, metrics)
        assert gain == 0.0

    def test_different_community_no_prior_edges_returns_positive(self) -> None:
        """User with no outgoing edges → simulated edge weight = 1.0 → gain > 0."""
        graph = _make_graph({"alice": "A", "bob": "B"})
        metrics = _make_metrics("alice", "A", diversity_score=0.0)
        gain = estimate_diversity_gain("alice", "bob", graph, metrics)
        # Only 1 edge after simulation: 1 cross-community out of 1 total → new_score = 1.0
        # gain = 1.0 - 0.0 = 1.0 but clamped to [0, 1 - 0.0] = 1.0
        assert gain > 0.0
        assert gain <= 1.0 - metrics.diversityScore

    def test_different_community_with_existing_intra_edges_improves_score(
        self,
    ) -> None:
        """User with intra-community edges → adding cross-community edge improves score."""
        graph = _make_graph(
            {"alice": "A", "carol": "A", "bob": "B"},
            edges=[("alice", "carol", 1.0)],  # all intra-community
        )
        metrics = _make_metrics("alice", "A", diversity_score=0.0)
        gain = estimate_diversity_gain("alice", "bob", graph, metrics)
        assert gain > 0.0

    def test_gain_clamped_to_max(self) -> None:
        """gain is clamped to [0, 1 - metrics.diversityScore]."""
        # User already at 0.9 diversity → max gain = 0.1
        graph = _make_graph({"alice": "A", "bob": "B"})
        metrics = _make_metrics("alice", "A", diversity_score=0.9)
        gain = estimate_diversity_gain("alice", "bob", graph, metrics)
        assert 0.0 <= gain <= 0.1 + 1e-9  # small float tolerance

    def test_gain_never_negative(self) -> None:
        """gain is always >= 0."""
        graph = _make_graph({"alice": "A", "bob": "B"})
        metrics = _make_metrics("alice", "A", diversity_score=0.5)
        gain = estimate_diversity_gain("alice", "bob", graph, metrics)
        assert gain >= 0.0

    def test_user_not_in_graph_returns_zero(self) -> None:
        """If user_id not in graph → log warning and return 0.0."""
        graph = _make_graph({"bob": "B"})
        metrics = _make_metrics("alice", "A", diversity_score=0.0)
        gain = estimate_diversity_gain("alice", "bob", graph, metrics)
        assert gain == 0.0

    def test_candidate_not_in_graph_returns_zero(self) -> None:
        """If candidate_id not in graph → log warning and return 0.0."""
        graph = _make_graph({"alice": "A"})
        metrics = _make_metrics("alice", "A", diversity_score=0.0)
        gain = estimate_diversity_gain("alice", "ghost", graph, metrics)
        assert gain == 0.0

    def test_gain_in_range_for_mixed_existing_edges(self) -> None:
        """User with mix of intra and inter edges → gain is in [0, 1 - current_score]."""
        graph = _make_graph(
            {"alice": "A", "carol": "A", "dave": "B", "bob": "C"},
            edges=[
                ("alice", "carol", 0.5),  # intra
                ("alice", "dave", 0.5),   # inter (community B)
            ],
        )
        # diversityScore = 0.5 / 1.0 = 0.5 (half cross)
        metrics = _make_metrics("alice", "A", diversity_score=0.5)
        gain = estimate_diversity_gain("alice", "bob", graph, metrics)
        assert 0.0 <= gain <= 1.0 - 0.5

    def test_average_edge_weight_used_for_simulation(self) -> None:
        """New edge weight equals average of existing outgoing edge weights."""
        # alice has two outgoing edges with weights 0.2 and 0.8 → avg = 0.5
        graph = _make_graph(
            {"alice": "A", "carol": "A", "dave": "A", "bob": "B"},
            edges=[
                ("alice", "carol", 0.2),
                ("alice", "dave", 0.8),
            ],
        )
        metrics = _make_metrics("alice", "A", diversity_score=0.0)
        gain = estimate_diversity_gain("alice", "bob", graph, metrics)
        # current_total = 1.0, current_cross = 0.0, new_edge_weight = 0.5
        # simulated_total = 1.5, simulated_cross = 0.5
        # new_score = 0.5/1.5 ≈ 0.333, gain ≈ 0.333
        assert abs(gain - (0.5 / 1.5)) < 1e-9

    def test_perfect_diversity_gives_zero_gain(self) -> None:
        """User with diversity_score = 1.0 → max gain = 0.0."""
        graph = _make_graph({"alice": "A", "bob": "B"})
        metrics = _make_metrics("alice", "A", diversity_score=1.0)
        gain = estimate_diversity_gain("alice", "bob", graph, metrics)
        assert gain == 0.0


# ===========================================================================
# generate_recommendations — unit tests
# ===========================================================================


def _make_rec_graph(
    user_community: str = "A",
    n_bridge_candidates: int = 3,
) -> tuple[InteractionGraph, UserMetrics]:
    """Build a standard test graph with a user and multiple bridge candidates."""
    user_topic = [1.0, 0.0, 0.0]
    candidate_topic = [0.8, 0.2, 0.0]  # high cosine similarity

    nodes: dict[str, Optional[str]] = {"user1": user_community}
    bw: dict[str, float] = {"user1": 0.0}
    tv: dict[str, list[float]] = {"user1": user_topic}

    for i in range(n_bridge_candidates):
        cid = f"B{i}"
        uid = f"cand{i}"
        nodes[uid] = cid
        bw[uid] = 0.5  # well above BRIDGE_CENTRALITY_THRESHOLD
        tv[uid] = candidate_topic

    graph = _make_graph(nodes, betweenness=bw, topic_vectors=tv)
    metrics = _make_metrics("user1", user_community, diversity_score=0.0)
    return graph, metrics


class TestGenerateRecommendations:
    """generate_recommendations covers all Algorithm 6 requirements."""

    def test_top_k_zero_returns_empty(self) -> None:
        """topK = 0 → always return empty list (Req 6.5)."""
        graph, metrics = _make_rec_graph()
        result = generate_recommendations("user1", graph, metrics, top_k=0)
        assert result == []

    def test_negative_top_k_returns_empty(self) -> None:
        """topK < 0 → always return empty list (guard on Req 6.5)."""
        graph, metrics = _make_rec_graph()
        result = generate_recommendations("user1", graph, metrics, top_k=-1)
        assert result == []

    def test_no_bridge_candidates_returns_empty(self) -> None:
        """No nodes with betweenness > threshold → empty result."""
        # All candidates have betweenness = 0 (below threshold)
        graph = _make_graph(
            {"user1": "A", "bob": "B"},
            betweenness={"user1": 0.0, "bob": 0.0},
            topic_vectors={"user1": [1.0, 0.0], "bob": [0.9, 0.1]},
        )
        metrics = _make_metrics("user1", "A", diversity_score=0.0)
        result = generate_recommendations("user1", graph, metrics, top_k=5)
        assert result == []

    def test_returns_at_most_top_k(self) -> None:
        """result length never exceeds topK (Req 6.5)."""
        graph, metrics = _make_rec_graph(n_bridge_candidates=10)
        result = generate_recommendations("user1", graph, metrics, top_k=3)
        assert len(result) <= 3

    def test_all_recs_from_different_community(self) -> None:
        """Every recommendation comes from a different community (Req 6.1)."""
        graph, metrics = _make_rec_graph(n_bridge_candidates=5)
        result = generate_recommendations("user1", graph, metrics, top_k=5)
        user_community = metrics.communityId
        for rec in result:
            assert rec.communityId != user_community

    def test_sorted_descending_by_diversity_gain(self) -> None:
        """Recommendations are sorted descending by diversityGain (Req 6.4)."""
        graph, metrics = _make_rec_graph(n_bridge_candidates=5)
        result = generate_recommendations("user1", graph, metrics, top_k=5)
        gains = [rec.diversityGain for rec in result]
        assert gains == sorted(gains, reverse=True)

    def test_reason_string_is_non_empty(self) -> None:
        """Each recommendation has a non-empty human-readable reason (Req 6.8)."""
        graph, metrics = _make_rec_graph(n_bridge_candidates=3)
        result = generate_recommendations("user1", graph, metrics, top_k=3)
        for rec in result:
            assert isinstance(rec.reason, str)
            assert len(rec.reason) > 0

    def test_recommendation_fields_populated(self) -> None:
        """Each Recommendation object has all required fields set."""
        graph, metrics = _make_rec_graph(n_bridge_candidates=2)
        result = generate_recommendations("user1", graph, metrics, top_k=2)
        assert len(result) >= 1
        for rec in result:
            assert rec.recommendationId  # non-empty UUID
            assert rec.targetUserId == "user1"
            assert rec.recommendedUserId
            assert rec.communityId
            assert 0.0 <= rec.diversityGain <= 1.0
            assert 0.0 <= rec.topicRelevance <= 1.0

    def test_topic_relevance_at_least_threshold(self) -> None:
        """All returned candidates have topicRelevance >= MIN_TOPIC_RELEVANCE_THRESHOLD
        (Req 6.3)."""
        graph, metrics = _make_rec_graph(n_bridge_candidates=4)
        result = generate_recommendations("user1", graph, metrics, top_k=4)
        for rec in result:
            assert rec.topicRelevance >= MIN_TOPIC_RELEVANCE_THRESHOLD

    def test_candidates_with_low_topic_relevance_excluded(self) -> None:
        """Candidates with topicRelevance below threshold are excluded (Req 6.3)."""
        # Orthogonal vectors → cosine similarity = 0.0
        graph = _make_graph(
            {"user1": "A", "bob": "B"},
            betweenness={"user1": 0.0, "bob": 0.5},
            topic_vectors={"user1": [1.0, 0.0], "bob": [0.0, 1.0]},
        )
        metrics = _make_metrics("user1", "A", diversity_score=0.0)
        result = generate_recommendations("user1", graph, metrics, top_k=5)
        # bob's cosine similarity = 0.0 < MIN_TOPIC_RELEVANCE_THRESHOLD → excluded
        assert all(rec.recommendedUserId != "bob" for rec in result)

    def test_wiki_rfa_hostile_community_excluded(self) -> None:
        """Candidates from communities with net_sentiment_index < 0 are excluded
        for wiki-RfA graphs."""
        graph = _make_graph(
            {"user1": "A", "carol": "B"},
            betweenness={"user1": 0.0, "carol": 0.5},
            topic_vectors={"user1": [1.0, 0.0], "carol": [0.9, 0.1]},
            dataset_source="wiki_rfa",
        )
        metrics = _make_metrics("user1", "A", diversity_score=0.0)

        # Community "B" is hostile (net_sentiment_index < 0)
        hostile_signed_metrics = [
            SignedMetrics(
                snapshotId="s",
                communityId="B",
                positiveEdgeRatio=0.2,
                negativeEdgeRatio=0.8,
                netSentimentIndex=-0.5,
                crossCommunityNegativity=0.3,
                computedAt=datetime.now(timezone.utc),
            )
        ]
        result = generate_recommendations(
            "user1", graph, metrics, top_k=5, signed_metrics=hostile_signed_metrics
        )
        assert all(rec.recommendedUserId != "carol" for rec in result)

    def test_wiki_rfa_positive_community_included(self) -> None:
        """Candidates from communities with net_sentiment_index >= 0 are NOT excluded."""
        graph = _make_graph(
            {"user1": "A", "carol": "B"},
            betweenness={"user1": 0.0, "carol": 0.5},
            topic_vectors={"user1": [1.0, 0.0], "carol": [0.9, 0.1]},
            dataset_source="wiki_rfa",
        )
        metrics = _make_metrics("user1", "A", diversity_score=0.0)

        positive_signed_metrics = [
            SignedMetrics(
                snapshotId="s",
                communityId="B",
                positiveEdgeRatio=0.8,
                negativeEdgeRatio=0.2,
                netSentimentIndex=0.5,
                crossCommunityNegativity=0.1,
                computedAt=datetime.now(timezone.utc),
            )
        ]
        result = generate_recommendations(
            "user1", graph, metrics, top_k=5, signed_metrics=positive_signed_metrics
        )
        assert any(rec.recommendedUserId == "carol" for rec in result)

    def test_no_topic_vector_user_returns_empty(self) -> None:
        """User with no topic vector and no community centroid → empty list."""
        graph = _make_graph(
            {"user1": "A", "bob": "B"},
            betweenness={"bob": 0.5},
            topic_vectors={"user1": [], "bob": [0.9, 0.1]},
        )
        metrics = _make_metrics("user1", "A", diversity_score=0.0)
        result = generate_recommendations("user1", graph, metrics, top_k=5)
        assert result == []

    def test_sparse_user_uses_community_centroid(self) -> None:
        """User with fewer than 5 interactions uses community centroid topic vector."""
        # user1 has 1 outgoing edge (< 5 threshold) and no own topic vector
        # community A centroid: carol has [1.0, 0.0]
        graph = _make_graph(
            {"user1": "A", "carol": "A", "bob": "B"},
            edges=[("user1", "carol", 1.0)],  # 1 edge < SPARSE_USER_INTERACTION_THRESHOLD
            betweenness={"user1": 0.0, "carol": 0.0, "bob": 0.5},
            topic_vectors={
                "user1": [],              # no personal vector
                "carol": [1.0, 0.0],     # community A centroid-member
                "bob": [0.9, 0.1],       # good topic match
            },
        )
        metrics = _make_metrics("user1", "A", diversity_score=0.0)
        result = generate_recommendations("user1", graph, metrics, top_k=5)
        # bob should appear because community centroid ≈ [1.0, 0.0] which matches bob
        assert any(rec.recommendedUserId == "bob" for rec in result)

    def test_reason_string_contains_community_and_gain(self) -> None:
        """Reason string mentions community and diversity gain (Req 6.8)."""
        graph, metrics = _make_rec_graph(n_bridge_candidates=1)
        result = generate_recommendations("user1", graph, metrics, top_k=1)
        assert len(result) == 1
        reason = result[0].reason
        # Should mention the community and gain in some form
        assert "B0" in reason or "community" in reason.lower()
        assert "diversity" in reason.lower() or "%" in reason


# ===========================================================================
# Property-based tests (Hypothesis) — Properties 17–20
# ===========================================================================

# Strategy: generate node IDs as short unique strings
_node_id_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="_",
    ),
    min_size=1,
    max_size=15,
)

_float_01_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def _build_graph_and_metrics(
    user_id: str,
    user_community: str,
    candidate_ids: list[str],
    candidate_community: str,
    diversity_score: float,
) -> tuple[InteractionGraph, UserMetrics]:
    """Build a small graph with a user and several bridge candidates."""
    user_topic = [1.0, 0.0]
    candidate_topic = [0.8, 0.2]  # high cosine similarity to user

    nodes: dict[str, Optional[str]] = {user_id: user_community}
    bw: dict[str, float] = {user_id: 0.0}
    tv: dict[str, list[float]] = {user_id: user_topic}

    for cid_str in candidate_ids:
        nodes[cid_str] = candidate_community
        bw[cid_str] = 0.5  # above threshold
        tv[cid_str] = candidate_topic

    graph = _make_graph(nodes, betweenness=bw, topic_vectors=tv)
    metrics = _make_metrics(user_id, user_community, diversity_score=diversity_score)
    return graph, metrics


@given(
    user_id=_node_id_st,
    diversity_score=_float_01_st,
    candidate_count=st.integers(min_value=1, max_value=5),
    top_k=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=30, deadline=5_000)
def test_property_17_cross_community_invariant(
    user_id: str,
    diversity_score: float,
    candidate_count: int,
    top_k: int,
) -> None:
    """**Validates: Requirements 6.1**

    Property 17: For any recommendation r generated for user u, the
    recommendedUserId must belong to a community different from u's community.
    """
    # Ensure candidate IDs don't collide with user_id
    candidate_ids = [f"cand_{i}_{user_id[:3]}" for i in range(candidate_count)]
    assume(user_id not in candidate_ids)

    graph, metrics = _build_graph_and_metrics(
        user_id=user_id,
        user_community="X",
        candidate_ids=candidate_ids,
        candidate_community="Y",
        diversity_score=diversity_score,
    )

    recs = generate_recommendations(user_id, graph, metrics, top_k=top_k)
    for rec in recs:
        assert rec.communityId != "X", (
            f"Recommendation for user in community 'X' points to community {rec.communityId!r}"
        )


@given(
    user_id=_node_id_st,
    diversity_score=_float_01_st,
    candidate_count=st.integers(min_value=1, max_value=5),
    top_k=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=30, deadline=5_000)
def test_property_18_candidate_quality_invariant(
    user_id: str,
    diversity_score: float,
    candidate_count: int,
    top_k: int,
) -> None:
    """**Validates: Requirements 6.2, 6.3**

    Property 18: For any recommendation generated, the recommended account
    must have betweenness centrality above BRIDGE_CENTRALITY_THRESHOLD, and
    topicRelevance must be >= MIN_TOPIC_RELEVANCE_THRESHOLD.
    """
    candidate_ids = [f"cand_{i}_{user_id[:3]}" for i in range(candidate_count)]
    assume(user_id not in candidate_ids)

    graph, metrics = _build_graph_and_metrics(
        user_id=user_id,
        user_community="X",
        candidate_ids=candidate_ids,
        candidate_community="Y",
        diversity_score=diversity_score,
    )

    recs = generate_recommendations(user_id, graph, metrics, top_k=top_k)
    for rec in recs:
        rec_node = graph.nodes[rec.recommendedUserId]
        assert rec_node.betweenness > BRIDGE_CENTRALITY_THRESHOLD, (
            f"Recommended node '{rec.recommendedUserId}' betweenness "
            f"{rec_node.betweenness} is not above threshold {BRIDGE_CENTRALITY_THRESHOLD}"
        )
        assert rec.topicRelevance >= MIN_TOPIC_RELEVANCE_THRESHOLD, (
            f"topicRelevance {rec.topicRelevance} is below "
            f"MIN_TOPIC_RELEVANCE_THRESHOLD {MIN_TOPIC_RELEVANCE_THRESHOLD}"
        )


@given(
    user_id=_node_id_st,
    diversity_score=_float_01_st,
    candidate_count=st.integers(min_value=2, max_value=6),
    top_k=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=30, deadline=5_000)
def test_property_19_sorted_by_diversity_gain(
    user_id: str,
    diversity_score: float,
    candidate_count: int,
    top_k: int,
) -> None:
    """**Validates: Requirements 6.4**

    Property 19: For any list of recommendations returned, the list must be
    sorted in non-increasing order of diversityGain.
    """
    candidate_ids = [f"cand_{i}_{user_id[:3]}" for i in range(candidate_count)]
    assume(user_id not in candidate_ids)

    graph, metrics = _build_graph_and_metrics(
        user_id=user_id,
        user_community="X",
        candidate_ids=candidate_ids,
        candidate_community="Y",
        diversity_score=diversity_score,
    )

    recs = generate_recommendations(user_id, graph, metrics, top_k=top_k)
    gains = [rec.diversityGain for rec in recs]
    assert gains == sorted(gains, reverse=True), (
        f"Recommendations not sorted descending by diversityGain: {gains}"
    )


@given(
    user_id=_node_id_st,
    diversity_score=_float_01_st,
    candidate_count=st.integers(min_value=0, max_value=8),
    top_k=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=40, deadline=5_000)
def test_property_20_size_bound(
    user_id: str,
    diversity_score: float,
    candidate_count: int,
    top_k: int,
) -> None:
    """**Validates: Requirements 6.5**

    Property 20: For any call to generateRecommendations with parameter topK,
    the returned list must have length at most topK.
    """
    candidate_ids = [f"cand_{i}_{user_id[:3]}" for i in range(candidate_count)]
    assume(user_id not in candidate_ids)

    if candidate_count == 0:
        # No candidates: build minimal graph with just the user
        graph = _make_graph(
            {user_id: "X"},
            topic_vectors={user_id: [1.0, 0.0]},
        )
        metrics = _make_metrics(user_id, "X", diversity_score=diversity_score)
    else:
        graph, metrics = _build_graph_and_metrics(
            user_id=user_id,
            user_community="X",
            candidate_ids=candidate_ids,
            candidate_community="Y",
            diversity_score=diversity_score,
        )

    recs = generate_recommendations(user_id, graph, metrics, top_k=top_k)
    assert len(recs) <= top_k, (
        f"Expected len(recs) <= {top_k}, got {len(recs)}"
    )
