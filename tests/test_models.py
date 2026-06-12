"""Tests for graph/models.py — core data models for the Echo Chamber Detector.

Covers:
- Valid InteractionRecord construction for each dataset (reddit, congress, wiki-rfa)
- Validation errors: empty userId, self-loop, future timestamp, invalid votePolarity,
  invalid voteResult
- Edge weight < 0 raises ValueError
- Node defaults are correct
- InteractionGraph nodeCount and edgeCount properties
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from graph.models import (
    CommunityPartition,
    Edge,
    InteractionGraph,
    InteractionRecord,
    InteractionType,
    Node,
    PolarizationMetrics,
    Recommendation,
    UserMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAST = datetime(2020, 1, 1, tzinfo=timezone.utc)
_FUTURE = datetime.now(timezone.utc) + timedelta(days=365)


def _new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# InteractionType enum
# ---------------------------------------------------------------------------


def test_interaction_type_values() -> None:
    assert InteractionType.HYPERLINK.value == "HYPERLINK"
    assert InteractionType.RETWEET.value == "RETWEET"
    assert InteractionType.VOTE.value == "VOTE"


# ---------------------------------------------------------------------------
# Valid InteractionRecord construction — per dataset
# ---------------------------------------------------------------------------


def test_reddit_title_record() -> None:
    """Reddit Title dataset: HYPERLINK interaction with sentimentScore."""
    record = InteractionRecord(
        id=_new_id(),
        sourceUserId="subreddit_a",
        targetUserId="subreddit_b",
        interactionType=InteractionType.HYPERLINK,
        datasetSource="reddit_title",
        timestamp=_PAST,
        contentId="post_123",
        topicTags=["politics", "news"],
        sentimentScore=-0.5,
    )
    assert record.sourceUserId == "subreddit_a"
    assert record.targetUserId == "subreddit_b"
    assert record.interactionType == InteractionType.HYPERLINK
    assert record.datasetSource == "reddit_title"
    assert record.sentimentScore == -0.5
    assert record.votePolarity is None
    assert record.voteResult is None
    assert record.bodyText is None


def test_reddit_body_record() -> None:
    """Reddit Body dataset: HYPERLINK with bodyText extracted from PROPERTIES."""
    record = InteractionRecord(
        id=_new_id(),
        sourceUserId="subreddit_x",
        targetUserId="subreddit_y",
        interactionType=InteractionType.HYPERLINK,
        datasetSource="reddit_body",
        timestamp=_PAST,
        bodyText="Example post body text extracted from PROPERTIES JSON",
    )
    assert record.bodyText == "Example post body text extracted from PROPERTIES JSON"
    assert record.topicTags == []  # default


def test_congress_record() -> None:
    """Congress Network dataset: RETWEET with no timestamp; pre-normalized weight."""
    record = InteractionRecord(
        id=_new_id(),
        sourceUserId="senator_alice",
        targetUserId="senator_bob",
        interactionType=InteractionType.RETWEET,
        datasetSource="congress",
        # No timestamp — Congress dataset omits it
    )
    assert record.timestamp is None
    assert record.interactionType == InteractionType.RETWEET
    assert record.datasetSource == "congress"
    assert record.topicTags == []


def test_wiki_rfa_record_positive_vote() -> None:
    """Wiki-RfA dataset: VOTE with votePolarity=+1, voteResult=1 (granted)."""
    record = InteractionRecord(
        id=_new_id(),
        sourceUserId="editor_alice",
        targetUserId="editor_bob",
        interactionType=InteractionType.VOTE,
        datasetSource="wiki_rfa",
        timestamp=_PAST,
        votePolarity=1,
        voteResult=1,
        sentimentScore=1.0,
        bodyText="Support: great contributions.",
    )
    assert record.votePolarity == 1
    assert record.voteResult == 1
    assert record.bodyText == "Support: great contributions."


def test_wiki_rfa_record_negative_vote() -> None:
    """Wiki-RfA dataset: VOTE with votePolarity=-1, voteResult=0 (denied)."""
    record = InteractionRecord(
        id=_new_id(),
        sourceUserId="editor_carol",
        targetUserId="editor_dave",
        interactionType=InteractionType.VOTE,
        datasetSource="wiki_rfa",
        timestamp=_PAST,
        votePolarity=-1,
        voteResult=0,
        sentimentScore=-1.0,
        bodyText="Oppose: insufficient track record.",
    )
    assert record.votePolarity == -1
    assert record.voteResult == 0


# ---------------------------------------------------------------------------
# InteractionRecord — validation errors
# ---------------------------------------------------------------------------


def test_empty_source_user_id_raises() -> None:
    with pytest.raises(ValueError, match="sourceUserId must be non-empty"):
        InteractionRecord(
            id=_new_id(),
            sourceUserId="",
            targetUserId="user_b",
            interactionType=InteractionType.RETWEET,
            datasetSource="congress",
        )


def test_empty_target_user_id_raises() -> None:
    with pytest.raises(ValueError, match="targetUserId must be non-empty"):
        InteractionRecord(
            id=_new_id(),
            sourceUserId="user_a",
            targetUserId="",
            interactionType=InteractionType.RETWEET,
            datasetSource="congress",
        )


def test_self_loop_raises() -> None:
    with pytest.raises(ValueError, match="sourceUserId must not equal targetUserId"):
        InteractionRecord(
            id=_new_id(),
            sourceUserId="user_a",
            targetUserId="user_a",
            interactionType=InteractionType.HYPERLINK,
            datasetSource="reddit_title",
            timestamp=_PAST,
        )


def test_future_timestamp_raises() -> None:
    with pytest.raises(ValueError, match="timestamp must be a past datetime"):
        InteractionRecord(
            id=_new_id(),
            sourceUserId="user_a",
            targetUserId="user_b",
            interactionType=InteractionType.HYPERLINK,
            datasetSource="reddit_title",
            timestamp=_FUTURE,
        )


def test_invalid_vote_polarity_zero_raises() -> None:
    with pytest.raises(ValueError, match="votePolarity must be \\+1 or -1"):
        InteractionRecord(
            id=_new_id(),
            sourceUserId="editor_a",
            targetUserId="editor_b",
            interactionType=InteractionType.VOTE,
            datasetSource="wiki_rfa",
            timestamp=_PAST,
            votePolarity=0,
        )


def test_invalid_vote_polarity_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="votePolarity must be \\+1 or -1"):
        InteractionRecord(
            id=_new_id(),
            sourceUserId="editor_a",
            targetUserId="editor_b",
            interactionType=InteractionType.VOTE,
            datasetSource="wiki_rfa",
            timestamp=_PAST,
            votePolarity=2,
        )


def test_invalid_vote_result_raises() -> None:
    with pytest.raises(ValueError, match="voteResult must be 0 or 1"):
        InteractionRecord(
            id=_new_id(),
            sourceUserId="editor_a",
            targetUserId="editor_b",
            interactionType=InteractionType.VOTE,
            datasetSource="wiki_rfa",
            timestamp=_PAST,
            votePolarity=1,
            voteResult=2,
        )


def test_valid_vote_polarity_none_does_not_raise() -> None:
    """votePolarity=None should not raise — it is optional."""
    record = InteractionRecord(
        id=_new_id(),
        sourceUserId="user_a",
        targetUserId="user_b",
        interactionType=InteractionType.HYPERLINK,
        datasetSource="reddit_title",
        timestamp=_PAST,
        votePolarity=None,
    )
    assert record.votePolarity is None


def test_naive_past_datetime_accepted() -> None:
    """Naive datetimes should be treated as UTC and accepted if in the past."""
    naive_past = datetime(2020, 6, 1)  # no tzinfo
    record = InteractionRecord(
        id=_new_id(),
        sourceUserId="user_a",
        targetUserId="user_b",
        interactionType=InteractionType.RETWEET,
        datasetSource="congress",
        timestamp=naive_past,
    )
    assert record.timestamp == naive_past


# ---------------------------------------------------------------------------
# Edge — validation
# ---------------------------------------------------------------------------


def test_edge_negative_weight_raises() -> None:
    with pytest.raises(ValueError, match="Edge weight must be >= 0"):
        Edge(
            sourceUserId="user_a",
            targetUserId="user_b",
            weight=-0.1,
        )


def test_edge_zero_weight_is_valid() -> None:
    edge = Edge(sourceUserId="user_a", targetUserId="user_b", weight=0.0)
    assert edge.weight == 0.0


def test_edge_defaults() -> None:
    edge = Edge(sourceUserId="user_a", targetUserId="user_b", weight=0.5)
    assert edge.isCrossCommunity is False
    assert edge.signedPolarity is None


def test_edge_with_signed_polarity() -> None:
    edge = Edge(
        sourceUserId="editor_a",
        targetUserId="editor_b",
        weight=1.0,
        signedPolarity=-1,
    )
    assert edge.signedPolarity == -1


# ---------------------------------------------------------------------------
# Node — defaults
# ---------------------------------------------------------------------------


def test_node_defaults() -> None:
    node = Node(userId="user_42")
    assert node.userId == "user_42"
    assert node.communityId is None
    assert node.betweenness == 0.0
    assert node.diversityScore == 0.0
    assert node.topicVector == []


def test_node_topic_vector_is_independent() -> None:
    """Mutable default (topicVector) should not be shared across instances."""
    node_a = Node(userId="user_a")
    node_b = Node(userId="user_b")
    node_a.topicVector.append(0.1)
    assert node_b.topicVector == [], "topicVector should not be shared between Node instances"


def test_node_with_community_and_scores() -> None:
    node = Node(
        userId="user_99",
        communityId="community_1",
        betweenness=0.42,
        diversityScore=0.75,
        topicVector=[0.1, 0.2, 0.3],
    )
    assert node.communityId == "community_1"
    assert node.betweenness == 0.42
    assert node.diversityScore == 0.75
    assert node.topicVector == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# InteractionGraph — nodeCount and edgeCount properties
# ---------------------------------------------------------------------------


def test_interaction_graph_empty() -> None:
    graph = InteractionGraph(
        nodes={},
        edges=[],
        snapshotId="snap-001",
        createdAt=_PAST,
    )
    assert graph.nodeCount == 0
    assert graph.edgeCount == 0


def test_interaction_graph_counts() -> None:
    nodes = {
        "user_a": Node(userId="user_a"),
        "user_b": Node(userId="user_b"),
        "user_c": Node(userId="user_c"),
    }
    edges = [
        Edge(sourceUserId="user_a", targetUserId="user_b", weight=0.8),
        Edge(sourceUserId="user_b", targetUserId="user_c", weight=0.5),
    ]
    graph = InteractionGraph(
        nodes=nodes,
        edges=edges,
        snapshotId="snap-002",
        createdAt=_PAST,
        datasetSource="reddit_title",
    )
    assert graph.nodeCount == 3
    assert graph.edgeCount == 2


def test_interaction_graph_nodecount_reflects_dict_size() -> None:
    """Adding a node to the dict should update nodeCount immediately."""
    nodes: dict = {}
    graph = InteractionGraph(
        nodes=nodes,
        edges=[],
        snapshotId="snap-003",
        createdAt=_PAST,
    )
    assert graph.nodeCount == 0
    nodes["user_a"] = Node(userId="user_a")
    assert graph.nodeCount == 1


def test_interaction_graph_default_dataset_source() -> None:
    graph = InteractionGraph(
        nodes={},
        edges=[],
        snapshotId="snap-004",
        createdAt=_PAST,
    )
    assert graph.datasetSource == ""


# ---------------------------------------------------------------------------
# CommunityPartition — construction
# ---------------------------------------------------------------------------


def test_community_partition_construction() -> None:
    partition = CommunityPartition(
        communityId="c_1",
        memberIds={"user_a", "user_b", "user_c"},
        modularity=0.42,
        intraEdges=5,
        interEdges=2,
        centroidNode="user_a",
    )
    assert partition.communityId == "c_1"
    assert "user_b" in partition.memberIds
    assert partition.centroidNode == "user_a"


# ---------------------------------------------------------------------------
# PolarizationMetrics — construction
# ---------------------------------------------------------------------------


def test_polarization_metrics_construction() -> None:
    metrics = PolarizationMetrics(
        snapshotId="snap-001",
        polarizationIndex=0.85,
        modularity=0.43,
        communityCount=2,
        avgCommunitySize=750.0,
        interCommunityEdgeRatio=0.15,
        computedAt=_PAST,
        datasetSource="congress",
    )
    assert metrics.polarizationIndex == 0.85
    assert metrics.datasetSource == "congress"


def test_polarization_metrics_default_dataset_source() -> None:
    metrics = PolarizationMetrics(
        snapshotId="snap-002",
        polarizationIndex=0.5,
        modularity=0.3,
        communityCount=3,
        avgCommunitySize=100.0,
        interCommunityEdgeRatio=0.5,
        computedAt=_PAST,
    )
    assert metrics.datasetSource == ""


# ---------------------------------------------------------------------------
# UserMetrics — construction
# ---------------------------------------------------------------------------


def test_user_metrics_construction() -> None:
    um = UserMetrics(
        userId="user_1",
        communityId="c_2",
        diversityScore=0.3,
        intraEdgeCount=10,
        interEdgeCount=2,
        betweennessCentrality=0.05,
        snapshotId="snap-001",
        computedAt=_PAST,
    )
    assert um.userId == "user_1"
    assert um.intraEdgeCount == 10
    assert um.interEdgeCount == 2


# ---------------------------------------------------------------------------
# Recommendation — construction
# ---------------------------------------------------------------------------


def test_recommendation_construction() -> None:
    rec = Recommendation(
        recommendationId=_new_id(),
        targetUserId="user_low_diversity",
        recommendedUserId="bridge_user",
        diversityGain=0.25,
        topicRelevance=0.7,
        communityId="c_3",
        reason="bridge_user is a connector with high betweenness in community c_3",
        contentId="post_456",
    )
    assert rec.diversityGain == 0.25
    assert rec.contentId == "post_456"


def test_recommendation_default_content_id() -> None:
    rec = Recommendation(
        recommendationId=_new_id(),
        targetUserId="user_a",
        recommendedUserId="user_b",
        diversityGain=0.1,
        topicRelevance=0.6,
        communityId="c_1",
        reason="Cross-community bridge node.",
    )
    assert rec.contentId is None
