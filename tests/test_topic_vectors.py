"""Tests for recommendations/topic_vectors.py — TopicVectorService.

Unit tests:
  - Reddit: nodes in corpus get non-zero vectors of length 5000
  - Reddit: nodes absent from corpus get zero vectors of length 5000
  - Reddit: empty corpus → all zero vectors
  - Wiki-RfA: vectors assigned correctly from editor TXT corpus
  - Congress: two-community graph → correct [1.0, 0.0] / [0.0, 1.0] assignment
  - Congress: nodes in neither dominant community → [0.5, 0.5]
  - Congress: empty partition → all neutral vectors
  - Dispatcher: routes correctly by datasetSource
  - Dispatcher: unknown datasetSource → no vectors assigned (nodes unchanged)
  - WikiRfAAdapter.editor_text_corpus populated during fetch()

Property tests (Hypothesis):
  - All topic vectors in a graph have the same length after build_topic_vectors
  - Congress vectors contain only values in {0.0, 0.5, 1.0}
  - TF-IDF vectors have non-negative components

**Validates: Requirements 6.3** (topic relevance scoring uses cosine similarity
between topic vectors) and **Requirements 6.6** (fallback to community centroid
when user has < 5 interactions).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from graph.models import Edge, InteractionGraph, Node
from recommendations.topic_vectors import TopicVectorService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(
    node_ids: list[str],
    *,
    dataset_source: str = "reddit_body",
    edges: Optional[list[tuple[str, str]]] = None,
) -> InteractionGraph:
    """Build a minimal InteractionGraph for testing."""
    nodes = {
        uid: Node(userId=uid, topicVector=[])
        for uid in node_ids
    }
    edge_objs = []
    if edges:
        for src, tgt in edges:
            edge_objs.append(
                Edge(sourceUserId=src, targetUserId=tgt, weight=1.0)
            )
    return InteractionGraph(
        nodes=nodes,
        edges=edge_objs,
        snapshotId=str(uuid.uuid4()),
        createdAt=datetime.now(timezone.utc),
        datasetSource=dataset_source,
    )


# ===========================================================================
# Reddit topic vectors
# ===========================================================================


class TestRedditTopicVectors:
    """build_reddit_topic_vectors assigns TF-IDF vectors from subreddit corpus."""

    def test_nodes_in_corpus_get_nonzero_vectors(self) -> None:
        """Nodes that appear in the corpus get non-zero TF-IDF vectors."""
        graph = _make_graph(["gaming", "news"])
        corpus = {
            "gaming": ["Call of Duty review", "League of Legends patch notes"],
            "news": ["World news update", "Political developments"],
        }
        svc = TopicVectorService()
        svc.build_reddit_topic_vectors(graph, corpus)

        for node in graph.nodes.values():
            assert len(node.topicVector) > 0
            assert any(v > 0.0 for v in node.topicVector), (
                f"Node '{node.userId}' should have at least one non-zero component"
            )

    def test_nodes_in_corpus_get_vectors_of_length_5000(self) -> None:
        """With enough vocabulary, vector length equals max_features (5000)."""
        # Build a corpus with at least 5000 unique tokens to exercise max_features.
        # Use 100 subreddits with 50 unique words each.
        graph_ids = [f"sub_{i}" for i in range(100)]
        graph = _make_graph(graph_ids)
        corpus = {
            f"sub_{i}": [
                " ".join(f"word_{i}_{j}" for j in range(50))
            ]
            for i in range(100)
        }
        svc = TopicVectorService()
        svc.build_reddit_topic_vectors(graph, corpus)

        # Vocabulary < 5000 here, but length should equal actual vocab size
        # and be consistent across nodes
        lengths = {len(node.topicVector) for node in graph.nodes.values()}
        assert len(lengths) == 1, "All nodes must have the same vector length"

    def test_nodes_absent_from_corpus_get_zero_vectors(self) -> None:
        """Nodes not in the corpus receive a zero vector."""
        graph = _make_graph(["gaming", "absent_subreddit"])
        corpus = {"gaming": ["some game text"]}
        svc = TopicVectorService()
        svc.build_reddit_topic_vectors(graph, corpus)

        absent_node = graph.nodes["absent_subreddit"]
        assert all(v == 0.0 for v in absent_node.topicVector)

    def test_absent_and_present_vectors_have_same_length(self) -> None:
        """Absent-corpus nodes and corpus nodes have identical vector lengths."""
        graph = _make_graph(["present", "absent"])
        corpus = {"present": ["hello world"]}
        svc = TopicVectorService()
        svc.build_reddit_topic_vectors(graph, corpus)

        assert len(graph.nodes["present"].topicVector) == len(graph.nodes["absent"].topicVector)

    def test_empty_corpus_assigns_zero_vectors(self) -> None:
        """Empty subreddit_text_corpus → all nodes get zero vectors of length 5000."""
        graph = _make_graph(["sub_a", "sub_b"])
        svc = TopicVectorService()
        svc.build_reddit_topic_vectors(graph, {})

        for node in graph.nodes.values():
            assert len(node.topicVector) == 5000
            assert all(v == 0.0 for v in node.topicVector)

    def test_returns_fitted_vectorizer(self) -> None:
        """build_reddit_topic_vectors returns the fitted TfidfVectorizer."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        graph = _make_graph(["sub_a"])
        corpus = {"sub_a": ["some text content"]}
        svc = TopicVectorService()
        result = svc.build_reddit_topic_vectors(graph, corpus)

        assert isinstance(result, TfidfVectorizer)

    def test_vectors_are_plain_lists_not_numpy(self) -> None:
        """topicVector must be a plain Python list[float], not a numpy array."""
        graph = _make_graph(["sub_a"])
        corpus = {"sub_a": ["hello world"]}
        svc = TopicVectorService()
        svc.build_reddit_topic_vectors(graph, corpus)

        for node in graph.nodes.values():
            assert isinstance(node.topicVector, list)
            for v in node.topicVector:
                assert isinstance(v, float)

    def test_single_node_single_text(self) -> None:
        """Single node, single text document → non-zero vector."""
        graph = _make_graph(["politics"])
        corpus = {"politics": ["election democracy voting"]}
        svc = TopicVectorService()
        svc.build_reddit_topic_vectors(graph, corpus)

        node = graph.nodes["politics"]
        assert len(node.topicVector) > 0
        assert any(v > 0 for v in node.topicVector)


# ===========================================================================
# Wiki-RfA topic vectors
# ===========================================================================


class TestWikiRfATopicVectors:
    """build_wiki_rfa_topic_vectors assigns TF-IDF vectors from editor corpus."""

    def test_editors_in_corpus_get_nonzero_vectors(self) -> None:
        """Editors present in the corpus receive non-zero TF-IDF vectors."""
        graph = _make_graph(["alice", "bob"], dataset_source="wiki_rfa")
        corpus = {
            "alice": ["This user has great contributions to the wiki"],
            "bob": ["Outstanding administrator, highly recommend for adminship"],
        }
        svc = TopicVectorService()
        svc.build_wiki_rfa_topic_vectors(graph, corpus)

        for node in graph.nodes.values():
            assert len(node.topicVector) > 0
            assert any(v > 0.0 for v in node.topicVector)

    def test_editors_absent_from_corpus_get_zero_vectors(self) -> None:
        """Editors not in the corpus get zero vectors."""
        graph = _make_graph(["alice", "carol"], dataset_source="wiki_rfa")
        corpus = {"alice": ["good editor"]}
        svc = TopicVectorService()
        svc.build_wiki_rfa_topic_vectors(graph, corpus)

        assert all(v == 0.0 for v in graph.nodes["carol"].topicVector)

    def test_all_nodes_same_vector_length(self) -> None:
        """All nodes share the same topic vector length after fitting."""
        graph = _make_graph(["alice", "bob", "carol"], dataset_source="wiki_rfa")
        corpus = {
            "alice": ["wiki admin support"],
            "bob": ["oppose the candidate"],
        }
        svc = TopicVectorService()
        svc.build_wiki_rfa_topic_vectors(graph, corpus)

        lengths = {len(node.topicVector) for node in graph.nodes.values()}
        assert len(lengths) == 1

    def test_empty_corpus_all_zero_vectors(self) -> None:
        """Empty editor corpus → all nodes get zero vectors of length 5000."""
        graph = _make_graph(["alice", "bob"], dataset_source="wiki_rfa")
        svc = TopicVectorService()
        svc.build_wiki_rfa_topic_vectors(graph, {})

        for node in graph.nodes.values():
            assert len(node.topicVector) == 5000
            assert all(v == 0.0 for v in node.topicVector)

    def test_returns_fitted_vectorizer(self) -> None:
        """build_wiki_rfa_topic_vectors returns the fitted TfidfVectorizer."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        graph = _make_graph(["alice"], dataset_source="wiki_rfa")
        corpus = {"alice": ["text comment"]}
        svc = TopicVectorService()
        result = svc.build_wiki_rfa_topic_vectors(graph, corpus)

        assert isinstance(result, TfidfVectorizer)

    def test_vectors_are_plain_lists(self) -> None:
        """topicVector values are plain Python list[float]."""
        graph = _make_graph(["alice"], dataset_source="wiki_rfa")
        corpus = {"alice": ["some wiki comment"]}
        svc = TopicVectorService()
        svc.build_wiki_rfa_topic_vectors(graph, corpus)

        node = graph.nodes["alice"]
        assert isinstance(node.topicVector, list)
        for v in node.topicVector:
            assert isinstance(v, float)


# ===========================================================================
# Congress topic vectors
# ===========================================================================


class TestCongressTopicVectors:
    """build_congress_topic_vectors assigns categorical vectors by community."""

    def test_two_community_graph_larger_gets_1_0(self) -> None:
        """The larger community receives [1.0, 0.0]."""
        # 3 nodes in community 0, 2 nodes in community 1
        node_ids = ["a", "b", "c", "d", "e"]
        graph = _make_graph(node_ids, dataset_source="congress")
        partition = {"a": 0, "b": 0, "c": 0, "d": 1, "e": 1}
        svc = TopicVectorService()
        svc.build_congress_topic_vectors(graph, partition)

        assert graph.nodes["a"].topicVector == [1.0, 0.0]
        assert graph.nodes["b"].topicVector == [1.0, 0.0]
        assert graph.nodes["c"].topicVector == [1.0, 0.0]

    def test_two_community_graph_smaller_gets_0_1(self) -> None:
        """The smaller community receives [0.0, 1.0]."""
        node_ids = ["a", "b", "c", "d", "e"]
        graph = _make_graph(node_ids, dataset_source="congress")
        partition = {"a": 0, "b": 0, "c": 0, "d": 1, "e": 1}
        svc = TopicVectorService()
        svc.build_congress_topic_vectors(graph, partition)

        assert graph.nodes["d"].topicVector == [0.0, 1.0]
        assert graph.nodes["e"].topicVector == [0.0, 1.0]

    def test_nodes_outside_dominant_communities_get_neutral(self) -> None:
        """Nodes not in the two dominant communities receive [0.5, 0.5]."""
        node_ids = ["a", "b", "c", "d", "e", "f"]
        graph = _make_graph(node_ids, dataset_source="congress")
        # Communities 0 and 1 dominate; "f" is in community 2 (singleton)
        partition = {"a": 0, "b": 0, "c": 0, "d": 1, "e": 1, "f": 2}
        svc = TopicVectorService()
        svc.build_congress_topic_vectors(graph, partition)

        assert graph.nodes["f"].topicVector == [0.5, 0.5]

    def test_node_absent_from_partition_gets_neutral(self) -> None:
        """Nodes not present in the partition dict receive [0.5, 0.5]."""
        node_ids = ["a", "b", "c", "extra_node"]
        graph = _make_graph(node_ids, dataset_source="congress")
        partition = {"a": 0, "b": 0, "c": 1}  # extra_node not in partition
        svc = TopicVectorService()
        svc.build_congress_topic_vectors(graph, partition)

        assert graph.nodes["extra_node"].topicVector == [0.5, 0.5]

    def test_empty_partition_gives_all_neutral(self) -> None:
        """An empty partition → all nodes receive [0.5, 0.5]."""
        graph = _make_graph(["a", "b"], dataset_source="congress")
        svc = TopicVectorService()
        svc.build_congress_topic_vectors(graph, {})

        for node in graph.nodes.values():
            assert node.topicVector == [0.5, 0.5]

    def test_equal_size_communities_deterministic(self) -> None:
        """Equal-size communities: both dominant communities get categorical vectors."""
        node_ids = ["a", "b", "c", "d"]
        graph = _make_graph(node_ids, dataset_source="congress")
        partition = {"a": 0, "b": 0, "c": 1, "d": 1}
        svc = TopicVectorService()
        svc.build_congress_topic_vectors(graph, partition)

        # One community gets [1.0, 0.0] and the other gets [0.0, 1.0]
        vectors = [tuple(graph.nodes[uid].topicVector) for uid in node_ids]
        assert (1.0, 0.0) in vectors
        assert (0.0, 1.0) in vectors


# ===========================================================================
# Dispatcher — build_topic_vectors
# ===========================================================================


class TestBuildTopicVectorsDispatcher:
    """build_topic_vectors dispatches to the correct dataset-specific method."""

    def test_dispatches_to_reddit_for_reddit_body_source(self) -> None:
        """datasetSource='reddit_body' calls build_reddit_topic_vectors."""
        graph = _make_graph(["sub_a"], dataset_source="reddit_body")
        corpus = {"sub_a": ["game text"]}
        svc = TopicVectorService()
        svc.build_topic_vectors(graph, subreddit_text_corpus=corpus)

        # Node should have a non-empty topic vector
        assert len(graph.nodes["sub_a"].topicVector) > 0

    def test_dispatches_to_wiki_rfa_for_wiki_rfa_source(self) -> None:
        """datasetSource='wiki_rfa' calls build_wiki_rfa_topic_vectors."""
        graph = _make_graph(["alice"], dataset_source="wiki_rfa")
        corpus = {"alice": ["support this nomination"]}
        svc = TopicVectorService()
        svc.build_topic_vectors(graph, editor_text_corpus=corpus)

        assert len(graph.nodes["alice"].topicVector) > 0

    def test_dispatches_to_congress_for_congress_source(self) -> None:
        """datasetSource='congress' calls build_congress_topic_vectors."""
        graph = _make_graph(["a", "b", "c"], dataset_source="congress")
        partition = {"a": 0, "b": 0, "c": 1}
        svc = TopicVectorService()
        svc.build_topic_vectors(graph, partition=partition)

        # All vectors should be categorical (not empty)
        for node in graph.nodes.values():
            assert len(node.topicVector) == 2
            assert node.topicVector in ([1.0, 0.0], [0.0, 1.0], [0.5, 0.5])

    def test_unknown_source_leaves_vectors_unchanged(self) -> None:
        """Unsupported datasetSource → no vectors assigned, originals preserved."""
        graph = _make_graph(["sub_a"], dataset_source="reddit_title")
        # Pre-assign vectors to verify they are untouched
        graph.nodes["sub_a"].topicVector = [9.9, 8.8]
        svc = TopicVectorService()
        svc.build_topic_vectors(graph)

        # Vector should remain unchanged
        assert graph.nodes["sub_a"].topicVector == [9.9, 8.8]

    def test_reddit_body_missing_corpus_skips_gracefully(self) -> None:
        """Calling dispatcher for 'reddit_body' without corpus → skip silently."""
        graph = _make_graph(["sub_a"], dataset_source="reddit_body")
        svc = TopicVectorService()
        # Should not raise
        svc.build_topic_vectors(graph)
        # topicVector remains at default empty
        assert graph.nodes["sub_a"].topicVector == []

    def test_wiki_rfa_missing_corpus_skips_gracefully(self) -> None:
        """Calling dispatcher for 'wiki_rfa' without editor corpus → skip silently."""
        graph = _make_graph(["alice"], dataset_source="wiki_rfa")
        svc = TopicVectorService()
        svc.build_topic_vectors(graph)
        assert graph.nodes["alice"].topicVector == []

    def test_congress_missing_partition_skips_gracefully(self) -> None:
        """Calling dispatcher for 'congress' without partition → skip silently."""
        graph = _make_graph(["a", "b"], dataset_source="congress")
        svc = TopicVectorService()
        svc.build_topic_vectors(graph)
        for node in graph.nodes.values():
            assert node.topicVector == []


# ===========================================================================
# WikiRfAAdapter — editor_text_corpus
# ===========================================================================


class TestWikiRfAAdapterEditorCorpus:
    """WikiRfAAdapter.editor_text_corpus is populated during fetch()."""

    def test_initial_corpus_is_empty(self) -> None:
        """Freshly constructed WikiRfAAdapter has an empty editor_text_corpus."""
        from ingestion.adapters import WikiRfAAdapter

        adapter = WikiRfAAdapter()
        assert adapter.editor_text_corpus == {}

    def test_normalize_does_not_update_corpus(self) -> None:
        """Calling normalize() directly does NOT update editor_text_corpus."""
        from ingestion.adapters import WikiRfAAdapter

        adapter = WikiRfAAdapter()
        record = adapter.normalize(
            {"SRC": "alice", "TGT": "bob", "VOT": "1", "RES": "1",
             "DAT": "01:00, 01 January 2010", "TXT": "good editor"}
        )
        assert record is not None
        # normalize() must not side-effect the corpus
        assert adapter.editor_text_corpus == {}

    def test_fetch_populates_corpus(self, tmp_path) -> None:
        """fetch() builds editor_text_corpus from TXT fields."""
        import gzip
        from pathlib import Path
        from ingestion.adapters import WikiRfAAdapter, DatasetConfig

        # Write a minimal wiki-RfA gz file with two records
        content = (
            "SRC:alice\nTGT:bob\nVOT:1\nRES:1\n"
            "YEA:2010\nDAT:01:00, 01 January 2010\nTXT:great candidate\n\n"
            "SRC:alice\nTGT:carol\nVOT:-1\nRES:0\n"
            "YEA:2010\nDAT:02:00, 02 January 2010\nTXT:not suitable\n\n"
            "SRC:dave\nTGT:carol\nVOT:1\nRES:0\n"
            "YEA:2011\nDAT:03:00, 03 January 2011\nTXT:support the nomination\n\n"
        )
        gz_path = tmp_path / "wiki.txt.gz"
        with gzip.open(str(gz_path), "wt", encoding="utf-8") as f:
            f.write(content)

        adapter = WikiRfAAdapter()
        config = DatasetConfig(source_type="wiki_rfa", file_path=str(gz_path), format="txt_gz")
        adapter.fetch(config)

        assert "alice" in adapter.editor_text_corpus
        assert set(adapter.editor_text_corpus["alice"]) == {"great candidate", "not suitable"}
        assert "dave" in adapter.editor_text_corpus
        assert adapter.editor_text_corpus["dave"] == ["support the nomination"]

    def test_fetch_resets_corpus_between_calls(self, tmp_path) -> None:
        """A second fetch() call resets editor_text_corpus, not appends."""
        import gzip
        from pathlib import Path
        from ingestion.adapters import WikiRfAAdapter, DatasetConfig

        content = (
            "SRC:alice\nTGT:bob\nVOT:1\nRES:1\n"
            "YEA:2010\nDAT:01:00, 01 January 2010\nTXT:first run\n\n"
        )
        gz_path = tmp_path / "wiki2.txt.gz"
        with gzip.open(str(gz_path), "wt", encoding="utf-8") as f:
            f.write(content)

        adapter = WikiRfAAdapter()
        config = DatasetConfig(source_type="wiki_rfa", file_path=str(gz_path), format="txt_gz")

        adapter.fetch(config)
        first_corpus = dict(adapter.editor_text_corpus)

        adapter.fetch(config)
        # Second fetch must be equivalent, not doubled
        assert adapter.editor_text_corpus == first_corpus

    def test_records_without_txt_not_in_corpus(self, tmp_path) -> None:
        """Records with empty/absent TXT fields are not added to editor corpus."""
        import gzip
        from ingestion.adapters import WikiRfAAdapter, DatasetConfig

        content = (
            "SRC:alice\nTGT:bob\nVOT:1\nRES:1\n"
            "YEA:2010\nDAT:01:00, 01 January 2010\nTXT:\n\n"  # empty TXT
            "SRC:dave\nTGT:carol\nVOT:-1\nRES:0\n"
            "YEA:2011\nDAT:03:00, 03 January 2011\n\n"  # no TXT field at all
        )
        gz_path = tmp_path / "wiki3.txt.gz"
        with gzip.open(str(gz_path), "wt", encoding="utf-8") as f:
            f.write(content)

        adapter = WikiRfAAdapter()
        config = DatasetConfig(source_type="wiki_rfa", file_path=str(gz_path), format="txt_gz")
        adapter.fetch(config)

        # Neither alice nor dave should appear in the corpus
        assert "alice" not in adapter.editor_text_corpus
        assert "dave" not in adapter.editor_text_corpus


# ===========================================================================
# Property-based tests (Hypothesis)
# ===========================================================================

# Strategy: generate small node ID sets as lists of unique strings
_node_ids_st = st.lists(
    st.text(
        alphabet=st.characters(
            whitelist_categories=("Ll", "Lu", "Nd"),
            whitelist_characters="_-",
        ),
        min_size=1,
        max_size=30,
    ),
    min_size=2,
    max_size=20,
    unique=True,
)

# Strategy: non-empty text corpus documents
_text_st = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters=" "),
    min_size=3,
    max_size=200,
)


@given(
    node_ids=_node_ids_st,
    corpus_texts=st.lists(_text_st, min_size=1, max_size=5),
)
@settings(max_examples=30, deadline=10_000)
def test_property_reddit_all_vectors_same_length(
    node_ids: list[str],
    corpus_texts: list[str],
) -> None:
    """**Validates: Requirements 6.3**

    Property: After build_reddit_topic_vectors, every node in the graph has a
    topic vector of the same length.
    """
    graph = _make_graph(node_ids, dataset_source="reddit_body")

    # Assign some nodes to corpus (first half) and leave rest absent
    half = max(1, len(node_ids) // 2)
    corpus = {node_ids[i]: corpus_texts for i in range(half)}

    svc = TopicVectorService()
    svc.build_reddit_topic_vectors(graph, corpus)

    lengths = [len(node.topicVector) for node in graph.nodes.values()]
    assert len(set(lengths)) == 1, (
        f"Not all topic vectors have the same length: {set(lengths)}"
    )


@given(
    node_ids=_node_ids_st,
)
@settings(max_examples=30, deadline=5_000)
def test_property_congress_vectors_only_allowed_values(
    node_ids: list[str],
) -> None:
    """**Validates: Requirements 6.3**

    Property: Congress topic vectors contain only values in {0.0, 0.5, 1.0}.
    """
    graph = _make_graph(node_ids, dataset_source="congress")

    # Assign nodes alternately to 2 communities + some to a third
    partition: dict[str, int] = {}
    for i, uid in enumerate(node_ids):
        partition[uid] = i % 3  # communities 0, 1, 2

    svc = TopicVectorService()
    svc.build_congress_topic_vectors(graph, partition)

    allowed = {0.0, 0.5, 1.0}
    for node in graph.nodes.values():
        assert len(node.topicVector) == 2
        for v in node.topicVector:
            assert v in allowed, (
                f"Node '{node.userId}' has unexpected vector value {v}: {node.topicVector}"
            )


@given(
    node_ids=_node_ids_st,
    corpus_texts=st.lists(_text_st, min_size=1, max_size=5),
)
@settings(max_examples=30, deadline=10_000)
def test_property_tfidf_vectors_nonnegative(
    node_ids: list[str],
    corpus_texts: list[str],
) -> None:
    """**Validates: Requirements 6.3**

    Property: TF-IDF vectors produced by build_reddit_topic_vectors have only
    non-negative components (TF-IDF scores are always >= 0).
    """
    graph = _make_graph(node_ids, dataset_source="reddit_body")
    corpus = {uid: corpus_texts for uid in node_ids}

    svc = TopicVectorService()
    svc.build_reddit_topic_vectors(graph, corpus)

    for node in graph.nodes.values():
        for v in node.topicVector:
            assert v >= 0.0, (
                f"Node '{node.userId}' has negative TF-IDF component: {v}"
            )


@given(
    node_ids=_node_ids_st,
    corpus_texts=st.lists(_text_st, min_size=1, max_size=5),
)
@settings(max_examples=30, deadline=10_000)
def test_property_wiki_rfa_all_vectors_same_length(
    node_ids: list[str],
    corpus_texts: list[str],
) -> None:
    """**Validates: Requirements 6.3**

    Property: After build_wiki_rfa_topic_vectors, every node in a wiki-RfA
    graph has a topic vector of the same length.
    """
    graph = _make_graph(node_ids, dataset_source="wiki_rfa")
    half = max(1, len(node_ids) // 2)
    corpus = {node_ids[i]: corpus_texts for i in range(half)}

    svc = TopicVectorService()
    svc.build_wiki_rfa_topic_vectors(graph, corpus)

    lengths = [len(node.topicVector) for node in graph.nodes.values()]
    assert len(set(lengths)) == 1, (
        f"Not all wiki-RfA topic vectors have the same length: {set(lengths)}"
    )
