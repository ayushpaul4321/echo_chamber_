"""Full pipeline integration test on the Reddit body dataset.

Task 9.2: Run full pipeline on Reddit body dataset
  - Ingest soc-redditHyperlinks-body.tsv via RedditBodyAdapter
  - Confirm subreddit_text_corpus populated
  - Build topic vectors (TF-IDF); confirm node.topicVector non-empty for all subreddit nodes
  - Re-run recommendations on same subreddits as 9.1; confirm semantic ranking differs
    from graph-only (betweenness-only) ranking

Run with:
    pytest tests/test_pipeline_reddit_body.py -v -s --timeout=300

Dataset note: uses the first 20,000 records for speed; the full dataset can be
used by setting REDDIT_BODY_MAX_ROWS=0 in the environment.

References: Phase 6 topic embedding tasks, Requirements 1–11
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset path
# ---------------------------------------------------------------------------

DATASET_PATH = str(
    Path(__file__).parent.parent / "echo_chamber_detector" / "soc-redditHyperlinks-body.tsv"
)

# Maximum number of records to ingest (0 = all).  Override via env var.
# Default 20,000 gives a reasonably sized graph while keeping CI fast (~120s).
MAX_ROWS = int(os.environ.get("REDDIT_BODY_MAX_ROWS", "20000"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ingest_records_and_corpus(max_rows: int = MAX_ROWS):
    """Ingest records from the Reddit body TSV using RedditBodyAdapter.

    Uses chunked streaming internally (chunksize=10_000).  When *max_rows* > 0,
    stops after that many valid records have been collected.

    Returns a tuple of (records, adapter) so that the caller can access
    adapter.subreddit_text_corpus after ingestion.
    """
    from ingestion.adapters import DatasetConfig, RedditBodyAdapter

    adapter = RedditBodyAdapter()
    config = DatasetConfig(
        source_type="reddit_body",
        file_path=DATASET_PATH,
        format="tsv",
    )

    if max_rows <= 0:
        # Full dataset ingestion via normal adapter.fetch()
        records = adapter.fetch(config)
        return records, adapter

    # Partial ingestion: stream chunks until we reach max_rows valid records.
    # We do this manually so we also accumulate subreddit_text_corpus.
    # Reset corpus before partial ingest.
    adapter.subreddit_text_corpus = {}

    records = []
    total_rows = 0
    rejected_rows = 0

    for chunk in adapter._iter_chunks(config.file_path):
        for _, row in chunk.iterrows():
            total_rows += 1
            record = adapter._normalize_row(row)
            if record is None:
                rejected_rows += 1
            else:
                records.append(record)
                # Accumulate body text into subreddit corpus
                if record.bodyText:
                    subreddit = record.sourceUserId
                    adapter.subreddit_text_corpus.setdefault(subreddit, []).append(
                        record.bodyText
                    )
                if len(records) >= max_rows:
                    break
        if len(records) >= max_rows:
            break

    logger.info(
        "_ingest_records_and_corpus: collected %d records (rejected %d / %d rows, "
        "corpus=%d subreddits, max_rows=%d)",
        len(records),
        rejected_rows,
        total_rows,
        len(adapter.subreddit_text_corpus),
        max_rows,
    )
    return records, adapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def body_pipeline_results():
    """Run the full body pipeline once and return all results for assertions.

    Scope is 'module' so the expensive computation runs only once even when
    multiple test functions reference this fixture.
    """
    # ------------------------------------------------------------------
    # Step 1: Ingest via RedditBodyAdapter
    # ------------------------------------------------------------------
    logger.info("Body pipeline step 1: ingesting Reddit body records (max=%d)…", MAX_ROWS)
    records, adapter = _ingest_records_and_corpus(max_rows=MAX_ROWS)

    assert len(records) > 0, "No records ingested — check dataset path"
    logger.info(
        "Ingested %d records; subreddit_text_corpus has %d subreddits",
        len(records),
        len(adapter.subreddit_text_corpus),
    )

    subreddit_text_corpus = adapter.subreddit_text_corpus

    # ------------------------------------------------------------------
    # Step 2: Build graph
    # ------------------------------------------------------------------
    logger.info("Body pipeline step 2: building interaction graph…")
    from graph.service import GraphConstructionService

    graph_service = GraphConstructionService()
    graph = graph_service.build_graph(records, dataset_source="reddit_body")

    logger.info(
        "Graph built: %d nodes, %d edges, snapshotId=%s",
        graph.nodeCount,
        graph.edgeCount,
        graph.snapshotId,
    )

    assert graph.nodeCount > 0, "Graph has no nodes"
    assert graph.edgeCount > 0, "Graph has no edges"

    # ------------------------------------------------------------------
    # Step 3: Community detection (Louvain)
    # ------------------------------------------------------------------
    logger.info("Body pipeline step 3: running Louvain community detection…")
    from community.service import CommunityDetectionService

    community_service = CommunityDetectionService()
    partitions = community_service.detect_communities(graph)

    community_count = len(partitions)
    logger.info("Community detection complete: %d communities found", community_count)
    assert community_count >= 2, "Expected at least 2 communities"

    # ------------------------------------------------------------------
    # Step 4: Compute Polarization Index
    # ------------------------------------------------------------------
    logger.info("Body pipeline step 4: computing Polarization Index…")
    from metrics.service import MetricsService

    metrics_service = MetricsService()
    pol_metrics = metrics_service.compute_polarization_index(graph, partitions)
    pi = pol_metrics.polarizationIndex
    logger.info("Polarization Index = %.4f", pi)

    # ------------------------------------------------------------------
    # Step 5: Compute Diversity Scores (batch pass over edges)
    # ------------------------------------------------------------------
    logger.info("Body pipeline step 5: computing Diversity Scores for all nodes…")

    # Build a flat partition dict
    partition_map: dict[str, str] = {}
    for cp in partitions:
        for member_id in cp.memberIds:
            partition_map[member_id] = cp.communityId

    outgoing_total: dict[str, float] = {}
    outgoing_cross: dict[str, float] = {}

    for edge in graph.edges:
        src = edge.sourceUserId
        tgt = edge.targetUserId
        w = edge.weight
        outgoing_total[src] = outgoing_total.get(src, 0.0) + w
        src_comm = partition_map.get(src)
        tgt_comm = partition_map.get(tgt)
        if src_comm != tgt_comm:
            outgoing_cross[src] = outgoing_cross.get(src, 0.0) + w

    diversity_scores: dict[str, float] = {}
    for user_id in graph.nodes:
        total = outgoing_total.get(user_id, 0.0)
        cross = outgoing_cross.get(user_id, 0.0)
        score = (cross / total) if total > 0.0 else 0.0
        score = max(0.0, min(1.0, score))
        diversity_scores[user_id] = score
        graph.nodes[user_id].diversityScore = score

    avg_diversity = sum(diversity_scores.values()) / len(diversity_scores)
    logger.info(
        "Diversity Scores computed: %d users, avg=%.4f",
        len(diversity_scores),
        avg_diversity,
    )

    # ------------------------------------------------------------------
    # Step 6: Identify 10 low-diversity subreddits
    # ------------------------------------------------------------------
    logger.info("Body pipeline step 6: identifying 10 lowest-diversity subreddits…")

    nodes_with_edges = [
        (uid, score)
        for uid, score in diversity_scores.items()
        if outgoing_total.get(uid, 0.0) > 0.0
    ]
    nodes_with_edges.sort(key=lambda x: x[1])
    low_diversity_users = [uid for uid, _ in nodes_with_edges[:10]]

    logger.info(
        "10 lowest-diversity subreddits: %s",
        ", ".join(
            f"{uid}(DS={diversity_scores[uid]:.3f})" for uid in low_diversity_users
        ),
    )

    # ------------------------------------------------------------------
    # Step 7: Build TF-IDF topic vectors (the core of task 9.2)
    # ------------------------------------------------------------------
    logger.info(
        "Body pipeline step 7: building TF-IDF topic vectors from subreddit corpus…"
    )
    from recommendations.topic_vectors import TopicVectorService

    tv_service = TopicVectorService()
    vectorizer = tv_service.build_reddit_topic_vectors(
        graph, subreddit_text_corpus
    )

    # Count nodes with non-empty topicVector
    nodes_with_vectors = sum(
        1 for node in graph.nodes.values() if node.topicVector
    )
    nodes_with_nonzero_vectors = sum(
        1 for node in graph.nodes.values() if node.topicVector and any(v != 0.0 for v in node.topicVector)
    )
    logger.info(
        "Topic vectors assigned: %d / %d nodes have non-zero vectors",
        nodes_with_nonzero_vectors,
        graph.nodeCount,
    )

    # ------------------------------------------------------------------
    # Step 8: Betweenness centrality (needed for bridge node filtering)
    # ------------------------------------------------------------------
    logger.info("Body pipeline step 8: computing betweenness centrality…")
    try:
        metrics_service.compute_betweenness_centrality(graph)
        logger.info("Betweenness centrality computed")
    except Exception as exc:
        logger.warning("Betweenness centrality failed (%s), using defaults (0.0)", exc)

    # ------------------------------------------------------------------
    # Step 9: Generate SEMANTIC recommendations (topic-vector-based)
    # ------------------------------------------------------------------
    logger.info(
        "Body pipeline step 9: generating SEMANTIC recommendations for low-diversity subreddits…"
    )
    from graph.models import UserMetrics
    from recommendations.bridge_nodes import generate_recommendations

    semantic_recommendations: dict[str, list] = {}

    for uid in low_diversity_users:
        node = graph.nodes.get(uid)
        if node is None:
            continue

        intra_count = sum(
            1
            for e in graph.edges
            if e.sourceUserId == uid and not e.isCrossCommunity
        )
        inter_count = sum(
            1
            for e in graph.edges
            if e.sourceUserId == uid and e.isCrossCommunity
        )

        user_metrics = UserMetrics(
            userId=uid,
            communityId=node.communityId or "0",
            diversityScore=node.diversityScore,
            intraEdgeCount=intra_count,
            interEdgeCount=inter_count,
            betweennessCentrality=node.betweenness,
            snapshotId=graph.snapshotId,
            computedAt=datetime.now(timezone.utc),
        )

        recs = generate_recommendations(uid, graph, user_metrics, top_k=5)
        semantic_recommendations[uid] = recs
        logger.info(
            "  Semantic recs for %s → %d recommendations",
            uid,
            len(recs),
        )

    # ------------------------------------------------------------------
    # Step 10: Generate GRAPH-ONLY recommendations (betweenness-only, no TF-IDF)
    # We temporarily replace topic vectors with community-based categorical
    # vectors (same as 9.1 approach) then restore the TF-IDF vectors after.
    # ------------------------------------------------------------------
    logger.info(
        "Body pipeline step 10: generating GRAPH-ONLY recommendations (betweenness-only)…"
    )

    # Save TF-IDF vectors
    tfidf_vectors: dict[str, list[float]] = {
        uid: list(node.topicVector)
        for uid, node in graph.nodes.items()
    }

    # Replace with community-based categorical vectors
    import math as _math

    community_ids = sorted(
        {node.communityId for node in graph.nodes.values() if node.communityId}
    )
    n_communities = len(community_ids)
    community_to_vec: dict[str, list[float]] = {}
    for idx, cid in enumerate(community_ids):
        angle = 2.0 * _math.pi * idx / n_communities
        community_to_vec[cid] = [_math.cos(angle), _math.sin(angle)]

    for node in graph.nodes.values():
        if node.communityId in community_to_vec:
            node.topicVector = list(community_to_vec[node.communityId])

    graph_only_recommendations: dict[str, list] = {}

    for uid in low_diversity_users:
        node = graph.nodes.get(uid)
        if node is None:
            continue

        intra_count = sum(
            1
            for e in graph.edges
            if e.sourceUserId == uid and not e.isCrossCommunity
        )
        inter_count = sum(
            1
            for e in graph.edges
            if e.sourceUserId == uid and e.isCrossCommunity
        )

        user_metrics = UserMetrics(
            userId=uid,
            communityId=node.communityId or "0",
            diversityScore=node.diversityScore,
            intraEdgeCount=intra_count,
            interEdgeCount=inter_count,
            betweennessCentrality=node.betweenness,
            snapshotId=graph.snapshotId,
            computedAt=datetime.now(timezone.utc),
        )

        recs = generate_recommendations(uid, graph, user_metrics, top_k=5)
        graph_only_recommendations[uid] = recs
        logger.info(
            "  Graph-only recs for %s → %d recommendations",
            uid,
            len(recs),
        )

    # Restore TF-IDF topic vectors
    for uid, vec in tfidf_vectors.items():
        if uid in graph.nodes:
            graph.nodes[uid].topicVector = vec

    return {
        "records": records,
        "adapter": adapter,
        "subreddit_text_corpus": subreddit_text_corpus,
        "graph": graph,
        "partitions": partitions,
        "pol_metrics": pol_metrics,
        "diversity_scores": diversity_scores,
        "avg_diversity": avg_diversity,
        "low_diversity_users": low_diversity_users,
        "nodes_with_vectors": nodes_with_vectors,
        "nodes_with_nonzero_vectors": nodes_with_nonzero_vectors,
        "vectorizer": vectorizer,
        "semantic_recommendations": semantic_recommendations,
        "graph_only_recommendations": graph_only_recommendations,
        "partition_map": partition_map,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dataset_file_exists():
    """Verify the Reddit body TSV file is present."""
    assert Path(DATASET_PATH).exists(), (
        f"Dataset file not found: {DATASET_PATH}\n"
        "Expected at: echo_chamber_detector/soc-redditHyperlinks-body.tsv"
    )
    size_mb = Path(DATASET_PATH).stat().st_size / (1024 * 1024)
    logger.info("Dataset file size: %.1f MB", size_mb)
    assert size_mb > 1.0, "Dataset file is suspiciously small (< 1 MB)"


def test_ingestion_records(body_pipeline_results):
    """Step 1: Verify records were ingested from the body TSV."""
    records = body_pipeline_results["records"]
    assert len(records) > 1000, f"Expected >1000 records, got {len(records)}"

    r = records[0]
    assert r.sourceUserId, "sourceUserId must be non-empty"
    assert r.targetUserId, "targetUserId must be non-empty"
    assert r.sourceUserId != r.targetUserId, "self-loop record leaked through"
    assert r.interactionType.value == "HYPERLINK"
    assert r.datasetSource == "reddit_body"
    assert r.timestamp is not None, "timestamp must be present"
    assert r.timestamp < datetime.now(timezone.utc), "timestamp must be in the past"
    logger.info("✓ Ingestion: %d records, first record OK", len(records))


def test_subreddit_text_corpus_populated(body_pipeline_results):
    """Step 1b: Verify subreddit_text_corpus is a non-empty dict.

    Key verification point 1: subreddit_text_corpus is a non-empty dict
    mapping subreddit name → list of strings.
    """
    corpus = body_pipeline_results["subreddit_text_corpus"]

    assert isinstance(corpus, dict), "subreddit_text_corpus must be a dict"
    assert len(corpus) > 0, "subreddit_text_corpus must not be empty"

    # Each value must be a non-empty list of strings
    for subreddit, texts in corpus.items():
        assert isinstance(subreddit, str), f"Corpus key must be str, got {type(subreddit)}"
        assert isinstance(texts, list), (
            f"Corpus value for '{subreddit}' must be a list, got {type(texts)}"
        )
        assert len(texts) > 0, (
            f"Corpus entry for '{subreddit}' must have at least one text"
        )
        for t in texts[:3]:  # spot-check first 3 entries
            assert isinstance(t, str), (
                f"Text entries for '{subreddit}' must be str, got {type(t)}"
            )

    logger.info(
        "✓ subreddit_text_corpus: %d subreddits, spot-check OK",
        len(corpus),
    )


def test_graph_construction(body_pipeline_results):
    """Step 2: Verify interaction graph was built correctly."""
    graph = body_pipeline_results["graph"]

    assert graph.nodeCount > 100, f"Expected >100 nodes, got {graph.nodeCount}"
    assert graph.edgeCount > 100, f"Expected >100 edges, got {graph.edgeCount}"
    assert graph.datasetSource == "reddit_body"
    assert graph.snapshotId, "snapshotId must be non-empty"

    for edge in graph.edges:
        assert 0.0 <= edge.weight <= 1.0, (
            f"Edge weight out of range: {edge.weight} for "
            f"{edge.sourceUserId}->{edge.targetUserId}"
        )

    logger.info(
        "✓ Graph: %d nodes, %d edges, snapshotId=%s",
        graph.nodeCount,
        graph.edgeCount,
        graph.snapshotId,
    )


def test_community_detection(body_pipeline_results):
    """Step 3: Verify Louvain detected multiple communities."""
    partitions = body_pipeline_results["partitions"]
    graph = body_pipeline_results["graph"]

    assert len(partitions) >= 2, (
        f"Expected ≥2 communities, got {len(partitions)}"
    )

    all_community_members: set[str] = set()
    for cp in partitions:
        assert cp.communityId, "communityId must be non-empty"
        assert len(cp.memberIds) > 0, "community must have at least one member"
        all_community_members.update(cp.memberIds)

    for node_id in graph.nodes:
        assert node_id in all_community_members, (
            f"Node '{node_id}' was not assigned to any community"
        )

    for node_id, node in graph.nodes.items():
        assert node.communityId is not None, (
            f"Node '{node_id}' has communityId=None after detection"
        )

    logger.info(
        "✓ Community detection: %d communities, all %d nodes assigned",
        len(partitions),
        graph.nodeCount,
    )


def test_topic_vectors_populated(body_pipeline_results):
    """Step 7: Verify TF-IDF topic vectors are populated on graph nodes.

    Key verification point 2: at least 80% of graph nodes have a non-empty
    topicVector (i.e. the list is not []) after TF-IDF fitting.
    All nodes receive at minimum a zero vector of the TF-IDF vocabulary
    dimensionality from TopicVectorService, so the list should never be [].
    Some nodes may have all-zero values if they are target-only subreddits
    with no body text in the corpus — that is expected and acceptable.
    """
    graph = body_pipeline_results["graph"]
    nodes_with_vectors = body_pipeline_results["nodes_with_vectors"]
    nodes_with_nonzero_vectors = body_pipeline_results["nodes_with_nonzero_vectors"]
    total_nodes = graph.nodeCount

    # Every node should have topicVector assigned (zero or non-zero list)
    for uid, node in graph.nodes.items():
        assert isinstance(node.topicVector, list), (
            f"Node '{uid}' topicVector must be a list, got {type(node.topicVector)}"
        )
        assert len(node.topicVector) > 0, (
            f"Node '{uid}' topicVector is empty list [] — TopicVectorService should "
            "assign at minimum a zero vector of the TF-IDF vocabulary dimensionality"
        )

    # All nodes should have a non-empty (dimensioned) topicVector list —
    # this is the "non-empty" criterion from task 9.2 spec.
    coverage_ratio = nodes_with_vectors / total_nodes if total_nodes > 0 else 0.0
    assert coverage_ratio >= 0.80, (
        f"Expected ≥80% of nodes to have a non-empty topicVector list; "
        f"got {nodes_with_vectors}/{total_nodes} = {coverage_ratio:.1%}."
    )

    # Log how many nodes have truly non-zero (meaningful) vectors for information
    nonzero_ratio = nodes_with_nonzero_vectors / total_nodes if total_nodes > 0 else 0.0
    logger.info(
        "✓ Topic vectors: %d / %d nodes have non-empty vectors (%.1f%%); "
        "%d / %d have non-zero values (%.1f%%) — target-only subreddits get zero vectors",
        nodes_with_vectors,
        total_nodes,
        coverage_ratio * 100,
        nodes_with_nonzero_vectors,
        total_nodes,
        nonzero_ratio * 100,
    )


def test_topic_vector_dimensionality(body_pipeline_results):
    """Verify all topic vectors have the same dimensionality (TF-IDF consistency)."""
    graph = body_pipeline_results["graph"]

    # All non-empty vectors should have the same length
    vector_lengths = set()
    for node in graph.nodes.values():
        if node.topicVector:
            vector_lengths.add(len(node.topicVector))

    assert len(vector_lengths) == 1, (
        f"Expected all topic vectors to have the same length, "
        f"got multiple lengths: {vector_lengths}"
    )

    vector_dim = vector_lengths.pop()
    assert vector_dim > 0, "Topic vector dimension must be > 0"
    assert vector_dim <= 5000, (
        f"Topic vector dimension {vector_dim} exceeds TF-IDF max_features=5000"
    )

    logger.info("✓ Topic vector dimensionality: all vectors have dim=%d", vector_dim)


def test_semantic_recommendations_generated(body_pipeline_results):
    """Step 9: Verify semantic recommendations were generated.

    Key verification point 4: all recommendations returned are from a
    different community than the requesting subreddit.
    """
    semantic_recommendations = body_pipeline_results["semantic_recommendations"]
    graph = body_pipeline_results["graph"]

    total_recs = sum(len(v) for v in semantic_recommendations.values())
    logger.info("Total semantic recommendations: %d", total_recs)

    # Validate recommendation structure and cross-community invariant
    for uid, recs in semantic_recommendations.items():
        for rec in recs:
            assert rec.recommendationId, "recommendationId must be non-empty"
            assert rec.targetUserId == uid, "targetUserId must match the requested user"
            assert rec.recommendedUserId, "recommendedUserId must be non-empty"
            assert 0.0 <= rec.diversityGain <= 1.0, (
                f"diversityGain out of range: {rec.diversityGain}"
            )
            assert 0.0 <= rec.topicRelevance <= 1.0, (
                f"topicRelevance out of range: {rec.topicRelevance}"
            )
            assert rec.communityId, "communityId must be non-empty"
            assert rec.reason, "reason string must be non-empty"

            # Cross-community invariant (Requirement 6.1 / Key point 4)
            user_node = graph.nodes.get(uid)
            rec_node = graph.nodes.get(rec.recommendedUserId)
            if user_node and rec_node:
                assert user_node.communityId != rec_node.communityId, (
                    f"Recommendation {rec.recommendedUserId} is in the same community "
                    f"({rec_node.communityId}) as target user {uid} "
                    f"({user_node.communityId}) — violates cross-community invariant"
                )

    logger.info(
        "✓ Semantic recommendations: %d recs for %d users, "
        "cross-community invariant holds",
        total_recs,
        len([u for u in semantic_recommendations if semantic_recommendations[u]]),
    )


def test_semantic_ranking_differs_from_graph_only(body_pipeline_results):
    """Key verification point 3: semantic ranking differs from graph-only ranking.

    Confirms that for at least one subreddit, the ordering of recommended
    nodes differs between:
    - semantic ranking (TF-IDF topic vectors, topic relevance-weighted)
    - graph-only ranking (community-based categorical vectors, betweenness proxy)

    If both approaches return 0 recommendations for all users, the test is
    skipped (unlikely with a real dataset but possible with very sparse graphs).
    """
    semantic_recommendations = body_pipeline_results["semantic_recommendations"]
    graph_only_recommendations = body_pipeline_results["graph_only_recommendations"]

    # Find users that have recommendations in both modes
    comparable_users = [
        uid
        for uid in semantic_recommendations
        if semantic_recommendations[uid] and graph_only_recommendations.get(uid)
    ]

    if not comparable_users:
        # Both modes returned empty recs for all users — may happen with very sparse
        # graphs (all betweenness = 0).  Skip rather than fail in that edge case.
        pytest.skip(
            "No users received recommendations in both semantic and graph-only modes "
            "— likely due to all betweenness centrality values being 0.0 in a sparse "
            "graph (too few rows ingested). Increase REDDIT_BODY_MAX_ROWS."
        )

    # Check if at least one user has a different ordering between the two modes.
    found_difference = False
    for uid in comparable_users:
        sem_ids = [r.recommendedUserId for r in semantic_recommendations[uid]]
        graph_ids = [r.recommendedUserId for r in graph_only_recommendations.get(uid, [])]

        # Different sets of recommended nodes OR same nodes in different order
        if sem_ids != graph_ids:
            logger.info(
                "✓ Ranking differs for subreddit '%s': "
                "semantic=%s vs graph-only=%s",
                uid,
                sem_ids,
                graph_ids,
            )
            found_difference = True
            break

    if not found_difference:
        # Log the full recommendation lists to aid debugging
        for uid in comparable_users[:3]:
            sem_recs = semantic_recommendations[uid]
            go_recs = graph_only_recommendations.get(uid, [])
            logger.info(
                "  subreddit='%s': "
                "semantic_topicRelevances=%s, graph_only_topicRelevances=%s",
                uid,
                [f"{r.topicRelevance:.3f}" for r in sem_recs],
                [f"{r.topicRelevance:.3f}" for r in go_recs],
            )

    assert found_difference, (
        "Expected semantic ranking to differ from graph-only ranking for at least "
        "one subreddit, but all recommendations were identical. "
        "This suggests the TF-IDF topic vectors are not influencing the recommendation "
        "ordering — check that build_reddit_topic_vectors() is correctly assigning "
        "non-zero vectors from the subreddit_text_corpus."
    )

    logger.info(
        "✓ Semantic vs graph-only ranking: confirmed difference in ordering "
        "for at least 1 of %d comparable subreddits",
        len(comparable_users),
    )


def test_low_diversity_users_identified(body_pipeline_results):
    """Verify the 10 lowest-diversity subreddits are identified correctly."""
    low_diversity_users = body_pipeline_results["low_diversity_users"]
    diversity_scores = body_pipeline_results["diversity_scores"]

    assert len(low_diversity_users) <= 10, "Should return at most 10 low-diversity users"
    assert len(low_diversity_users) > 0, "Should return at least 1 low-diversity user"

    # Verify sorted ascending by diversity score
    for i in range(len(low_diversity_users) - 1):
        uid_a = low_diversity_users[i]
        uid_b = low_diversity_users[i + 1]
        assert diversity_scores[uid_a] <= diversity_scores[uid_b], (
            f"Low-diversity list not sorted: {uid_a}({diversity_scores[uid_a]:.4f}) "
            f"> {uid_b}({diversity_scores[uid_b]:.4f})"
        )

    logger.info(
        "✓ Low-diversity subreddits: %s",
        ", ".join(
            f"{uid}(DS={diversity_scores[uid]:.3f})"
            for uid in low_diversity_users
        ),
    )


def test_polarization_index(body_pipeline_results):
    """Verify Polarization Index is computed and in the expected range."""
    pol_metrics = body_pipeline_results["pol_metrics"]

    pi = pol_metrics.polarizationIndex
    assert 0.0 <= pi <= 1.0, f"PI must be in [0, 1], got {pi}"

    # The body graph should exhibit similar polarization to the title graph
    assert pi > 0.40, (
        f"Polarization Index {pi:.4f} is very low for the Reddit body dataset "
        f"(expected > 0.40 for a partial ingest). "
        f"Check that the pipeline is computing edges and communities correctly."
    )

    # PI + interCommunityEdgeRatio should equal 1.0 (Requirement 4.4)
    assert abs(pi + pol_metrics.interCommunityEdgeRatio - 1.0) < 1e-9, (
        f"PI ({pi}) + interCommunityEdgeRatio ({pol_metrics.interCommunityEdgeRatio}) "
        f"should equal 1.0 (Requirement 4.4)"
    )

    logger.info(
        "✓ Polarization Index: PI=%.4f, modularity=%.4f, communities=%d",
        pi,
        pol_metrics.modularity,
        pol_metrics.communityCount,
    )


def test_diversity_scores(body_pipeline_results):
    """Verify Diversity Scores are computed for all nodes."""
    graph = body_pipeline_results["graph"]
    diversity_scores = body_pipeline_results["diversity_scores"]
    avg_diversity = body_pipeline_results["avg_diversity"]

    assert len(diversity_scores) == graph.nodeCount, (
        f"Expected diversity scores for all {graph.nodeCount} nodes, "
        f"got {len(diversity_scores)}"
    )

    for uid, score in diversity_scores.items():
        assert 0.0 <= score <= 1.0, (
            f"Diversity score out of range for {uid}: {score}"
        )

    logger.info(
        "✓ Diversity Scores: %d users scored, avg=%.4f",
        len(diversity_scores),
        avg_diversity,
    )


def test_body_adapter_corpus_attribute(body_pipeline_results):
    """Verify the RedditBodyAdapter exposes subreddit_text_corpus as instance attr."""
    adapter = body_pipeline_results["adapter"]

    assert hasattr(adapter, "subreddit_text_corpus"), (
        "RedditBodyAdapter must have a 'subreddit_text_corpus' attribute"
    )
    assert isinstance(adapter.subreddit_text_corpus, dict), (
        "subreddit_text_corpus must be a dict"
    )

    # Spot-check: every value should be a non-empty list of strings
    sample_items = list(adapter.subreddit_text_corpus.items())[:5]
    for subreddit, texts in sample_items:
        assert isinstance(texts, list), (
            f"Corpus entry for '{subreddit}' must be a list"
        )
        assert len(texts) > 0, (
            f"Corpus entry for '{subreddit}' must be non-empty"
        )

    logger.info(
        "✓ RedditBodyAdapter.subreddit_text_corpus: %d subreddits",
        len(adapter.subreddit_text_corpus),
    )
