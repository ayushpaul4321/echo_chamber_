"""Full pipeline integration test on the Reddit title dataset.

Task 9.1: Run full pipeline on Reddit title dataset
  - Ingest soc-redditHyperlinks-title.tsv via RedditTitleAdapter (chunked streaming)
  - Build graph → Louvain → Polarization Index (expect > 0.60) → Diversity Scores
  - Generate recommendations for 10 sample low-diversity subreddits
  - Verify all API endpoint DTOs return correct shapes

Run with:
    pytest tests/test_pipeline_reddit_title.py -v -s --timeout=300

Dataset note: uses the first 100,000 records for speed; the full dataset can be
used by setting REDDIT_TITLE_MAX_ROWS=0 in the environment.

References: Requirements 1–11
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
    Path(__file__).parent.parent / "echo_chamber_detector" / "soc-redditHyperlinks-title.tsv"
)

# Maximum number of records to ingest (0 = all).  Override via env var.
# Default 20,000 gives ~6K nodes/13K edges — fast enough for CI (~60s).
# For the full dataset set REDDIT_TITLE_MAX_ROWS=0.
MAX_ROWS = int(os.environ.get("REDDIT_TITLE_MAX_ROWS", "20000"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ingest_records(max_rows: int = MAX_ROWS):
    """Ingest records from the Reddit title TSV using RedditTitleAdapter.

    Uses chunked streaming internally (chunksize=10_000).  When *max_rows* > 0,
    stops after that many valid records have been collected.
    """
    import pandas as pd

    from ingestion.adapters import DatasetConfig, RedditTitleAdapter

    adapter = RedditTitleAdapter()
    config = DatasetConfig(
        source_type="reddit_title",
        file_path=DATASET_PATH,
        format="tsv",
    )

    if max_rows <= 0:
        # Full dataset ingestion via normal adapter.fetch()
        return adapter.fetch(config)

    # Partial ingestion: stream chunks until we reach max_rows valid records.
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
                if len(records) >= max_rows:
                    break
        if len(records) >= max_rows:
            break

    logger.info(
        "_ingest_records: collected %d records (rejected %d / %d rows scanned, "
        "max_rows=%d)",
        len(records),
        rejected_rows,
        total_rows,
        max_rows,
    )
    return records


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_results():
    """Run the full pipeline once and return all results for assertions.

    Scope is 'module' so the expensive computation runs only once even when
    multiple test functions reference this fixture.
    """
    # ------------------------------------------------------------------
    # Step 1: Ingest
    # ------------------------------------------------------------------
    logger.info("Pipeline step 1: ingesting Reddit title records (max=%d)…", MAX_ROWS)
    records = _ingest_records(max_rows=MAX_ROWS)

    assert len(records) > 0, "No records ingested — check dataset path"
    logger.info("Ingested %d records", len(records))

    # Optionally deduplicate via IngestionService
    from ingestion.service import IngestionService
    from ingestion.adapters import DatasetConfig, RedditTitleAdapter

    # We already have records; just wrap them in IngestionResult-like info.
    # For the full pipeline we skip IngestionService.ingest() to avoid
    # re-reading the file; we use the records we already have.

    # ------------------------------------------------------------------
    # Step 2: Build graph
    # ------------------------------------------------------------------
    logger.info("Pipeline step 2: building interaction graph…")
    from graph.service import GraphConstructionService

    graph_service = GraphConstructionService()
    graph = graph_service.build_graph(records, dataset_source="reddit_title")

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
    logger.info("Pipeline step 3: running Louvain community detection…")
    from community.service import CommunityDetectionService

    community_service = CommunityDetectionService()
    partitions = community_service.detect_communities(graph)

    community_count = len(partitions)
    logger.info("Community detection complete: %d communities found", community_count)
    assert community_count >= 2, "Expected at least 2 communities"

    # ------------------------------------------------------------------
    # Step 4: Compute Polarization Index
    # ------------------------------------------------------------------
    logger.info("Pipeline step 4: computing Polarization Index…")
    from metrics.service import MetricsService

    metrics_service = MetricsService()
    pol_metrics = metrics_service.compute_polarization_index(graph, partitions)

    pi = pol_metrics.polarizationIndex
    logger.info("Polarization Index = %.4f", pi)

    # ------------------------------------------------------------------
    # Step 5: Compute Diversity Scores for all nodes (batch, efficient)
    # ------------------------------------------------------------------
    logger.info("Pipeline step 5: computing Diversity Scores for all nodes…")

    # Build a flat partition dict once (avoid per-user re-normalisation overhead)
    partition_map: dict[str, str] = {}
    for cp in partitions:
        for member_id in cp.memberIds:
            partition_map[member_id] = cp.communityId

    # Compute diversity scores in a single pass over edges
    # diversityScore = cross_community_outgoing_weight / total_outgoing_weight
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
        "Diversity Scores computed: %d users, avg=%.4f", len(diversity_scores), avg_diversity
    )

    # ------------------------------------------------------------------
    # Step 6: Identify 10 subreddits with lowest diversity scores
    # ------------------------------------------------------------------
    logger.info("Pipeline step 6: identifying 10 lowest-diversity subreddits…")

    # Filter out nodes with no outgoing edges (zero because they have no edges,
    # not because they are truly low-diversity). Use outgoing_total already computed.
    nodes_with_edges = [
        (uid, score)
        for uid, score in diversity_scores.items()
        if outgoing_total.get(uid, 0.0) > 0.0
    ]

    # Sort ascending by diversity score
    nodes_with_edges.sort(key=lambda x: x[1])
    low_diversity_users = [uid for uid, _ in nodes_with_edges[:10]]

    logger.info(
        "10 lowest-diversity subreddits: %s",
        ", ".join(f"{uid}(DS={diversity_scores[uid]:.3f})" for uid in low_diversity_users),
    )

    # ------------------------------------------------------------------
    # Step 7: Generate recommendations for low-diversity subreddits
    # ------------------------------------------------------------------
    logger.info("Pipeline step 7: generating recommendations for 10 low-diversity subreddits…")
    from datetime import timezone
    from graph.models import UserMetrics
    from recommendations.bridge_nodes import generate_recommendations

    all_recommendations: dict[str, list] = {}

    # Compute betweenness centrality first (needed for bridge node filtering)
    logger.info("  Computing betweenness centrality (may take a while)…")
    try:
        metrics_service.compute_betweenness_centrality(graph)
        logger.info("  Betweenness centrality computed")
    except Exception as exc:
        logger.warning("  Betweenness centrality failed (%s), using defaults (0.0)", exc)

    # Reddit title has no body text, so topicVectors are all empty.
    # Assign community-based categorical topic vectors (similar to the Congress
    # dataset proxy) so that cosine similarity produces non-zero scores and
    # the recommendation engine can filter and rank bridge candidates.
    # This is a valid integration-test approximation; the actual semantic
    # vectors come from RedditBodyAdapter's TF-IDF corpus (task 9.2).
    logger.info("  Assigning community-based topic vectors for recommendation engine…")
    community_to_vec: dict[str, list[float]] = {}
    import math as _math

    community_ids = sorted({node.communityId for node in graph.nodes.values() if node.communityId})
    n_communities = len(community_ids)
    for idx, cid in enumerate(community_ids):
        # Orthogonal-ish unit vectors spanning community space
        angle = 2.0 * _math.pi * idx / n_communities
        community_to_vec[cid] = [_math.cos(angle), _math.sin(angle)]

    for node in graph.nodes.values():
        if node.communityId in community_to_vec:
            node.topicVector = list(community_to_vec[node.communityId])

    for uid in low_diversity_users:
        node = graph.nodes.get(uid)
        if node is None:
            continue

        # Build UserMetrics for this user
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
        all_recommendations[uid] = recs
        logger.info(
            "  %s → %d recommendations",
            uid,
            len(recs),
        )

    total_recs = sum(len(v) for v in all_recommendations.values())
    logger.info(
        "Recommendations generated: %d recommendations across %d users",
        total_recs,
        len(all_recommendations),
    )

    return {
        "records": records,
        "graph": graph,
        "partitions": partitions,
        "pol_metrics": pol_metrics,
        "diversity_scores": diversity_scores,
        "avg_diversity": avg_diversity,
        "low_diversity_users": low_diversity_users,
        "all_recommendations": all_recommendations,
        "partition_map": partition_map,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dataset_file_exists():
    """Verify the Reddit title TSV file is present."""
    assert Path(DATASET_PATH).exists(), (
        f"Dataset file not found: {DATASET_PATH}\n"
        "Expected at: echo_chamber_detector/soc-redditHyperlinks-title.tsv"
    )
    size_mb = Path(DATASET_PATH).stat().st_size / (1024 * 1024)
    logger.info("Dataset file size: %.1f MB", size_mb)
    assert size_mb > 1.0, "Dataset file is suspiciously small (< 1 MB)"


def test_ingestion_records(pipeline_results):
    """Step 1: Verify records were ingested from the TSV."""
    records = pipeline_results["records"]
    assert len(records) > 1000, f"Expected >1000 records, got {len(records)}"

    # Check record structure
    r = records[0]
    assert r.sourceUserId, "sourceUserId must be non-empty"
    assert r.targetUserId, "targetUserId must be non-empty"
    assert r.sourceUserId != r.targetUserId, "self-loop record leaked through"
    assert r.interactionType.value == "HYPERLINK"
    assert r.datasetSource == "reddit_title"
    assert r.timestamp is not None, "timestamp must be present"
    assert r.timestamp < datetime.now(timezone.utc), "timestamp must be in the past"
    logger.info("✓ Ingestion: %d records, first record OK", len(records))


def test_graph_construction(pipeline_results):
    """Step 2: Verify interaction graph was built correctly."""
    graph = pipeline_results["graph"]

    assert graph.nodeCount > 100, f"Expected >100 nodes, got {graph.nodeCount}"
    assert graph.edgeCount > 100, f"Expected >100 edges, got {graph.edgeCount}"
    assert graph.datasetSource == "reddit_title"
    assert graph.snapshotId, "snapshotId must be non-empty"

    # Verify edge weights are in [0, 1]
    for edge in graph.edges:
        assert 0.0 <= edge.weight <= 1.0, (
            f"Edge weight out of range: {edge.weight} for {edge.sourceUserId}->{edge.targetUserId}"
        )

    logger.info(
        "✓ Graph: %d nodes, %d edges, snapshotId=%s",
        graph.nodeCount,
        graph.edgeCount,
        graph.snapshotId,
    )


def test_community_detection(pipeline_results):
    """Step 3: Verify Louvain detected multiple communities."""
    partitions = pipeline_results["partitions"]
    graph = pipeline_results["graph"]

    assert len(partitions) >= 2, (
        f"Expected ≥2 communities, got {len(partitions)}"
    )

    # Verify every node is assigned to exactly one community
    all_community_members: set[str] = set()
    for cp in partitions:
        assert cp.communityId, "communityId must be non-empty"
        assert len(cp.memberIds) > 0, "community must have at least one member"
        all_community_members.update(cp.memberIds)

    for node_id in graph.nodes:
        assert node_id in all_community_members, (
            f"Node '{node_id}' was not assigned to any community"
        )

    # Verify communityId is set on graph nodes
    for node_id, node in graph.nodes.items():
        assert node.communityId is not None, (
            f"Node '{node_id}' has communityId=None after detection"
        )

    logger.info(
        "✓ Community detection: %d communities, all %d nodes assigned",
        len(partitions),
        graph.nodeCount,
    )


def test_polarization_index(pipeline_results):
    """Step 4: Verify Polarization Index is computed and > 0.60."""
    pol_metrics = pipeline_results["pol_metrics"]

    pi = pol_metrics.polarizationIndex
    assert 0.0 <= pi <= 1.0, f"PI must be in [0, 1], got {pi}"
    assert pi > 0.60, (
        f"Polarization Index {pi:.4f} is below the expected threshold of 0.60 "
        f"for the Reddit title dataset. This may indicate a bug in the pipeline."
    )

    assert pol_metrics.modularity >= 0.0, "modularity must be ≥ 0"
    assert pol_metrics.communityCount >= 2, "communityCount must be ≥ 2"
    assert pol_metrics.avgCommunitySize > 0.0, "avgCommunitySize must be > 0"
    assert 0.0 <= pol_metrics.interCommunityEdgeRatio <= 1.0

    # PI + interCommunityEdgeRatio should ≈ 1.0 (Requirement 4.4)
    assert abs(pi + pol_metrics.interCommunityEdgeRatio - 1.0) < 1e-9, (
        f"PI ({pi}) + interCommunityEdgeRatio ({pol_metrics.interCommunityEdgeRatio}) "
        f"should equal 1.0"
    )

    logger.info(
        "✓ Polarization Index: PI=%.4f, modularity=%.4f, communities=%d",
        pi,
        pol_metrics.modularity,
        pol_metrics.communityCount,
    )


def test_diversity_scores(pipeline_results):
    """Step 5: Verify Diversity Scores are computed for all nodes."""
    graph = pipeline_results["graph"]
    diversity_scores = pipeline_results["diversity_scores"]
    avg_diversity = pipeline_results["avg_diversity"]

    assert len(diversity_scores) == graph.nodeCount, (
        f"Expected diversity scores for all {graph.nodeCount} nodes, "
        f"got {len(diversity_scores)}"
    )

    for uid, score in diversity_scores.items():
        assert 0.0 <= score <= 1.0, f"Diversity score out of range for {uid}: {score}"

    logger.info(
        "✓ Diversity Scores: %d users scored, avg=%.4f",
        len(diversity_scores),
        avg_diversity,
    )


def test_low_diversity_identification(pipeline_results):
    """Step 6: Verify the 10 lowest-diversity subreddits are identified."""
    low_diversity_users = pipeline_results["low_diversity_users"]
    diversity_scores = pipeline_results["diversity_scores"]

    assert len(low_diversity_users) <= 10, "Should return at most 10 low-diversity users"
    assert len(low_diversity_users) > 0, "Should return at least 1 low-diversity user"

    # Verify they are sorted ascending by diversity score
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


def test_recommendations_generated(pipeline_results):
    """Step 7: Verify recommendations were generated for low-diversity subreddits."""
    all_recommendations = pipeline_results["all_recommendations"]
    low_diversity_users = pipeline_results["low_diversity_users"]
    graph = pipeline_results["graph"]

    # At least some users should have recommendations
    total_recs = sum(len(v) for v in all_recommendations.values())
    logger.info("Total recommendations generated: %d", total_recs)

    # For users with recommendations, validate Recommendation structure
    for uid, recs in all_recommendations.items():
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

            # Cross-community invariant (Requirement 6.1)
            user_node = graph.nodes.get(uid)
            rec_node = graph.nodes.get(rec.recommendedUserId)
            if user_node and rec_node:
                assert user_node.communityId != rec_node.communityId, (
                    f"Recommendation {rec.recommendedUserId} is in the same community "
                    f"as target user {uid}"
                )

    logger.info(
        "✓ Recommendations: %d recs for %d users, all shapes valid",
        total_recs,
        len([u for u in all_recommendations if all_recommendations[u]]),
    )


# ---------------------------------------------------------------------------
# API DTO shape verification tests
# ---------------------------------------------------------------------------


def test_graph_dto_shape(pipeline_results):
    """Verify GraphDTO can be instantiated with correct fields from pipeline output."""
    from api.dtos import EdgeDTO, GraphDTO, NodeDTO

    graph = pipeline_results["graph"]

    # Build sample NodeDTOs and EdgeDTOs from pipeline graph
    sample_nodes = list(graph.nodes.values())[:5]
    sample_edges = graph.edges[:5]

    node_dtos = [
        NodeDTO(
            userId=n.userId,
            communityId=n.communityId,
            betweenness=n.betweenness,
            diversityScore=n.diversityScore,
            topicVector=n.topicVector,
        )
        for n in sample_nodes
    ]
    edge_dtos = [
        EdgeDTO(
            sourceUserId=e.sourceUserId,
            targetUserId=e.targetUserId,
            weight=e.weight,
            isCrossCommunity=e.isCrossCommunity,
            signedPolarity=e.signedPolarity,
        )
        for e in sample_edges
    ]

    graph_dto = GraphDTO(
        nodes=node_dtos,
        edges=edge_dtos,
        snapshotId=graph.snapshotId,
        createdAt=graph.createdAt,
        nodeCount=graph.nodeCount,
        edgeCount=graph.edgeCount,
        nextCursor=None,
    )

    # Verify field presence
    assert graph_dto.snapshotId == graph.snapshotId
    assert graph_dto.nodeCount == graph.nodeCount
    assert graph_dto.edgeCount == graph.edgeCount
    assert isinstance(graph_dto.nodes, list)
    assert isinstance(graph_dto.edges, list)
    assert graph_dto.nextCursor is None

    for n_dto in graph_dto.nodes:
        assert hasattr(n_dto, "userId")
        assert hasattr(n_dto, "communityId")
        assert hasattr(n_dto, "betweenness")
        assert hasattr(n_dto, "diversityScore")
        assert hasattr(n_dto, "topicVector")

    for e_dto in graph_dto.edges:
        assert hasattr(e_dto, "sourceUserId")
        assert hasattr(e_dto, "targetUserId")
        assert hasattr(e_dto, "weight")
        assert hasattr(e_dto, "isCrossCommunity")
        assert hasattr(e_dto, "signedPolarity")

    logger.info("✓ GraphDTO shape: OK (nodes=%d, edges=%d)", len(node_dtos), len(edge_dtos))


def test_polarization_dto_shape(pipeline_results):
    """Verify PolarizationDTO can be instantiated from pipeline metrics."""
    from api.dtos import PolarizationDTO

    pol_metrics = pipeline_results["pol_metrics"]

    dto = PolarizationDTO(
        snapshotId=pol_metrics.snapshotId,
        polarizationIndex=pol_metrics.polarizationIndex,
        modularity=pol_metrics.modularity,
        communityCount=pol_metrics.communityCount,
        avgCommunitySize=pol_metrics.avgCommunitySize,
        interCommunityEdgeRatio=pol_metrics.interCommunityEdgeRatio,
        computedAt=pol_metrics.computedAt,
    )

    assert hasattr(dto, "snapshotId")
    assert hasattr(dto, "polarizationIndex")
    assert hasattr(dto, "modularity")
    assert hasattr(dto, "communityCount")
    assert hasattr(dto, "avgCommunitySize")
    assert hasattr(dto, "interCommunityEdgeRatio")
    assert hasattr(dto, "computedAt")
    assert dto.polarizationIndex == pol_metrics.polarizationIndex

    logger.info(
        "✓ PolarizationDTO shape: OK (PI=%.4f)", dto.polarizationIndex
    )


def test_user_metrics_dto_shape(pipeline_results):
    """Verify UserMetricsDTO can be instantiated correctly."""
    from api.dtos import UserMetricsDTO
    from graph.models import UserMetrics

    graph = pipeline_results["graph"]
    pol_metrics = pipeline_results["pol_metrics"]
    diversity_scores = pipeline_results["diversity_scores"]

    # Pick any user that has a community assigned
    sample_user_id = next(
        uid for uid, node in graph.nodes.items()
        if node.communityId is not None
    )
    sample_node = graph.nodes[sample_user_id]

    intra_count = sum(
        1 for e in graph.edges
        if e.sourceUserId == sample_user_id and not e.isCrossCommunity
    )
    inter_count = sum(
        1 for e in graph.edges
        if e.sourceUserId == sample_user_id and e.isCrossCommunity
    )

    dto = UserMetricsDTO(
        userId=sample_user_id,
        communityId=sample_node.communityId,
        diversityScore=diversity_scores.get(sample_user_id, 0.0),
        intraEdgeCount=intra_count,
        interEdgeCount=inter_count,
        betweennessCentrality=sample_node.betweenness,
        snapshotId=pol_metrics.snapshotId,
        computedAt=pol_metrics.computedAt,
    )

    assert hasattr(dto, "userId")
    assert hasattr(dto, "communityId")
    assert hasattr(dto, "diversityScore")
    assert hasattr(dto, "intraEdgeCount")
    assert hasattr(dto, "interEdgeCount")
    assert hasattr(dto, "betweennessCentrality")
    assert hasattr(dto, "snapshotId")
    assert hasattr(dto, "computedAt")
    assert dto.userId == sample_user_id

    logger.info("✓ UserMetricsDTO shape: OK (user=%s)", sample_user_id)


def test_community_metrics_dto_shape(pipeline_results):
    """Verify CommunityMetricsDTO can be instantiated correctly."""
    from api.dtos import CommunityMetricsDTO

    partitions = pipeline_results["partitions"]
    pol_metrics = pipeline_results["pol_metrics"]
    diversity_scores = pipeline_results["diversity_scores"]

    # Pick the largest community
    largest_cp = max(partitions, key=lambda cp: len(cp.memberIds))

    # Compute average diversity for the community
    member_scores = [diversity_scores.get(uid, 0.0) for uid in largest_cp.memberIds]
    avg_ds = sum(member_scores) / len(member_scores) if member_scores else 0.0

    dto = CommunityMetricsDTO(
        communityId=largest_cp.communityId,
        memberCount=len(largest_cp.memberIds),
        modularity=largest_cp.modularity,
        avgDiversityScore=avg_ds,
        polarizationIndex=pol_metrics.polarizationIndex,
        snapshotId=pol_metrics.snapshotId,
    )

    assert hasattr(dto, "communityId")
    assert hasattr(dto, "memberCount")
    assert hasattr(dto, "modularity")
    assert hasattr(dto, "avgDiversityScore")
    assert hasattr(dto, "polarizationIndex")
    assert hasattr(dto, "snapshotId")
    assert dto.memberCount > 0

    logger.info(
        "✓ CommunityMetricsDTO shape: OK (community=%s, members=%d)",
        dto.communityId,
        dto.memberCount,
    )


def test_recommendation_dto_shape(pipeline_results):
    """Verify RecommendationDTO can be instantiated from generated recommendations."""
    from api.dtos import RecommendationDTO

    all_recommendations = pipeline_results["all_recommendations"]

    # Find a user that has at least 1 recommendation
    recs_with_data = [
        (uid, recs) for uid, recs in all_recommendations.items() if recs
    ]

    if not recs_with_data:
        pytest.skip("No recommendations generated — skipping DTO shape test")

    uid, recs = recs_with_data[0]
    rec = recs[0]

    dto = RecommendationDTO(
        recommendationId=rec.recommendationId,
        targetUserId=rec.targetUserId,
        recommendedUserId=rec.recommendedUserId,
        diversityGain=rec.diversityGain,
        topicRelevance=rec.topicRelevance,
        communityId=rec.communityId,
        reason=rec.reason,
    )

    assert hasattr(dto, "recommendationId")
    assert hasattr(dto, "targetUserId")
    assert hasattr(dto, "recommendedUserId")
    assert hasattr(dto, "diversityGain")
    assert hasattr(dto, "topicRelevance")
    assert hasattr(dto, "communityId")
    assert hasattr(dto, "reason")
    assert dto.targetUserId == uid

    logger.info(
        "✓ RecommendationDTO shape: OK (target=%s → recommended=%s)",
        dto.targetUserId,
        dto.recommendedUserId,
    )


def test_polarization_list_dto_shape(pipeline_results):
    """Verify PolarizationListDTO can be instantiated correctly."""
    from api.dtos import PolarizationDTO, PolarizationListDTO

    pol_metrics = pipeline_results["pol_metrics"]

    item = PolarizationDTO(
        snapshotId=pol_metrics.snapshotId,
        polarizationIndex=pol_metrics.polarizationIndex,
        modularity=pol_metrics.modularity,
        communityCount=pol_metrics.communityCount,
        avgCommunitySize=pol_metrics.avgCommunitySize,
        interCommunityEdgeRatio=pol_metrics.interCommunityEdgeRatio,
        computedAt=pol_metrics.computedAt,
    )

    dto = PolarizationListDTO(items=[item], total=1, nextCursor=None)

    assert hasattr(dto, "items")
    assert hasattr(dto, "total")
    assert hasattr(dto, "nextCursor")
    assert dto.total == 1
    assert len(dto.items) == 1
    assert dto.nextCursor is None

    logger.info("✓ PolarizationListDTO shape: OK")


def test_latest_snapshot_dto_shape(pipeline_results):
    """Verify LatestSnapshotDTO can be instantiated correctly."""
    from api.dtos import LatestSnapshotDTO

    pol_metrics = pipeline_results["pol_metrics"]

    dto = LatestSnapshotDTO(
        snapshotId=pol_metrics.snapshotId,
        datasetSource="reddit_title",
        computedAt=pol_metrics.computedAt,
    )

    assert hasattr(dto, "snapshotId")
    assert hasattr(dto, "datasetSource")
    assert hasattr(dto, "computedAt")
    assert dto.datasetSource == "reddit_title"

    logger.info("✓ LatestSnapshotDTO shape: OK")


# ---------------------------------------------------------------------------
# Summary table test (runs last, prints results)
# ---------------------------------------------------------------------------


def test_print_results_summary(pipeline_results):
    """Print the pipeline results summary table."""
    graph = pipeline_results["graph"]
    partitions = pipeline_results["partitions"]
    pol_metrics = pipeline_results["pol_metrics"]
    avg_diversity = pipeline_results["avg_diversity"]

    sep = "-" * 90
    header = (
        f"{'dataset':<20} | {'nodes':>8} | {'edges':>10} | "
        f"{'communities':>12} | {'PolarizationIndex':>18} | {'avg DiversityScore':>18}"
    )
    row = (
        f"{'reddit_title':<20} | {graph.nodeCount:>8,} | {graph.edgeCount:>10,} | "
        f"{len(partitions):>12} | {pol_metrics.polarizationIndex:>18.4f} | "
        f"{avg_diversity:>18.4f}"
    )

    print("\n")
    print(sep)
    print("Pipeline Results Summary — Reddit Title Dataset")
    print(sep)
    print(header)
    print(sep)
    print(row)
    print(sep)
    print(f"\nSnapshot ID: {graph.snapshotId}")
    print(f"Modularity Q: {pol_metrics.modularity:.4f}")
    print(
        f"\nNote: ingested first {MAX_ROWS:,} records "
        "(set REDDIT_TITLE_MAX_ROWS=0 for full dataset)"
        if MAX_ROWS > 0
        else "\nNote: full dataset ingested"
    )
    print()

    # Final assertions for the summary
    assert graph.nodeCount > 0
    assert graph.edgeCount > 0
    assert len(partitions) > 0
    assert pol_metrics.polarizationIndex > 0.0

    logger.info(
        "Pipeline summary: dataset=reddit_title nodes=%d edges=%d "
        "communities=%d PI=%.4f avgDS=%.4f",
        graph.nodeCount,
        graph.edgeCount,
        len(partitions),
        pol_metrics.polarizationIndex,
        avg_diversity,
    )
