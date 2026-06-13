"""Bridge Node Identification and Candidate Scoring for the Echo Chamber Detector.

Identifies cross-community bridge nodes that are topically relevant to a
given user, and scores them for recommendation purposes.

Algorithm 6 (design.md):

1. Filter candidates from *other* communities whose betweenness centrality
   exceeds ``BRIDGE_CENTRALITY_THRESHOLD``.
2. Score each candidate by ``cosine_similarity(user_topic_vector,
   candidate.topicVector)``; exclude candidates below
   ``MIN_TOPIC_RELEVANCE_THRESHOLD``.
3. Wiki-RfA extra filter: exclude candidates from communities whose
   ``net_sentiment_index < 0`` (hostile communities).
4. For users with fewer than ``SPARSE_USER_INTERACTION_THRESHOLD``
   interactions, fall back to the community centroid topic vector (Req 6.6).
5. Estimate diversity gain analytically for each candidate (task 6.3).
6. Sort candidates descending by diversity gain; return ≤ topK Recommendation
   objects with human-readable reason strings (task 6.3).

References: Requirements 6.2, 6.3, 6.4, 6.5, 6.6, 6.8; design.md Algorithm 6.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Optional

from graph.models import InteractionGraph, Node, Recommendation, SignedMetrics, UserMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BRIDGE_CENTRALITY_THRESHOLD: float = 0.01
MIN_TOPIC_RELEVANCE_THRESHOLD: float = 0.1
SPARSE_USER_INTERACTION_THRESHOLD: int = 5  # users with fewer interactions use centroid fallback


# ---------------------------------------------------------------------------
# BridgeNodeService
# ---------------------------------------------------------------------------


class BridgeNodeService:
    """Identifies bridge node candidates and scores them by topic relevance.

    This service is stateless; no ``__init__`` is required.

    Usage::

        service = BridgeNodeService()
        candidates = service.identify_bridge_candidates(user_id, graph)
        # Returns list of (Node, relevance_score) sorted descending by score.
    """

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
        """Compute the cosine similarity between two vectors.

        Returns the dot product divided by the product of magnitudes.
        Returns 0.0 when either vector is empty or has zero magnitude (i.e.
        the zero vector), which avoids division-by-zero.

        For non-negative topic vectors (e.g. TF-IDF outputs) the result is
        in [0, 1].  For arbitrary real-valued vectors it is in [-1, 1].

        Args:
            vector_a: First vector as a list of floats.
            vector_b: Second vector as a list of floats.

        Returns:
            Cosine similarity in [-1, 1], or 0.0 if either vector is
            empty or has zero magnitude.
        """
        if not vector_a or not vector_b:
            return 0.0

        # Truncate to the shorter length so mismatched dimensions don't crash
        length = min(len(vector_a), len(vector_b))
        a = vector_a[:length]
        b = vector_b[:length]

        dot_product = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))

        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0

        return dot_product / (mag_a * mag_b)

    # ------------------------------------------------------------------
    # Interaction count helper
    # ------------------------------------------------------------------

    def get_user_interaction_count(
        self, user_id: str, graph: InteractionGraph
    ) -> int:
        """Count the number of outgoing edges for a user in the graph.

        Args:
            user_id: The userId to look up.
            graph:   The :class:`~graph.models.InteractionGraph` to search.

        Returns:
            Number of edges whose ``sourceUserId`` equals ``user_id``.
        """
        return sum(1 for edge in graph.edges if edge.sourceUserId == user_id)

    # ------------------------------------------------------------------
    # Community centroid helper
    # ------------------------------------------------------------------

    def compute_community_centroid(
        self, community_id: str, graph: InteractionGraph
    ) -> list[float]:
        """Compute the average topic vector for all members of a community.

        Members without a topic vector (empty list) are skipped.

        Args:
            community_id: The community identifier to aggregate over.
            graph:        The :class:`~graph.models.InteractionGraph`.

        Returns:
            Element-wise mean topic vector as a list of floats.
            Returns an empty list if the community has no members or none
            of the members have topic vectors.
        """
        members_with_vectors = [
            node.topicVector
            for node in graph.nodes.values()
            if node.communityId == community_id and node.topicVector
        ]

        if not members_with_vectors:
            return []

        vector_length = len(members_with_vectors[0])
        centroid: list[float] = [0.0] * vector_length

        for vec in members_with_vectors:
            # Handle vectors of different lengths gracefully
            for i in range(min(len(vec), vector_length)):
                centroid[i] += vec[i]

        n = len(members_with_vectors)
        return [v / n for v in centroid]

    # ------------------------------------------------------------------
    # Effective topic vector (with sparse-user fallback)
    # ------------------------------------------------------------------

    def get_effective_topic_vector(
        self, user_id: str, graph: InteractionGraph
    ) -> list[float]:
        """Return the topic vector to use for a user, with centroid fallback.

        If the user has fewer than ``SPARSE_USER_INTERACTION_THRESHOLD``
        interactions, the community centroid topic vector is used instead of
        the user's own vector (Requirement 6.6).

        Args:
            user_id: The userId to retrieve a topic vector for.
            graph:   The :class:`~graph.models.InteractionGraph`.

        Returns:
            A list[float] topic vector, or an empty list if none is
            available (user not in graph, no topic vector, no community
            centroid).
        """
        node = graph.nodes.get(user_id)
        if node is None:
            return []

        interaction_count = self.get_user_interaction_count(user_id, graph)

        if interaction_count < SPARSE_USER_INTERACTION_THRESHOLD:
            logger.debug(
                "BridgeNodeService.get_effective_topic_vector: user '%s' has "
                "only %d interactions (threshold=%d) — falling back to "
                "community centroid for community '%s'.",
                user_id,
                interaction_count,
                SPARSE_USER_INTERACTION_THRESHOLD,
                node.communityId,
            )
            if node.communityId is not None:
                centroid = self.compute_community_centroid(node.communityId, graph)
                if centroid:
                    return centroid

        # Use the user's own topic vector (or return empty if not populated)
        return node.topicVector if node.topicVector else []

    # ------------------------------------------------------------------
    # Main bridge candidate identification
    # ------------------------------------------------------------------

    def identify_bridge_candidates(
        self,
        user_id: str,
        graph: InteractionGraph,
        signed_metrics: Optional[list[SignedMetrics]] = None,
    ) -> list[tuple[Node, float]]:
        """Identify and score cross-community bridge node candidates.

        Steps (Algorithm 6):

        1. Retrieve the user node and effective topic vector.
        2. Filter candidate nodes: ``node.communityId != user.communityId``
           AND ``node.betweenness > BRIDGE_CENTRALITY_THRESHOLD``.
        3. Score each candidate: ``cosine_similarity(user_vector,
           candidate.topicVector)``; exclude scores below
           ``MIN_TOPIC_RELEVANCE_THRESHOLD``.
        4. Wiki-RfA filter (when ``signed_metrics`` is provided and
           ``graph.datasetSource == "wiki_rfa"``): exclude candidates whose
           community has ``net_sentiment_index < 0``.
        5. Return remaining candidates sorted descending by topic relevance.

        Args:
            user_id:        The userId for whom to find bridge candidates.
            graph:          The :class:`~graph.models.InteractionGraph`.
            signed_metrics: Optional list of
                            :class:`~graph.models.SignedMetrics` for wiki-RfA
                            signed-graph filtering.  Only consulted when
                            ``graph.datasetSource == "wiki_rfa"``.

        Returns:
            List of ``(Node, topic_relevance_score)`` tuples sorted descending
            by score.  Returns an empty list when the user is not found in
            the graph or has no effective topic vector.

        Raises:
            ValueError: If ``user_id`` is not present in ``graph.nodes``.
        """
        user_node = graph.nodes.get(user_id)
        if user_node is None:
            raise ValueError(
                f"BridgeNodeService.identify_bridge_candidates: "
                f"user '{user_id}' not found in graph."
            )

        user_vector = self.get_effective_topic_vector(user_id, graph)
        if not user_vector:
            logger.debug(
                "BridgeNodeService.identify_bridge_candidates: user '%s' has "
                "no effective topic vector — returning empty candidates.",
                user_id,
            )
            return []

        # Build sentiment lookup for wiki-RfA filter
        sentiment_by_community: dict[str, float] = {}
        apply_sentiment_filter = (
            signed_metrics is not None
            and graph.datasetSource == "wiki_rfa"
        )
        if apply_sentiment_filter and signed_metrics:
            sentiment_by_community = {
                sm.communityId: sm.netSentimentIndex for sm in signed_metrics
            }

        results: list[tuple[Node, float]] = []

        for candidate_id, candidate_node in graph.nodes.items():
            if candidate_id == user_id:
                continue

            # Filter 1: must be from a different community
            if candidate_node.communityId == user_node.communityId:
                continue

            # Filter 2: must exceed bridge centrality threshold
            if candidate_node.betweenness <= BRIDGE_CENTRALITY_THRESHOLD:
                continue

            # Score by topic relevance
            if not candidate_node.topicVector:
                continue

            score = self.cosine_similarity(user_vector, candidate_node.topicVector)

            # Filter 3: exclude below minimum topic relevance
            if score < MIN_TOPIC_RELEVANCE_THRESHOLD:
                continue

            # Filter 4 (wiki-RfA only): exclude hostile communities
            if apply_sentiment_filter:
                community_sentiment = sentiment_by_community.get(
                    candidate_node.communityId, 0.0
                )
                if community_sentiment < 0:
                    continue

            results.append((candidate_node, score))

        # Sort descending by topic relevance score
        results.sort(key=lambda item: item[1], reverse=True)

        return results


# ---------------------------------------------------------------------------
# Diversity Gain Estimation (Algorithm 6, estimateDiversityGain)
# ---------------------------------------------------------------------------


def estimate_diversity_gain(
    user_id: str,
    candidate_id: str,
    graph: InteractionGraph,
    metrics: UserMetrics,
) -> float:
    """Estimate the increase in diversity score if user_id followed candidate_id.

    This is an *analytic* simulation — the graph is NOT mutated.

    Algorithm (from design.md):
        current_total    = sum of all outgoing edge weights from user_id
        current_cross    = sum of outgoing weights to nodes in different communities
        simulated_total  = current_total + new_edge_weight
        simulated_cross  = current_cross + new_edge_weight  (candidate is in a
                           different community, so it always counts as cross)
        new_score        = simulated_cross / simulated_total
        gain             = new_score - metrics.diversityScore
        return clamp(gain, 0.0, 1.0 - metrics.diversityScore)

    The ``new_edge_weight`` for the simulated edge is the average existing
    outgoing edge weight for ``user_id``.  When the user has no outgoing edges
    the simulated weight is 1.0 (treating the new edge as the only one).

    Args:
        user_id:      The user for whom to estimate diversity gain.
        candidate_id: The candidate user to simulate following.
        graph:        The :class:`~graph.models.InteractionGraph`.
        metrics:      Pre-computed :class:`~graph.models.UserMetrics` for
                      ``user_id``, including the current ``diversityScore`` and
                      ``communityId``.

    Returns:
        Estimated gain in [0, 1 - metrics.diversityScore].
        Returns 0.0 if the candidate belongs to the same community as the user.

    Preconditions:
        Both ``user_id`` and ``candidate_id`` must be present in ``graph.nodes``.
        ``metrics.diversityScore`` must be in [0, 1].
    """
    user_node = graph.nodes.get(user_id)
    candidate_node = graph.nodes.get(candidate_id)

    if user_node is None or candidate_node is None:
        logger.warning(
            "estimate_diversity_gain: user '%s' or candidate '%s' not in graph.",
            user_id,
            candidate_id,
        )
        return 0.0

    # Postcondition: same community → gain = 0.0
    if candidate_node.communityId == user_node.communityId:
        return 0.0

    # Collect all outgoing edges from user_id
    outgoing_edges = [e for e in graph.edges if e.sourceUserId == user_id]

    current_total = sum(e.weight for e in outgoing_edges)
    user_community = user_node.communityId
    current_cross = sum(
        e.weight
        for e in outgoing_edges
        if graph.nodes.get(e.targetUserId, Node(userId=e.targetUserId)).communityId != user_community
    )

    # Determine a sensible simulated edge weight
    if outgoing_edges:
        new_edge_weight = current_total / len(outgoing_edges)  # average existing weight
    else:
        new_edge_weight = 1.0  # no prior edges: treat new edge as the only one

    simulated_total = current_total + new_edge_weight
    simulated_cross = current_cross + new_edge_weight  # candidate is cross-community

    if simulated_total == 0.0:
        return 0.0

    new_score = simulated_cross / simulated_total
    gain = new_score - metrics.diversityScore

    # Clamp to [0, 1 - current_score]
    max_gain = max(0.0, 1.0 - metrics.diversityScore)
    return max(0.0, min(gain, max_gain))


# ---------------------------------------------------------------------------
# Recommendation Generation (Algorithm 6, generate_recommendations)
# ---------------------------------------------------------------------------


def generate_recommendations(
    user_id: str,
    graph: InteractionGraph,
    metrics: UserMetrics,
    top_k: int,
    signed_metrics: Optional[list[SignedMetrics]] = None,
) -> list[Recommendation]:
    """Generate cross-community recommendations for a low-diversity user.

    Implements Algorithm 6 from design.md:

    1. Identify bridge node candidates from other communities (via
       :meth:`BridgeNodeService.identify_bridge_candidates`).
    2. For each candidate compute ``estimateDiversityGain``.
    3. Sort descending by ``diversityGain``.
    4. Return at most ``topK`` :class:`~graph.models.Recommendation` objects,
       each with a human-readable ``reason`` string.

    Args:
        user_id:        The user for whom to generate recommendations.
        graph:          The :class:`~graph.models.InteractionGraph`.
        metrics:        Pre-computed :class:`~graph.models.UserMetrics` for
                        ``user_id`` (must include ``diversityScore``,
                        ``communityId``).
        top_k:          Maximum number of recommendations to return.
                        Returns an empty list when ``top_k <= 0``.
        signed_metrics: Optional signed-graph metrics for wiki-RfA hostile-
                        community filtering (forwarded to
                        :meth:`BridgeNodeService.identify_bridge_candidates`).

    Returns:
        List of :class:`~graph.models.Recommendation` objects sorted
        descending by ``diversityGain``, with ``len(result) <= top_k``.

    Requirements:
        6.1  All recommendations are from a different community than the user.
        6.4  Sorted descending by diversityGain.
        6.5  At most topK results returned.
        6.8  Each recommendation has a human-readable reason string.
    """
    if top_k <= 0:
        return []

    service = BridgeNodeService()

    # Step 1 + 2: identify candidates with topic relevance scores
    bridge_candidates: list[tuple[Node, float]] = service.identify_bridge_candidates(
        user_id, graph, signed_metrics=signed_metrics
    )

    if not bridge_candidates:
        logger.debug(
            "generate_recommendations: no bridge candidates found for user '%s'.",
            user_id,
        )
        return []

    # Step 2 (cont.): score each candidate by diversity gain
    scored: list[tuple[Node, float, float]] = []  # (node, topic_relevance, diversity_gain)
    for candidate_node, topic_relevance in bridge_candidates:
        gain = estimate_diversity_gain(user_id, candidate_node.userId, graph, metrics)
        scored.append((candidate_node, topic_relevance, gain))

    # Step 3: sort descending by diversity_gain (ties broken by topic_relevance)
    scored.sort(key=lambda item: (item[2], item[1]), reverse=True)

    # Step 4: build Recommendation objects, up to top_k
    recommendations: list[Recommendation] = []
    user_community = metrics.communityId

    for candidate_node, topic_relevance, gain in scored[:top_k]:
        candidate_community = candidate_node.communityId or "unknown"
        reason = _build_reason_string(
            candidate_node.userId,
            candidate_community,
            topic_relevance,
            gain,
        )

        rec = Recommendation(
            recommendationId=str(uuid.uuid4()),
            targetUserId=user_id,
            recommendedUserId=candidate_node.userId,
            diversityGain=gain,
            topicRelevance=topic_relevance,
            communityId=candidate_community,
            reason=reason,
        )
        recommendations.append(rec)

    logger.info(
        "generate_recommendations: generated %d recommendations for user '%s' "
        "(top_k=%d, community='%s').",
        len(recommendations),
        user_id,
        top_k,
        user_community,
    )

    return recommendations


def _build_reason_string(
    candidate_id: str,
    community_id: str,
    topic_relevance: float,
    diversity_gain: float,
) -> str:
    """Build a human-readable reason string for a recommendation.

    Produces a generic string that works across all dataset types.

    Args:
        candidate_id:    userId of the recommended account.
        community_id:    Community the recommended account belongs to.
        topic_relevance: Cosine similarity score in [0, 1].
        diversity_gain:  Estimated increase in diversity score in [0, 1].

    Returns:
        A human-readable explanation string.
    """
    return (
        f"{candidate_id} is a bridge node in community {community_id} "
        f"with {topic_relevance:.0%} topic overlap — "
        f"following them would increase your diversity score by an estimated {diversity_gain:.0%}"
    )
