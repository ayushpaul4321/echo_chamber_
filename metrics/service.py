"""Metrics & Analysis Service for the Echo Chamber Detector pipeline.

Implements ``computePolarizationIndex`` (Algorithm 3) that takes an
``InteractionGraph`` and a community partition and produces
``PolarizationMetrics``.

Dataset-specific notes
-----------------------
All datasets:
    - Standard PI = 1.0 − (interEdgeWeight / totalEdgeWeight)
    - Sets ``edge.isCrossCommunity`` flag as side effect on each edge
    - Returns ``polarizationIndex = 0.0`` and ``interCommunityEdgeRatio = 0.0``
      when totalEdgeWeight == 0 (empty graph or all zero-weight edges)

Expected benchmark ranges (approximate):
    - Reddit title  : PI ≈ 0.65–0.75
    - Congress      : PI ≈ 0.85–0.91
    - Wiki-RfA (unsigned): PI ≈ 0.50–0.65

References: design.md Algorithm 3, Requirements 4.1–4.5
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Union

from graph.models import (
    CommunityPartition,
    InteractionGraph,
    PolarizationMetrics,
    SignedMetrics,
    UserMetrics,
)
from graph.redis_keys import DEFAULT_TTL_SECONDS, polarization_key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MetricsFilter
# ---------------------------------------------------------------------------


@dataclass
class MetricsFilter:
    """Predicates used by :meth:`MetricsService.query_metrics`.

    All fields are optional; omitting a field means no filter is applied for
    that predicate.

    Attributes:
        snapshot_id:       Exact ``snapshot_id`` to match.
        dataset_source:    Exact ``dataset_source`` to match.
        from_date:         Lower bound on ``computed_at`` (inclusive).
        to_date:           Upper bound on ``computed_at`` (inclusive).
        community_id:      Filter UserMetrics rows by ``community_id``.
        min_polarization:  Minimum ``polarization_index`` for PolarizationMetrics rows.
    """

    snapshot_id: Optional[str] = None
    dataset_source: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    community_id: Optional[str] = None
    min_polarization: Optional[float] = None


# ---------------------------------------------------------------------------
# Type alias for the partition argument
# ---------------------------------------------------------------------------

# Accepts either a list of CommunityPartition objects (as returned by
# CommunityDetectionService.detect_communities) or a raw dict mapping
# userId → communityId (for flexibility / testing).
PartitionInput = Union[list[CommunityPartition], dict[str, str]]


# ---------------------------------------------------------------------------
# MetricsService
# ---------------------------------------------------------------------------


class MetricsService:
    """Service that computes Polarization Index and (future) Diversity Scores.

    Usage::

        service = MetricsService()
        metrics = service.compute_polarization_index(graph, partition)

    The service is stateless; each call to :meth:`compute_polarization_index`
    is independent.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_polarization_index(
        self,
        graph: InteractionGraph,
        partition: PartitionInput,
    ) -> PolarizationMetrics:
        """Compute the Polarization Index for *graph* given a community *partition*.

        Implements Algorithm 3 from design.md.

        Side effects:
            Sets ``edge.isCrossCommunity = True/False`` on every edge in
            *graph* based on whether source and target belong to different
            communities.

        Args:
            graph:     :class:`InteractionGraph` to analyse.  May be empty
                       (zero nodes / zero edges).
            partition: Community assignment, either:
                       - a ``list[CommunityPartition]`` (from
                         :class:`~community.service.CommunityDetectionService`), or
                       - a ``dict[str, str]`` mapping ``userId → communityId``.

        Returns:
            :class:`PolarizationMetrics` with:
            - ``polarizationIndex``          in [0, 1]
            - ``interCommunityEdgeRatio``    in [0, 1]
            - ``modularity``                 from the partition (if available)
            - ``communityCount``             number of distinct communities
            - ``avgCommunitySize``           average nodes per community
            - ``computedAt``                 current UTC timestamp
            - ``snapshotId``                 copied from *graph*
            - ``datasetSource``              copied from *graph*

        Requirements:
            4.1 polarizationIndex ∈ [0, 1]
            4.2 all-intra edges → polarizationIndex = 1.0
            4.3 all-inter edges → polarizationIndex = 0.0
            4.4 polarizationIndex + interCommunityEdgeRatio = 1.0
            4.5 totalEdgeWeight = 0 → polarizationIndex = 0.0,
                                       interCommunityEdgeRatio = 0.0
        """
        # --- Normalise the partition into a flat userId → communityId dict ---
        partition_map, modularity, community_member_sets = (
            self._normalise_partition(partition)
        )

        # --- Edge classification and weight accumulation (Algorithm 3) ---
        intra_edge_weight = 0.0
        inter_edge_weight = 0.0

        for edge in graph.edges:
            src_community = partition_map.get(edge.sourceUserId)
            tgt_community = partition_map.get(edge.targetUserId)

            if src_community is not None and src_community == tgt_community:
                # Intra-community edge
                intra_edge_weight += edge.weight
                edge.isCrossCommunity = False
            else:
                # Inter-community edge (also covers nodes not in partition)
                inter_edge_weight += edge.weight
                edge.isCrossCommunity = True

        total_weight = intra_edge_weight + inter_edge_weight

        # --- Handle empty/zero-weight case (Requirement 4.5) ---
        if total_weight == 0.0:
            logger.debug(
                "MetricsService.compute_polarization_index: totalEdgeWeight=0 "
                "for graph '%s'; returning polarizationIndex=0.0",
                graph.snapshotId,
            )
            inter_ratio = 0.0
            polarization_index = 0.0
        else:
            inter_ratio = inter_edge_weight / total_weight
            polarization_index = 1.0 - inter_ratio

        # Clamp to [0, 1] as a safety net for floating-point edge cases
        polarization_index = max(0.0, min(1.0, polarization_index))
        inter_ratio = max(0.0, min(1.0, inter_ratio))

        # --- Compute modularity if not already available from partition ---
        if modularity is None:
            modularity = self._compute_modularity(graph, partition_map)

        # --- Community-level statistics ---
        community_count, avg_community_size = self._community_stats(
            community_member_sets, graph
        )

        computed_at = datetime.now(timezone.utc)

        logger.info(
            "MetricsService.compute_polarization_index: "
            "snapshotId='%s' dataset='%s' "
            "PI=%.4f interRatio=%.4f modularity=%.4f "
            "communityCount=%d avgCommunitySize=%.1f",
            graph.snapshotId,
            graph.datasetSource,
            polarization_index,
            inter_ratio,
            modularity,
            community_count,
            avg_community_size,
        )

        return PolarizationMetrics(
            snapshotId=graph.snapshotId,
            polarizationIndex=polarization_index,
            modularity=modularity,
            communityCount=community_count,
            avgCommunitySize=avg_community_size,
            interCommunityEdgeRatio=inter_ratio,
            computedAt=computed_at,
            datasetSource=graph.datasetSource,
        )

    def compute_diversity_score(
        self,
        user_id: str,
        graph: InteractionGraph,
        partition: PartitionInput,
    ) -> float:
        """Compute the Diversity Score for a single user.

        Implements Algorithm 4 from design.md.

        The diversity score is the fraction of *outgoing* edge weight that
        crosses community boundaries:

            diversityScore = crossCommunityWeight / totalOutgoingWeight

        Side effects:
            Sets ``graph.nodes[user_id].diversityScore`` to the computed value
            if *user_id* exists in the graph.

        Args:
            user_id:   The user whose diversity score to compute.
            graph:     :class:`InteractionGraph` containing edges.
            partition: Community assignment (list or dict, same as
                       :meth:`compute_polarization_index`).

        Returns:
            Float in [0, 1]:
            - 0.0 if the user has no outgoing edges (Req 5.4)
            - 0.0 if totalOutgoingWeight == 0.0
            - 0.0 if all outgoing edges are intra-community (Req 5.2)
            - 1.0 if all outgoing edges are inter-community (Req 5.3)

        Requirements:
            5.1 result ∈ [0, 1]
            5.2 all outgoing intra-community → 0.0
            5.3 all outgoing inter-community → 1.0
            5.4 no outgoing edges → 0.0
        """
        partition_map, _, _ = self._normalise_partition(partition)

        # Collect outgoing edges for this user
        outgoing_edges = [e for e in graph.edges if e.sourceUserId == user_id]

        if not outgoing_edges:
            score = 0.0
            if user_id in graph.nodes:
                graph.nodes[user_id].diversityScore = score
            return score

        user_community = partition_map.get(user_id)

        cross_community_weight = 0.0
        total_outgoing_weight = 0.0

        for edge in outgoing_edges:
            total_outgoing_weight += edge.weight
            tgt_community = partition_map.get(edge.targetUserId)
            if tgt_community != user_community:
                cross_community_weight += edge.weight

        if total_outgoing_weight == 0.0:
            score = 0.0
        else:
            score = cross_community_weight / total_outgoing_weight

        # Clamp to [0, 1] as a safety net for floating-point edge cases
        score = max(0.0, min(1.0, score))

        # Side effect: persist diversity score on the node
        if user_id in graph.nodes:
            graph.nodes[user_id].diversityScore = score

        return score

    def compute_betweenness_centrality(
        self,
        graph: InteractionGraph,
    ) -> dict[str, float]:
        """Compute betweenness centrality for every node in *graph* (Algorithm 5).

        Uses Brandes' algorithm via NetworkX:
        - Exact computation for graphs with ≤ 100,000 nodes.
        - Approximate computation (k=500 pivot nodes) for graphs with > 100,000 nodes.

        Side effects:
            Sets ``node.betweenness`` on every :class:`~graph.models.Node` in
            *graph* to its normalized centrality value.

        Args:
            graph: :class:`InteractionGraph` to analyse.

        Returns:
            ``dict[userId, betweenness_float]`` — normalized betweenness
            centrality values in [0, 1] using the standard (n-1)(n-2)
            normalization factor.

        Requirements:
            5.6 betweenness centrality computed for every node, normalized to
                [0, 1] using (n-1)(n-2).
        """
        import networkx as nx  # noqa: PLC0415

        # Build a directed NetworkX graph from the InteractionGraph
        nx_graph: nx.DiGraph = nx.DiGraph()

        for uid in graph.nodes:
            nx_graph.add_node(uid)

        for edge in graph.edges:
            src, tgt = edge.sourceUserId, edge.targetUserId
            if src == tgt:
                continue  # skip self-loops
            if nx_graph.has_edge(src, tgt):
                nx_graph[src][tgt]["weight"] += edge.weight
            else:
                nx_graph.add_edge(src, tgt, weight=edge.weight)

        node_count = nx_graph.number_of_nodes()

        if node_count <= 100_000:
            centrality: dict[str, float] = nx.betweenness_centrality(
                nx_graph, normalized=True
            )
        else:
            # Approximate for large graphs (k pivot nodes)
            centrality = nx.betweenness_centrality(
                nx_graph, k=500, normalized=True
            )

        # Persist values back onto the InteractionGraph nodes
        for uid, node in graph.nodes.items():
            node.betweenness = centrality.get(uid, 0.0)

        logger.info(
            "MetricsService.compute_betweenness_centrality: "
            "snapshotId='%s' nodeCount=%d approximate=%s",
            graph.snapshotId,
            node_count,
            node_count > 100_000,
        )

        return centrality

    def compute_community_diversity_score(
        self,
        community_id: str,
        graph: InteractionGraph,
        partition: PartitionInput,
    ) -> float:
        """Compute the community-level Diversity Score.

        The community score is the arithmetic mean of each member user's
        individual :meth:`compute_diversity_score`.

        Args:
            community_id: The community whose aggregate diversity to compute.
            graph:        :class:`InteractionGraph` containing edges.
            partition:    Community assignment (list or dict).

        Returns:
            Float in [0, 1]; 0.0 if the community has no members.

        Requirements:
            5.5 community-level score = arithmetic mean of member scores
        """
        partition_map, _, community_member_sets = self._normalise_partition(partition)

        # Collect members for the requested community
        member_ids: list[str] = [
            uid for uid, cid in partition_map.items() if cid == community_id
        ]

        if not member_ids:
            return 0.0

        scores = [
            self.compute_diversity_score(uid, graph, partition)
            for uid in member_ids
        ]

        return sum(scores) / len(scores)

    def compute_signed_metrics(
        self,
        graph: InteractionGraph,
        partition: PartitionInput,
    ) -> list[SignedMetrics]:
        """Compute per-community signed-edge sentiment metrics for wiki-RfA graphs.

        Only meaningful for wiki-RfA (``graph.datasetSource == "wiki_rfa"``).
        For other datasets a warning is logged and an empty list is returned.

        For each community the following metrics are computed using only
        *intra-community* edges (both source and target are members):

        - ``positiveEdgeRatio``  — sum of weights of positive edges
          (``signedPolarity == +1``) divided by total intra-community weight.
          Set to 0.0 when total weight is 0.
        - ``negativeEdgeRatio``  — ``1.0 - positiveEdgeRatio``
        - ``netSentimentIndex``  — arithmetic mean of ``signedPolarity``
          values for intra-community edges where ``signedPolarity is not None``.
          Set to 0.0 when no such edge exists.

        ``crossCommunityNegativity`` is a graph-level value (same for every
        community in the output):

        - Fraction of negative edges (``signedPolarity == -1``) across the
          entire graph that cross community boundaries.
        - Set to 0.0 when there are no negative edges.

        The method uses ``edge.isCrossCommunity`` when already set (e.g. by a
        prior call to :meth:`compute_polarization_index`), and falls back to
        computing it from *partition* when the flag is not populated.

        Args:
            graph:     :class:`InteractionGraph` to analyse.
            partition: Community assignment (list or dict).

        Returns:
            ``list[SignedMetrics]``, one entry per community.  Empty list for
            non-wiki-RfA datasets.

        References: Requirements 4.6
        """
        if graph.datasetSource != "wiki_rfa":
            logger.warning(
                "MetricsService.compute_signed_metrics: called on non-wiki-RfA "
                "dataset '%s' (snapshotId='%s'); returning empty list.",
                graph.datasetSource,
                graph.snapshotId,
            )
            return []

        partition_map, _, community_member_sets = self._normalise_partition(partition)

        # Build a set-based lookup: communityId → set of memberIds
        community_members: dict[str, set[str]] = {}
        for uid, cid in partition_map.items():
            community_members.setdefault(cid, set()).add(uid)

        # ------------------------------------------------------------------
        # Graph-level: cross_community_negativity
        # ------------------------------------------------------------------
        # Collect all negative edges across the entire graph
        negative_edges = [e for e in graph.edges if e.signedPolarity == -1]
        total_negative = len(negative_edges)

        if total_negative == 0:
            cross_community_negativity = 0.0
        else:
            cross_negative_count = 0
            for edge in negative_edges:
                # Use pre-set flag if available, otherwise derive from partition
                is_cross: bool
                if edge.isCrossCommunity is not None:
                    # isCrossCommunity is always a bool on the dataclass; but
                    # it defaults to False, so we can only trust it when
                    # compute_polarization_index has been called.  We use the
                    # fallback unconditionally to stay safe.
                    src_c = partition_map.get(edge.sourceUserId)
                    tgt_c = partition_map.get(edge.targetUserId)
                    is_cross = (src_c != tgt_c) if (src_c is not None and tgt_c is not None) else edge.isCrossCommunity
                else:
                    src_c = partition_map.get(edge.sourceUserId)
                    tgt_c = partition_map.get(edge.targetUserId)
                    is_cross = src_c != tgt_c
                if is_cross:
                    cross_negative_count += 1
            cross_community_negativity = cross_negative_count / total_negative

        # ------------------------------------------------------------------
        # Per-community metrics
        # ------------------------------------------------------------------
        results: list[SignedMetrics] = []
        computed_at = datetime.now(timezone.utc)

        for community_id, members in community_members.items():
            # Intra-community edges: both endpoints in this community
            intra_edges = [
                e for e in graph.edges
                if e.sourceUserId in members and e.targetUserId in members
            ]

            # positive_edge_ratio
            total_intra_weight = sum(e.weight for e in intra_edges)
            if total_intra_weight == 0.0:
                positive_edge_ratio = 0.0
            else:
                positive_weight = sum(
                    e.weight for e in intra_edges if e.signedPolarity == 1
                )
                positive_edge_ratio = positive_weight / total_intra_weight

            negative_edge_ratio = 1.0 - positive_edge_ratio

            # net_sentiment_index: mean signedPolarity for edges that have it
            polarity_values = [
                e.signedPolarity for e in intra_edges if e.signedPolarity is not None
            ]
            if polarity_values:
                net_sentiment_index = sum(polarity_values) / len(polarity_values)
            else:
                net_sentiment_index = 0.0

            results.append(
                SignedMetrics(
                    snapshotId=graph.snapshotId,
                    communityId=community_id,
                    positiveEdgeRatio=positive_edge_ratio,
                    negativeEdgeRatio=negative_edge_ratio,
                    netSentimentIndex=net_sentiment_index,
                    crossCommunityNegativity=cross_community_negativity,
                    computedAt=computed_at,
                    datasetSource=graph.datasetSource,
                )
            )

        logger.info(
            "MetricsService.compute_signed_metrics: "
            "snapshotId='%s' dataset='%s' communityCount=%d "
            "crossCommunityNegativity=%.4f",
            graph.snapshotId,
            graph.datasetSource,
            len(results),
            cross_community_negativity,
        )

        return results

    # ------------------------------------------------------------------
    # Persistence and querying (Requirements 4.6, 4.7, 5.7)
    # ------------------------------------------------------------------

    def persist_metrics(
        self,
        polarization_metrics: PolarizationMetrics,
        user_metrics_list: list[UserMetrics],
        db_session: Any,
        redis_client: Any = None,
    ) -> None:
        """Persist ``PolarizationMetrics`` and a list of ``UserMetrics`` to PostgreSQL.

        Writes one :class:`~graph.db_models.PolarizationMetricRow` and one
        :class:`~graph.db_models.UserMetricRow` per entry to the database via
        the supplied SQLAlchemy session.  The session is flushed but **not**
        committed — the caller is responsible for committing the transaction.

        Additionally caches the polarization metrics as a JSON string in Redis
        (if *redis_client* is provided) using the key from
        :func:`~graph.redis_keys.polarization_key` with the default TTL
        (:data:`~graph.redis_keys.DEFAULT_TTL_SECONDS`).

        Args:
            polarization_metrics: Graph-level metrics to persist.
            user_metrics_list:    Per-user metrics to persist (may be empty).
            db_session:           Active SQLAlchemy ``Session`` (or compatible).
            redis_client:         Optional Redis client (e.g. ``redis.Redis``).
                                  When ``None``, the Redis caching step is skipped.

        Requirements:
            4.6  Persist PolarizationMetrics per snapshot per datasetSource.
            4.7  One record per Snapshot (time-series).
            5.7  Persist UserMetrics per user per snapshot.
        """
        from graph.db_models import PolarizationMetricRow, UserMetricRow  # noqa: PLC0415

        # --- Persist PolarizationMetrics row ---
        pm_row = PolarizationMetricRow(
            snapshot_id=polarization_metrics.snapshotId,
            dataset_source=polarization_metrics.datasetSource,
            polarization_index=polarization_metrics.polarizationIndex,
            modularity=polarization_metrics.modularity,
            community_count=polarization_metrics.communityCount,
            avg_community_size=polarization_metrics.avgCommunitySize,
            inter_community_edge_ratio=polarization_metrics.interCommunityEdgeRatio,
            computed_at=polarization_metrics.computedAt,
        )
        db_session.add(pm_row)

        # --- Persist UserMetrics rows ---
        for um in user_metrics_list:
            um_row = UserMetricRow(
                snapshot_id=um.snapshotId,
                dataset_source=polarization_metrics.datasetSource,
                user_id=um.userId,
                community_id=um.communityId,
                diversity_score=um.diversityScore,
                intra_edge_count=um.intraEdgeCount,
                inter_edge_count=um.interEdgeCount,
                betweenness_centrality=um.betweennessCentrality,
                computed_at=um.computedAt,
            )
            db_session.add(um_row)

        db_session.flush()

        logger.info(
            "MetricsService.persist_metrics: persisted PolarizationMetrics "
            "snapshotId='%s' datasetSource='%s' and %d UserMetrics rows",
            polarization_metrics.snapshotId,
            polarization_metrics.datasetSource,
            len(user_metrics_list),
        )

        # --- Cache polarization metrics in Redis ---
        if redis_client is not None:
            try:
                cache_payload = json.dumps({
                    "snapshotId": polarization_metrics.snapshotId,
                    "datasetSource": polarization_metrics.datasetSource,
                    "polarizationIndex": polarization_metrics.polarizationIndex,
                    "modularity": polarization_metrics.modularity,
                    "communityCount": polarization_metrics.communityCount,
                    "avgCommunitySize": polarization_metrics.avgCommunitySize,
                    "interCommunityEdgeRatio": polarization_metrics.interCommunityEdgeRatio,
                    "computedAt": polarization_metrics.computedAt.isoformat(),
                })
                redis_key = polarization_key(polarization_metrics.snapshotId)
                redis_client.setex(redis_key, DEFAULT_TTL_SECONDS, cache_payload)
                logger.debug(
                    "MetricsService.persist_metrics: cached polarization metrics "
                    "in Redis key='%s' ttl=%ds",
                    redis_key,
                    DEFAULT_TTL_SECONDS,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MetricsService.persist_metrics: Redis caching failed (%s); "
                    "continuing without cache.",
                    exc,
                )

    def query_metrics(
        self,
        filter: "MetricsFilter",
        db_session: Any,
    ) -> dict[str, list[Any]]:
        """Query persisted metrics from PostgreSQL using the supplied *filter*.

        Supports filtering by:

        - ``snapshotId``       — exact match on ``snapshot_id``
        - ``datasetSource``    — exact match on ``dataset_source``
        - ``from_date``        — ``computed_at >= from_date``
        - ``to_date``          — ``computed_at <= to_date``
        - ``communityId``      — filter UserMetrics rows by ``community_id``
        - ``min_polarization`` — filter PolarizationMetrics rows where
                                 ``polarization_index >= min_polarization``

        Args:
            filter:     :class:`MetricsFilter` specifying query predicates.
            db_session: Active SQLAlchemy ``Session`` (or compatible).

        Returns:
            A dict with keys:

            - ``"polarization"`` — list of :class:`~graph.db_models.PolarizationMetricRow`
            - ``"user_metrics"`` — list of :class:`~graph.db_models.UserMetricRow`

        Requirements:
            4.6  Query PolarizationMetrics per snapshot per datasetSource.
            4.7  Time-series querying by date range.
            5.7  Query UserMetrics per snapshot / community.
        """
        from graph.db_models import PolarizationMetricRow, UserMetricRow  # noqa: PLC0415
        from sqlalchemy import and_  # noqa: PLC0415

        # --- Build PolarizationMetrics query ---
        pm_query = db_session.query(PolarizationMetricRow)
        pm_filters = []

        if filter.snapshot_id is not None:
            pm_filters.append(
                PolarizationMetricRow.snapshot_id == filter.snapshot_id
            )
        if filter.dataset_source is not None:
            pm_filters.append(
                PolarizationMetricRow.dataset_source == filter.dataset_source
            )
        if filter.from_date is not None:
            pm_filters.append(
                PolarizationMetricRow.computed_at >= filter.from_date
            )
        if filter.to_date is not None:
            pm_filters.append(
                PolarizationMetricRow.computed_at <= filter.to_date
            )
        if filter.min_polarization is not None:
            pm_filters.append(
                PolarizationMetricRow.polarization_index >= filter.min_polarization
            )

        if pm_filters:
            pm_query = pm_query.filter(and_(*pm_filters))

        polarization_rows = pm_query.all()

        # --- Build UserMetrics query ---
        um_query = db_session.query(UserMetricRow)
        um_filters = []

        if filter.snapshot_id is not None:
            um_filters.append(
                UserMetricRow.snapshot_id == filter.snapshot_id
            )
        if filter.dataset_source is not None:
            um_filters.append(
                UserMetricRow.dataset_source == filter.dataset_source
            )
        if filter.from_date is not None:
            um_filters.append(
                UserMetricRow.computed_at >= filter.from_date
            )
        if filter.to_date is not None:
            um_filters.append(
                UserMetricRow.computed_at <= filter.to_date
            )
        if filter.community_id is not None:
            um_filters.append(
                UserMetricRow.community_id == filter.community_id
            )

        if um_filters:
            um_query = um_query.filter(and_(*um_filters))

        user_metric_rows = um_query.all()

        logger.info(
            "MetricsService.query_metrics: found %d polarization rows and "
            "%d user_metrics rows for filter %r",
            len(polarization_rows),
            len(user_metric_rows),
            filter,
        )

        return {
            "polarization": polarization_rows,
            "user_metrics": user_metric_rows,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_partition(
        partition: PartitionInput,
    ) -> tuple[dict[str, str], float | None, list[set[str]]]:
        """Normalise *partition* into a flat ``{userId: communityId}`` dict.

        Args:
            partition: Either a ``list[CommunityPartition]`` or a
                       ``dict[str, str]``.

        Returns:
            A 3-tuple:
            - ``partition_map``         — flat userId → communityId dict
            - ``modularity``            — overall Q if available (else None)
            - ``community_member_sets`` — list of member-id sets, one per
                                          community (used for avgCommunitySize)
        """
        if isinstance(partition, dict):
            # Raw dict: userId → communityId
            partition_map: dict[str, str] = {
                str(uid): str(cid) for uid, cid in partition.items()
            }
            modularity = None
            # Build community member sets from the dict
            community_sets: dict[str, set[str]] = {}
            for uid, cid in partition_map.items():
                community_sets.setdefault(cid, set()).add(uid)
            community_member_sets: list[set[str]] = list(community_sets.values())
            return partition_map, modularity, community_member_sets

        # list[CommunityPartition]
        partition_map = {}
        modularity: float | None = None
        community_member_sets = []

        for cp in partition:
            for member_id in cp.memberIds:
                partition_map[str(member_id)] = str(cp.communityId)
            community_member_sets.append(set(cp.memberIds))
            # Use the first non-zero modularity we encounter (all CPs in one
            # run share the same overall Q)
            if modularity is None and cp.modularity != 0.0:
                modularity = cp.modularity

        if modularity is None and partition:
            # Fall back to the first CP's modularity (even if it is 0.0)
            modularity = partition[0].modularity

        return partition_map, modularity, community_member_sets

    @staticmethod
    def _compute_modularity(
        graph: InteractionGraph,
        partition_map: dict[str, str],
    ) -> float:
        """Compute modularity Q using python-louvain (or NetworkX fallback).

        Args:
            graph:         Source :class:`InteractionGraph`.
            partition_map: Flat userId → communityId dict.

        Returns:
            Modularity Q as a float.  Returns 0.0 on failure.
        """
        if not graph.edges:
            return 0.0

        try:
            import networkx as nx  # noqa: PLC0415

            nx_graph = nx.Graph()
            for uid in graph.nodes:
                nx_graph.add_node(uid)
            for edge in graph.edges:
                src, tgt = edge.sourceUserId, edge.targetUserId
                if src == tgt:
                    continue
                if nx_graph.has_edge(src, tgt):
                    nx_graph[src][tgt]["weight"] += edge.weight
                else:
                    nx_graph.add_edge(src, tgt, weight=edge.weight)

            if nx_graph.number_of_edges() == 0:
                return 0.0

            # Build integer partition for python-louvain / networkx
            cid_to_int: dict[str, int] = {}
            next_int = 0
            int_partition: dict[str, int] = {}
            for uid, cid in partition_map.items():
                if cid not in cid_to_int:
                    cid_to_int[cid] = next_int
                    next_int += 1
                int_partition[uid] = cid_to_int[cid]

            # Try python-louvain first
            try:
                import importlib
                import sys

                saved = sys.modules.pop("community", None)
                try:
                    import importlib.util

                    spec = importlib.util.find_spec("community")
                    if spec is not None and spec.origin is not None:
                        real_community = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(real_community)  # type: ignore[union-attr]
                        if hasattr(real_community, "modularity"):
                            q = real_community.modularity(
                                int_partition, nx_graph, weight="weight"
                            )
                            return max(0.0, float(q))
                finally:
                    if saved is not None:
                        sys.modules["community"] = saved
            except Exception:  # noqa: BLE001
                pass

            # NetworkX fallback
            communities_list: list[set[str]] = []
            groups: dict[int, set[str]] = {}
            for uid, cid_int in int_partition.items():
                groups.setdefault(cid_int, set()).add(uid)
            communities_list = list(groups.values())

            if not communities_list:
                return 0.0

            q = nx.community.modularity(nx_graph, communities_list, weight="weight")
            return max(0.0, float(q))

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MetricsService._compute_modularity: failed (%s); returning 0.0",
                exc,
            )
            return 0.0

    @staticmethod
    def _community_stats(
        community_member_sets: list[set[str]],
        graph: InteractionGraph,
    ) -> tuple[int, float]:
        """Compute community count and average community size.

        Args:
            community_member_sets: List of member-id sets (one per community).
            graph:                 Source graph (used when partition is empty).

        Returns:
            ``(community_count, avg_community_size)``
        """
        if not community_member_sets:
            # No partition provided — treat entire graph as one community
            n = graph.nodeCount
            return (1, float(n)) if n > 0 else (0, 0.0)

        community_count = len(community_member_sets)
        total_members = sum(len(s) for s in community_member_sets)
        avg_size = total_members / community_count if community_count > 0 else 0.0
        return community_count, avg_size
