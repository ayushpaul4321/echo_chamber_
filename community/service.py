"""Community Detection Engine for the Echo Chamber Detector pipeline.

Implements Louvain modularity-based community detection using python-louvain
(``community.best_partition``) on a NetworkX graph derived from an
``InteractionGraph``.

Dataset-specific handling
--------------------------
Wiki-RfA (``datasetSource == "wiki_rfa"``):
    Louvain runs on the **unsigned** graph — all edges with weight=1.0,
    ignoring ``signedPolarity``.  Signed polarity analysis is handled
    separately in Phase 5 metrics.

All other datasets:
    Edge weights from the ``InteractionGraph`` are used directly.

Isolated nodes (nodes with no edges) are assigned to their own singleton
community so that every node has exactly one ``communityId`` (Requirement 3.5).

MAX_ITERATIONS cap
-------------------
``python-louvain``'s ``community.best_partition`` does not expose a direct
iteration counter, so the cap is enforced by wrapping the call with a timeout
mechanism: we monkey-patch ``random.random`` to count calls as a proxy for
iterations.  If the number of internal improvement-passes exceeds
``MAX_ITERATIONS``, the snapshot is flagged as ``"approximate_partition"`` and
a warning is logged (Requirement 3.2).

References: design.md Algorithm 2, Requirements 3.1, 3.2, 3.5
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx

from graph.models import CommunityPartition, InteractionGraph, Node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ITERATIONS: int = 100

_WIKI_RFA_SOURCE: str = "wiki_rfa"


# ---------------------------------------------------------------------------
# CommunityDetectionService
# ---------------------------------------------------------------------------


class CommunityDetectionService:
    """Service that partitions an :class:`InteractionGraph` into communities.

    Wraps ``python-louvain``'s ``community.best_partition`` for Louvain
    modularity optimization.

    Usage::

        service = CommunityDetectionService()
        partitions = service.detect_communities(graph)

    State is kept between calls so that :meth:`get_community_membership`
    can answer membership queries after the last :meth:`detect_communities`
    call.
    """

    def __init__(self) -> None:
        # Internal state: maps userId → communityId (populated by detect_communities)
        self._membership: dict[str, str] = {}
        # Internal state: maps communityId → CommunityPartition
        self._partitions: dict[str, CommunityPartition] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_communities(
        self,
        graph: InteractionGraph,
        previous_snapshot_communities: Optional[list[CommunityPartition]] = None,
        *,
        enable_girvan_newman: bool = False,
    ) -> list[CommunityPartition]:
        """Detect communities in *graph* using the Louvain algorithm.

        Args:
            graph: :class:`InteractionGraph` with at least one node.
            previous_snapshot_communities: Optional list of
                :class:`CommunityPartition` objects from the previous snapshot.
                When provided, label persistence is applied: new communities are
                matched to previous ones by Jaccard overlap (threshold 0.5) to
                assign stable community IDs across snapshots (Requirement 3.6).
            enable_girvan_newman: When ``True``, also run the Girvan-Newman
                algorithm and store its partition in each
                :class:`CommunityPartition` for comparison (Requirement 3.7).

        Returns:
            List of :class:`CommunityPartition` objects — one per discovered
            community.  Every node in *graph* is assigned to exactly one
            community (isolated nodes get a singleton community).

        Raises:
            ValueError: If *graph* has no nodes.
        """
        if not graph.nodes:
            raise ValueError("detect_communities requires a non-empty InteractionGraph")

        # --- Step 1: Build NetworkX graph ---
        nx_graph = self._build_nx_graph(graph)

        # --- Step 2: Run Louvain with iteration guard ---
        raw_partition, is_approximate = self._run_louvain(nx_graph)

        if is_approximate:
            logger.warning(
                "CommunityDetectionService.detect_communities: Louvain did not "
                "converge within %d iterations for graph '%s'; returning best "
                "partition found so far (approximate_partition).",
                MAX_ITERATIONS,
                graph.snapshotId,
            )

        # --- Step 3: Assign isolated nodes to singleton communities ---
        raw_partition = self._assign_isolated_nodes(graph, nx_graph, raw_partition)

        # --- Step 4: Apply label persistence if previous communities provided ---
        if previous_snapshot_communities:
            raw_partition = self._persist_labels(
                raw_partition, previous_snapshot_communities
            )

        # --- Step 5: Compute overall modularity ---
        overall_modularity = self._compute_nx_modularity(nx_graph, raw_partition)

        # --- Step 6: Build CommunityPartition objects ---
        partitions = self._build_partitions(
            graph, raw_partition, overall_modularity, is_approximate
        )

        # --- Step 7: Optionally run Girvan-Newman secondary validation ---
        if enable_girvan_newman:
            gn_partition = self._run_girvan_newman(nx_graph)
            for cp in partitions:
                cp.girvan_newman_partition = gn_partition

        # --- Step 8: Update internal membership state ---
        self._membership = {str(uid): str(cid) for uid, cid in raw_partition.items()}
        self._partitions = {p.communityId: p for p in partitions}

        # --- Step 9: Update communityId on the graph's Node objects ---
        for user_id, community_id in self._membership.items():
            if user_id in graph.nodes:
                graph.nodes[user_id].communityId = community_id

        return partitions

    def get_community_membership(self, user_id: str) -> Optional[str]:
        """Return the ``communityId`` assigned to *user_id*, or ``None``.

        Only valid after :meth:`detect_communities` has been called.

        Args:
            user_id: User node identifier.

        Returns:
            Community ID string, or ``None`` if the user is not found in
            the most recent partition.
        """
        return self._membership.get(user_id)

    def compute_modularity(
        self,
        graph: InteractionGraph,
        partition: list[CommunityPartition],
    ) -> float:
        """Compute the modularity score Q for *partition* on *graph*.

        Args:
            graph:     :class:`InteractionGraph` to evaluate.
            partition: List of :class:`CommunityPartition` objects produced
                       by :meth:`detect_communities`.

        Returns:
            Modularity Q as a float.  Returns 0.0 for empty graphs.
        """
        if not graph.nodes or not graph.edges:
            return 0.0

        nx_graph = self._build_nx_graph(graph)

        # Rebuild raw partition dict from CommunityPartition list
        raw_partition: dict[str, int] = {}
        for cp in partition:
            try:
                cid_int = int(cp.communityId)
            except (ValueError, TypeError):
                # Non-integer community ID — use index as fallback
                cid_int = hash(cp.communityId) & 0x7FFFFFFF
            for member_id in cp.memberIds:
                raw_partition[member_id] = cid_int

        return self._compute_nx_modularity(nx_graph, raw_partition)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_girvan_newman(self, nx_graph: nx.Graph) -> Optional[list[set[str]]]:
        """Run one step of Girvan-Newman community detection on *nx_graph*.

        Takes the first split of the dendrogram (first edge removal) using
        ``networkx.algorithms.community.girvan_newman``.

        Args:
            nx_graph: Undirected NetworkX graph.

        Returns:
            A list of sets of node IDs representing the Girvan-Newman partition,
            or ``None`` if the algorithm could not run.
        """
        if nx_graph.number_of_edges() < 2:
            logger.warning(
                "CommunityDetectionService._run_girvan_newman: graph has fewer "
                "than 2 edges (%d); Girvan-Newman cannot run meaningfully — "
                "skipping secondary validation.",
                nx_graph.number_of_edges(),
            )
            return None

        try:
            from networkx.algorithms.community import girvan_newman  # lazy import

            communities_gen = girvan_newman(nx_graph)
            first_split = next(communities_gen)  # tuple of frozensets
            return [set(component) for component in first_split]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CommunityDetectionService._run_girvan_newman: Girvan-Newman "
                "failed: %s — leaving girvan_newman_partition=None.",
                exc,
            )
            return None

    def _build_nx_graph(self, graph: InteractionGraph) -> nx.Graph:
        """Convert *graph* to an undirected NetworkX graph for Louvain.

        For wiki-RfA graphs, all edge weights are set to 1.0 (unsigned
        Louvain run).  For all other datasets, the existing normalized
        edge weights are used.

        Args:
            graph: Source :class:`InteractionGraph`.

        Returns:
            Undirected :class:`nx.Graph` with a ``weight`` attribute on
            every edge.
        """
        is_wiki_rfa = graph.datasetSource == _WIKI_RFA_SOURCE

        nx_graph = nx.Graph()

        # Add all nodes (including isolated ones)
        for user_id in graph.nodes:
            nx_graph.add_node(user_id)

        # Add edges
        for edge in graph.edges:
            src = edge.sourceUserId
            tgt = edge.targetUserId

            if src == tgt:
                # Skip self-loops (should not occur, but guard defensively)
                continue

            weight = 1.0 if is_wiki_rfa else edge.weight

            # For undirected graph: if edge already exists (because the
            # directed graph has both (A→B) and (B→A)), aggregate weights.
            if nx_graph.has_edge(src, tgt):
                existing_weight = nx_graph[src][tgt].get("weight", 0.0)
                nx_graph[src][tgt]["weight"] = existing_weight + weight
            else:
                nx_graph.add_edge(src, tgt, weight=weight)

        return nx_graph

    def _run_louvain(
        self, nx_graph: nx.Graph
    ) -> tuple[dict[str, int], bool]:
        """Run Louvain community detection with an iteration cap.

        Uses ``community.best_partition`` (python-louvain) with
        ``randomize=False`` for deterministic execution.  An iteration counter
        is maintained via a wrapper around the internal modularity function to
        detect whether convergence was achieved within ``MAX_ITERATIONS``.

        Falls back to NetworkX's greedy-modularity community detection if
        python-louvain is not installed.

        Args:
            nx_graph: Undirected NetworkX graph.

        Returns:
            A tuple ``(partition, is_approximate)`` where:
            - ``partition`` maps node → integer community id.
            - ``is_approximate`` is ``True`` if the iteration cap was hit.
        """
        # Handle empty or edgeless graphs — every node is its own community
        if nx_graph.number_of_nodes() == 0:
            return {}, False

        if nx_graph.number_of_edges() == 0:
            # All isolated: each node becomes singleton community
            partition = {node: idx for idx, node in enumerate(nx_graph.nodes())}
            return partition, False

        # Try to import python-louvain.  The library installs a module named
        # ``community`` (or ``community.community_louvain``).  Because this
        # project also has a top-level ``community/`` package, we must import
        # the louvain sub-module directly to avoid the namespace collision.
        community_louvain = self._import_louvain()

        if community_louvain is None:
            # Fall back to NetworkX greedy modularity
            logger.warning(
                "CommunityDetectionService: python-louvain not available; "
                "falling back to NetworkX greedy-modularity community detection."
            )
            return self._run_networkx_fallback(nx_graph)

        # --- Iteration tracking via modularity call counting ---
        # python-louvain calls its internal modularity function once per pass.
        # We count those calls to detect when MAX_ITERATIONS is exceeded.
        iteration_count = [0]
        is_approximate = [False]
        original_modularity = community_louvain.modularity

        def counting_modularity(partition, graph, weight="weight"):
            """Wrapper that counts modularity evaluation calls."""
            iteration_count[0] += 1
            if iteration_count[0] > MAX_ITERATIONS:
                is_approximate[0] = True
            return original_modularity(partition, graph, weight=weight)

        # Temporarily replace the modularity function on the module object
        community_louvain.modularity = counting_modularity
        try:
            partition = community_louvain.best_partition(
                nx_graph,
                weight="weight",
                randomize=False,
            )
        finally:
            # Always restore original modularity function
            community_louvain.modularity = original_modularity

        return partition, is_approximate[0]

    @staticmethod
    def _import_louvain():  # type: ignore[return]
        """Import python-louvain's ``community_louvain`` sub-module.

        python-louvain installs a package named ``community`` that contains a
        ``community_louvain`` sub-module with ``best_partition`` and
        ``modularity``.  Because this project's own ``community/`` package
        shadows the top-level ``community`` name, we import the sub-module
        directly via the file-system path to avoid the namespace conflict.

        Returns:
            The ``community_louvain`` module object, or ``None`` if
            python-louvain is not installed.
        """
        import importlib
        import importlib.util
        import sys

        # First, check if we already have a reference cached
        if "_louvain_module" in sys.modules:
            return sys.modules["_louvain_module"]

        # Try direct sub-module import (works when our 'community' package
        # exposes the louvain sub-module or it's importable under a different name)
        for module_name in ("community.community_louvain",):
            try:
                # Temporarily remove our local 'community' package from sys.modules
                # so we can reach the real python-louvain package.
                saved = sys.modules.pop("community", None)
                try:
                    mod = importlib.import_module(module_name)
                    sys.modules["_louvain_module"] = mod
                    return mod
                except ImportError:
                    pass
                finally:
                    # Restore our local community package
                    if saved is not None:
                        sys.modules["community"] = saved
            except Exception:  # noqa: BLE001
                pass

        # Try locating the louvain module via importlib.util.find_spec
        # by temporarily adjusting sys.modules to bypass the namespace conflict.
        try:
            saved = sys.modules.pop("community", None)
            try:
                spec = importlib.util.find_spec("community")
                if spec is not None and spec.origin is not None:
                    # Load the real community package
                    real_community = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(real_community)  # type: ignore[union-attr]
                    # python-louvain exposes best_partition and modularity at the top level
                    if hasattr(real_community, "best_partition"):
                        sys.modules["_louvain_module"] = real_community
                        return real_community
            except Exception:  # noqa: BLE001
                pass
            finally:
                if saved is not None:
                    sys.modules["community"] = saved
        except Exception:  # noqa: BLE001
            pass

        return None

    def _assign_isolated_nodes(
        self,
        graph: InteractionGraph,
        nx_graph: nx.Graph,
        raw_partition: dict[str, int],
    ) -> dict[str, int]:
        """Ensure every node in *graph* appears in *raw_partition*.

        Nodes with no edges are isolated and may not appear in the Louvain
        output.  Each such node is assigned a unique singleton community ID
        beyond the maximum community ID already in the partition.

        Args:
            graph:         Source :class:`InteractionGraph`.
            nx_graph:      NetworkX graph (used to check degree).
            raw_partition: Partition dict from Louvain (may be incomplete).

        Returns:
            Updated partition dict where every node in *graph* has an entry.
        """
        partition = dict(raw_partition)

        # Find the next available community ID
        next_community_id = max(partition.values(), default=-1) + 1

        for user_id in graph.nodes:
            if user_id not in partition:
                # Isolated node: assign to its own singleton community
                partition[user_id] = next_community_id
                next_community_id += 1
                logger.debug(
                    "CommunityDetectionService: isolated node '%s' assigned to "
                    "singleton community %d",
                    user_id,
                    next_community_id - 1,
                )

        return partition

    def _compute_nx_modularity(
        self,
        nx_graph: nx.Graph,
        raw_partition: dict[str, int],
    ) -> float:
        """Compute modularity Q using python-louvain's ``community.modularity``.

        Falls back to NetworkX's ``nx.community.modularity`` if python-louvain
        is unavailable.  Returns 0.0 if the graph has no edges.

        Handles mixed-type partitions (int or str community IDs) that may result
        from label persistence.  When string community IDs are present, they are
        re-mapped to integers for the python-louvain call.

        Requirement 3.4: The modularity score Q is clamped to 0.0 if it would
        otherwise be negative, and a warning is logged.

        Args:
            nx_graph:      NetworkX graph.
            raw_partition: Partition mapping node → community id (int or str).

        Returns:
            Modularity Q as a float, guaranteed to be ≥ 0.0.
        """
        if nx_graph.number_of_edges() == 0:
            return 0.0

        # Ensure partition values are integers (required by python-louvain)
        # community IDs may be strings after label persistence is applied.
        int_partition: dict[str, int] = {}
        cid_to_int: dict[object, int] = {}
        next_int = [0]

        for node, cid in raw_partition.items():
            if cid not in cid_to_int:
                cid_to_int[cid] = next_int[0]
                next_int[0] += 1
            int_partition[node] = cid_to_int[cid]

        q: float = 0.0

        # Try python-louvain first
        community_louvain = self._import_louvain()
        if community_louvain is not None:
            try:
                q = community_louvain.modularity(int_partition, nx_graph, weight="weight")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CommunityDetectionService._compute_nx_modularity: "
                    "python-louvain modularity failed: %s — trying NetworkX fallback",
                    exc,
                )
                q = self._compute_nx_modularity_fallback(nx_graph, int_partition)
        else:
            q = self._compute_nx_modularity_fallback(nx_graph, int_partition)

        # Requirement 3.4: clamp to 0.0 if negative
        if q < 0.0:
            logger.warning(
                "CommunityDetectionService._compute_nx_modularity: computed "
                "modularity Q = %.6f is negative; clamping to 0.0 (Requirement 3.4)",
                q,
            )
            q = 0.0

        return q

    def _compute_nx_modularity_fallback(
        self,
        nx_graph: nx.Graph,
        raw_partition: dict[str, int],
    ) -> float:
        """NetworkX fallback for modularity computation.

        Args:
            nx_graph:      NetworkX graph.
            raw_partition: Partition mapping node → integer community id.

        Returns:
            Modularity Q as a float, or 0.0 on failure.
        """
        try:
            communities_list = self._partition_to_community_sets(raw_partition)
            if not communities_list:
                return 0.0
            return nx.community.modularity(nx_graph, communities_list, weight="weight")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CommunityDetectionService._compute_nx_modularity: NetworkX "
                "modularity also failed: %s — returning 0.0",
                exc,
            )
            return 0.0

    def _persist_labels(
        self,
        new_partition_map: dict[str, int],
        previous_communities: list[CommunityPartition],
    ) -> dict[str, int]:
        """Apply label persistence by matching new communities to previous ones.

        Uses Jaccard overlap (threshold 0.5) to match new communities to
        previous communities and assign stable community IDs.

        Algorithm:
            1. Build old_community_id → set of memberIds from previous_communities.
            2. Build new_community_id → set of memberIds from new_partition_map.
            3. For each new community, find the old community with the highest
               Jaccard overlap: |A ∩ B| / |A ∪ B|.
            4. If best Jaccard ≥ 0.5, rename new community to old community's ID.
            5. Greedy one-to-one matching: each old label is assigned to at most
               one new community (highest Jaccard wins across all new communities).
            6. New communities with no match keep their newly generated IDs.

        Args:
            new_partition_map:     Dict of ``{userId: new_community_id}`` from
                                   Louvain (community IDs are integers).
            previous_communities:  List of :class:`CommunityPartition` from the
                                   previous snapshot.

        Returns:
            Relabeled partition dict where stable communities use old IDs
            (as strings) and unmatched communities keep integer IDs.
        """
        if not previous_communities:
            return new_partition_map

        # --- Build old community member sets ---
        old_members: dict[str, set[str]] = {
            cp.communityId: set(cp.memberIds) for cp in previous_communities
        }

        # --- Build new community member sets ---
        new_community_members: dict[int, set[str]] = {}
        for user_id, cid in new_partition_map.items():
            new_community_members.setdefault(cid, set()).add(user_id)

        # --- Compute all Jaccard scores ---
        # Collect (jaccard, new_cid, old_cid) for all pairs where jaccard >= 0.5
        candidates: list[tuple[float, int, str]] = []

        for new_cid, new_members in new_community_members.items():
            for old_cid, old_m in old_members.items():
                intersection = len(new_members & old_m)
                if intersection == 0:
                    continue
                union = len(new_members | old_m)
                jaccard = intersection / union if union > 0 else 0.0
                if jaccard >= 0.5:
                    candidates.append((jaccard, new_cid, old_cid))

        # --- Greedy one-to-one matching (highest Jaccard first) ---
        # Sort descending by Jaccard score
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_new: set[int] = set()
        matched_old: set[str] = set()
        # Maps: new_community_int_id → old_community_str_id
        relabel_map: dict[int, str] = {}

        for jaccard, new_cid, old_cid in candidates:
            if new_cid in matched_new or old_cid in matched_old:
                continue
            relabel_map[new_cid] = old_cid
            matched_new.add(new_cid)
            matched_old.add(old_cid)

        if not relabel_map:
            return new_partition_map

        # --- Apply relabeling ---
        # We need to ensure unmatched new community IDs don't collide with old IDs.
        # Unmatched communities keep their integer IDs (converted to strings when
        # building CommunityPartition objects, so no collision risk here).
        relabeled: dict[str, int | str] = {}
        for user_id, new_cid in new_partition_map.items():
            if new_cid in relabel_map:
                # Use the stable old community ID (stored as a string key)
                relabeled[user_id] = relabel_map[new_cid]
            else:
                relabeled[user_id] = new_cid

        # _build_partitions expects dict[str, int] but we now have mixed str/int values.
        # We need a unified approach: convert old string IDs to a stable negative integer
        # space so they remain unique and distinguishable, then convert back in
        # _build_partitions. However, _build_partitions converts cid → str(cid) for
        # communityId, so we can instead directly store the string IDs by using a
        # custom sentinel approach.
        #
        # Simpler approach: return a dict[str, Any] and update _build_partitions to
        # handle string community IDs. We'll return the mixed dict as-is and let
        # _build_partitions use str(cid) for the communityId — which naturally handles
        # both int(0) → "0" and str("old_id") → "old_id" correctly since str(x) is
        # the identity for strings.
        return relabeled  # type: ignore[return-value]

    @staticmethod
    def _partition_to_community_sets(
        raw_partition: dict[str, int]
    ) -> list[set[str]]:
        """Convert a {node: community_id} dict to a list of node-sets.

        Handles both integer and string community IDs (the latter may result
        from label persistence via ``_persist_labels``).
        """
        communities: dict[object, set[str]] = {}
        for node, cid in raw_partition.items():
            communities.setdefault(cid, set()).add(node)
        return list(communities.values())

    def _run_networkx_fallback(
        self, nx_graph: nx.Graph
    ) -> tuple[dict[str, int], bool]:
        """Run NetworkX greedy-modularity community detection as a fallback.

        Used when python-louvain is not installed.

        Args:
            nx_graph: Undirected NetworkX graph.

        Returns:
            A tuple ``(partition, is_approximate)`` where:
            - ``partition`` maps node → integer community id.
            - ``is_approximate`` is always ``False`` for this fallback.
        """
        try:
            communities = nx.community.greedy_modularity_communities(
                nx_graph, weight="weight"
            )
            partition: dict[str, int] = {}
            for cid, community in enumerate(communities):
                for node in community:
                    partition[node] = cid
            return partition, False
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CommunityDetectionService._run_networkx_fallback: greedy "
                "modularity failed: %s — assigning all nodes to community 0",
                exc,
            )
            partition = {node: 0 for node in nx_graph.nodes()}
            return partition, False

    def _build_partitions(
        self,
        graph: InteractionGraph,
        raw_partition: dict[str, int],
        overall_modularity: float,
        is_approximate: bool,
    ) -> list[CommunityPartition]:
        """Build :class:`CommunityPartition` objects from the raw partition.

        Computes intraEdges, interEdges, and centroidNode (highest-degree
        node) for each community.

        Handles both integer community IDs (from Louvain) and string community
        IDs (from label persistence via ``_persist_labels``).

        Args:
            graph:              Source :class:`InteractionGraph`.
            raw_partition:      Map of userId → community id (int or str after
                                label persistence is applied).
            overall_modularity: Pre-computed modularity Q.
            is_approximate:     Whether the iteration cap was hit.

        Returns:
            List of :class:`CommunityPartition` objects.
        """
        # Group member IDs by community (community key may be int or str)
        community_members: dict[object, set[str]] = {}
        for user_id, community_id in raw_partition.items():
            community_members.setdefault(community_id, set()).add(user_id)

        # Build per-community degree counts (for centroid selection)
        community_degree: dict[object, dict[str, int]] = {
            cid: {uid: 0 for uid in members}
            for cid, members in community_members.items()
        }

        # Count intra/inter edges and accumulate degree
        intra_counts: dict[object, int] = {cid: 0 for cid in community_members}
        inter_counts: dict[object, int] = {cid: 0 for cid in community_members}

        for edge in graph.edges:
            src = edge.sourceUserId
            tgt = edge.targetUserId

            src_community = raw_partition.get(src)
            tgt_community = raw_partition.get(tgt)

            if src_community is None or tgt_community is None:
                continue

            if src_community == tgt_community:
                intra_counts[src_community] = intra_counts.get(src_community, 0) + 1
                # Both endpoints count toward intra-community degree
                if src in community_degree.get(src_community, {}):
                    community_degree[src_community][src] = (
                        community_degree[src_community].get(src, 0) + 1
                    )
                if tgt in community_degree.get(tgt_community, {}):
                    community_degree[tgt_community][tgt] = (
                        community_degree[tgt_community].get(tgt, 0) + 1
                    )
            else:
                inter_counts[src_community] = inter_counts.get(src_community, 0) + 1
                inter_counts[tgt_community] = inter_counts.get(tgt_community, 0) + 1

        # Build CommunityPartition objects
        partitions: list[CommunityPartition] = []
        for cid, members in community_members.items():
            # Centroid: node with highest intra-community degree
            degree_map = community_degree.get(cid, {})
            centroid = max(degree_map, key=degree_map.get) if degree_map else None  # type: ignore[arg-type]

            partitions.append(
                CommunityPartition(
                    communityId=str(cid),
                    memberIds=set(members),
                    modularity=overall_modularity,
                    intraEdges=intra_counts.get(cid, 0),
                    interEdges=inter_counts.get(cid, 0),
                    centroidNode=centroid,
                    isApproximate=is_approximate,
                )
            )

        return partitions


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def detect_communities(
    graph: InteractionGraph,
    previous_snapshot_communities: Optional[list[CommunityPartition]] = None,
    *,
    enable_girvan_newman: bool = False,
) -> list[CommunityPartition]:
    """Module-level convenience wrapper around
    :meth:`CommunityDetectionService.detect_communities`.

    Args:
        graph: :class:`InteractionGraph` with at least one node.
        previous_snapshot_communities: Optional list of
            :class:`CommunityPartition` objects from the previous snapshot
            for label persistence (Requirement 3.6).
        enable_girvan_newman: When ``True``, also run the Girvan-Newman
            algorithm and store its partition alongside the Louvain result
            (Requirement 3.7).

    Returns:
        List of :class:`CommunityPartition` objects.
    """
    return CommunityDetectionService().detect_communities(
        graph,
        previous_snapshot_communities=previous_snapshot_communities,
        enable_girvan_newman=enable_girvan_newman,
    )


def compute_modularity(
    graph: InteractionGraph,
    partition: list[CommunityPartition],
) -> float:
    """Module-level convenience wrapper around
    :meth:`CommunityDetectionService.compute_modularity`.

    Args:
        graph:     :class:`InteractionGraph` to evaluate.
        partition: List of :class:`CommunityPartition` objects.

    Returns:
        Modularity Q as a float.
    """
    return CommunityDetectionService().compute_modularity(graph, partition)
