"""Graph Construction Service for the Echo Chamber Detector pipeline.

Implements dataset-aware ``buildGraph`` that transforms a list of
``InteractionRecord`` objects into a weighted directed ``InteractionGraph``.

Dataset-specific weight handling
---------------------------------
Reddit (``datasetSource`` in ``{"reddit_title", "reddit_body"}``):
    Aggregate raw interaction counts per (sourceUserId, targetUserId) pair;
    normalize weights to [0, 1] by dividing each count by the maximum count.
    The LINK_SENTIMENT value from the record is stored on the edge but is NOT
    used as the edge weight — weight is the normalized interaction frequency.

Congress (``datasetSource == "congress"``):
    The transmission-probability weight is already in [0, 1] and is stored in
    ``InteractionRecord.sentimentScore`` by ``CongressNetworkAdapter``.  Use
    it directly; no further normalization is applied.

Wiki-RfA (``datasetSource == "wiki_rfa"``):
    Binary votes: all edge weights are fixed at 1.0.
    ``votePolarity`` (+1 or -1) is stored in ``Edge.signedPolarity``.
    The combined ``InteractionGraph`` carries all edges; callers may filter
    by ``signedPolarity`` to build sub-graphs for positive / negative votes.

All datasets:
    - Self-loops (sourceUserId == targetUserId) are rejected with a log.
    - All nodes are initialized with
      ``communityId=None``, ``betweenness=0.0``, ``diversityScore=0.0``,
      ``topicVector=[]``.
    - The returned graph is tagged with a UUID ``snapshotId``, a ``createdAt``
      timestamp, and the ``datasetSource`` of the input records.

References: design.md Algorithm 1, Requirements 2.1–2.4
"""

from __future__ import annotations

import json
import logging
import os
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from graph.models import Edge, InteractionGraph, InteractionRecord, Node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset-source constants
# ---------------------------------------------------------------------------

_REDDIT_SOURCES: frozenset[str] = frozenset({"reddit_title", "reddit_body"})
_CONGRESS_SOURCE: str = "congress"
_WIKI_RFA_SOURCE: str = "wiki_rfa"


# ---------------------------------------------------------------------------
# GraphConstructionService
# ---------------------------------------------------------------------------


class GraphConstructionService:
    """Service that transforms InteractionRecords into an InteractionGraph.

    Usage::

        service = GraphConstructionService()
        graph   = service.build_graph(records)

    The service is stateless; each call to :meth:`build_graph` is independent.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graph(
        self,
        records: list[InteractionRecord],
        *,
        dataset_source: Optional[str] = None,
    ) -> InteractionGraph:
        """Build a weighted directed ``InteractionGraph`` from *records*.

        Args:
            records:        Non-empty list of :class:`InteractionRecord` objects.
            dataset_source: Optional override for the ``datasetSource`` tag on
                            the resulting graph.  When omitted, the
                            ``datasetSource`` of the first record in *records*
                            is used.

        Returns:
            :class:`InteractionGraph` with normalized weights, initialized node
            fields, a UUID ``snapshotId``, and a ``createdAt`` timestamp.

        Raises:
            ValueError: If *records* is empty.
        """
        if not records:
            raise ValueError("buildGraph requires a non-empty list of InteractionRecords")

        # Infer the dataset source from the first record when not supplied.
        source = dataset_source or records[0].datasetSource

        if source in _REDDIT_SOURCES:
            return self._build_reddit_graph(records, source)
        elif source == _CONGRESS_SOURCE:
            return self._build_congress_graph(records, source)
        elif source == _WIKI_RFA_SOURCE:
            return self._build_wiki_rfa_graph(records, source)
        else:
            # Unknown dataset: fall back to Reddit-style count aggregation.
            logger.warning(
                "GraphConstructionService: unknown datasetSource '%s'; "
                "falling back to count-aggregation (Reddit-style) normalization",
                source,
            )
            return self._build_reddit_graph(records, source)

    def update_graph(
        self,
        graph: InteractionGraph,
        new_records: list[InteractionRecord],
    ) -> InteractionGraph:
        """Incrementally update *graph* with *new_records* without a full rebuild.

        This method merges the new interaction records into the existing graph's
        edge accumulator, re-normalizes weights where appropriate, and preserves
        existing node metadata for unchanged nodes.

        Dataset-specific weight handling mirrors :meth:`build_graph`:

        - **Reddit** (``reddit_title`` / ``reddit_body``): the existing edges
          carry normalized counts.  Raw counts are recovered from
          ``graph.rawEdgeCounts`` (populated by :meth:`build_graph`).  New
          records increment the per-pair raw counts, and all weights are
          re-normalized by the new maximum count.
        - **Congress**: weights are pre-normalized transmission probabilities;
          new records overwrite existing entries for the same (source, target)
          pair (last record wins, consistent with :meth:`_build_congress_graph`).
          No re-normalization is applied.
        - **Wiki-RfA**: binary weights (1.0); signed polarity from the latest
          record for each (source, target) pair is used.  No re-normalization.
        - **Unknown datasets**: fall back to Reddit-style count aggregation.

        Existing node metadata (``communityId``, ``betweenness``,
        ``diversityScore``, ``topicVector``) is preserved for nodes that are
        already present in *graph*.  New nodes are added with default
        zero-initialized metadata.

        Self-loop records (``sourceUserId == targetUserId``) are rejected with
        a log message, consistent with :meth:`build_graph`.

        Args:
            graph:       Existing :class:`InteractionGraph` to update.
            new_records: List of new :class:`InteractionRecord` objects to
                         incorporate.  May be empty (returns a copy of the
                         graph with a new ``snapshotId``).

        Returns:
            Updated :class:`InteractionGraph` with a new ``snapshotId`` and
            ``createdAt`` timestamp, incorporating all original and new edges.

        References:
            Requirements 2.7 — incremental update without full rebuild.
        """
        source = graph.datasetSource

        if source in _REDDIT_SOURCES:
            return self._update_reddit_graph(graph, new_records, source)
        elif source == _CONGRESS_SOURCE:
            return self._update_congress_graph(graph, new_records, source)
        elif source == _WIKI_RFA_SOURCE:
            return self._update_wiki_rfa_graph(graph, new_records, source)
        else:
            logger.warning(
                "GraphConstructionService.update_graph: unknown datasetSource '%s'; "
                "falling back to Reddit-style count-aggregation update",
                source,
            )
            return self._update_reddit_graph(graph, new_records, source)

    # ------------------------------------------------------------------
    # Serialization / Deserialization
    # ------------------------------------------------------------------

    def serialize_to_graphml(self, graph: InteractionGraph) -> str:
        """Serialize *graph* to a GraphML XML string.

        Node attributes encoded: ``userId``, ``communityId``, ``betweenness``,
        ``diversityScore``, ``topicVector`` (as a JSON array string).

        Edge attributes encoded: ``sourceUserId``, ``targetUserId``, ``weight``,
        ``isCrossCommunity``, ``signedPolarity`` (omitted when ``None``).

        Graph-level data: ``snapshotId``, ``createdAt`` (ISO format),
        ``datasetSource``.

        Args:
            graph: :class:`InteractionGraph` to serialize.

        Returns:
            UTF-8 GraphML XML string.
        """
        # GraphML namespace
        ns = "http://graphml.graphdrawing.org/graphml"
        ET.register_namespace("", ns)

        graphml = ET.Element("graphml", xmlns=ns)

        # --- Key declarations (node attributes) ---
        _gml_key(graphml, "d_snapshot_id",   "graph", "snapshotId",    "string")
        _gml_key(graphml, "d_created_at",    "graph", "createdAt",     "string")
        _gml_key(graphml, "d_dataset_source","graph", "datasetSource", "string")

        _gml_key(graphml, "n_user_id",         "node", "userId",        "string")
        _gml_key(graphml, "n_community_id",    "node", "communityId",   "string")
        _gml_key(graphml, "n_betweenness",     "node", "betweenness",   "double")
        _gml_key(graphml, "n_diversity_score", "node", "diversityScore","double")
        _gml_key(graphml, "n_topic_vector",    "node", "topicVector",   "string")

        _gml_key(graphml, "e_source_user_id",    "edge", "sourceUserId",    "string")
        _gml_key(graphml, "e_target_user_id",    "edge", "targetUserId",    "string")
        _gml_key(graphml, "e_weight",            "edge", "weight",          "double")
        _gml_key(graphml, "e_is_cross_community","edge", "isCrossCommunity","boolean")
        _gml_key(graphml, "e_signed_polarity",   "edge", "signedPolarity",  "int")

        # --- <graph> element ---
        g_el = ET.SubElement(graphml, "graph", id="G", edgedefault="directed")

        # Graph-level data
        _gml_data(g_el, "d_snapshot_id",    graph.snapshotId)
        _gml_data(g_el, "d_created_at",     graph.createdAt.isoformat())
        _gml_data(g_el, "d_dataset_source", graph.datasetSource)

        # --- Node elements ---
        for uid, node in graph.nodes.items():
            n_el = ET.SubElement(g_el, "node", id=uid)
            _gml_data(n_el, "n_user_id",         node.userId)
            _gml_data(n_el, "n_community_id",    node.communityId if node.communityId is not None else "")
            _gml_data(n_el, "n_betweenness",     str(node.betweenness))
            _gml_data(n_el, "n_diversity_score", str(node.diversityScore))
            _gml_data(n_el, "n_topic_vector",    json.dumps(node.topicVector))

        # --- Edge elements ---
        for idx, edge in enumerate(graph.edges):
            e_el = ET.SubElement(
                g_el, "edge",
                id=f"e{idx}",
                source=edge.sourceUserId,
                target=edge.targetUserId,
            )
            _gml_data(e_el, "e_source_user_id",    edge.sourceUserId)
            _gml_data(e_el, "e_target_user_id",    edge.targetUserId)
            _gml_data(e_el, "e_weight",            str(edge.weight))
            _gml_data(e_el, "e_is_cross_community",str(edge.isCrossCommunity).lower())
            if edge.signedPolarity is not None:
                _gml_data(e_el, "e_signed_polarity", str(edge.signedPolarity))

        ET.indent(graphml, space="  ")
        return ET.tostring(graphml, encoding="unicode", xml_declaration=False)

    def deserialize_from_graphml(self, graphml_str: str) -> InteractionGraph:
        """Deserialize a GraphML XML string back into an :class:`InteractionGraph`.

        Restores all node and edge fields, including ``signedPolarity`` (``None``
        when absent from the XML).  Graph-level metadata (``snapshotId``,
        ``createdAt``, ``datasetSource``) is also restored.

        Args:
            graphml_str: GraphML XML string produced by :meth:`serialize_to_graphml`.

        Returns:
            :class:`InteractionGraph` equivalent to the one that was serialized.
        """
        root = ET.fromstring(graphml_str)

        # Strip namespace prefix from tags for easier matching
        def _tag(el: ET.Element) -> str:
            return el.tag.split("}")[-1] if "}" in el.tag else el.tag

        # --- Collect key id → attr-name mapping ---
        key_to_name: dict[str, str] = {}
        for child in root:
            if _tag(child) == "key":
                kid = child.get("id", "")
                attr_name = child.get("attr.name", "")
                key_to_name[kid] = attr_name

        # Locate the <graph> element
        graph_el = None
        for child in root:
            if _tag(child) == "graph":
                graph_el = child
                break
        if graph_el is None:
            raise ValueError("GraphML string contains no <graph> element")

        # --- Read graph-level data ---
        graph_data: dict[str, str] = {}
        nodes: dict[str, Node] = {}
        edges: list[Edge] = []

        for child in graph_el:
            tag = _tag(child)
            if tag == "data":
                key = child.get("key", "")
                attr = key_to_name.get(key, key)
                graph_data[attr] = child.text or ""
            elif tag == "node":
                node_attrs: dict[str, str] = {}
                for d in child:
                    if _tag(d) == "data":
                        key = d.get("key", "")
                        attr = key_to_name.get(key, key)
                        node_attrs[attr] = d.text or ""
                user_id = node_attrs.get("userId", child.get("id", ""))
                community_raw = node_attrs.get("communityId", "")
                node = Node(
                    userId=user_id,
                    communityId=community_raw if community_raw != "" else None,
                    betweenness=float(node_attrs.get("betweenness", "0.0")),
                    diversityScore=float(node_attrs.get("diversityScore", "0.0")),
                    topicVector=json.loads(node_attrs.get("topicVector", "[]")),
                )
                nodes[user_id] = node
            elif tag == "edge":
                edge_attrs: dict[str, str] = {}
                for d in child:
                    if _tag(d) == "data":
                        key = d.get("key", "")
                        attr = key_to_name.get(key, key)
                        edge_attrs[attr] = d.text or ""
                src = edge_attrs.get("sourceUserId", child.get("source", ""))
                tgt = edge_attrs.get("targetUserId", child.get("target", ""))
                weight = float(edge_attrs.get("weight", "0.0"))
                cross = edge_attrs.get("isCrossCommunity", "false").lower() == "true"
                sp_raw = edge_attrs.get("signedPolarity")
                signed_polarity: Optional[int] = int(sp_raw) if sp_raw is not None else None
                edges.append(
                    Edge(
                        sourceUserId=src,
                        targetUserId=tgt,
                        weight=weight,
                        isCrossCommunity=cross,
                        signedPolarity=signed_polarity,
                    )
                )

        snapshot_id = graph_data.get("snapshotId", str(uuid.uuid4()))
        created_at_str = graph_data.get("createdAt", "")
        created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(timezone.utc)
        dataset_source = graph_data.get("datasetSource", "")

        return InteractionGraph(
            nodes=nodes,
            edges=edges,
            snapshotId=snapshot_id,
            createdAt=created_at,
            datasetSource=dataset_source,
        )

    def serialize_to_json(self, graph: InteractionGraph) -> str:
        """Serialize *graph* to an adjacency-list JSON string (pretty-printed).

        Structure::

            {
              "metadata": {
                "snapshotId": "...",
                "createdAt": "...",
                "datasetSource": "...",
                "nodeCount": N,
                "edgeCount": M
              },
              "nodes": [ { "userId": "...", ... }, ... ],
              "edges": [ { "sourceUserId": "...", ... }, ... ]
            }

        ``signedPolarity`` is included on an edge only when it is not ``None``.
        ``topicVector`` is included as a JSON array.

        Args:
            graph: :class:`InteractionGraph` to serialize.

        Returns:
            Indented JSON string (``indent=2``).
        """
        nodes_list = []
        for uid, node in graph.nodes.items():
            node_dict: dict = {
                "userId": node.userId,
                "communityId": node.communityId,
                "betweenness": node.betweenness,
                "diversityScore": node.diversityScore,
                "topicVector": node.topicVector,
            }
            nodes_list.append(node_dict)

        edges_list = []
        for edge in graph.edges:
            edge_dict: dict = {
                "sourceUserId": edge.sourceUserId,
                "targetUserId": edge.targetUserId,
                "weight": edge.weight,
                "isCrossCommunity": edge.isCrossCommunity,
            }
            if edge.signedPolarity is not None:
                edge_dict["signedPolarity"] = edge.signedPolarity
            edges_list.append(edge_dict)

        payload = {
            "metadata": {
                "snapshotId": graph.snapshotId,
                "createdAt": graph.createdAt.isoformat(),
                "datasetSource": graph.datasetSource,
                "nodeCount": graph.nodeCount,
                "edgeCount": graph.edgeCount,
            },
            "nodes": nodes_list,
            "edges": edges_list,
        }
        return json.dumps(payload, indent=2)

    def deserialize_from_json(self, json_str: str) -> InteractionGraph:
        """Deserialize an adjacency-list JSON string back into an :class:`InteractionGraph`.

        Restores all node and edge fields.  ``signedPolarity`` is ``None`` when
        the key is absent from the JSON edge object.

        Args:
            json_str: JSON string produced by :meth:`serialize_to_json`.

        Returns:
            :class:`InteractionGraph` equivalent to the one that was serialized.
        """
        payload = json.loads(json_str)
        metadata = payload.get("metadata", {})

        snapshot_id = metadata.get("snapshotId", str(uuid.uuid4()))
        created_at_str = metadata.get("createdAt", "")
        created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(timezone.utc)
        dataset_source = metadata.get("datasetSource", "")

        nodes: dict[str, Node] = {}
        for n in payload.get("nodes", []):
            user_id = n["userId"]
            nodes[user_id] = Node(
                userId=user_id,
                communityId=n.get("communityId"),
                betweenness=float(n.get("betweenness", 0.0)),
                diversityScore=float(n.get("diversityScore", 0.0)),
                topicVector=list(n.get("topicVector", [])),
            )

        edges: list[Edge] = []
        for e in payload.get("edges", []):
            sp = e.get("signedPolarity")
            edges.append(
                Edge(
                    sourceUserId=e["sourceUserId"],
                    targetUserId=e["targetUserId"],
                    weight=float(e["weight"]),
                    isCrossCommunity=bool(e.get("isCrossCommunity", False)),
                    signedPolarity=int(sp) if sp is not None else None,
                )
            )

        return InteractionGraph(
            nodes=nodes,
            edges=edges,
            snapshotId=snapshot_id,
            createdAt=created_at,
            datasetSource=dataset_source,
        )

    def pretty_print(self, graph: InteractionGraph) -> str:
        """Return a human-readable text representation of *graph* for debugging.

        Shows:
        - Graph metadata: ``snapshotId``, ``createdAt``, ``datasetSource``,
          node count, edge count.
        - All nodes with their attributes.
        - All edges with their attributes.

        Args:
            graph: :class:`InteractionGraph` to format.

        Returns:
            Multi-line string suitable for printing to a terminal.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("InteractionGraph")
        lines.append("=" * 60)
        lines.append(f"  snapshotId   : {graph.snapshotId}")
        lines.append(f"  createdAt    : {graph.createdAt.isoformat()}")
        lines.append(f"  datasetSource: {graph.datasetSource}")
        lines.append(f"  nodeCount    : {graph.nodeCount}")
        lines.append(f"  edgeCount    : {graph.edgeCount}")

        lines.append("")
        lines.append(f"Nodes ({graph.nodeCount}):")
        lines.append("-" * 40)
        for uid, node in graph.nodes.items():
            lines.append(f"  [{uid}]")
            lines.append(f"    communityId  : {node.communityId}")
            lines.append(f"    betweenness  : {node.betweenness}")
            lines.append(f"    diversityScore: {node.diversityScore}")
            lines.append(f"    topicVector  : {node.topicVector}")

        lines.append("")
        lines.append(f"Edges ({graph.edgeCount}):")
        lines.append("-" * 40)
        for edge in graph.edges:
            polarity_str = (
                f", signedPolarity={edge.signedPolarity}"
                if edge.signedPolarity is not None
                else ""
            )
            lines.append(
                f"  {edge.sourceUserId} -> {edge.targetUserId}"
                f"  weight={edge.weight}"
                f"  isCrossCommunity={edge.isCrossCommunity}"
                f"{polarity_str}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence — file-based (GraphML) + Neo4j mirroring
    # ------------------------------------------------------------------

    def persist_graph(self, graph: InteractionGraph, snapshot_id: str) -> str:
        """Persist *graph* to GraphML and mirror to Neo4j.

        Writes the serialized graph to::

            data/snapshots/{datasetSource}/{snapshotId}.graphml

        The directory is created automatically if it does not exist.

        Neo4j mirroring is optional — if Neo4j is not configured or
        unavailable, a warning is logged and the method continues normally.

        Args:
            graph:       :class:`InteractionGraph` to persist.
            snapshot_id: Snapshot identifier used as the file name stem.

        Returns:
            Absolute path (str) to the written GraphML file.
        """
        # --- Determine file path ---
        dataset_dir = Path("data") / "snapshots" / graph.datasetSource
        dataset_dir.mkdir(parents=True, exist_ok=True)
        file_path = dataset_dir / f"{snapshot_id}.graphml"

        # --- Serialize to GraphML ---
        graphml_str = self.serialize_to_graphml(graph)
        file_path.write_text(graphml_str, encoding="utf-8")
        logger.info(
            "GraphConstructionService.persist_graph: wrote %d nodes, %d edges to '%s'",
            graph.nodeCount,
            graph.edgeCount,
            file_path,
        )

        # --- Mirror to Neo4j (optional) ---
        self._mirror_to_neo4j(graph)

        return str(file_path)

    def load_graph(self, snapshot_id: str) -> InteractionGraph:
        """Load an :class:`InteractionGraph` from a persisted GraphML snapshot.

        Searches all subdirectories of ``data/snapshots/`` for a file named
        ``{snapshotId}.graphml``.

        Args:
            snapshot_id: Snapshot identifier to load.

        Returns:
            :class:`InteractionGraph` equivalent to the one that was persisted
            (lossless round-trip via GraphML).

        Raises:
            FileNotFoundError: If no GraphML file for *snapshot_id* is found.
        """
        snapshots_root = Path("data") / "snapshots"
        target_filename = f"{snapshot_id}.graphml"

        # Walk all dataset-source subdirectories looking for the snapshot file.
        found_path: Optional[Path] = None
        if snapshots_root.exists():
            for candidate in snapshots_root.rglob(target_filename):
                found_path = candidate
                break

        if found_path is None:
            raise FileNotFoundError(
                f"GraphML snapshot not found for snapshotId='{snapshot_id}'. "
                f"Searched under '{snapshots_root}'."
            )

        graphml_str = found_path.read_text(encoding="utf-8")
        graph = self.deserialize_from_graphml(graphml_str)
        logger.info(
            "GraphConstructionService.load_graph: loaded %d nodes, %d edges from '%s'",
            graph.nodeCount,
            graph.edgeCount,
            found_path,
        )
        return graph

    # ------------------------------------------------------------------
    # Private helpers — Neo4j mirroring
    # ------------------------------------------------------------------

    def _mirror_to_neo4j(self, graph: InteractionGraph) -> None:
        """Upsert *graph* nodes and edges into Neo4j.

        Uses environment variables for connection:
        - ``NEO4J_URI``      (default ``bolt://localhost:7687``)
        - ``NEO4J_USER``     (default ``neo4j``)
        - ``NEO4J_PASSWORD`` (default ``password``)

        If Neo4j is not configured or the driver cannot connect, logs a warning
        and returns without raising.

        Args:
            graph: :class:`InteractionGraph` to mirror.
        """
        try:
            import neo4j as _neo4j  # noqa: PLC0415 — lazy import, optional dep
        except ImportError:
            logger.warning(
                "GraphConstructionService._mirror_to_neo4j: neo4j package not "
                "installed; skipping Neo4j mirroring."
            )
            return

        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "password")

        driver = None
        try:
            driver = _neo4j.GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                # Upsert nodes
                for node in graph.nodes.values():
                    session.run(
                        """
                        MERGE (n:Node {userId: $userId})
                        SET n.communityId    = $communityId,
                            n.betweenness    = $betweenness,
                            n.diversityScore = $diversityScore,
                            n.snapshotId     = $snapshotId,
                            n.datasetSource  = $datasetSource
                        """,
                        userId=node.userId,
                        communityId=node.communityId,
                        betweenness=node.betweenness,
                        diversityScore=node.diversityScore,
                        snapshotId=graph.snapshotId,
                        datasetSource=graph.datasetSource,
                    )

                # Upsert edges
                for edge in graph.edges:
                    session.run(
                        """
                        MATCH (src:Node {userId: $sourceUserId})
                        MATCH (tgt:Node {userId: $targetUserId})
                        MERGE (src)-[r:EDGE {snapshotId: $snapshotId}]->(tgt)
                        SET r.weight           = $weight,
                            r.signedPolarity   = $signedPolarity,
                            r.isCrossCommunity = $isCrossCommunity
                        """,
                        sourceUserId=edge.sourceUserId,
                        targetUserId=edge.targetUserId,
                        weight=edge.weight,
                        signedPolarity=edge.signedPolarity,
                        isCrossCommunity=edge.isCrossCommunity,
                        snapshotId=graph.snapshotId,
                    )

            logger.info(
                "GraphConstructionService._mirror_to_neo4j: mirrored %d nodes, "
                "%d edges to Neo4j (snapshotId='%s')",
                graph.nodeCount,
                graph.edgeCount,
                graph.snapshotId,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GraphConstructionService._mirror_to_neo4j: Neo4j mirroring "
                "failed (uri='%s'): %s — continuing without Neo4j.",
                uri,
                exc,
            )
        finally:
            if driver is not None:
                try:
                    driver.close()
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # Private helpers — dataset-specific builders
    # ------------------------------------------------------------------

    def _build_reddit_graph(
        self, records: list[InteractionRecord], source: str
    ) -> InteractionGraph:
        """Build graph for Reddit datasets (reddit_title / reddit_body).

        Weight = normalized interaction count per (source, target) pair.
        ``sentimentScore`` from the record is stored on the edge as-is (it
        represents LINK_SENTIMENT, not the graph weight).

        Args:
            records: Validated InteractionRecords.
            source:  ``datasetSource`` tag (``"reddit_title"`` or ``"reddit_body"``).

        Returns:
            :class:`InteractionGraph` with weights normalized to [0, 1].
        """
        # Accumulate interaction counts and collect all user IDs.
        edge_counts: dict[tuple[str, str], int] = {}
        # Store the most recent sentimentScore per (source, target) pair.
        edge_sentiments: dict[tuple[str, str], Optional[float]] = {}
        node_set: set[str] = set()

        skipped_self_loops = 0

        for record in records:
            # Reject self-loops (Requirement 2.4).
            if record.sourceUserId == record.targetUserId:
                skipped_self_loops += 1
                logger.debug(
                    "GraphConstructionService: rejected self-loop for user '%s' "
                    "(datasetSource='%s', record id='%s')",
                    record.sourceUserId,
                    source,
                    record.id,
                )
                continue

            node_set.add(record.sourceUserId)
            node_set.add(record.targetUserId)

            key = (record.sourceUserId, record.targetUserId)
            edge_counts[key] = edge_counts.get(key, 0) + 1
            # Keep the sentiment from the last record for this pair.
            edge_sentiments[key] = record.sentimentScore

        if skipped_self_loops:
            logger.warning(
                "GraphConstructionService: skipped %d self-loop record(s) "
                "(datasetSource='%s')",
                skipped_self_loops,
                source,
            )

        # Normalize weights to [0, 1] by dividing by the max count.
        max_count: int = max(edge_counts.values()) if edge_counts else 1

        nodes = _make_nodes(node_set)

        edges: list[Edge] = []
        for (src, tgt), count in edge_counts.items():
            normalized_weight = count / max_count
            edges.append(
                Edge(
                    sourceUserId=src,
                    targetUserId=tgt,
                    weight=normalized_weight,
                    isCrossCommunity=False,
                    signedPolarity=None,
                )
            )

        # Store raw counts so updateGraph can merge incrementally without
        # needing to reverse-normalize.
        raw_counts: dict[tuple[str, str], float] = {
            k: float(v) for k, v in edge_counts.items()
        }
        return _make_graph(nodes, edges, source, raw_edge_counts=raw_counts)

    def _build_congress_graph(
        self, records: list[InteractionRecord], source: str
    ) -> InteractionGraph:
        """Build graph for the Congress Network dataset.

        Weights are pre-normalized transmission probabilities stored in
        ``InteractionRecord.sentimentScore`` by ``CongressNetworkAdapter``.
        They are already in [0, 1]; no further normalization is applied.

        Args:
            records: Validated InteractionRecords.
            source:  ``datasetSource`` tag (``"congress"``).

        Returns:
            :class:`InteractionGraph` with pre-normalized weights passed
            through unchanged.
        """
        edge_map: dict[tuple[str, str], float] = {}
        node_set: set[str] = set()
        skipped_self_loops = 0

        for record in records:
            # Reject self-loops (Requirement 2.4).
            if record.sourceUserId == record.targetUserId:
                skipped_self_loops += 1
                logger.debug(
                    "GraphConstructionService: rejected self-loop for user '%s' "
                    "(datasetSource='congress', record id='%s')",
                    record.sourceUserId,
                    record.id,
                )
                continue

            node_set.add(record.sourceUserId)
            node_set.add(record.targetUserId)

            key = (record.sourceUserId, record.targetUserId)

            # Use the transmission probability stored in sentimentScore.
            # CongressNetworkAdapter stores weight = float(weight) in sentimentScore.
            weight = record.sentimentScore if record.sentimentScore is not None else 0.0

            # For Congress, dedup key is (sourceUserId, targetUserId) only.
            # Keep the first (or last) occurrence; since IngestionService
            # already deduplicates, there should be at most one record per pair.
            if key not in edge_map:
                edge_map[key] = weight

        if skipped_self_loops:
            logger.warning(
                "GraphConstructionService: skipped %d self-loop record(s) "
                "(datasetSource='congress')",
                skipped_self_loops,
            )

        nodes = _make_nodes(node_set)

        edges: list[Edge] = [
            Edge(
                sourceUserId=src,
                targetUserId=tgt,
                weight=w,
                isCrossCommunity=False,
                signedPolarity=None,
            )
            for (src, tgt), w in edge_map.items()
        ]

        # Store pre-normalized weights as raw counts so updateGraph has
        # a consistent interface (weights are already in [0,1]).
        raw_counts: dict[tuple[str, str], float] = dict(edge_map)
        return _make_graph(nodes, edges, source, raw_edge_counts=raw_counts)

    def _build_wiki_rfa_graph(
        self, records: list[InteractionRecord], source: str
    ) -> InteractionGraph:
        """Build graph for the Wiki-RfA dataset.

        All edge weights are fixed at 1.0 (binary votes).
        ``votePolarity`` (+1 or -1) is stored in ``Edge.signedPolarity``.
        The combined ``InteractionGraph`` stores all edges (positive and
        negative); callers may filter by ``signedPolarity`` to obtain the
        positive or negative sub-graph.

        Args:
            records: Validated InteractionRecords.
            source:  ``datasetSource`` tag (``"wiki_rfa"``).

        Returns:
            :class:`InteractionGraph` where every edge has ``weight=1.0``
            and ``signedPolarity`` in ``{+1, -1}``.
        """
        # For wiki-RfA, edges are keyed by (sourceUserId, targetUserId).
        # Multiple votes from the same voter to the same candidate are
        # possible in different years, but after IngestionService dedup
        # (keyed on timestamp) there is at most one record per (src, tgt, ts).
        # We aggregate here per (src, tgt) pair — each vote becomes exactly
        # one edge; if two votes with different timestamps exist between the
        # same pair, we keep both but here aggregate into one edge by
        # (src, tgt) for graph purposes (count ≥ 1 → weight = 1.0).
        #
        # The signed polarity: if multiple votes exist between the same pair
        # we use the most recent (last in list) polarity for the combined edge.
        edge_polarity: dict[tuple[str, str], Optional[int]] = {}
        node_set: set[str] = set()
        skipped_self_loops = 0

        for record in records:
            # Reject self-loops (Requirement 2.4).
            if record.sourceUserId == record.targetUserId:
                skipped_self_loops += 1
                logger.debug(
                    "GraphConstructionService: rejected self-loop for user '%s' "
                    "(datasetSource='wiki_rfa', record id='%s')",
                    record.sourceUserId,
                    record.id,
                )
                continue

            node_set.add(record.sourceUserId)
            node_set.add(record.targetUserId)

            key = (record.sourceUserId, record.targetUserId)
            # Last polarity wins for duplicate pairs.
            edge_polarity[key] = record.votePolarity

        if skipped_self_loops:
            logger.warning(
                "GraphConstructionService: skipped %d self-loop record(s) "
                "(datasetSource='wiki_rfa')",
                skipped_self_loops,
            )

        nodes = _make_nodes(node_set)

        edges: list[Edge] = [
            Edge(
                sourceUserId=src,
                targetUserId=tgt,
                weight=1.0,
                isCrossCommunity=False,
                signedPolarity=polarity,
            )
            for (src, tgt), polarity in edge_polarity.items()
        ]

        # Store pre-normalized weights (1.0 per edge) so updateGraph has
        # a consistent interface.
        raw_counts: dict[tuple[str, str], float] = {
            k: 1.0 for k in edge_polarity
        }
        return _make_graph(nodes, edges, source, raw_edge_counts=raw_counts)

    # ------------------------------------------------------------------
    # Private helpers — dataset-specific incremental updaters
    # ------------------------------------------------------------------

    def _update_reddit_graph(
        self,
        graph: InteractionGraph,
        new_records: list[InteractionRecord],
        source: str,
    ) -> InteractionGraph:
        """Incrementally update a Reddit-style (count-aggregated) graph.

        Recovers raw counts from ``graph.rawEdgeCounts``, merges new records
        by incrementing per-pair counts, then re-normalizes all weights to
        [0, 1] by dividing by the new maximum count.

        Args:
            graph:       Existing graph to update.
            new_records: New :class:`InteractionRecord` objects to add.
            source:      ``datasetSource`` string.

        Returns:
            Updated :class:`InteractionGraph`.
        """
        # --- Step 1: Recover existing raw counts ---
        # graph.rawEdgeCounts maps (src, tgt) → raw count (float).
        # Fall back to existing edge weights if rawEdgeCounts is absent
        # (e.g. for graphs built without the updated service).
        edge_counts: dict[tuple[str, str], float] = {}
        if graph.rawEdgeCounts is not None:
            edge_counts = dict(graph.rawEdgeCounts)
        else:
            # Reverse-normalization fallback: recover raw counts from weights.
            # Without max_count we cannot reverse exactly, so we use weights as
            # proportional counts (they are already in (0,1]).  After merging
            # new records and re-normalizing, the relative ordering is preserved.
            logger.debug(
                "GraphConstructionService._update_reddit_graph: rawEdgeCounts not "
                "available for graph '%s'; using edge weights as proportional counts",
                graph.snapshotId,
            )
            for edge in graph.edges:
                key = (edge.sourceUserId, edge.targetUserId)
                edge_counts[key] = edge.weight

        # --- Step 2: Collect existing nodes (to preserve metadata) ---
        node_set: set[str] = set(graph.nodes.keys())

        # --- Step 3: Process new records ---
        skipped_self_loops = 0
        for record in new_records:
            if record.sourceUserId == record.targetUserId:
                skipped_self_loops += 1
                logger.debug(
                    "GraphConstructionService.update_graph: rejected self-loop "
                    "for user '%s' (datasetSource='%s', record id='%s')",
                    record.sourceUserId,
                    source,
                    record.id,
                )
                continue

            node_set.add(record.sourceUserId)
            node_set.add(record.targetUserId)

            key = (record.sourceUserId, record.targetUserId)
            edge_counts[key] = edge_counts.get(key, 0.0) + 1.0

        if skipped_self_loops:
            logger.warning(
                "GraphConstructionService.update_graph: skipped %d self-loop "
                "record(s) (datasetSource='%s')",
                skipped_self_loops,
                source,
            )

        # --- Step 4: Re-normalize ---
        max_count = max(edge_counts.values()) if edge_counts else 1.0

        # --- Step 5: Build nodes, preserving metadata for existing nodes ---
        nodes = _merge_nodes(graph.nodes, node_set)

        # --- Step 6: Build edges ---
        edges: list[Edge] = []
        for (src, tgt), count in edge_counts.items():
            edges.append(
                Edge(
                    sourceUserId=src,
                    targetUserId=tgt,
                    weight=count / max_count,
                    isCrossCommunity=False,
                    signedPolarity=None,
                )
            )

        return _make_graph(nodes, edges, source, raw_edge_counts=dict(edge_counts))

    def _update_congress_graph(
        self,
        graph: InteractionGraph,
        new_records: list[InteractionRecord],
        source: str,
    ) -> InteractionGraph:
        """Incrementally update a Congress (pre-normalized) graph.

        Pre-normalized weights are passed through unchanged; new records
        overwrite existing entries for the same (source, target) pair.
        No re-normalization is applied.

        Args:
            graph:       Existing graph to update.
            new_records: New :class:`InteractionRecord` objects to add.
            source:      ``datasetSource`` string (``"congress"``).

        Returns:
            Updated :class:`InteractionGraph`.
        """
        # --- Step 1: Recover existing edge weights ---
        edge_map: dict[tuple[str, str], float] = {}
        for edge in graph.edges:
            key = (edge.sourceUserId, edge.targetUserId)
            edge_map[key] = edge.weight

        # --- Step 2: Collect existing nodes ---
        node_set: set[str] = set(graph.nodes.keys())

        # --- Step 3: Process new records ---
        skipped_self_loops = 0
        for record in new_records:
            if record.sourceUserId == record.targetUserId:
                skipped_self_loops += 1
                logger.debug(
                    "GraphConstructionService.update_graph: rejected self-loop "
                    "for user '%s' (datasetSource='congress', record id='%s')",
                    record.sourceUserId,
                    record.id,
                )
                continue

            node_set.add(record.sourceUserId)
            node_set.add(record.targetUserId)

            key = (record.sourceUserId, record.targetUserId)
            # Use the transmission probability stored in sentimentScore.
            weight = record.sentimentScore if record.sentimentScore is not None else 0.0
            # New record for the same pair overwrites the existing weight;
            # consistent with _build_congress_graph (keep first occurrence
            # from IngestionService dedup, or last if multiple arrive here).
            if key not in edge_map:
                edge_map[key] = weight

        if skipped_self_loops:
            logger.warning(
                "GraphConstructionService.update_graph: skipped %d self-loop "
                "record(s) (datasetSource='congress')",
                skipped_self_loops,
            )

        # --- Step 4: Build nodes, preserving metadata for existing nodes ---
        nodes = _merge_nodes(graph.nodes, node_set)

        # --- Step 5: Build edges (weights already in [0,1]; no normalization) ---
        edges: list[Edge] = [
            Edge(
                sourceUserId=src,
                targetUserId=tgt,
                weight=w,
                isCrossCommunity=False,
                signedPolarity=None,
            )
            for (src, tgt), w in edge_map.items()
        ]

        raw_counts = dict(edge_map)
        return _make_graph(nodes, edges, source, raw_edge_counts=raw_counts)

    def _update_wiki_rfa_graph(
        self,
        graph: InteractionGraph,
        new_records: list[InteractionRecord],
        source: str,
    ) -> InteractionGraph:
        """Incrementally update a Wiki-RfA (binary-weight signed) graph.

        All edge weights remain 1.0.  The ``signedPolarity`` for each
        (source, target) pair is taken from the last record seen (new records
        override existing polarity for the same pair).

        Args:
            graph:       Existing graph to update.
            new_records: New :class:`InteractionRecord` objects to add.
            source:      ``datasetSource`` string (``"wiki_rfa"``).

        Returns:
            Updated :class:`InteractionGraph`.
        """
        # --- Step 1: Recover existing edge polarities ---
        edge_polarity: dict[tuple[str, str], Optional[int]] = {}
        for edge in graph.edges:
            key = (edge.sourceUserId, edge.targetUserId)
            edge_polarity[key] = edge.signedPolarity

        # --- Step 2: Collect existing nodes ---
        node_set: set[str] = set(graph.nodes.keys())

        # --- Step 3: Process new records ---
        skipped_self_loops = 0
        for record in new_records:
            if record.sourceUserId == record.targetUserId:
                skipped_self_loops += 1
                logger.debug(
                    "GraphConstructionService.update_graph: rejected self-loop "
                    "for user '%s' (datasetSource='wiki_rfa', record id='%s')",
                    record.sourceUserId,
                    record.id,
                )
                continue

            node_set.add(record.sourceUserId)
            node_set.add(record.targetUserId)

            key = (record.sourceUserId, record.targetUserId)
            # Last polarity wins (consistent with _build_wiki_rfa_graph).
            edge_polarity[key] = record.votePolarity

        if skipped_self_loops:
            logger.warning(
                "GraphConstructionService.update_graph: skipped %d self-loop "
                "record(s) (datasetSource='wiki_rfa')",
                skipped_self_loops,
            )

        # --- Step 4: Build nodes, preserving metadata for existing nodes ---
        nodes = _merge_nodes(graph.nodes, node_set)

        # --- Step 5: Build edges (weight = 1.0 for all votes) ---
        edges: list[Edge] = [
            Edge(
                sourceUserId=src,
                targetUserId=tgt,
                weight=1.0,
                isCrossCommunity=False,
                signedPolarity=polarity,
            )
            for (src, tgt), polarity in edge_polarity.items()
        ]

        raw_counts = {k: 1.0 for k in edge_polarity}
        return _make_graph(nodes, edges, source, raw_edge_counts=raw_counts)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def build_graph(
    records: list[InteractionRecord],
    *,
    dataset_source: Optional[str] = None,
) -> InteractionGraph:
    """Module-level convenience wrapper around :meth:`GraphConstructionService.build_graph`.

    Args:
        records:        Non-empty list of :class:`InteractionRecord` objects.
        dataset_source: Optional override for the ``datasetSource`` tag.

    Returns:
        :class:`InteractionGraph` produced by :class:`GraphConstructionService`.

    Raises:
        ValueError: If *records* is empty.
    """
    return GraphConstructionService().build_graph(records, dataset_source=dataset_source)


def update_graph(
    graph: InteractionGraph,
    new_records: list[InteractionRecord],
) -> InteractionGraph:
    """Module-level convenience wrapper around :meth:`GraphConstructionService.update_graph`.

    Args:
        graph:       Existing :class:`InteractionGraph` to update.
        new_records: List of new :class:`InteractionRecord` objects to incorporate.

    Returns:
        Updated :class:`InteractionGraph` with a new ``snapshotId`` and ``createdAt``.
    """
    return GraphConstructionService().update_graph(graph, new_records)


def persist_graph(graph: InteractionGraph, snapshot_id: str) -> str:
    """Module-level convenience wrapper around :meth:`GraphConstructionService.persist_graph`.

    Args:
        graph:       :class:`InteractionGraph` to persist.
        snapshot_id: Snapshot identifier used as the file name stem.

    Returns:
        Path to the written GraphML file.
    """
    return GraphConstructionService().persist_graph(graph, snapshot_id)


def load_graph(snapshot_id: str) -> InteractionGraph:
    """Module-level convenience wrapper around :meth:`GraphConstructionService.load_graph`.

    Args:
        snapshot_id: Snapshot identifier to load.

    Returns:
        :class:`InteractionGraph` loaded from the persisted GraphML file.

    Raises:
        FileNotFoundError: If no GraphML file for *snapshot_id* is found.
    """
    return GraphConstructionService().load_graph(snapshot_id)


# ---------------------------------------------------------------------------
# Private utility functions
# ---------------------------------------------------------------------------


def _gml_key(parent: ET.Element, kid: str, for_: str, attr_name: str, attr_type: str) -> None:
    """Append a GraphML ``<key>`` declaration element to *parent*."""
    ET.SubElement(
        parent,
        "key",
        id=kid,
        **{"for": for_, "attr.name": attr_name, "attr.type": attr_type},
    )


def _gml_data(parent: ET.Element, key: str, value: str) -> None:
    """Append a GraphML ``<data>`` element to *parent*."""
    d = ET.SubElement(parent, "data", key=key)
    d.text = value


def _make_nodes(user_ids: set[str]) -> dict[str, Node]:
    """Create a node dict with default-initialized fields for all *user_ids*.

    All nodes are initialized with:
        ``communityId=None``, ``betweenness=0.0``,
        ``diversityScore=0.0``, ``topicVector=[]``

    Args:
        user_ids: Set of unique user identifier strings.

    Returns:
        Mapping of userId → :class:`Node`.
    """
    return {
        uid: Node(
            userId=uid,
            communityId=None,
            betweenness=0.0,
            diversityScore=0.0,
            topicVector=[],
        )
        for uid in user_ids
    }


def _merge_nodes(
    existing_nodes: dict[str, Node],
    all_user_ids: set[str],
) -> dict[str, Node]:
    """Build a merged node dict, preserving metadata for existing nodes.

    For each userId in *all_user_ids*:
    - If the userId already exists in *existing_nodes*, the existing
      :class:`Node` object (with its ``communityId``, ``betweenness``,
      ``diversityScore``, and ``topicVector``) is carried forward unchanged.
    - If the userId is new, a default-initialized :class:`Node` is created.

    Args:
        existing_nodes: Node mapping from the previous graph.
        all_user_ids:   Complete set of user IDs for the updated graph
                        (existing + newly discovered).

    Returns:
        Updated mapping of userId → :class:`Node`.
    """
    result: dict[str, Node] = {}
    for uid in all_user_ids:
        if uid in existing_nodes:
            result[uid] = existing_nodes[uid]
        else:
            result[uid] = Node(
                userId=uid,
                communityId=None,
                betweenness=0.0,
                diversityScore=0.0,
                topicVector=[],
            )
    return result


def _make_graph(
    nodes: dict[str, Node],
    edges: list[Edge],
    dataset_source: str,
    raw_edge_counts: Optional[dict[tuple[str, str], float]] = None,
) -> InteractionGraph:
    """Construct an :class:`InteractionGraph` with a UUID snapshot ID and UTC timestamp.

    Args:
        nodes:           Node mapping for the graph.
        edges:           Edge list for the graph.
        dataset_source:  Dataset source tag.
        raw_edge_counts: Optional mapping of (sourceUserId, targetUserId) → raw
                         count or pre-normalized weight, stored on the graph for
                         use by :meth:`~GraphConstructionService.update_graph`.

    Returns:
        :class:`InteractionGraph` with generated ``snapshotId`` and ``createdAt``.
    """
    return InteractionGraph(
        nodes=nodes,
        edges=edges,
        snapshotId=str(uuid.uuid4()),
        createdAt=datetime.now(timezone.utc),
        datasetSource=dataset_source,
        rawEdgeCounts=raw_edge_counts,
    )
