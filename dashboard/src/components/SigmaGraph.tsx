import { useEffect, useRef, useState, useCallback } from "react";
import Graph from "graphology";
import { Sigma } from "sigma";
import type { NodeDTO, EdgeDTO, DatasetSource, RecommendationDTO } from "../api/client";
import { fetchRecommendations } from "../api/client";

// ---------------------------------------------------------------------------
// Color palette for communities
// ---------------------------------------------------------------------------

const COMMUNITY_COLORS = [
  "#2563eb", // blue
  "#16a34a", // green
  "#dc2626", // red
  "#d97706", // amber
  "#7c3aed", // violet
  "#0891b2", // cyan
  "#db2777", // pink
  "#65a30d", // lime
  "#ea580c", // orange
  "#0284c7", // sky
];

function getCommunityColor(communityId: string | null): string {
  if (!communityId) return "#9ca3af"; // gray fallback
  let hash = 0;
  for (let i = 0; i < communityId.length; i++) {
    hash = (hash * 31 + communityId.charCodeAt(i)) >>> 0;
  }
  return COMMUNITY_COLORS[hash % COMMUNITY_COLORS.length];
}

// ---------------------------------------------------------------------------
// Edge color based on signedPolarity
// ---------------------------------------------------------------------------

function getEdgeColor(signedPolarity: number | null): string {
  if (signedPolarity === 1) return "#16a34a";   // green — positive vote
  if (signedPolarity === -1) return "#dc2626";  // red   — negative vote
  return "#9ca3af";                              // gray  — neutral / no polarity
}

// ---------------------------------------------------------------------------
// Node size based on betweenness centrality
// ---------------------------------------------------------------------------

const MIN_NODE_SIZE = 4;
const MAX_NODE_SIZE = 20;

function getNodeSize(betweenness: number): number {
  const clamped = Math.max(0, Math.min(1, betweenness));
  return MIN_NODE_SIZE + clamped * (MAX_NODE_SIZE - MIN_NODE_SIZE);
}

// ---------------------------------------------------------------------------
// Circular layout — places nodes evenly on a circle, avoids random positions
// ---------------------------------------------------------------------------

function applyCircularLayout(graph: Graph): void {
  const nodes = graph.nodes();
  const count = nodes.length;
  if (count === 0) return;
  const radius = 100;
  nodes.forEach((nodeId, index) => {
    const angle = (2 * Math.PI * index) / count;
    graph.setNodeAttribute(nodeId, "x", radius * Math.cos(angle));
    graph.setNodeAttribute(nodeId, "y", radius * Math.sin(angle));
  });
}

// ---------------------------------------------------------------------------
// Sidebar state types
// ---------------------------------------------------------------------------

interface SelectedNodeInfo {
  userId: string;
  communityId: string | null;
  diversityScore: number;
  betweenness: number;
}

// ---------------------------------------------------------------------------
// Component props
// ---------------------------------------------------------------------------

export interface SigmaGraphProps {
  nodes: NodeDTO[];
  edges: EdgeDTO[];
  datasetSource: DatasetSource;
}

/**
 * Renders an interactive graph using sigma.js v3 and graphology.
 *
 * - Nodes are colored by communityId (stable palette hash).
 * - Node size is proportional to betweenness centrality.
 * - Edges are colored: green for signedPolarity=1, red for -1, gray otherwise.
 * - Clicking a node opens a sidebar with diversity score, community label,
 *   and top-5 recommendations.
 */
export function SigmaGraph({ nodes, edges, datasetSource }: SigmaGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const graphContainerRef = useRef<HTMLDivElement>(null);

  const [selectedNode, setSelectedNode] = useState<SelectedNodeInfo | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationDTO[]>([]);
  const [recsLoading, setRecsLoading] = useState(false);
  const [recsError, setRecsError] = useState<string | null>(null);

  // ---------------------------------------------------------------------------
  // Close sidebar handler
  // ---------------------------------------------------------------------------

  const closeSidebar = useCallback(() => {
    setSelectedNode(null);
    setRecommendations([]);
    setRecsError(null);
    // Return focus to graph container
    graphContainerRef.current?.focus();
  }, []);

  // ---------------------------------------------------------------------------
  // Keyboard handler: Escape closes sidebar
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (!selectedNode) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        closeSidebar();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedNode, closeSidebar]);

  // ---------------------------------------------------------------------------
  // Focus sidebar when it opens
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (selectedNode && sidebarRef.current) {
      sidebarRef.current.focus();
    }
  }, [selectedNode]);

  // ---------------------------------------------------------------------------
  // Build and mount sigma when nodes/edges change
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (!containerRef.current) return;

    const graph = new Graph({ multi: false, type: "directed" });

    for (const node of nodes) {
      graph.addNode(node.userId, {
        label: node.userId,
        size: getNodeSize(node.betweenness),
        color: getCommunityColor(node.communityId),
        communityId: node.communityId,
        betweenness: node.betweenness,
        diversityScore: node.diversityScore,
        // Positions are set by layout below
        x: 0,
        y: 0,
      });
    }

    for (const edge of edges) {
      if (!graph.hasNode(edge.sourceUserId) || !graph.hasNode(edge.targetUserId)) {
        continue;
      }
      try {
        graph.addEdge(edge.sourceUserId, edge.targetUserId, {
          // Store color as attribute; edgeReducer will pass it to sigma
          color: getEdgeColor(edge.signedPolarity),
          size: 1,
          weight: edge.weight,
          signedPolarity: edge.signedPolarity,
          isCrossCommunity: edge.isCrossCommunity,
        });
      } catch {
        // Duplicate edge — skip silently
      }
    }

    // Apply deterministic circular layout
    applyCircularLayout(graph);

    // Mount sigma with edgeReducer to forward stored color attribute
    const renderer = new Sigma(graph, containerRef.current, {
      renderEdgeLabels: false,
      defaultEdgeColor: "#9ca3af",
      defaultNodeColor: "#9ca3af",
      // edgeReducer: read the stored color from the graph and pass to sigma
      edgeReducer: (_edge, attrs) => {
        return {
          ...attrs,
          color: (attrs.color as string | undefined) ?? "#9ca3af",
        };
      },
    });

    // Node click handler
    renderer.on("clickNode", ({ node }) => {
      const attrs = graph.getNodeAttributes(node);
      const info: SelectedNodeInfo = {
        userId: node,
        communityId: (attrs.communityId as string | null) ?? null,
        diversityScore: (attrs.diversityScore as number) ?? 0,
        betweenness: (attrs.betweenness as number) ?? 0,
      };
      setSelectedNode(info);
      setRecommendations([]);
      setRecsError(null);
      setRecsLoading(true);

      fetchRecommendations(node)
        .then((recs) => {
          setRecommendations(recs.slice(0, 5));
          setRecsLoading(false);
        })
        .catch(() => {
          // Recommendations require DB persistence — show a friendly message
          // instead of a red error when the DB has no data yet.
          setRecommendations([]);
          setRecsError(null);
          setRecsLoading(false);
        });
    });

    sigmaRef.current = renderer;

    return () => {
      renderer.kill();
      sigmaRef.current = null;
    };
  }, [nodes, edges]);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div
      ref={graphContainerRef}
      style={styles.wrapper}
    >
      {/* Sigma canvas */}
      <div
        ref={containerRef}
        role="application"
        aria-label="Interactive graph visualization. Click a node to view details."
        tabIndex={0}
        style={styles.canvas}
      />

      {/* Edge color legend — always visible for wiki_rfa */}
      {datasetSource === "wiki_rfa" && (
        <div style={styles.legend} aria-label="Edge color legend">
          <span style={styles.legendTitle}>Edge polarity:</span>
          <span style={styles.legendItem}>
            <span style={{ ...styles.legendSwatch, background: "#16a34a" }} />
            Positive (+1)
          </span>
          <span style={styles.legendItem}>
            <span style={{ ...styles.legendSwatch, background: "#dc2626" }} />
            Negative (−1)
          </span>
          <span style={styles.legendItem}>
            <span style={{ ...styles.legendSwatch, background: "#9ca3af" }} />
            Neutral
          </span>
        </div>
      )}

      {/* Node detail sidebar */}
      {selectedNode && (
        <aside
          ref={sidebarRef}
          role="complementary"
          aria-label="Node details"
          tabIndex={-1}
          style={styles.sidebar}
        >
          {/* Header */}
          <div style={styles.sidebarHeader}>
            <h2 style={styles.sidebarTitle} title={selectedNode.userId}>
              {selectedNode.userId.length > 24
                ? selectedNode.userId.slice(0, 22) + "…"
                : selectedNode.userId}
            </h2>
            <button
              onClick={closeSidebar}
              aria-label="Close node details"
              style={styles.closeButton}
            >
              ×
            </button>
          </div>

          {/* Community */}
          <div style={styles.field}>
            <span style={styles.fieldLabel}>Community</span>
            <span style={styles.fieldValue}>
              {selectedNode.communityId ?? "—"}
            </span>
          </div>

          {/* Diversity Score with progress bar */}
          <div style={styles.field}>
            <span style={styles.fieldLabel}>Diversity Score</span>
            <div style={styles.progressContainer}>
              <div
                style={{
                  ...styles.progressBar,
                  width: `${(selectedNode.diversityScore * 100).toFixed(0)}%`,
                }}
              />
            </div>
            <span style={styles.fieldValue}>
              {selectedNode.diversityScore.toFixed(2)}
            </span>
          </div>

          {/* Betweenness Centrality */}
          <div style={styles.field}>
            <span style={styles.fieldLabel}>Betweenness</span>
            <span style={styles.fieldValue}>
              {selectedNode.betweenness.toFixed(4)}
            </span>
          </div>

          {/* Recommendations */}
          <div style={styles.recsSection}>
            <h3 style={styles.recsTitle}>Top Recommendations</h3>

            {recsLoading && (
              <p style={styles.recsStatus} role="status" aria-live="polite">
                Loading recommendations…
              </p>
            )}

            {recsError && (
              <p style={styles.recsError} role="alert">
                {recsError}
              </p>
            )}

            {!recsLoading && !recsError && recommendations.length === 0 && (
              <p style={styles.recsStatus}>No recommendations available.</p>
            )}

            {!recsLoading && !recsError && recommendations.length > 0 && (
              <ul style={styles.recsList} role="list">
                {recommendations.map((rec) => (
                  <li key={rec.recommendationId} role="listitem" style={styles.recItem}>
                    <span style={styles.recUserId}>{rec.recommendedUserId}</span>
                    <div style={styles.recMeta}>
                      <span>Diversity gain: {rec.diversityGain.toFixed(2)}</span>
                      <span>Topic relevance: {rec.topicRelevance.toFixed(2)}</span>
                    </div>
                    {rec.reason && (
                      <span style={styles.recReason}>{rec.reason}</span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Wiki-RfA legend inside sidebar */}
          {datasetSource === "wiki_rfa" && (
            <div style={styles.sidebarLegend}>
              <span style={styles.legendTitle}>Edges:</span>
              <span style={styles.legendItem}>
                <span style={{ ...styles.legendSwatch, background: "#16a34a" }} />
                +1
              </span>
              <span style={styles.legendItem}>
                <span style={{ ...styles.legendSwatch, background: "#dc2626" }} />
                −1
              </span>
            </div>
          )}
        </aside>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inline styles
// ---------------------------------------------------------------------------

const styles = {
  wrapper: {
    position: "relative",
    width: "100%",
    height: "100%",
    minHeight: "500px",
  } as React.CSSProperties,

  canvas: {
    width: "100%",
    height: "100%",
    minHeight: "500px",
    background: "#f8fafc",
    borderRadius: "8px",
    border: "1px solid #e2e8f0",
    outline: "none",
  } as React.CSSProperties,

  legend: {
    position: "absolute",
    bottom: "12px",
    left: "12px",
    display: "flex",
    alignItems: "center",
    gap: "10px",
    background: "rgba(255,255,255,0.92)",
    border: "1px solid #e2e8f0",
    borderRadius: "6px",
    padding: "6px 10px",
    fontSize: "12px",
    color: "#475569",
    backdropFilter: "blur(4px)",
  } as React.CSSProperties,

  legendTitle: {
    fontWeight: 600,
    marginRight: "4px",
  } as React.CSSProperties,

  legendItem: {
    display: "flex",
    alignItems: "center",
    gap: "4px",
  } as React.CSSProperties,

  legendSwatch: {
    display: "inline-block",
    width: "10px",
    height: "10px",
    borderRadius: "50%",
    flexShrink: 0,
  } as React.CSSProperties,

  sidebar: {
    position: "absolute",
    top: "8px",
    right: "8px",
    width: "280px",
    maxHeight: "calc(100% - 16px)",
    overflowY: "auto",
    background: "#ffffff",
    border: "1px solid #e2e8f0",
    borderRadius: "8px",
    boxShadow: "0 4px 16px rgba(0,0,0,0.10)",
    padding: "16px",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
    outline: "none",
    zIndex: 10,
  } as React.CSSProperties,

  sidebarHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "8px",
  } as React.CSSProperties,

  sidebarTitle: {
    fontSize: "15px",
    fontWeight: 700,
    color: "#1e293b",
    margin: 0,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  } as React.CSSProperties,

  closeButton: {
    background: "none",
    border: "none",
    cursor: "pointer",
    fontSize: "20px",
    lineHeight: 1,
    color: "#64748b",
    padding: "2px 6px",
    borderRadius: "4px",
    flexShrink: 0,
  } as React.CSSProperties,

  field: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  } as React.CSSProperties,

  fieldLabel: {
    fontSize: "11px",
    fontWeight: 600,
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: "0.04em",
  } as React.CSSProperties,

  fieldValue: {
    fontSize: "14px",
    color: "#1e293b",
    wordBreak: "break-all",
  } as React.CSSProperties,

  progressContainer: {
    height: "6px",
    background: "#f1f5f9",
    borderRadius: "3px",
    overflow: "hidden",
  } as React.CSSProperties,

  progressBar: {
    height: "100%",
    background: "#2563eb",
    borderRadius: "3px",
    transition: "width 0.3s ease",
  } as React.CSSProperties,

  recsSection: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    borderTop: "1px solid #f1f5f9",
    paddingTop: "12px",
  } as React.CSSProperties,

  recsTitle: {
    fontSize: "13px",
    fontWeight: 600,
    color: "#475569",
    margin: 0,
  } as React.CSSProperties,

  recsStatus: {
    fontSize: "13px",
    color: "#94a3b8",
    margin: 0,
  } as React.CSSProperties,

  recsError: {
    fontSize: "13px",
    color: "#b91c1c",
    background: "#fef2f2",
    padding: "8px",
    borderRadius: "4px",
    margin: 0,
  } as React.CSSProperties,

  recsList: {
    listStyle: "none",
    margin: 0,
    padding: 0,
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  } as React.CSSProperties,

  recItem: {
    display: "flex",
    flexDirection: "column",
    gap: "3px",
    padding: "8px",
    background: "#f8fafc",
    borderRadius: "6px",
    border: "1px solid #e2e8f0",
  } as React.CSSProperties,

  recUserId: {
    fontSize: "13px",
    fontWeight: 600,
    color: "#1e293b",
    wordBreak: "break-all",
  } as React.CSSProperties,

  recMeta: {
    display: "flex",
    flexDirection: "column",
    gap: "1px",
    fontSize: "12px",
    color: "#475569",
  } as React.CSSProperties,

  recReason: {
    fontSize: "11px",
    color: "#64748b",
    fontStyle: "italic",
  } as React.CSSProperties,

  sidebarLegend: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    borderTop: "1px solid #f1f5f9",
    paddingTop: "10px",
    fontSize: "12px",
    color: "#475569",
    flexWrap: "wrap",
  } as React.CSSProperties,
} satisfies Record<string, React.CSSProperties>;
