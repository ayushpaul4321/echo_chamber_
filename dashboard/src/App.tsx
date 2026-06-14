import React, { useState, useCallback } from "react";
import { DatasetSelector } from "./components/DatasetSelector";
import { SigmaGraph } from "./components/SigmaGraph";
import { MetricsPanel } from "./components/MetricsPanel";
import { PolarizationChart } from "./components/PolarizationChart";
import type { PolarizationSnapshotEntry } from "./components/PolarizationChart";
import { DiversityHistogram } from "./components/DiversityHistogram";
import type {
  DatasetSource,
  NodeDTO,
  EdgeDTO,
  PolarizationDTO,
  SignedMetricsDTO,
} from "./api/client";
import {
  fetchGraphSnapshot,
  fetchPolarizationMetrics,
  fetchSignedMetrics,
  fetchAllPolarizationMetrics,
  fetchUserMetricsList,
} from "./api/client";
import { useSnapshotPoller } from "./hooks/useSnapshotPoller";

// ---------------------------------------------------------------------------
// App state types
// ---------------------------------------------------------------------------

type LoadStatus = "idle" | "loading" | "success" | "error";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function App() {
  const [datasetSource, setDatasetSource] = useState<DatasetSource>("reddit_title");
  const [snapshotId, setSnapshotId] = useState<string>("");

  // The snapshotId that was last successfully loaded — used by the poller.
  const [activeSnapshotId, setActiveSnapshotId] = useState<string | null>(null);

  // Graph state
  const [nodes, setNodes] = useState<NodeDTO[]>([]);
  const [edges, setEdges] = useState<EdgeDTO[]>([]);
  const [graphStatus, setGraphStatus] = useState<LoadStatus>("idle");

  // Metrics state
  const [metricsStatus, setMetricsStatus] = useState<LoadStatus>("idle");
  const [polarizationMetrics, setPolarizationMetrics] = useState<PolarizationDTO | null>(null);
  const [signedMetrics, setSignedMetrics] = useState<SignedMetricsDTO[] | null>(null);
  const [timeSeriesSnapshots, setTimeSeriesSnapshots] = useState<PolarizationSnapshotEntry[]>([]);
  const [diversityScores, setDiversityScores] = useState<number[]>([]);

  const [errorMessage, setErrorMessage] = useState<string>("");

  // Live refresh indicator: true while an auto-refresh triggered by the poller
  // is in progress, or briefly after new data is detected.
  const [refreshing, setRefreshing] = useState(false);

  // ---------------------------------------------------------------------------
  // Core load logic (shared by manual Load and auto-refresh)
  // ---------------------------------------------------------------------------

  const loadMetrics = useCallback(async (sid: string) => {
    // 1. Polarization metrics for the selected snapshot
    const polMetrics = await fetchPolarizationMetrics(sid);
    setPolarizationMetrics(polMetrics);

    // 2. Signed metrics (wiki-RfA only) — soft failure: not available for other datasets
    if (datasetSource === "wiki_rfa") {
      try {
        const signed = await fetchSignedMetrics(sid);
        setSignedMetrics(signed);
      } catch {
        setSignedMetrics(null);
      }
    } else {
      setSignedMetrics(null);
    }

    // 3. Time-series: fetch polarization metrics for all three sources (parallel)
    const [redditList, congressList, wikiList] = await Promise.allSettled([
      fetchAllPolarizationMetrics("reddit_title", undefined, 200),
      fetchAllPolarizationMetrics("congress", undefined, 200),
      fetchAllPolarizationMetrics("wiki_rfa", undefined, 200),
    ]);

    const allEntries: PolarizationSnapshotEntry[] = [];

    for (const [result, source] of [
      [redditList, "reddit_title"],
      [congressList, "congress"],
      [wikiList, "wiki_rfa"],
    ] as const) {
      if (result.status === "fulfilled") {
        for (const item of result.value.items) {
          allEntries.push({
            snapshotId: item.snapshotId,
            datasetSource: source as PolarizationSnapshotEntry["datasetSource"],
            polarizationIndex: item.polarizationIndex,
            computedAt: item.computedAt,
          });
        }
      }
    }

    setTimeSeriesSnapshots(allEntries);

    // 4. Diversity scores — fetch user metrics for this snapshot
    try {
      const userMetrics = await fetchUserMetricsList(sid, undefined, 500);
      setDiversityScores(userMetrics.items.map((u) => u.diversityScore));
    } catch {
      setDiversityScores([]);
    }

    setMetricsStatus("success");
  }, [datasetSource]);

  const loadSnapshot = useCallback(async (sid: string, isAutoRefresh = false) => {
    if (isAutoRefresh) {
      setRefreshing(true);
    } else {
      setGraphStatus("loading");
      setMetricsStatus("loading");
      setErrorMessage("");
    }
    setNodes([]);
    setEdges([]);
    setPolarizationMetrics(null);
    setSignedMetrics(null);
    setTimeSeriesSnapshots([]);
    setDiversityScores([]);

    // Fetch graph and metrics in parallel for efficiency
    const [graphResult, metricsResult] = await Promise.allSettled([
      fetchGraphSnapshot(sid),
      loadMetrics(sid),
    ]);

    // Handle graph result
    if (graphResult.status === "fulfilled") {
      setNodes(graphResult.value.nodes);
      setEdges(graphResult.value.edges);
      setGraphStatus("success");
      setActiveSnapshotId(sid);
    } else {
      const message =
        graphResult.reason instanceof Error
          ? graphResult.reason.message
          : String(graphResult.reason);
      setErrorMessage(message);
      setGraphStatus("error");
    }

    // Metrics result is handled inside loadMetrics via state setters
    if (metricsResult.status === "rejected") {
      setMetricsStatus("error");
    }

    if (isAutoRefresh) {
      setRefreshing(false);
    }
  }, [loadMetrics]);

  // ---------------------------------------------------------------------------
  // Load handler (manual button click)
  // ---------------------------------------------------------------------------

  async function handleLoad() {
    const trimmed = snapshotId.trim();
    if (!trimmed) {
      setErrorMessage("Please enter a snapshot ID.");
      setGraphStatus("error");
      return;
    }
    await loadSnapshot(trimmed, false);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      void handleLoad();
    }
  }

  // ---------------------------------------------------------------------------
  // Live snapshot poller — fires every 60 s when a snapshot is loaded
  // ---------------------------------------------------------------------------

  useSnapshotPoller({
    currentSnapshotId: activeSnapshotId,
    datasetSource,
    enabled: graphStatus === "success",
    onNewSnapshot: (newSnapshotId) => {
      // Update the input field so the user can see the new snapshot ID
      setSnapshotId(newSnapshotId);
      // Trigger a re-fetch of all data using the new snapshot
      void loadSnapshot(newSnapshotId, true);
    },
  });

  // Combined loading indicator
  const isLoading = graphStatus === "loading" || metricsStatus === "loading";
  return (
    <div style={styles.root}>
      {/* ------------------------------------------------------------------ */}
      {/* Header / controls                                                   */}
      {/* ------------------------------------------------------------------ */}
      <header style={styles.header}>
        <h1 style={styles.title}>Echo Chamber Detector</h1>

        <div style={styles.controls}>
          {/* Dataset source toggle */}
          <DatasetSelector
            selected={datasetSource}
            onChange={setDatasetSource}
          />

          {/* Snapshot ID input */}
          <div style={styles.inputGroup}>
            <label htmlFor="snapshot-input" style={styles.label}>
              Snapshot ID
            </label>
            <input
              id="snapshot-input"
              type="text"
              value={snapshotId}
              onChange={(e) => setSnapshotId(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="e.g. 550e8400-e29b-41d4-a716-446655440000"
              style={styles.input}
              aria-describedby={graphStatus === "error" ? "error-msg" : undefined}
            />
          </div>

          {/* Load button */}
          <button
            onClick={() => void handleLoad()}
            disabled={isLoading}
            style={{
              ...styles.button,
              ...(isLoading ? styles.buttonDisabled : {}),
            }}
          >
            {isLoading ? "Loading…" : "Load Graph"}
          </button>
        </div>

        {/* Dataset source indicator */}
        <p style={styles.sourceBadge}>
          Source: <strong>{datasetSource}</strong>
        </p>
      </header>

      {/* ------------------------------------------------------------------ */}
      {/* Status messages                                                      */}
      {/* ------------------------------------------------------------------ */}
      {graphStatus === "error" && (
        <div id="error-msg" role="alert" style={styles.errorBanner}>
          {errorMessage}
        </div>
      )}

      {isLoading && (
        <div role="status" aria-live="polite" style={styles.statusBanner}>
          Loading graph data…
        </div>
      )}

      {graphStatus === "success" && nodes.length === 0 && (
        <div role="status" style={styles.statusBanner}>
          No nodes returned for this snapshot.
        </div>
      )}

      {/* Live-refresh indicator — shown while the poller is fetching new data */}
      {refreshing && (
        <div role="status" aria-live="polite" style={styles.refreshBanner}>
          🔄 New snapshot detected — refreshing data…
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Metrics panels — shown when metrics loaded successfully             */}
      {/* ------------------------------------------------------------------ */}
      {metricsStatus === "success" && polarizationMetrics && (
        <div style={styles.metricsSection}>
          {/* Metric cards */}
          <MetricsPanel
            polarizationMetrics={polarizationMetrics}
            signedMetrics={signedMetrics ?? undefined}
          />

          {/* Chart + Histogram row */}
          <div style={styles.chartsRow}>
            <div style={styles.chartCell}>
              <PolarizationChart snapshots={timeSeriesSnapshots} />
            </div>
            <div style={styles.chartCell}>
              <DiversityHistogram diversityScores={diversityScores} />
            </div>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Graph canvas                                                         */}
      {/* ------------------------------------------------------------------ */}
      <main style={styles.graphContainer}>
        {graphStatus === "idle" && (
          <div style={styles.emptyState}>
            <p>Enter a snapshot ID above and click <strong>Load Graph</strong> to visualize.</p>
          </div>
        )}

        {(graphStatus === "success" || graphStatus === "loading") && nodes.length > 0 && (
          <SigmaGraph nodes={nodes} edges={edges} datasetSource={datasetSource} />
        )}
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inline styles
// ---------------------------------------------------------------------------

const styles = {
  root: {
    display: "flex",
    flexDirection: "column",
    minHeight: "100vh",
    padding: "16px",
    gap: "16px",
    background: "#ffffff",
  } as React.CSSProperties,

  header: {
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  } as React.CSSProperties,

  title: {
    fontSize: "22px",
    fontWeight: 700,
    color: "#1e293b",
  } as React.CSSProperties,

  controls: {
    display: "flex",
    alignItems: "flex-end",
    gap: "16px",
    flexWrap: "wrap",
  } as React.CSSProperties,

  inputGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  } as React.CSSProperties,

  label: {
    fontSize: "13px",
    fontWeight: 500,
    color: "#475569",
  } as React.CSSProperties,

  input: {
    padding: "8px 12px",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    fontSize: "14px",
    width: "360px",
    outline: "none",
    color: "#1e293b",
    background: "#f9fafb",
  } as React.CSSProperties,

  button: {
    padding: "8px 20px",
    backgroundColor: "#2563eb",
    color: "#ffffff",
    border: "none",
    borderRadius: "6px",
    fontSize: "14px",
    fontWeight: 500,
    cursor: "pointer",
    alignSelf: "flex-end",
  } as React.CSSProperties,

  buttonDisabled: {
    backgroundColor: "#93c5fd",
    cursor: "not-allowed",
  } as React.CSSProperties,

  sourceBadge: {
    fontSize: "13px",
    color: "#64748b",
  } as React.CSSProperties,

  errorBanner: {
    padding: "10px 14px",
    backgroundColor: "#fef2f2",
    border: "1px solid #fca5a5",
    borderRadius: "6px",
    color: "#b91c1c",
    fontSize: "14px",
  } as React.CSSProperties,

  statusBanner: {
    padding: "10px 14px",
    backgroundColor: "#f0f9ff",
    border: "1px solid #bae6fd",
    borderRadius: "6px",
    color: "#0369a1",
    fontSize: "14px",
  } as React.CSSProperties,

  refreshBanner: {
    padding: "10px 14px",
    backgroundColor: "#f0fdf4",
    border: "1px solid #86efac",
    borderRadius: "6px",
    color: "#15803d",
    fontSize: "14px",
  } as React.CSSProperties,

  metricsSection: {
    display: "flex",
    flexDirection: "column",
    gap: "16px",
  } as React.CSSProperties,

  chartsRow: {
    display: "flex",
    gap: "16px",
    flexWrap: "wrap",
  } as React.CSSProperties,

  chartCell: {
    flex: "1 1 420px",
    minWidth: "320px",
  } as React.CSSProperties,

  graphContainer: {
    flex: 1,
    minHeight: "500px",
  } as React.CSSProperties,

  emptyState: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    minHeight: "300px",
    color: "#94a3b8",
    fontSize: "15px",
    background: "#f8fafc",
    borderRadius: "8px",
    border: "1px dashed #cbd5e1",
  } as React.CSSProperties,
} satisfies Record<string, React.CSSProperties>;
