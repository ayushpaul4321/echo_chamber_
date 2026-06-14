import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PolarizationSnapshotEntry {
  snapshotId: string;
  datasetSource: "reddit_title" | "congress" | "wiki_rfa";
  polarizationIndex: number;
  computedAt: string;
}

export interface PolarizationChartProps {
  snapshots: PolarizationSnapshotEntry[];
}

// ---------------------------------------------------------------------------
// Chart data types
// ---------------------------------------------------------------------------

interface ChartDataPoint {
  /** Formatted date string used as X-axis tick */
  date: string;
  reddit_title?: number;
  congress?: number;
  wiki_rfa?: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format an ISO datetime string to a short readable date. */
function fmtDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "2-digit",
    });
  } catch {
    return iso;
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Time-series line chart showing PolarizationIndex across snapshots.
 * Renders three separate series: Reddit (blue), Congress (red), Wiki-RfA (green).
 *
 * References: Requirements 8.2, 8.3
 */
export function PolarizationChart({ snapshots }: PolarizationChartProps) {
  // Group snapshots by date bucket; within a date keep the latest per source
  const chartData = useMemo<ChartDataPoint[]>(() => {
    if (snapshots.length === 0) return [];

    // Sort by computedAt ascending so the chart reads left-to-right chronologically
    const sorted = [...snapshots].sort(
      (a, b) => new Date(a.computedAt).getTime() - new Date(b.computedAt).getTime()
    );

    // Collect unique dates (one data point per snapshot, keyed by snapshotId+date combo)
    const dateMap = new Map<string, ChartDataPoint>();
    for (const entry of sorted) {
      const dateKey = fmtDate(entry.computedAt);
      if (!dateMap.has(dateKey)) {
        dateMap.set(dateKey, { date: dateKey });
      }
      const point = dateMap.get(dateKey)!;
      // Overwrite with latest value for that date bucket and source
      point[entry.datasetSource] = entry.polarizationIndex;
    }

    return Array.from(dateMap.values());
  }, [snapshots]);

  if (snapshots.length === 0) {
    return (
      <div style={styles.empty}>
        <p>No polarization time-series data available for this snapshot.</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Polarization Index Over Time</h3>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart
          data={chartData}
          margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11, fill: "#64748b" }}
            label={{
              value: "Snapshot Date",
              position: "insideBottom",
              offset: -4,
              fontSize: 12,
              fill: "#64748b",
            }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            tick={{ fontSize: 11, fill: "#64748b" }}
            label={{
              value: "Polarization Index",
              angle: -90,
              position: "insideLeft",
              offset: 12,
              fontSize: 12,
              fill: "#64748b",
            }}
          />
          <Tooltip
            formatter={(value: number, name: string) => [
              `${(value * 100).toFixed(1)}%`,
              SERIES_CONFIG[name as keyof typeof SERIES_CONFIG]?.label ?? name,
            ]}
            contentStyle={{
              fontSize: "13px",
              borderRadius: "6px",
              border: "1px solid #e2e8f0",
            }}
          />
          <Legend
            formatter={(value: string) =>
              SERIES_CONFIG[value as keyof typeof SERIES_CONFIG]?.label ?? value
            }
            wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }}
          />
          <Line
            type="monotone"
            dataKey="reddit_title"
            stroke="#2563eb"
            strokeWidth={2}
            dot={{ r: 3, fill: "#2563eb" }}
            activeDot={{ r: 5 }}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="congress"
            stroke="#dc2626"
            strokeWidth={2}
            dot={{ r: 3, fill: "#dc2626" }}
            activeDot={{ r: 5 }}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="wiki_rfa"
            stroke="#16a34a"
            strokeWidth={2}
            dot={{ r: 3, fill: "#16a34a" }}
            activeDot={{ r: 5 }}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Series configuration
// ---------------------------------------------------------------------------

const SERIES_CONFIG = {
  reddit_title: { label: "Reddit" },
  congress: { label: "Congress" },
  wiki_rfa: { label: "Wiki-RfA" },
} as const;

// ---------------------------------------------------------------------------
// Inline styles
// ---------------------------------------------------------------------------

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    padding: "16px",
    background: "#f8fafc",
    border: "1px solid #e2e8f0",
    borderRadius: "8px",
  } as React.CSSProperties,

  title: {
    fontSize: "14px",
    fontWeight: 600,
    color: "#1e293b",
    margin: 0,
  } as React.CSSProperties,

  empty: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "200px",
    background: "#f8fafc",
    border: "1px solid #e2e8f0",
    borderRadius: "8px",
    color: "#94a3b8",
    fontSize: "13px",
    textAlign: "center",
  } as React.CSSProperties,
} satisfies Record<string, React.CSSProperties>;
