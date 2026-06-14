import React, { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DiversityHistogramProps {
  /** Array of raw DiversityScore values (each in [0, 1]). */
  diversityScores: number[];
}

// ---------------------------------------------------------------------------
// Bucket configuration
// ---------------------------------------------------------------------------

const BUCKET_COUNT = 10;

interface HistogramBucket {
  /** Label displayed on the X-axis, e.g. "0.2–0.3" */
  range: string;
  /** User count falling in this bucket */
  count: number;
  /** Lower bound (inclusive) */
  low: number;
  /** Upper bound (exclusive, or inclusive for the last bucket) */
  high: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildBuckets(scores: number[]): HistogramBucket[] {
  const buckets: HistogramBucket[] = [];
  const step = 1 / BUCKET_COUNT;

  for (let i = 0; i < BUCKET_COUNT; i++) {
    const low = i * step;
    const high = (i + 1) * step;
    const label = `${low.toFixed(1)}–${high.toFixed(1)}`;
    buckets.push({ range: label, count: 0, low, high });
  }

  for (const score of scores) {
    const clamped = Math.max(0, Math.min(1, score));
    // For the last bucket, include 1.0 (boundary case)
    let idx = Math.floor(clamped / step);
    if (idx >= BUCKET_COUNT) idx = BUCKET_COUNT - 1;
    buckets[idx].count += 1;
  }

  return buckets;
}

// ---------------------------------------------------------------------------
// Custom tooltip
// ---------------------------------------------------------------------------

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const count = payload[0].value;
  return (
    <div style={tooltipStyles.box}>
      <span style={tooltipStyles.range}>Score: {label}</span>
      <span style={tooltipStyles.count}>
        {count} {count === 1 ? "user" : "users"}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Histogram of DiversityScore distribution.
 * X-axis: score buckets (0.0–0.1 … 0.9–1.0).
 * Y-axis: user count per bucket.
 *
 * References: Requirements 8.2, 8.5
 */
export function DiversityHistogram({ diversityScores }: DiversityHistogramProps) {
  const buckets = useMemo(() => buildBuckets(diversityScores), [diversityScores]);

  if (diversityScores.length === 0) {
    return (
      <div style={styles.empty}>
        <p>No diversity score data available for this snapshot.</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Diversity Score Distribution</h3>
      <p style={styles.subtitle}>
        {diversityScores.length} user{diversityScores.length !== 1 ? "s" : ""}
      </p>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart
          data={buckets}
          margin={{ top: 8, right: 16, left: 8, bottom: 24 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
          <XAxis
            dataKey="range"
            tick={{ fontSize: 10, fill: "#64748b" }}
            angle={-30}
            textAnchor="end"
            interval={0}
            label={{
              value: "Diversity Score",
              position: "insideBottom",
              offset: -16,
              fontSize: 12,
              fill: "#64748b",
            }}
          />
          <YAxis
            allowDecimals={false}
            tick={{ fontSize: 11, fill: "#64748b" }}
            label={{
              value: "User Count",
              angle: -90,
              position: "insideLeft",
              offset: 12,
              fontSize: 12,
              fill: "#64748b",
            }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar
            dataKey="count"
            fill="#2563eb"
            radius={[3, 3, 0, 0]}
            aria-label="Diversity score bucket"
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inline styles
// ---------------------------------------------------------------------------

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
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

  subtitle: {
    fontSize: "11px",
    color: "#94a3b8",
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

const tooltipStyles = {
  box: {
    padding: "8px 12px",
    background: "#ffffff",
    border: "1px solid #e2e8f0",
    borderRadius: "6px",
    display: "flex",
    flexDirection: "column",
    gap: "2px",
    fontSize: "13px",
    boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
  } as React.CSSProperties,

  range: {
    fontWeight: 600,
    color: "#1e293b",
  } as React.CSSProperties,

  count: {
    color: "#475569",
  } as React.CSSProperties,
} satisfies Record<string, React.CSSProperties>;
