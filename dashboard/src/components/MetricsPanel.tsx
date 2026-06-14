import React from "react";
import type { PolarizationDTO, SignedMetricsDTO } from "../api/client";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface MetricsPanelProps {
  /** Polarization metrics for the current snapshot. */
  polarizationMetrics: PolarizationDTO;
  /**
   * Aggregated signed-edge metrics for the current snapshot (wiki-RfA only).
   * When present, additional wiki-RfA metric cards are rendered.
   */
  signedMetrics?: SignedMetricsDTO[];
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/** Format a value in [0, 1] as a percentage string, e.g. 0.724 → "72.4%" */
function fmtPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** Format a number to 2 decimal places */
function fmtDec2(value: number): string {
  return value.toFixed(2);
}

/** Format as an integer */
function fmtInt(value: number): string {
  return Math.round(value).toString();
}

// ---------------------------------------------------------------------------
// MetricCard sub-component
// ---------------------------------------------------------------------------

interface MetricCardProps {
  label: string;
  value: string;
  /** When true, shows a "Wiki-RfA" badge in the top-right corner. */
  wikiRfaOnly?: boolean;
  /** Optional description shown as a subtitle. */
  description?: string;
}

function MetricCard({ label, value, wikiRfaOnly, description }: MetricCardProps) {
  return (
    <div style={cardStyles.card} role="figure" aria-label={`${label}: ${value}`}>
      <div style={cardStyles.header}>
        <span style={cardStyles.label}>{label}</span>
        {wikiRfaOnly && (
          <span style={cardStyles.badge} aria-label="Wiki-RfA only metric">
            Wiki-RfA
          </span>
        )}
      </div>
      <span style={cardStyles.value}>{value}</span>
      {description && (
        <span style={cardStyles.description}>{description}</span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Aggregated signed metrics helper
// ---------------------------------------------------------------------------

interface AggregatedSigned {
  avgPositiveEdgeRatio: number;
  avgNegativeEdgeRatio: number;
  avgCrossCommunityNegativity: number;
}

function aggregateSignedMetrics(metrics: SignedMetricsDTO[]): AggregatedSigned {
  if (metrics.length === 0) {
    return { avgPositiveEdgeRatio: 0, avgNegativeEdgeRatio: 0, avgCrossCommunityNegativity: 0 };
  }
  const sum = metrics.reduce(
    (acc, m) => ({
      pos: acc.pos + m.positiveEdgeRatio,
      neg: acc.neg + m.negativeEdgeRatio,
      ccn: acc.ccn + m.crossCommunityNegativity,
    }),
    { pos: 0, neg: 0, ccn: 0 }
  );
  const n = metrics.length;
  return {
    avgPositiveEdgeRatio: sum.pos / n,
    avgNegativeEdgeRatio: sum.neg / n,
    avgCrossCommunityNegativity: sum.ccn / n,
  };
}

// ---------------------------------------------------------------------------
// MetricsPanel
// ---------------------------------------------------------------------------

/**
 * Renders a row of metric cards for graph-level polarization metrics.
 * Conditionally renders additional wiki-RfA specific cards when
 * `signedMetrics` is provided.
 *
 * References: Requirements 8.2, 8.3, 8.5
 */
export function MetricsPanel({ polarizationMetrics, signedMetrics }: MetricsPanelProps) {
  const agg = signedMetrics ? aggregateSignedMetrics(signedMetrics) : null;

  return (
    <section
      style={panelStyles.section}
      aria-label="Metrics overview"
    >
      <h2 style={panelStyles.sectionTitle}>Metrics</h2>

      {/* Primary metrics row */}
      <div style={panelStyles.cardsRow} role="list" aria-label="Primary metrics">
        <div role="listitem">
          <MetricCard
            label="Polarization Index"
            value={fmtPct(polarizationMetrics.polarizationIndex)}
            description="Echo chamber strength [0–100%]"
          />
        </div>
        <div role="listitem">
          <MetricCard
            label="Modularity Q"
            value={fmtDec2(polarizationMetrics.modularity)}
            description="Community separation quality"
          />
        </div>
        <div role="listitem">
          <MetricCard
            label="Communities"
            value={fmtInt(polarizationMetrics.communityCount)}
            description="Detected community count"
          />
        </div>
        <div role="listitem">
          <MetricCard
            label="Avg Community Size"
            value={fmtDec2(polarizationMetrics.avgCommunitySize)}
            description="Mean users per community"
          />
        </div>
        <div role="listitem">
          <MetricCard
            label="Cross-Community Edge Ratio"
            value={fmtPct(polarizationMetrics.interCommunityEdgeRatio)}
            description="Fraction of inter-community edges"
          />
        </div>
      </div>

      {/* Wiki-RfA specific metrics */}
      {agg && (
        <>
          <h3 style={panelStyles.subTitle}>Signed-Edge Metrics (Wiki-RfA)</h3>
          <div style={panelStyles.cardsRow} role="list" aria-label="Wiki-RfA metrics">
            <div role="listitem">
              <MetricCard
                label="Positive Edge Ratio"
                value={fmtPct(agg.avgPositiveEdgeRatio)}
                wikiRfaOnly
                description="Avg fraction of positive votes"
              />
            </div>
            <div role="listitem">
              <MetricCard
                label="Negative Edge Ratio"
                value={fmtPct(agg.avgNegativeEdgeRatio)}
                wikiRfaOnly
                description="Avg fraction of negative votes"
              />
            </div>
            <div role="listitem">
              <MetricCard
                label="Cross-Community Negativity"
                value={fmtPct(agg.avgCrossCommunityNegativity)}
                wikiRfaOnly
                description="Avg negativity between communities"
              />
            </div>
          </div>
        </>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Inline styles
// ---------------------------------------------------------------------------

const panelStyles = {
  section: {
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  } as React.CSSProperties,

  sectionTitle: {
    fontSize: "16px",
    fontWeight: 700,
    color: "#1e293b",
    margin: 0,
  } as React.CSSProperties,

  subTitle: {
    fontSize: "13px",
    fontWeight: 600,
    color: "#475569",
    margin: 0,
    marginTop: "4px",
  } as React.CSSProperties,

  cardsRow: {
    display: "flex",
    flexWrap: "wrap",
    gap: "12px",
  } as React.CSSProperties,
} satisfies Record<string, React.CSSProperties>;

const cardStyles = {
  card: {
    display: "flex",
    flexDirection: "column",
    gap: "6px",
    padding: "14px 18px",
    background: "#f8fafc",
    border: "1px solid #e2e8f0",
    borderRadius: "8px",
    minWidth: "160px",
    flex: "0 0 auto",
  } as React.CSSProperties,

  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "8px",
  } as React.CSSProperties,

  label: {
    fontSize: "11px",
    fontWeight: 600,
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: "0.04em",
  } as React.CSSProperties,

  badge: {
    fontSize: "10px",
    fontWeight: 600,
    color: "#7c3aed",
    background: "#ede9fe",
    border: "1px solid #c4b5fd",
    borderRadius: "4px",
    padding: "1px 5px",
    flexShrink: 0,
  } as React.CSSProperties,

  value: {
    fontSize: "24px",
    fontWeight: 700,
    color: "#1e293b",
    lineHeight: 1.1,
  } as React.CSSProperties,

  description: {
    fontSize: "11px",
    color: "#94a3b8",
  } as React.CSSProperties,
} satisfies Record<string, React.CSSProperties>;
