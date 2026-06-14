import React from "react";
import type { DatasetSource } from "../api/client";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DatasetOption {
  label: string;
  value: DatasetSource;
}

const DATASET_OPTIONS: DatasetOption[] = [
  { label: "Reddit", value: "reddit_title" },
  { label: "Congress", value: "congress" },
  { label: "Wiki-RfA", value: "wiki_rfa" },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface DatasetSelectorProps {
  /** The currently selected dataset source value. */
  selected: DatasetSource;
  /** Called when the user clicks a different dataset option. */
  onChange: (value: DatasetSource) => void;
}

/**
 * A tab-style toggle that lets the user switch between the three supported
 * dataset sources: Reddit (reddit_title), Congress, and Wiki-RfA (wiki_rfa).
 */
export function DatasetSelector({ selected, onChange }: DatasetSelectorProps) {
  return (
    <div style={styles.container} role="tablist" aria-label="Dataset source">
      {DATASET_OPTIONS.map((option) => {
        const isSelected = option.value === selected;
        return (
          <button
            key={option.value}
            role="tab"
            aria-selected={isSelected}
            onClick={() => onChange(option.value)}
            style={{
              ...styles.tab,
              ...(isSelected ? styles.tabActive : styles.tabInactive),
            }}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inline styles (no external CSS dependency for scaffold)
// ---------------------------------------------------------------------------

const styles = {
  container: {
    display: "flex",
    gap: "0",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    overflow: "hidden",
    width: "fit-content",
  } as React.CSSProperties,
  tab: {
    padding: "8px 20px",
    border: "none",
    cursor: "pointer",
    fontSize: "14px",
    fontWeight: 500,
    transition: "background-color 0.15s, color 0.15s",
    outline: "none",
  } as React.CSSProperties,
  tabActive: {
    backgroundColor: "#2563eb",
    color: "#ffffff",
  } as React.CSSProperties,
  tabInactive: {
    backgroundColor: "#f9fafb",
    color: "#374151",
  } as React.CSSProperties,
} satisfies Record<string, React.CSSProperties>;
