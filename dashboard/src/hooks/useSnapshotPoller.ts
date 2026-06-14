/**
 * useSnapshotPoller — live snapshot refresh hook.
 *
 * Polls GET /api/snapshots/latest?datasetSource=<source> every 60 s.
 * When the returned snapshotId differs from the currently-loaded one, it
 * calls the provided `onNewSnapshot` callback so the parent can trigger a
 * full data re-fetch without a page reload.
 *
 * References: Requirements 8.6
 */

import { useEffect, useRef } from "react";
import type { DatasetSource } from "../api/client";
import { fetchLatestSnapshot } from "../api/client";

export const POLL_INTERVAL_MS = 60_000; // 60 seconds

export interface UseSnapshotPollerOptions {
  /** Currently loaded snapshot ID — the poller compares against this. */
  currentSnapshotId: string | null;
  /** Dataset source to query when polling. */
  datasetSource: DatasetSource;
  /** Whether polling should be active (false when no snapshot is loaded yet). */
  enabled: boolean;
  /**
   * Called when a newer snapshotId is detected.
   * The caller is responsible for re-fetching data.
   */
  onNewSnapshot: (newSnapshotId: string) => void;
}

/**
 * Polls the latest-snapshot endpoint and calls `onNewSnapshot` whenever a
 * newer snapshot is available for the selected dataset source.
 *
 * The hook cleans up the interval on unmount or when dependencies change,
 * so there are no dangling timers.
 */
export function useSnapshotPoller({
  currentSnapshotId,
  datasetSource,
  enabled,
  onNewSnapshot,
}: UseSnapshotPollerOptions): void {
  // Keep a stable ref to the latest callback to avoid re-scheduling the
  // interval on every render when the function identity changes.
  const onNewSnapshotRef = useRef(onNewSnapshot);
  onNewSnapshotRef.current = onNewSnapshot;

  const currentSnapshotIdRef = useRef(currentSnapshotId);
  currentSnapshotIdRef.current = currentSnapshotId;

  useEffect(() => {
    if (!enabled) return;

    const poll = async () => {
      try {
        const latest = await fetchLatestSnapshot(datasetSource);
        const current = currentSnapshotIdRef.current;
        if (current !== null && latest.snapshotId !== current) {
          onNewSnapshotRef.current(latest.snapshotId);
        }
      } catch {
        // Silently ignore network errors during background polling —
        // the next tick will retry automatically.
      }
    };

    const intervalId = setInterval(() => {
      void poll();
    }, POLL_INTERVAL_MS);

    return () => {
      clearInterval(intervalId);
    };
  }, [datasetSource, enabled]);
}
