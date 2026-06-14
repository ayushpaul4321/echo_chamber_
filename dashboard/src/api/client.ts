/**
 * API client for the Echo Chamber Detector REST API.
 *
 * Base URL is configured via the VITE_API_BASE_URL environment variable
 * (defaults to http://localhost:8000).
 *
 * All requests attach an Authorization: Bearer <token> header where the
 * token is read from localStorage key "echo_chamber_token".
 */

// ---------------------------------------------------------------------------
// Response types — mirroring the backend DTOs from api/dtos.py
// ---------------------------------------------------------------------------

export interface PolarizationListDTO {
  items: PolarizationDTO[];
  total: number;
  nextCursor: string | null;
}

export interface UserMetricsListDTO {
  items: UserMetricsDTO[];
  total: number;
  nextCursor: string | null;
}

export interface NodeDTO {
  userId: string;
  communityId: string | null;
  betweenness: number;
  diversityScore: number;
  topicVector: number[];
}

export interface EdgeDTO {
  sourceUserId: string;
  targetUserId: string;
  weight: number;
  isCrossCommunity: boolean;
  signedPolarity: number | null;
}

export interface GraphDTO {
  nodes: NodeDTO[];
  edges: EdgeDTO[];
  snapshotId: string;
  createdAt: string;
  nodeCount: number;
  edgeCount: number;
  nextCursor: string | null;
}

export interface PolarizationDTO {
  snapshotId: string;
  polarizationIndex: number;
  modularity: number;
  communityCount: number;
  avgCommunitySize: number;
  interCommunityEdgeRatio: number;
  computedAt: string;
}

export interface SignedMetricsDTO {
  snapshotId: string;
  communityId: string;
  positiveEdgeRatio: number;
  negativeEdgeRatio: number;
  netSentimentIndex: number;
  crossCommunityNegativity: number;
  computedAt: string;
}

export interface UserMetricsDTO {
  userId: string;
  communityId: string;
  diversityScore: number;
  intraEdgeCount: number;
  interEdgeCount: number;
  betweennessCentrality: number;
  snapshotId: string;
  computedAt: string;
}

export interface CommunityMetricsDTO {
  communityId: string;
  memberCount: number;
  modularity: number;
  avgDiversityScore: number;
  polarizationIndex: number;
  snapshotId: string;
}

export interface RecommendationDTO {
  recommendationId: string;
  targetUserId: string;
  recommendedUserId: string;
  diversityGain: number;
  topicRelevance: number;
  communityId: string;
  reason: string;
}

export interface LatestSnapshotDTO {
  snapshotId: string;
  datasetSource: string;
  computedAt: string;
}

// ---------------------------------------------------------------------------
// Dataset source values (matches backend datasetSource strings)
// ---------------------------------------------------------------------------

export type DatasetSource = "reddit_title" | "congress" | "wiki_rfa";

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

const BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://localhost:8000";

function getToken(): string | null {
  return localStorage.getItem("echo_chamber_token");
}

async function apiFetch<T>(path: string, params?: Record<string, string>): Promise<T> {
  const token = getToken();
  const url = new URL(`${BASE_URL}${path}`);

  if (params) {
    for (const [key, value] of Object.entries(params)) {
      url.searchParams.set(key, value);
    }
  }

  const headers: HeadersInit = {
    "Content-Type": "application/json",
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(url.toString(), { headers });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`API error ${response.status}: ${text}`);
  }

  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/**
 * Fetch a paginated graph snapshot.
 *
 * @param snapshotId - The snapshot ID to load.
 * @param cursor     - Optional pagination cursor (userId of the last node on
 *                     the previous page).
 */
export async function fetchGraphSnapshot(
  snapshotId: string,
  cursor?: string
): Promise<GraphDTO> {
  const params: Record<string, string> = {};
  if (cursor) {
    params["cursor"] = cursor;
  }
  return apiFetch<GraphDTO>(`/api/snapshots/${encodeURIComponent(snapshotId)}/graph`, params);
}

/**
 * Fetch polarization metrics for a snapshot.
 */
export async function fetchPolarizationMetrics(
  snapshotId: string
): Promise<PolarizationDTO> {
  return apiFetch<PolarizationDTO>(
    `/api/snapshots/${encodeURIComponent(snapshotId)}/metrics/polarization`
  );
}

/**
 * Fetch signed-edge sentiment metrics for a snapshot (wiki-RfA only).
 * Throws if the snapshot has no signed metrics (non-wiki-RfA datasets).
 */
export async function fetchSignedMetrics(
  snapshotId: string
): Promise<SignedMetricsDTO[]> {
  return apiFetch<SignedMetricsDTO[]>(
    `/api/snapshots/${encodeURIComponent(snapshotId)}/metrics/signed`
  );
}

/**
 * Fetch per-user diversity and centrality metrics.
 */
export async function fetchUserMetrics(userId: string): Promise<UserMetricsDTO> {
  return apiFetch<UserMetricsDTO>(`/api/users/${encodeURIComponent(userId)}/metrics`);
}

/**
 * Fetch aggregated community-level metrics.
 */
export async function fetchCommunityMetrics(
  communityId: string
): Promise<CommunityMetricsDTO> {
  return apiFetch<CommunityMetricsDTO>(
    `/api/communities/${encodeURIComponent(communityId)}/metrics`
  );
}

/**
 * Fetch cross-community recommendations for a user.
 * Requires the authenticated caller to match the requested userId.
 */
export async function fetchRecommendations(
  userId: string
): Promise<RecommendationDTO[]> {
  return apiFetch<RecommendationDTO[]>(
    `/api/users/${encodeURIComponent(userId)}/recommendations`
  );
}

/**
 * Fetch a filtered, paginated list of polarization metrics across all snapshots.
 * Used to populate the time-series chart.
 *
 * @param datasetSource - Optional filter by dataset source.
 * @param cursor        - Optional pagination cursor.
 * @param pageSize      - Records per page (default 50, max 500).
 */
export async function fetchAllPolarizationMetrics(
  datasetSource?: DatasetSource,
  cursor?: string,
  pageSize: number = 50
): Promise<PolarizationListDTO> {
  const params: Record<string, string> = {
    page_size: String(pageSize),
  };
  if (datasetSource) params["datasetSource"] = datasetSource;
  if (cursor) params["cursor"] = cursor;
  return apiFetch<PolarizationListDTO>("/api/metrics/polarization", params);
}

/**
 * Fetch a paginated list of user metrics for a snapshot.
 * Used to gather diversity scores for the histogram.
 *
 * @param snapshotId - Snapshot to filter by.
 * @param cursor     - Optional pagination cursor.
 * @param pageSize   - Records per page (default 500, max 1000).
 */
export async function fetchUserMetricsList(
  snapshotId: string,
  cursor?: string,
  pageSize: number = 500
): Promise<UserMetricsListDTO> {
  const params: Record<string, string> = {
    snapshotId,
    page_size: String(pageSize),
  };
  if (cursor) params["cursor"] = cursor;
  return apiFetch<UserMetricsListDTO>("/api/users/metrics", params);
}

/**
 * Fetch the most recent snapshot descriptor for a dataset source.
 * Used by the live-refresh polling hook to detect new snapshots.
 *
 * @param datasetSource - Optional filter by dataset source.
 */
export async function fetchLatestSnapshot(
  datasetSource?: DatasetSource
): Promise<LatestSnapshotDTO> {
  const params: Record<string, string> = {};
  if (datasetSource) params["datasetSource"] = datasetSource;
  return apiFetch<LatestSnapshotDTO>("/api/snapshots/latest", params);
}
