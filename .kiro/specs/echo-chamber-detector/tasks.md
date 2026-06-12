# Execution Plan: Echo Chamber Detector

## Overview

This execution plan implements the Echo Chamber Detector end-to-end using all four local datasets. Each dataset plays a distinct, non-overlapping role across the pipeline — from primary graph construction through enrichment, validation, and signed-graph research extension.

**Dataset roles:**

| Dataset | File | Role | What it contributes |
|---|---|---|---|
| Reddit Title | `soc-redditHyperlinks-title.tsv` | Primary graph + sentiment edges | Subreddit→subreddit interactions, LINK_SENTIMENT score, timestamps |
| Reddit Body | `soc-redditHyperlinks-body.tsv` | Topic embedding source | PROPERTIES JSON with post body text → TF-IDF / sentence embeddings for recommendation engine |
| Congress Network | `congress_network/congress.edgelist` + `congress_network_data.json` | Political polarization validation | Pre-weighted Twitter influence graph, Democrat/Republican communities, bridge politicians |
| Wiki-RfA | `wiki-RfA.txt.gz` | Signed graph research extension | SRC→TGT VOT:(+1/-1) RES YEA DAT TXT — signed voting graph + comment text for sentiment-aware analysis |

**File format summary (confirmed from inspection):**
- `soc-redditHyperlinks-title.tsv`: columns `SOURCE_SUBREDDIT`, `TARGET_SUBREDDIT`, `POST_ID`, `TIMESTAMP`, `LINK_SENTIMENT`, `PROPERTIES`
- `soc-redditHyperlinks-body.tsv`: same schema; `PROPERTIES` contains richer body-text JSON
- `congress.edgelist`: `nodeA nodeB {'weight': float}` (transmission probabilities, already in [0,1])
- `congress_network_data.json`: `usernameList`, `inList`, `inWeight`, `outList`, `outWeight`
- `wiki-RfA.txt.gz`: record-per-blank-line format, fields `SRC:`, `TGT:`, `VOT:` (+1/-1), `RES:`, `YEA:`, `DAT:`, `TXT:`

---

## Phase 1: Project Foundation

- [x] 1.1 Initialize project structure and dependency configuration
  - Create directory layout: `ingestion/`, `graph/`, `community/`, `metrics/`, `recommendations/`, `api/`, `dashboard/`, `tests/`, `data/snapshots/`
  - Create `requirements.txt` with pinned versions:
    - `networkx==3.3`, `python-louvain==0.16`, `scikit-learn==1.5`
    - `fastapi==0.111`, `uvicorn==0.30`, `hypothesis==6.103`
    - `redis==5.0`, `psycopg2-binary==2.9`, `pandas==2.2`
    - `sentence-transformers==3.0` (for body TSV embeddings)
    - `alembic==1.13`, `neo4j==5.20`
  - Configure `pyproject.toml` with ruff (lint) and black (format)
  - **References**: design.md Dependencies

- [x] 1.2 Define core data models supporting all four datasets
  - Implement `InteractionRecord` dataclass with fields:
    - `id` (UUID), `sourceUserId`, `targetUserId`
    - `interactionType` (Enum: `HYPERLINK` | `RETWEET` | `VOTE`)
    - `timestamp` (datetime, optional — absent in Congress dataset)
    - `contentId` (str, optional), `topicTags` (list), `datasetSource` (str)
    - `sentimentScore` (float, optional) — `LINK_SENTIMENT` from Reddit; `VOT` (+1/-1) from wiki-RfA
    - `votePolarity` (int, optional: +1 / -1) — wiki-RfA only; distinct from sentiment score
    - `bodyText` (str, optional) — wiki-RfA `TXT:` field and Reddit body PROPERTIES
    - `voteResult` (int, optional: 0/1) — wiki-RfA `RES:` field (adminship granted or not)
  - Implement `Node` dataclass: `userId`, `communityId`, `betweenness`, `diversityScore`, `topicVector`
  - Implement `Edge` dataclass: `sourceUserId`, `targetUserId`, `weight`, `isCrossCommunity`, `signedPolarity` (int, optional — wiki-RfA)
  - Implement `InteractionGraph`, `CommunityPartition`, `PolarizationMetrics`, `UserMetrics`, `Recommendation` dataclasses
  - Add field-level validation: non-empty userIds, `sourceUserId ≠ targetUserId`, timestamp is past datetime if present, weight ≥ 0
  - **References**: design.md Data Models, Requirements 9.1–9.4

- [x] 1.3 Set up database schemas
  - PostgreSQL tables: `snapshots`, `polarization_metrics`, `user_metrics`, `recommendations` — all keyed by `snapshotId` and `datasetSource`
  - Add `signed_metrics` table for wiki-RfA: `positive_edge_ratio`, `negative_edge_ratio`, `net_sentiment` per community
  - Redis key convention: `metrics:{snapshotId}:polarization`, `user:{userId}:recommendations`, `graph:{snapshotId}:page:{n}`
  - Alembic migration scripts for all tables
  - **References**: Requirements 4.6, 4.7, 5.7, 6.7

---

## Phase 2: Ingestion Layer — All Four Dataset Adapters

- [ ] 2.1 Implement `RedditTitleAdapter` for `soc-redditHyperlinks-title.tsv`
  - Chunked streaming via `pandas.read_csv(chunksize=10000)` — file exceeds 50 MB
  - Column mapping: `SOURCE_SUBREDDIT → sourceUserId`, `TARGET_SUBREDDIT → targetUserId`, `POST_ID → contentId`, `TIMESTAMP → timestamp`, `LINK_SENTIMENT → sentimentScore`, `interactionType = HYPERLINK`, `datasetSource = "reddit_title"`
  - Parse `TIMESTAMP` as datetime; reject records with future timestamps
  - Store `sentimentScore` as float; edge weight in graph = interaction count (normalized later in Phase 3)
  - **References**: Requirements 1.1, 1.2

- [ ] 2.2 Implement `RedditBodyAdapter` for `soc-redditHyperlinks-body.tsv`
  - Same chunked streaming and column mapping as `RedditTitleAdapter`; `datasetSource = "reddit_body"`
  - Additionally parse `PROPERTIES` column as JSON; extract `title` and body text fields and store in `bodyText`
  - `bodyText` is used exclusively in Phase 6 for TF-IDF / sentence embedding construction — not stored in the main interaction store
  - Build a `subreddit_text_corpus` dict: `{subreddit_name: [list of body texts]}` during ingestion for downstream vectorization
  - **References**: Requirements 1.1, 1.2

- [ ] 2.3 Implement `CongressNetworkAdapter` for `congress.edgelist` + `congress_network_data.json`
  - Load `congress_network_data.json`; build `id_to_username = usernameList[i]` lookup map
  - Parse `congress.edgelist` line by line: extract `nodeA`, `nodeB`, and `weight` from `{'weight': float}` dict string using regex
  - Map: `sourceUserId = id_to_username[nodeA]`, `targetUserId = id_to_username[nodeB]`, `weight = transmission_probability`, `interactionType = RETWEET`, `datasetSource = "congress"`
  - Flag `pre_normalized = True` — weights already in [0,1]; skip normalization step in `buildGraph`
  - No `timestamp` field — deduplication key is `(sourceUserId, targetUserId)` only
  - **References**: Requirements 1.1, 1.2

- [ ] 2.4 Implement `WikiRfAAdapter` for `wiki-RfA.txt.gz`
  - Decompress with `gzip.open(..., 'rt', encoding='utf-8', errors='replace')`
  - Parse record-per-blank-line format: read lines, split on blank line to get records, extract fields by prefix (`SRC:`, `TGT:`, `VOT:`, `RES:`, `YEA:`, `DAT:`, `TXT:`)
  - Map: `sourceUserId = SRC`, `targetUserId = TGT`, `votePolarity = int(VOT)` (+1 or -1), `voteResult = int(RES)`, `timestamp = parse(DAT)`, `bodyText = TXT`, `interactionType = VOTE`, `datasetSource = "wiki_rfa"`
  - Edge weight = 1.0 for all votes (binary); sign carried in `votePolarity`, not weight
  - `bodyText` (TXT field) stored for later sentiment/topic analysis in Phase 6
  - **References**: Requirements 1.1, 1.2

- [ ] 2.5 Implement `IngestionService` with deduplication and retry logic
  - Deduplication keys by dataset:
    - Reddit: `(sourceUserId, targetUserId, timestamp)`
    - Congress: `(sourceUserId, targetUserId)`
    - Wiki-RfA: `(sourceUserId, targetUserId, timestamp)` — multiple votes by same user on same target in different years are valid
  - File-based adapters skip HTTP retry; log warning on read failure
  - Preserve previous Snapshot on failure; log warning on zero valid records
  - **References**: Requirements 1.3, 1.4, 1.5, 1.6

- [ ] 2.6 Implement input record validation
  - Reject: empty `sourceUserId` or `targetUserId`; unrecognized `interactionType`; future timestamps; self-loops
  - Wiki-RfA extra validation: `VOT` must be +1 or -1; `RES` must be 0 or 1
  - Log each rejection with record ID and reason without halting the job
  - **References**: Requirements 9.1–9.5

- [ ] 2.7 Write tests for all four adapters
  - Unit tests per adapter: correct column/field mapping; chunked streaming on small fixture produces same result as single-pass; Congress ID resolution to username; wiki-RfA blank-line record parsing
  - Property tests (Hypothesis): normalization completeness — every accepted record has all required fields (Property 6); deduplication per dataset key (Property 7); invalid record rejection including wiki-RfA polarity bounds (Property 5)
  - **References**: Properties 5, 6, 7

---

## Phase 3: Graph Construction Service

- [ ] 3.1 Implement `buildGraph` with dataset-aware weight handling
  - Reddit graphs: aggregate raw interaction counts per (source, target) pair; normalize to [0,1] by dividing by max count
  - Congress graph: use pre-normalized transmission probability weights directly; no normalization step
  - Wiki-RfA graph: build two sub-graphs — `positive_graph` (VOT=+1 edges) and `negative_graph` (VOT=-1 edges); combined `InteractionGraph` stores all edges with `signedPolarity` field; weight = 1.0 for all edges
  - Initialize all node fields (`communityId=None`, `betweenness=0.0`, `diversityScore=0.0`, `topicVector=[]`)
  - Assign UUID `snapshotId` and `createdAt`; tag graph with `datasetSource`
  - **References**: design.md Algorithm 1, Requirements 2.1–2.4

- [ ] 3.2 Implement `updateGraph` for incremental updates
  - Merge new records into existing edge accumulator; re-normalize weights for Reddit; preserve pre-normalized weights for Congress
  - Preserve node metadata for unchanged nodes
  - **References**: Requirements 2.7

- [ ] 3.3 Implement graph serialization / deserialization
  - Serialize to GraphML (primary) — include `signedPolarity` as edge attribute for wiki-RfA graphs
  - Adjacency-list JSON (secondary)
  - Lossless round-trip: deserializing must reproduce identical node set, edge set, weights, and signed polarity
  - Pretty-print formatter
  - **References**: Requirements 11.1–11.4

- [ ] 3.4 Implement `persistGraph` and `loadGraph`
  - Write to `data/snapshots/{datasetSource}/{snapshotId}.graphml`
  - Mirror graph to Neo4j: upsert `(:Node)` with all node properties and `[:EDGE]` relationships with `weight`, `signedPolarity`, `isCrossCommunity` attributes
  - `loadGraph` reads from GraphML (primary); Neo4j used for querying and visualization, not as the load source
  - Load by `snapshotId`
  - **References**: Requirements 2.6

- [ ] 3.5 Write tests for Graph Construction Service
  - Unit tests: two Reddit records same (source, target) → single edge weight = 1.0; Congress pre-normalized weight passes through unchanged; wiki-RfA VOT=+1 → `signedPolarity=1` on edge; self-loop rejected across all dataset types
  - Property tests (Hypothesis): node coverage (Property 1); edge weights in [0,1] (Property 1); idempotence (Property 2); incremental equivalence (Property 3); serialization round-trip including `signedPolarity` (Property 4)
  - **References**: Properties 1, 2, 3, 4

---

## Phase 4: Community Detection Engine

- [ ] 4.1 Implement Louvain community detection
  - Use `python-louvain` (`community.best_partition`) on the NetworkX graph
  - For wiki-RfA: run Louvain on the unsigned graph (all edges, weight=1.0) first; signed polarity analysis done separately in Phase 5
  - Assign every node exactly one `communityId`; isolated nodes → singleton community
  - Enforce `MAX_ITERATIONS = 100`; log warning and flag snapshot as `"approximate_partition"` if cap is hit
  - **References**: design.md Algorithm 2, Requirements 3.1, 3.2, 3.5

- [ ] 4.2 Implement modularity computation and label persistence
  - Compute modularity Q using `community.modularity`; store in `CommunityPartition`
  - Label persistence: match new communities to previous Snapshot's communities by Jaccard overlap (threshold 0.5)
  - **References**: Requirements 3.3, 3.4, 3.6

- [ ] 4.3 Implement Girvan-Newman secondary validation
  - Run `networkx.algorithms.community.girvan_newman` when `enable_girvan_newman=True`
  - Store secondary partition in snapshot metadata alongside Louvain result
  - **References**: Requirements 3.7

- [ ] 4.4 Write tests for Community Detection Engine
  - Unit tests: K(5,5) bipartite → two communities of 5; isolated node → singleton; modularity Q in output
  - Property tests (Hypothesis): every node assigned exactly one CommunityId (Property 8); modularity Q ≥ 0 (Property 9)
  - Integration tests:
    - Reddit title graph → Louvain detects multiple topically distinct subreddit communities
    - Congress graph → Louvain detects 2 dominant communities (Democrat / Republican)
    - Wiki-RfA graph → Louvain detects communities of Wikipedia editors with similar voting patterns
  - **References**: Properties 8, 9

---

## Phase 5: Metrics & Analysis Service (including Signed Graph Metrics for wiki-RfA)

- [ ] 5.1 Implement Polarization Index computation
  - Standard PI = `1.0 − (interEdgeWeight / totalEdgeWeight)`; return 0.0 on empty graph
  - Set `edge.isCrossCommunity` flag as side effect
  - Expected benchmarks: Reddit title ≈ 0.65–0.75; Congress ≈ 0.85–0.91; wiki-RfA (unsigned) ≈ 0.50–0.65
  - **References**: design.md Algorithm 3, Requirements 4.1–4.5

- [ ] 5.2 Implement per-user and per-community Diversity Score
  - Per-user: `crossCommunityWeight / totalOutgoingWeight`; return 0.0 for no outgoing edges
  - Community-level: arithmetic mean of member users' scores
  - **References**: design.md Algorithm 4, Requirements 5.1–5.5

- [ ] 5.3 Implement betweenness centrality (Brandes' algorithm)
  - `networkx.betweenness_centrality(graph, normalized=True)` for graphs ≤ 100K nodes
  - `networkx.betweenness_centrality(graph, k=500, normalized=True)` (approximate) for > 100K nodes
  - Store in `node.betweenness`
  - **References**: design.md Algorithm 5, Requirements 5.6

- [ ] 5.4 Implement signed graph metrics for wiki-RfA (sentiment-aware analysis)
  - For wiki-RfA graphs only: compute `positive_edge_ratio` and `negative_edge_ratio` per community
    - `positive_edge_ratio = sum(weight where signedPolarity=+1) / sum(all weights)` within community
    - `negative_edge_ratio = 1.0 - positive_edge_ratio`
  - Compute `net_sentiment_index` per community: mean `votePolarity` of all intra-community edges
  - Compute `cross_community_negativity`: ratio of negative edges that cross community boundaries vs total negative edges
  - Persist signed metrics to `signed_metrics` table per snapshot
  - **References**: wiki-RfA unique contribution; Requirements 4.6

- [ ] 5.5 Implement metrics persistence and querying
  - Persist `PolarizationMetrics` per snapshot per `datasetSource`
  - Persist `UserMetrics` per user per snapshot
  - `queryMetrics(filter)` supports: `snapshotId`, `datasetSource`, date range, `communityId`, metric threshold
  - **References**: Requirements 4.6, 4.7, 5.7

- [ ] 5.6 Write tests for Metrics & Analysis Service
  - Unit tests: all-intra → PI = 1.0; all-inter → PI = 0.0; no-edge user → DS = 0.0; wiki-RfA all-positive community → `positive_edge_ratio = 1.0`
  - Property tests (Hypothesis): PI ∈ [0,1] (Property 10); PI + ICER = 1.0 (Property 11); PI boundary conditions (Property 12); DS ∈ [0,1] (Property 13); DS boundaries (Property 14); community averaging invariant (Property 15); betweenness ∈ [0,1] (Property 16)
  - Integration tests: Reddit PI > 0.60; Congress PI > 0.80; wiki-RfA `cross_community_negativity` > `intra_community_negativity` (negative votes cross boundaries more)
  - **References**: Properties 10–16

---

## Phase 6: Topic Embeddings and Semantic Recommendation Engine

- [ ] 6.1 Build topic vectors from Reddit body TSV and wiki-RfA comment text
  - **Reddit body source** (`soc-redditHyperlinks-body.tsv`):
    - Use `subreddit_text_corpus` built in Phase 2 task 2.2
    - Fit `TfidfVectorizer(max_features=5000)` on all subreddit text; store per-subreddit TF-IDF vector in `node.topicVector`
    - Optionally upgrade to `sentence-transformers` (`all-MiniLM-L6-v2`) for denser semantic vectors
  - **Wiki-RfA TXT source**:
    - Aggregate `TXT:` comment text per Wikipedia editor (userId)
    - Fit separate TF-IDF on wiki editor text corpus; store in `node.topicVector` for wiki-RfA graph nodes
  - **Congress fallback** (no body text):
    - Party affiliation proxy: assign `topicVector = [1.0, 0.0]` for Democrats and `[0.0, 1.0]` for Republicans based on community membership from Louvain result
  - **References**: Requirements 6.3, 6.6

- [ ] 6.2 Implement bridge node identification and candidate scoring
  - Filter candidates: `node.communityId ≠ user.communityId` AND `node.betweenness > BRIDGE_CENTRALITY_THRESHOLD` (default 0.01)
  - Score by `cosineSimilarity(user.topicVector, candidate.topicVector)`; exclude below `MIN_TOPIC_RELEVANCE_THRESHOLD` (default 0.1)
  - Wiki-RfA extra filter: exclude candidates with `net_sentiment_index < 0` (highly negative editors) from recommendations
  - Fallback to community centroid topic vector when user has < 5 interactions
  - **References**: design.md Algorithm 6, Requirements 6.2, 6.3, 6.6

- [ ] 6.3 Implement diversity gain estimation and recommendation ranking
  - `estimateDiversityGain(userId, candidateId, graph, metrics)`: simulate adding edge; compute new diversity score; return delta
  - Sort candidates descending by `diversityGain`; return ≤ `topK` as `Recommendation` objects with human-readable `reason` string
  - **References**: design.md Algorithm 6, Requirements 6.4, 6.5, 6.8

- [ ] 6.4 Implement recommendation persistence and retrieval
  - Persist each `Recommendation` to `recommendations` table
  - `fetchRecommendations(userId)` retrieval
  - **References**: Requirements 6.7

- [ ] 6.5 Write tests for Recommendation Engine
  - Unit tests: isolated subreddit → all recs from other communities; Congress node → categorical topic vector used; wiki-RfA negative editor excluded from recs; topK=0 → empty list
  - Property tests (Hypothesis): cross-community invariant (Property 17); bridge + topic threshold enforced (Property 18); sorted by diversityGain desc (Property 19); len(result) ≤ topK (Property 20)
  - **References**: Properties 17–20

---

## Phase 7: API Layer

- [ ] 7.1 Implement REST API endpoints with FastAPI
  - `GET /api/snapshots/{snapshotId}/graph` → paginated `GraphDTO`
  - `GET /api/snapshots/{snapshotId}/metrics/polarization` → `PolarizationDTO`
  - `GET /api/snapshots/{snapshotId}/metrics/signed` → `SignedMetricsDTO` (wiki-RfA only, returns 404 for other datasets)
  - `GET /api/users/{userId}/metrics` → `UserMetricsDTO`
  - `GET /api/communities/{communityId}/metrics` → `CommunityMetricsDTO`
  - `GET /api/users/{userId}/recommendations` → `List[RecommendationDTO]`
  - All endpoints require JWT or API key (`Authorization: Bearer <token>`)
  - **References**: design.md Component 6, Requirements 7.1–7.5, 7.8

- [ ] 7.2 Implement pagination, Redis caching, filtering, and rate limiting
  - Cursor-based pagination for graph endpoint (default 500 nodes per page)
  - Redis cache with TTL = 24h; return cached result on hit
  - Filter support: `?datasetSource=`, `?snapshotId=`, `?from=`, `?to=`, `?communityId=`, `?min_polarization=`
  - Rate limiting: 100 req/min per API key
  - **References**: Requirements 7.6, 7.7, 7.10, 7.11

- [ ] 7.3 Implement per-user access control
  - Recommendations endpoint: enforce `caller.userId == requested userId`; return HTTP 403 otherwise
  - **References**: Requirements 7.9

- [ ] 7.4 Write tests for API Layer
  - Unit tests: unauthenticated → 401; mismatched userId → 403; `signed` endpoint on Reddit snapshot → 404; valid request → 200 with correct DTO shape
  - Integration test: full pipeline for each dataset (ingest → build → detect → metrics → recommendations → API)
  - **References**: Requirements 7.1–7.11

---

## Phase 8: Visualization Dashboard

- [ ] 8.1 Scaffold React frontend with graph rendering library
  - Initialize React app with `sigma.js` (or `d3-force`) for graph rendering
  - Configure API client with `datasetSource` selector (Reddit / Congress / Wiki-RfA toggle)
  - **References**: design.md Dependencies

- [ ] 8.2 Implement interactive graph visualization
  - Nodes colored by community membership (Louvain partition)
  - Node size proportional to betweenness centrality (bridge nodes visually prominent)
  - Edge color: green for positive polarity (`signedPolarity=+1`), red for negative (`signedPolarity=-1`) on wiki-RfA; neutral gray for Reddit/Congress
  - On node click: sidebar showing `diversityScore`, community label, top-5 recommendations
  - **References**: Requirements 8.1, 8.4

- [ ] 8.3 Implement metrics panels
  - Metric cards: `PolarizationIndex`, `modularity Q`, community count, average `DiversityScore`
  - Wiki-RfA additional cards: `positive_edge_ratio`, `negative_edge_ratio`, `cross_community_negativity`
  - Time-series line chart: PolarizationIndex across snapshots; Reddit / Congress / wiki-RfA as separate series
  - Histogram: DiversityScore distribution
  - **References**: Requirements 8.2, 8.3, 8.5

- [ ] 8.4 Implement live snapshot refresh
  - Poll `GET /api/snapshots/latest?datasetSource=...` every 60 s
  - Re-render on new snapshot without full page reload
  - **References**: Requirements 8.6

---

## Phase 9: Integration and Four-Dataset Comparative Analysis

- [ ] 9.1 Run full pipeline on Reddit title dataset
  - Ingest `soc-redditHyperlinks-title.tsv` via `RedditTitleAdapter` (chunked streaming)
  - Build graph → Louvain → Polarization Index (expect > 0.60) → Diversity Scores
  - Generate recommendations for 10 sample low-diversity subreddits
  - Verify all API endpoints return correct shapes
  - **References**: Requirements 1–11

- [ ] 9.2 Run full pipeline on Reddit body dataset
  - Ingest `soc-redditHyperlinks-body.tsv` via `RedditBodyAdapter`; confirm `subreddit_text_corpus` populated
  - Build topic vectors (TF-IDF); confirm `node.topicVector` non-empty for all subreddit nodes
  - Re-run recommendations on same subreddits as 9.1; confirm semantic ranking differs from graph-only ranking
  - **References**: Phase 6 topic embedding tasks

- [ ] 9.3 Run full pipeline on Congress network dataset
  - Ingest via `CongressNetworkAdapter`; confirm integer IDs resolved to Twitter usernames
  - Build graph → Louvain → expect 2 dominant communities → Polarization Index (expect > 0.80)
  - Identify top-10 bridge politicians by betweenness centrality
  - **References**: Requirements 1–11

- [ ] 9.4 Run full pipeline on wiki-RfA dataset
  - Ingest via `WikiRfAAdapter`; parse all `SRC`, `TGT`, `VOT`, `RES`, `DAT`, `TXT` fields
  - Build signed graph → Louvain → Polarization Index → signed metrics (`positive_edge_ratio`, `cross_community_negativity`)
  - Confirm negative votes cross community boundaries at higher rate than positive votes
  - **References**: Phase 5 signed metrics tasks

- [ ] 9.5 Produce four-dataset comparative results table
  - Output table: `dataset | nodes | edges | communities | PolarizationIndex | avg DiversityScore | notes`
  - Expected row values:
    - Reddit title: ~50K nodes, ~860K edges, many communities, PI ≈ 0.65–0.75
    - Congress: ~475 nodes, ~13K+ edges, 2 dominant communities, PI ≈ 0.85–0.91
    - Wiki-RfA: ~10K nodes, ~200K+ edges, multiple editor communities, PI ≈ 0.50–0.65
  - This table is the primary presentable output validating the multi-dataset architecture
  - **References**: design.md Integration Testing Approach

- [ ] 9.6 Run incremental update integration test
  - Load Reddit title snapshot; add 10% new records via `updateGraph`; verify result equivalent to full rebuild
  - **References**: Property 3, Requirements 2.7

---

## Phase 10: Security Hardening and Performance Validation

- [ ] 10.1 Implement PII anonymization in Ingestion Layer
  - Congress: hash Twitter usernames → internal UUIDs before storage; maintain `username_lookup.json` locally (not exposed via API)
  - Wiki-RfA: hash Wikipedia editor usernames → internal UUIDs; `SRC` and `TGT` are real usernames
  - Reddit: subreddit names are pseudonymous; no additional hashing required
  - Verify no raw `bodyText` or `TXT` is stored in the database — only `topicVector` embeddings
  - **References**: Requirements 10.1, 10.3

- [ ] 10.2 Harden API authentication and rate limiting
  - Confirm JWT / API key required on all endpoints; verify no open routes exist
  - File-based adapters are exempt from upstream rate limit enforcement; document this
  - **References**: Requirements 10.2, 10.4

- [ ] 10.3 Validate performance across all datasets
  - Benchmark Louvain on full Reddit body graph (largest dataset); record wall-clock time
  - Enable approximate betweenness (`k=500`) automatically when node count > 100K
  - Benchmark wiki-RfA signed metrics computation; confirm it adds < 10% overhead vs unsigned pipeline
  - Verify Redis cache reduces repeated `/metrics/polarization` latency by ≥ 90% on all three dataset sources
  - **References**: design.md Performance Considerations
