# Requirements Document

## Introduction

The Echo Chamber Detector is a data pipeline and analysis system that ingests social media interaction data, constructs weighted directed interaction graphs, and applies community detection and polarization metrics to identify echo chambers — isolated clusters where users predominantly interact with ideologically similar peers. The system computes a Polarization Index (graph-level), Diversity Scores (user- and community-level), and exposes results through a REST/GraphQL API and a visualization dashboard. A recommendation engine surfaces balanced viewpoints to users trapped in low-diversity clusters. The system supports multiple public datasets (SNAP polblogs recommended as the starting point) and is designed for dataset-agnostic operation.

---

## Glossary

- **Ingestion_Layer**: The subsystem responsible for fetching, normalizing, and deduplicating raw interaction records from data sources.
- **Graph_Builder**: The service that transforms normalized interaction records into a weighted directed InteractionGraph.
- **Community_Detector**: The engine that partitions the InteractionGraph into communities using the Louvain algorithm.
- **Metrics_Service**: The service that computes Polarization Index and Diversity Scores from the graph and community partition.
- **Recommendation_Engine**: The service that generates balanced viewpoint recommendations for low-diversity users.
- **API_Layer**: The REST/GraphQL interface that exposes graph data, metrics, and recommendations to consumers.
- **Dashboard**: The frontend visualization interface displaying the interaction graph, metrics charts, and recommendations.
- **InteractionRecord**: A normalized, deduplicated record of a single user-to-user interaction (retweet, reply, mention, upvote, comment).
- **InteractionGraph**: A weighted directed graph where nodes are users and edge weights represent normalized interaction frequency.
- **CommunityPartition**: The assignment of every user node to a community ID, produced by community detection.
- **PolarizationIndex**: A scalar in [0, 1] representing the ratio of intra-community to total interaction weight; 1 = fully polarized.
- **DiversityScore**: A scalar in [0, 1] representing the fraction of a user's or community's interactions that cross community boundaries; 1 = fully diverse.
- **Modularity**: The partition quality metric Q measuring how well communities are separated relative to a random graph.
- **Bridge_Node**: A user node with high betweenness centrality that connects multiple communities.
- **DatasetConfig**: Configuration object specifying the data source type, path or endpoint, and format.
- **Snapshot**: A versioned, persisted state of the InteractionGraph and its computed metrics at a point in time.

---

## Requirements

### Requirement 1: Data Ingestion

**User Story:** As a data engineer, I want to ingest interaction data from multiple social media sources, so that the system can analyze real-world social network behavior.

#### Acceptance Criteria

1. THE Ingestion_Layer SHALL support pluggable adapter interfaces for at least SNAP TSV edge-list files, Reddit Pushshift JSONL files, and Twitter API v2 JSON streams.
2. WHEN the Ingestion_Layer fetches records from a data source, THE Ingestion_Layer SHALL normalize each raw record into a canonical InteractionRecord with fields: id, sourceUserId, targetUserId, interactionType, timestamp, contentId, topicTags, and datasetSource.
3. WHEN normalizing records, THE Ingestion_Layer SHALL deduplicate records by the composite key (sourceUserId, targetUserId, timestamp), discarding any record whose composite key already exists in the current ingestion batch.
4. IF a live API data source returns HTTP 429 or a connection timeout, THEN THE Ingestion_Layer SHALL retry with exponential backoff (1 s, 2 s, 4 s) up to 5 attempts before marking the ingestion job as failed.
5. WHEN an ingestion job fails after all retries, THE Ingestion_Layer SHALL log the failure with the DatasetConfig details and preserve the most recent successful Snapshot as the active dataset.
6. WHEN the ingestion source returns zero valid records, THE Ingestion_Layer SHALL log a warning with the DatasetConfig and notify the scheduler to retry after a configurable backoff interval.

---

### Requirement 2: Graph Construction

**User Story:** As a data engineer, I want interaction records transformed into a weighted directed graph, so that graph algorithms can analyze the social network structure.

#### Acceptance Criteria

1. WHEN the Graph_Builder receives a non-empty list of InteractionRecords, THE Graph_Builder SHALL produce an InteractionGraph where every unique userId appearing as sourceUserId or targetUserId is represented as a node.
2. WHEN multiple InteractionRecords share the same (sourceUserId, targetUserId) pair, THE Graph_Builder SHALL aggregate them into a single directed edge whose raw weight equals the count of such records.
3. WHEN constructing an InteractionGraph, THE Graph_Builder SHALL normalize all edge weights to the range [0, 1] by dividing each raw weight by the maximum raw weight across all edges.
4. THE Graph_Builder SHALL reject any InteractionRecord where sourceUserId equals targetUserId (self-loop), treating such records as invalid and logging the rejection.
5. WHEN the Graph_Builder is called with the same list of InteractionRecords more than once, THE Graph_Builder SHALL produce an equivalent InteractionGraph on each call (idempotent construction).
6. THE Graph_Builder SHALL persist a serialized InteractionGraph as a versioned Snapshot identified by a unique snapshotId and a createdAt timestamp.
7. WHEN an existing Snapshot is provided along with new InteractionRecords, THE Graph_Builder SHALL produce an updated InteractionGraph incorporating the new records without requiring a full rebuild from scratch.

---

### Requirement 3: Community Detection

**User Story:** As an analyst, I want the interaction graph partitioned into communities, so that I can identify isolated social clusters and measure their properties.

#### Acceptance Criteria

1. WHEN the Community_Detector receives a non-empty InteractionGraph, THE Community_Detector SHALL assign every node in the graph exactly one CommunityId using the Louvain modularity-optimization algorithm.
2. WHEN the Louvain algorithm does not converge within 100 passes, THE Community_Detector SHALL stop and return the best CommunityPartition found so far, logging a warning that the partition is approximate.
3. WHEN a CommunityPartition is produced, THE Community_Detector SHALL compute the modularity score Q for the partition and include it in the CommunityPartition output.
4. THE Community_Detector SHALL ensure the modularity score Q of the produced CommunityPartition is non-negative.
5. WHEN a node has no edges in the InteractionGraph, THE Community_Detector SHALL assign that node to a singleton community containing only that node.
6. WHEN a Snapshot is updated with new records, THE Community_Detector SHALL attempt to assign stable CommunityIds that are consistent with the previous Snapshot's partition (label persistence).
7. WHERE a secondary validation algorithm is configured, THE Community_Detector SHALL also run the Girvan-Newman algorithm and store its partition alongside the Louvain partition for comparison.

---

### Requirement 4: Polarization Index Computation

**User Story:** As a researcher, I want to measure the overall polarization of the interaction graph, so that I can quantify the degree to which the social network is fragmented into ideological silos.

#### Acceptance Criteria

1. WHEN the Metrics_Service computes the PolarizationIndex for an InteractionGraph and CommunityPartition, THE Metrics_Service SHALL produce a PolarizationIndex value in the range [0, 1].
2. WHEN all edges in the InteractionGraph are intra-community (source and target in the same community), THE Metrics_Service SHALL compute a PolarizationIndex equal to 1.0.
3. WHEN all edges in the InteractionGraph are inter-community (source and target in different communities), THE Metrics_Service SHALL compute a PolarizationIndex equal to 0.0.
4. THE Metrics_Service SHALL compute the interCommunityEdgeRatio as the sum of inter-community edge weights divided by the sum of all edge weights, and this value SHALL satisfy: PolarizationIndex + interCommunityEdgeRatio = 1.0.
5. WHEN the total edge weight of the InteractionGraph is zero, THE Metrics_Service SHALL return a PolarizationIndex of 0.0 and an interCommunityEdgeRatio of 0.0.
6. WHEN metrics are computed, THE Metrics_Service SHALL persist the PolarizationMetrics record (including polarizationIndex, modularity, communityCount, avgCommunitySize, interCommunityEdgeRatio, and computedAt) to the metrics store.
7. THE Metrics_Service SHALL store PolarizationMetrics as a time series, retaining one record per Snapshot, to support trend analysis.

---

### Requirement 5: Diversity Score Computation

**User Story:** As a researcher, I want per-user and per-community diversity scores, so that I can identify individual users and communities trapped in echo chambers.

#### Acceptance Criteria

1. WHEN the Metrics_Service computes the DiversityScore for a user, THE Metrics_Service SHALL produce a value in the range [0, 1].
2. WHEN all of a user's outgoing interactions are directed to users in the same community, THE Metrics_Service SHALL compute that user's DiversityScore as 0.0.
3. WHEN all of a user's outgoing interactions are directed to users in different communities, THE Metrics_Service SHALL compute that user's DiversityScore as 1.0.
4. WHEN a user has no outgoing edges in the InteractionGraph, THE Metrics_Service SHALL compute that user's DiversityScore as 0.0.
5. THE Metrics_Service SHALL compute a community-level DiversityScore for each community as the average DiversityScore of its member users.
6. THE Metrics_Service SHALL compute betweenness centrality for every node in the InteractionGraph, normalized to [0, 1] using the standard normalization factor (n-1)(n-2), where n is the number of nodes.
7. WHEN metrics are computed, THE Metrics_Service SHALL store each user's DiversityScore, intraEdgeCount, interEdgeCount, and betweennessCentrality in the UserMetrics record associated with the current Snapshot.

---

### Requirement 6: Recommendation Engine

**User Story:** As a user with a low diversity score, I want to receive recommendations for accounts and content outside my community, so that I can be exposed to balanced viewpoints.

#### Acceptance Criteria

1. WHEN the Recommendation_Engine generates recommendations for a user, THE Recommendation_Engine SHALL return only recommendations where the recommended account belongs to a community different from the requesting user's community.
2. WHEN generating recommendations, THE Recommendation_Engine SHALL identify Bridge_Nodes (nodes with betweenness centrality above a configurable threshold) from other communities as the candidate pool.
3. WHEN scoring recommendation candidates, THE Recommendation_Engine SHALL compute a topicRelevance score using cosine similarity between the user's topic vector and the candidate's topic vector, and SHALL exclude candidates with topicRelevance below a configurable minimum threshold.
4. WHEN ranking recommendations, THE Recommendation_Engine SHALL sort candidates in descending order of estimated diversityGain.
5. THE Recommendation_Engine SHALL return at most topK recommendations per user, where topK is a configurable parameter.
6. WHEN a user has no topic vector (new user with fewer than 5 interactions), THE Recommendation_Engine SHALL use the user's community centroid topic vector as a proxy for topic matching.
7. WHEN recommendations are generated, THE Recommendation_Engine SHALL persist each Recommendation record (recommendationId, targetUserId, recommendedUserId, diversityGain, topicRelevance, communityId, reason) to the metrics store.
8. THE Recommendation_Engine SHALL provide a human-readable reason string for each recommendation explaining why the account was recommended.

---

### Requirement 7: API Layer

**User Story:** As a developer, I want a REST/GraphQL API to access graph data, metrics, and recommendations, so that I can integrate the analysis results into dashboards and external applications.

#### Acceptance Criteria

1. THE API_Layer SHALL expose an endpoint to retrieve a serialized InteractionGraph Snapshot by snapshotId.
2. THE API_Layer SHALL expose an endpoint to retrieve PolarizationMetrics for a given snapshotId.
3. THE API_Layer SHALL expose an endpoint to retrieve UserMetrics for a given userId.
4. THE API_Layer SHALL expose an endpoint to retrieve CommunityMetrics for a given communityId.
5. THE API_Layer SHALL expose an endpoint to retrieve the list of Recommendations for a given userId.
6. WHEN returning large InteractionGraph responses, THE API_Layer SHALL paginate the results.
7. WHEN the same metrics query is repeated within the caching TTL window, THE API_Layer SHALL return the cached result from Redis rather than re-querying the metrics store.
8. THE API_Layer SHALL require authentication (JWT or API key) for all endpoints.
9. WHEN a request is made to the recommendations endpoint for a userId, THE API_Layer SHALL enforce that the authenticated caller's identity matches the requested userId (per-user access control).
10. THE API_Layer SHALL enforce rate limiting on all endpoints for external consumers.
11. WHEN filtering is requested, THE API_Layer SHALL support filtering metric and graph responses by date range, community ID, and metric threshold.

---

### Requirement 8: Visualization Dashboard

**User Story:** As a researcher or end user, I want an interactive visualization dashboard, so that I can explore community structure, polarization trends, and my personalized recommendations.

#### Acceptance Criteria

1. WHEN a user opens the Dashboard, THE Dashboard SHALL display an interactive graph visualization showing user nodes colored by community membership.
2. THE Dashboard SHALL display the current PolarizationIndex and modularity score as prominent metrics.
3. THE Dashboard SHALL display a time-series chart of PolarizationIndex across historical Snapshots.
4. WHEN a user selects a node in the graph visualization, THE Dashboard SHALL display that user's DiversityScore, community membership, and list of recommendations.
5. THE Dashboard SHALL display diversity score distribution across all users (e.g., histogram or heatmap).
6. WHEN new Snapshot data is available, THE Dashboard SHALL refresh the graph and metrics without requiring a full page reload.

---

### Requirement 9: Data Validation and Integrity

**User Story:** As a data engineer, I want all input data to be validated before processing, so that invalid records do not corrupt the graph or produce meaningless metrics.

#### Acceptance Criteria

1. THE Ingestion_Layer SHALL reject any InteractionRecord where sourceUserId or targetUserId is empty.
2. THE Ingestion_Layer SHALL reject any InteractionRecord where the interactionType is not one of the recognized enum values: RETWEET, REPLY, MENTION, UPVOTE, COMMENT.
3. THE Ingestion_Layer SHALL reject any InteractionRecord where the timestamp is not a valid past datetime.
4. THE Graph_Builder SHALL reject any InteractionRecord where an edge weight would be negative after aggregation.
5. WHEN an InteractionRecord is rejected, THE Ingestion_Layer SHALL log the rejection reason and the offending record identifier without halting the overall ingestion job.

---

### Requirement 10: Security and Compliance

**User Story:** As a system administrator, I want user data to be anonymized and API access to be secured, so that the system complies with data protection requirements and API terms of service.

#### Acceptance Criteria

1. THE Ingestion_Layer SHALL anonymize or hash all personally identifiable user identifiers (real usernames, email addresses) before storing records in the raw interaction store, using internal UUIDs as node identifiers.
2. THE API_Layer SHALL enforce authentication using JWT tokens or API keys on all endpoints.
3. WHEN raw interaction data is stored, THE system SHALL not store raw tweet or post text content that would violate the data source's Terms of Service.
4. THE Ingestion_Layer SHALL enforce upstream API rate limits for live data sources, not exceeding the rate limits specified in the respective API's terms.

---

### Requirement 11: Parser and Serialization Round-Trip

**User Story:** As a data engineer, I want graph snapshots to be serialized and deserialized losslessly, so that persisted graphs can be reloaded accurately for incremental updates and historical analysis.

#### Acceptance Criteria

1. WHEN the Graph_Builder serializes an InteractionGraph to a Snapshot, THE Graph_Builder SHALL encode the graph in GraphML or adjacency-list format.
2. WHEN the Graph_Builder deserializes a Snapshot, THE Graph_Builder SHALL produce an InteractionGraph equivalent to the one that was serialized (lossless round-trip).
3. FOR ALL valid InteractionGraph objects, serializing then deserializing SHALL produce an equivalent graph with the same node set, edge set, and edge weights (round-trip property).
4. THE Graph_Builder SHALL expose a pretty-print format for InteractionGraph objects suitable for human inspection and debugging.
