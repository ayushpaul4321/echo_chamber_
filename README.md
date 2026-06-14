# Echo Chamber Detector

A data pipeline and analysis system that ingests social network interaction data, constructs weighted directed graphs, detects communities, measures polarization, and surfaces cross-community recommendations — all visualized in an interactive dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Dashboard](#dashboard)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Results Summary](#results-summary)
- [Tech Stack](#tech-stack)

---

## Overview

Echo chambers are social network clusters where users interact predominantly with ideologically similar peers. This system:

1. **Ingests** four public datasets (Reddit, Congress Twitter network, Wikipedia RfA votes)
2. **Builds** weighted directed interaction graphs
3. **Detects communities** using the Louvain modularity algorithm
4. **Computes** a Polarization Index \[0–1\] and per-user Diversity Scores
5. **Generates** cross-community recommendations to reduce echo chambers
6. **Visualizes** results in a React + Sigma.js dashboard
7. **Exposes** all data via a authenticated REST API

---

## Architecture

```
Data Sources → Ingestion Layer → Graph Construction → Community Detection
                                                              ↓
Dashboard ← REST API ← Metrics Store ← Metrics & Analysis ←─┘
                              ↓
                    Recommendation Engine
```

| Layer | Technology |
|---|---|
| Ingestion | Python adapters (pandas, gzip streaming) |
| Graph Construction | NetworkX, GraphML serialization |
| Community Detection | python-louvain / NetworkX greedy modularity |
| Metrics | Custom Polarization Index, Diversity Score, Betweenness |
| API | FastAPI + Uvicorn |
| Storage | PostgreSQL (metrics), Redis (cache), Neo4j (graph mirror) |
| Dashboard | React 18, Sigma.js v3, Recharts |
| Testing | pytest + Hypothesis (property-based testing) |

---

## Datasets

| Dataset | File | Role | Size |
|---|---|---|---|
| Reddit Hyperlinks (Title) | `soc-redditHyperlinks-title.tsv` | Primary subreddit interaction graph | ~50 MB |
| Reddit Hyperlinks (Body) | `soc-redditHyperlinks-body.tsv` | Topic embedding source (TF-IDF) | ~50 MB |
| Congress Network | `congress_network/congress.edgelist` | Political polarization benchmark | ~1 MB |
| Wikipedia RfA | `wiki-RfA.txt.gz` | Signed voting graph (+1/−1) | ~6 MB |

Place all dataset files under `echo_chamber_detector/`.

---

## Project Structure

```
echo_chamber_detector/        # Raw dataset files
ingestion/
  adapters.py                 # RedditTitleAdapter, RedditBodyAdapter,
  │                           # CongressNetworkAdapter, WikiRfAAdapter
  service.py                  # IngestionService (dedup, validation)
  validation.py               # Input record validation rules
graph/
  models.py                   # InteractionRecord, Node, Edge, InteractionGraph, …
  service.py                  # GraphConstructionService (build, update, serialize)
  db_models.py                # SQLAlchemy ORM models
  redis_keys.py               # Redis key conventions
community/
  service.py                  # CommunityDetectionService (Louvain + Girvan-Newman)
metrics/
  service.py                  # MetricsService (Polarization Index, Diversity Score,
  │                           # Betweenness Centrality, Signed Metrics)
recommendations/
  bridge_nodes.py             # Bridge node identification + recommendation generation
  topic_vectors.py            # TF-IDF / sentence embedding vectorization
  service.py                  # RecommendationService (persist + fetch)
api/
  app.py                      # FastAPI application entry point
  router.py                   # All REST endpoints
  auth.py                     # JWT + API key authentication
  rate_limit.py               # Sliding-window rate limiter
  cache.py                    # Redis cache helpers
  dtos.py                     # Pydantic response models
dashboard/
  src/
    App.tsx                   # Main app (snapshot loader, live refresh)
    components/
      SigmaGraph.tsx          # Interactive graph (community colors, polarity edges)
      MetricsPanel.tsx        # Metric cards
      PolarizationChart.tsx   # Time-series chart
      DiversityHistogram.tsx  # Diversity score distribution
      DatasetSelector.tsx     # Reddit / Congress / Wiki-RfA toggle
    api/client.ts             # Typed API client
    hooks/useSnapshotPoller.ts # Live 60s refresh hook
tests/
  test_pipeline_reddit_title.py   # Full Reddit title pipeline test
  test_pipeline_reddit_body.py    # Reddit body + topic vectors test
  test_pipeline_congress.py       # Congress network pipeline test
  test_pipeline_wiki_rfa.py       # Wiki-RfA signed graph pipeline test
  test_adapters.py                # Unit tests for all four adapters
  test_graph_service.py           # Graph construction unit tests
  test_community_service.py       # Community detection unit tests
  test_metrics_service.py         # Metrics unit + property tests
  test_recommendations.py         # Recommendation engine tests
  test_api_pagination_cache_rate_limit.py  # API layer tests
alembic/                      # Database migration scripts
data/snapshots/               # Persisted GraphML snapshots
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL (optional — SQLite used as fallback)
- Redis (optional — in-process fallback used when unavailable)

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install dashboard dependencies

```bash
cd dashboard
npm install
```

### 3. Configure environment (optional)

```bash
cp dashboard/.env.example dashboard/.env
# Edit dashboard/.env if your API runs on a different port
```

---

## Running the Full Pipeline

Generate a graph snapshot from any dataset. Run these from the project root (`D:\webminig`):

### Congress Network (fastest, ~5 seconds)

```bash
python -c "
from ingestion.adapters import CongressNetworkAdapter, DatasetConfig
from ingestion.service import IngestionService
from graph.service import GraphConstructionService
from community.service import CommunityDetectionService

adapter = CongressNetworkAdapter()
svc = IngestionService()
result = svc.ingest(
    adapter, 'echo_chamber_detector/congress_network/congress.edgelist',
    config=DatasetConfig(
        source_type='congress',
        file_path='echo_chamber_detector/congress_network/congress.edgelist',
        format='edgelist',
        extra={'json_path': 'echo_chamber_detector/congress_network/congress_network_data.json'}
    )
)
graph_svc = GraphConstructionService()
graph = graph_svc.build_graph(result.records, dataset_source='congress')
comm_svc = CommunityDetectionService()
comm_svc.detect_communities(graph)
graph_svc.persist_graph(graph, graph.snapshotId)
print('SNAPSHOT ID:', graph.snapshotId)
"
```

### Wiki-RfA Signed Graph (~6 minutes)

```bash
python -c "
from ingestion.adapters import WikiRfAAdapter, DatasetConfig
from ingestion.service import IngestionService
from graph.service import GraphConstructionService
from community.service import CommunityDetectionService

adapter = WikiRfAAdapter()
svc = IngestionService()
result = svc.ingest(adapter, 'echo_chamber_detector/wiki-RfA.txt.gz')
graph_svc = GraphConstructionService()
graph = graph_svc.build_graph(result.records, dataset_source='wiki_rfa')
comm_svc = CommunityDetectionService()
comm_svc.detect_communities(graph)
graph_svc.persist_graph(graph, graph.snapshotId)
print('SNAPSHOT ID:', graph.snapshotId)
"
```

The snapshot ID printed at the end is what you paste into the dashboard.

---

## Dashboard

You need two terminals running simultaneously.

### Terminal 1 — Backend API

```bash
# From the project root (D:\webminig)
python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### Terminal 2 — Frontend

```bash
cd dashboard
npm run dev
```

Dashboard at: http://localhost:5173

### Authenticate

Generate a dev token and set it in the browser:

```bash
python -c "from api.auth import encode_jwt; print(encode_jwt({'sub': 'demo-user'}))"
```

Open http://localhost:5173, press **F12 → Console**, and paste:

```js
localStorage.setItem('echo_chamber_token', '<your token here>')
```

Refresh, enter your snapshot ID, and click **Load Graph**.

### Dashboard Features

| Feature | Description |
|---|---|
| Graph visualization | Nodes colored by community (13 distinct colors), sized by betweenness centrality |
| Edge polarity | Green = positive vote (+1), Red = negative vote (−1) for Wiki-RfA |
| Metric cards | Polarization Index, Modularity Q, community count, diversity score |
| Time-series chart | Polarization Index trend across snapshots per dataset |
| Diversity histogram | Score distribution across all users |
| Node click sidebar | Community, diversity score, betweenness, top recommendations |
| Live refresh | Polls for new snapshots every 60 seconds |
| Dataset toggle | Switch between Reddit / Congress / Wiki-RfA views |

---

## API Reference

All endpoints require `Authorization: Bearer <token>`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/snapshots/{id}/graph` | Paginated graph (500 nodes/page) |
| GET | `/api/snapshots/{id}/metrics/polarization` | Polarization metrics (cached 24h) |
| GET | `/api/snapshots/{id}/metrics/signed` | Signed-edge metrics (Wiki-RfA only) |
| GET | `/api/metrics/polarization` | Filtered list of all polarization metrics |
| GET | `/api/users/{userId}/metrics` | Per-user diversity + betweenness |
| GET | `/api/users/metrics` | Paginated list of user metrics |
| GET | `/api/communities/{id}/metrics` | Aggregated community metrics |
| GET | `/api/users/{userId}/recommendations` | Cross-community recommendations |
| GET | `/api/snapshots/latest` | Most recent snapshot ID for a dataset |
| GET | `/health` | Liveness probe |

Filter parameters: `?datasetSource=`, `?from=`, `?to=`, `?communityId=`, `?min_polarization=`

---

## Running Tests

```bash
# All unit tests
python -m pytest tests/ -v

# Individual pipeline integration tests
python -m pytest tests/test_pipeline_congress.py -v -s
python -m pytest tests/test_pipeline_wiki_rfa.py -v -s   # takes ~6 min
python -m pytest tests/test_pipeline_reddit_title.py -v -s

# Property-based tests (Hypothesis)
python -m pytest tests/test_graph_service.py tests/test_metrics_service.py -v
```

---

## Results Summary

| Dataset | Nodes | Edges | Communities | Polarization Index | Notes |
|---|---|---|---|---|---|
| Reddit Title | ~6K | ~13K | multiple | > 0.60 | Subreddit echo chambers |
| Congress | 475 | 13,289 | 4 | 0.79–0.91 | Democrat/Republican split |
| Wiki-RfA | 11,256 | 177,211 | 13 | 0.85 | Signed votes (+1/−1) |

**Key finding (Wiki-RfA):** Negative votes cluster predominantly *within* communities (intra-community negativity ≈ 84%). Louvain groups editors with similar voting patterns, so opposition votes stay inside communities rather than crossing boundaries.

---

## Tech Stack

**Backend**
- Python 3.11
- FastAPI 0.111 + Uvicorn 0.30
- NetworkX 3.3 (graph algorithms)
- python-louvain 0.16 (Louvain community detection)
- scikit-learn 1.5 (TF-IDF embeddings)
- SQLAlchemy + Alembic (PostgreSQL ORM + migrations)
- Redis 5.0 (caching)
- Neo4j 5.20 (graph database mirror)
- Hypothesis 6.103 (property-based testing)

**Frontend**
- React 18 + TypeScript
- Sigma.js v3 + Graphology (graph rendering)
- Recharts (time-series + histogram)
- Vite 5

---

## License

MIT
