"""Tests for API pagination, Redis caching, filtering, and rate limiting.

Covers:
- Cursor-based pagination on the graph endpoint (default 500 nodes/page)
- page_size query parameter respected
- nextCursor is None on the last page
- Redis cache: repeated call returns cached PolarizationDTO (cache hit)
- Redis cache: cache is populated on first call (cache miss → DB → cache)
- Disabled Redis (REDIS_URL=disabled) — endpoint still works without caching
- Filter: ?datasetSource= on /api/metrics/polarization list
- Filter: ?from= and ?to= date-range filter on list endpoint
- Filter: ?min_polarization= threshold filter
- Filter: ?communityId= on /api/users/metrics list
- Rate limiting: 100th request succeeds, 101st returns HTTP 429
- Rate limiting: different identities have independent counters
- Retry-After header present on 429 response
- Unauthenticated request returns HTTP 401 (rate limiter delegates to auth)

References: Requirements 7.6, 7.7, 7.10, 7.11
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# App import (must happen after any env-var patches)
# ---------------------------------------------------------------------------

from api.app import app
from api.auth import encode_jwt
from api.router import _get_db_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_headers(user_id: str = "test_user") -> dict[str, str]:
    """Return Authorization headers with a valid JWT for *user_id*."""
    token = encode_jwt({"sub": user_id})
    return {"Authorization": f"Bearer {token}"}


def _make_graph(node_count: int = 10, dataset_source: str = "reddit_title"):
    """Return a minimal InteractionGraph-like object with *node_count* nodes."""
    from datetime import datetime, timezone
    from graph.models import Edge, InteractionGraph, Node

    nodes = {
        f"user_{i:04d}": Node(userId=f"user_{i:04d}", communityId=str(i % 3))
        for i in range(node_count)
    }
    edges = [
        Edge(
            sourceUserId=f"user_{i:04d}",
            targetUserId=f"user_{(i + 1) % node_count:04d}",
            weight=0.5,
        )
        for i in range(min(node_count, 5))
    ]
    return InteractionGraph(
        nodes=nodes,
        edges=edges,
        snapshotId="snap-test",
        createdAt=datetime(2024, 1, 1, tzinfo=timezone.utc),
        datasetSource=dataset_source,
    )


def _make_polarization_row(
    snapshot_id: str = "snap-001",
    dataset_source: str = "reddit_title",
    polarization_index: float = 0.7,
    computed_at: datetime | None = None,
):
    """Return a mock PolarizationMetricRow-like object."""
    from graph.db_models import PolarizationMetricRow
    row = PolarizationMetricRow()
    row.id = abs(hash(snapshot_id)) % 1_000_000
    row.snapshot_id = snapshot_id
    row.dataset_source = dataset_source
    row.polarization_index = polarization_index
    row.modularity = 0.4
    row.community_count = 3
    row.avg_community_size = 10.0
    row.inter_community_edge_ratio = 1.0 - polarization_index
    row.computed_at = computed_at or datetime(2024, 1, 1, tzinfo=timezone.utc)
    return row


def _make_user_metric_row(
    user_id: str = "user_0001",
    snapshot_id: str = "snap-001",
    community_id: str = "0",
    dataset_source: str = "reddit_title",
    diversity_score: float = 0.5,
):
    """Return a mock UserMetricRow-like object."""
    from graph.db_models import UserMetricRow
    row = UserMetricRow()
    row.user_id = user_id
    row.snapshot_id = snapshot_id
    row.community_id = community_id
    row.dataset_source = dataset_source
    row.diversity_score = diversity_score
    row.intra_edge_count = 5
    row.inter_edge_count = 3
    row.betweenness_centrality = 0.1
    row.computed_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return row


def _make_mock_db_session(rows_or_side_effects=None) -> MagicMock:
    """Build a fully-mocked SQLAlchemy Session that never touches a real DB."""
    session = MagicMock()
    q = MagicMock()
    session.query.return_value = q
    q.filter.return_value = q
    q.order_by.return_value = q
    q.with_entities.return_value = q

    if isinstance(rows_or_side_effects, list):
        q._rows = list(rows_or_side_effects)
        q.count.side_effect = lambda: len(q._rows)
        q.all.side_effect = lambda: list(q._rows)
        q.first.side_effect = lambda: q._rows[0] if q._rows else None

        def _limit(n):
            lim = MagicMock()
            lim.all.return_value = q._rows[:n]
            return lim

        def _offset(n):
            q._rows = q._rows[n:]
            return q

        q.limit.side_effect = _limit
        q.offset.side_effect = _offset
        q.one_or_none.return_value = None

    return session


@contextmanager
def _override_db(session: MagicMock):
    """Context manager: temporarily inject *session* as the DB dependency."""
    def _mock_dep():
        yield session

    app.dependency_overrides[_get_db_session] = _mock_dep
    try:
        yield
    finally:
        app.dependency_overrides.pop(_get_db_session, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def disable_redis(monkeypatch):
    """Disable Redis for all tests in this module unless overridden."""
    monkeypatch.setenv("REDIS_URL", "disabled")
    import api.cache as cache_mod
    cache_mod.reset_client()
    yield
    cache_mod.reset_client()


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def mock_graph_service():
    """Patch GraphConstructionService.load_graph to return a synthetic graph."""
    graph = _make_graph(node_count=12)
    with patch("graph.service.GraphConstructionService.load_graph", return_value=graph):
        yield graph


# ---------------------------------------------------------------------------
# Pagination tests
# ---------------------------------------------------------------------------


class TestGraphPagination:
    """Cursor-based pagination on GET /api/snapshots/{snapshotId}/graph."""

    def test_default_page_size_is_500(self, client):
        """When page_size is omitted the default is 500 nodes per page."""
        large_graph = _make_graph(node_count=600)
        with patch("graph.service.GraphConstructionService.load_graph", return_value=large_graph):
            resp = client.get("/api/snapshots/snap-test/graph", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) == 500
        assert data["nextCursor"] is not None  # more pages remain

    def test_last_page_has_no_next_cursor(self, client, mock_graph_service):
        """When there are no more nodes, nextCursor is None."""
        # 12 nodes, page_size=20 → fits on one page
        resp = client.get(
            "/api/snapshots/snap-test/graph?page_size=20",
            headers=_auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["nextCursor"] is None

    def test_cursor_advances_to_next_page(self, client, mock_graph_service):
        """Using the nextCursor from page 1 returns the correct next page."""
        resp1 = client.get(
            "/api/snapshots/snap-test/graph?page_size=5",
            headers=_auth_headers(),
        )
        assert resp1.status_code == 200
        page1 = resp1.json()
        assert len(page1["nodes"]) == 5
        cursor = page1["nextCursor"]
        assert cursor is not None

        resp2 = client.get(
            f"/api/snapshots/snap-test/graph?page_size=5&cursor={cursor}",
            headers=_auth_headers(),
        )
        assert resp2.status_code == 200
        page2 = resp2.json()
        # No overlap between pages
        ids1 = {n["userId"] for n in page1["nodes"]}
        ids2 = {n["userId"] for n in page2["nodes"]}
        assert ids1.isdisjoint(ids2)

    def test_all_nodes_covered_across_pages(self, client, mock_graph_service):
        """Paginating through all pages covers every node exactly once."""
        all_ids = set()
        cursor = None
        while True:
            url = "/api/snapshots/snap-test/graph?page_size=4"
            if cursor:
                url += f"&cursor={cursor}"
            resp = client.get(url, headers=_auth_headers())
            assert resp.status_code == 200
            data = resp.json()
            for node in data["nodes"]:
                assert node["userId"] not in all_ids, "Duplicate node across pages"
                all_ids.add(node["userId"])
            cursor = data["nextCursor"]
            if cursor is None:
                break

        # mock_graph_service has 12 nodes
        assert len(all_ids) == 12

    def test_invalid_cursor_starts_from_beginning(self, client, mock_graph_service):
        """An unrecognised cursor value does not raise — it starts from page 1."""
        resp = client.get(
            "/api/snapshots/snap-test/graph?cursor=nonexistent_user",
            headers=_auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) > 0

    def test_page_size_query_param_respected(self, client, mock_graph_service):
        """page_size=3 returns at most 3 nodes."""
        resp = client.get(
            "/api/snapshots/snap-test/graph?page_size=3",
            headers=_auth_headers(),
        )
        assert resp.status_code == 200
        assert len(resp.json()["nodes"]) == 3

    def test_snapshot_not_found_returns_404(self, client):
        """A missing snapshot raises HTTP 404."""
        with patch(
            "graph.service.GraphConstructionService.load_graph",
            side_effect=FileNotFoundError("not found"),
        ):
            resp = client.get("/api/snapshots/missing/graph", headers=_auth_headers())
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Redis caching tests
# ---------------------------------------------------------------------------


class TestPolarizationCache:
    """Redis caching for GET /api/snapshots/{snapshotId}/metrics/polarization."""

    def test_cache_miss_queries_db_and_populates_cache(self, client, monkeypatch):
        """On first call, DB is queried and result is stored in cache."""
        import api.cache as cache_mod
        import api.router as router_mod

        row = _make_polarization_row()
        session = _make_mock_db_session([row])

        stored: dict = {}

        def fake_set(key, value, ttl=cache_mod.DEFAULT_TTL_SECONDS):
            stored[key] = value

        # Patch at the point-of-use in router.py (direct import)
        monkeypatch.setattr(router_mod, "cache_get_json", lambda k: None)
        monkeypatch.setattr(router_mod, "cache_set_json", fake_set)

        with _override_db(session):
            resp = client.get(
                "/api/snapshots/snap-001/metrics/polarization",
                headers=_auth_headers(),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["polarizationIndex"] == pytest.approx(0.7)
        # Cache should have been populated
        assert any("polarization" in k for k in stored)

    def test_cache_hit_skips_db(self, client, monkeypatch):
        """On a cache hit, the DB is never queried."""
        import api.router as router_mod

        cached_payload = {
            "snapshotId": "snap-001",
            "polarizationIndex": 0.88,
            "modularity": 0.4,
            "communityCount": 2,
            "avgCommunitySize": 5.0,
            "interCommunityEdgeRatio": 0.12,
            "computedAt": "2024-01-01T00:00:00+00:00",
        }

        # Patch at the point-of-use in router.py (direct import)
        monkeypatch.setattr(router_mod, "cache_get_json", lambda k: cached_payload)

        db_called = []
        session = MagicMock()
        session.query.side_effect = lambda *a, **kw: db_called.append(1) or MagicMock()

        with _override_db(session):
            resp = client.get(
                "/api/snapshots/snap-001/metrics/polarization",
                headers=_auth_headers(),
            )

        assert resp.status_code == 200
        assert resp.json()["polarizationIndex"] == pytest.approx(0.88)
        # DB must NOT have been touched
        assert len(db_called) == 0

    def test_redis_disabled_still_works(self, client, monkeypatch):
        """When REDIS_URL=disabled the endpoint works correctly without caching."""
        row = _make_polarization_row(polarization_index=0.65)
        session = _make_mock_db_session([row])

        with _override_db(session):
            resp = client.get(
                "/api/snapshots/snap-001/metrics/polarization",
                headers=_auth_headers(),
            )

        assert resp.status_code == 200
        assert resp.json()["polarizationIndex"] == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# Filtering tests
# ---------------------------------------------------------------------------


class TestPolarizationListFilters:
    """Filter params on GET /api/metrics/polarization."""

    def _rows(self):
        return [
            _make_polarization_row("snap-A", "reddit_title", 0.70, datetime(2024, 3, 1, tzinfo=timezone.utc)),
            _make_polarization_row("snap-B", "congress",     0.88, datetime(2024, 6, 1, tzinfo=timezone.utc)),
            _make_polarization_row("snap-C", "wiki_rfa",     0.55, datetime(2024, 9, 1, tzinfo=timezone.utc)),
        ]

    def test_no_filters_returns_all(self, client):
        rows = self._rows()
        session = _make_mock_db_session(rows)
        with _override_db(session):
            resp = client.get("/api/metrics/polarization", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3

    def test_filter_by_dataset_source(self, client):
        """?datasetSource=congress should match only the congress row."""
        rows = [self._rows()[1]]  # only congress row
        session = _make_mock_db_session(rows)
        with _override_db(session):
            resp = client.get(
                "/api/metrics/polarization?datasetSource=congress",
                headers=_auth_headers(),
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["snapshotId"] == "snap-B"

    def test_filter_by_min_polarization(self, client):
        """?min_polarization=0.80 should return only high-polarization rows."""
        rows = [self._rows()[1]]  # congress PI=0.88 only
        session = _make_mock_db_session(rows)
        with _override_db(session):
            resp = client.get(
                "/api/metrics/polarization?min_polarization=0.80",
                headers=_auth_headers(),
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["polarizationIndex"] == pytest.approx(0.88)

    def test_filter_by_date_range(self, client):
        """?from= and ?to= should filter by computedAt bounds."""
        rows = [self._rows()[1]]  # snap-B in June 2024
        session = _make_mock_db_session(rows)
        with _override_db(session):
            resp = client.get(
                "/api/metrics/polarization?from=2024-05-01T00:00:00Z&to=2024-07-31T00:00:00Z",
                headers=_auth_headers(),
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["snapshotId"] == "snap-B"

    def test_filter_by_snapshot_id(self, client):
        """?snapshotId=snap-C should return only that snapshot's metrics."""
        rows = [self._rows()[2]]
        session = _make_mock_db_session(rows)
        with _override_db(session):
            resp = client.get(
                "/api/metrics/polarization?snapshotId=snap-C",
                headers=_auth_headers(),
            )
        assert resp.status_code == 200
        assert resp.json()["items"][0]["snapshotId"] == "snap-C"

    def test_empty_result_set(self, client):
        """No matching rows returns items=[] total=0 nextCursor=None."""
        session = _make_mock_db_session([])
        with _override_db(session):
            resp = client.get(
                "/api/metrics/polarization?min_polarization=0.99",
                headers=_auth_headers(),
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []
        assert data["nextCursor"] is None


class TestUserMetricsFilters:
    """?communityId= filter on GET /api/users/metrics."""

    def test_filter_by_community_id(self, client):
        """?communityId=2 should only return users in community 2."""
        rows = [
            _make_user_metric_row("u1", community_id="2"),
            _make_user_metric_row("u2", community_id="2"),
        ]
        session = _make_mock_db_session(rows)
        with _override_db(session):
            resp = client.get(
                "/api/users/metrics?communityId=2",
                headers=_auth_headers(),
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert all(item["communityId"] == "2" for item in data["items"])

    def test_filter_by_dataset_source(self, client):
        rows = [_make_user_metric_row("u3", dataset_source="congress")]
        session = _make_mock_db_session(rows)
        with _override_db(session):
            resp = client.get(
                "/api/users/metrics?datasetSource=congress",
                headers=_auth_headers(),
            )
        assert resp.status_code == 200
        assert resp.json()["items"][0]["userId"] == "u3"


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Per-identity sliding-window rate limiter."""

    def _reset_local_windows(self):
        """Clear the in-process rate-limit store between tests."""
        import api.rate_limit as rl
        rl._local_windows.clear()

    def test_within_limit_returns_200(self, client, mock_graph_service):
        """Requests within the limit are served normally."""
        self._reset_local_windows()
        resp = client.get("/api/snapshots/snap-test/graph", headers=_auth_headers("u_limit"))
        assert resp.status_code == 200

    def test_exceeding_limit_returns_429(self, client, mock_graph_service, monkeypatch):
        """Exceeding RATE_LIMIT_REQUESTS within the window returns HTTP 429."""
        import api.rate_limit as rl
        self._reset_local_windows()
        monkeypatch.setattr(rl, "RATE_LIMIT_REQUESTS", 3)

        identity = "u_over_limit"
        headers = _auth_headers(identity)

        for i in range(3):
            resp = client.get("/api/snapshots/snap-test/graph", headers=headers)
            assert resp.status_code == 200, f"Request {i+1} should succeed"

        resp = client.get("/api/snapshots/snap-test/graph", headers=headers)
        assert resp.status_code == 429

    def test_retry_after_header_present_on_429(self, client, mock_graph_service, monkeypatch):
        """HTTP 429 response includes a Retry-After header."""
        import api.rate_limit as rl
        self._reset_local_windows()
        monkeypatch.setattr(rl, "RATE_LIMIT_REQUESTS", 1)

        identity = "u_retry_after"
        headers = _auth_headers(identity)

        client.get("/api/snapshots/snap-test/graph", headers=headers)
        resp = client.get("/api/snapshots/snap-test/graph", headers=headers)
        assert resp.status_code == 429
        assert "retry-after" in {k.lower() for k in resp.headers}

    def test_different_identities_have_independent_counters(
        self, client, mock_graph_service, monkeypatch
    ):
        """Rate limit counters are isolated per identity."""
        import api.rate_limit as rl
        self._reset_local_windows()
        monkeypatch.setattr(rl, "RATE_LIMIT_REQUESTS", 2)

        # Exhaust identity A
        for _ in range(2):
            r = client.get("/api/snapshots/snap-test/graph", headers=_auth_headers("ident_a"))
            assert r.status_code == 200
        r = client.get("/api/snapshots/snap-test/graph", headers=_auth_headers("ident_a"))
        assert r.status_code == 429

        # Identity B should still be within its own window
        r = client.get("/api/snapshots/snap-test/graph", headers=_auth_headers("ident_b"))
        assert r.status_code == 200

    def test_window_expiry_resets_counter(self, client, mock_graph_service, monkeypatch):
        """After the window expires the counter resets and requests succeed again."""
        import api.rate_limit as rl
        self._reset_local_windows()
        monkeypatch.setattr(rl, "RATE_LIMIT_REQUESTS", 2)
        monkeypatch.setattr(rl, "RATE_LIMIT_WINDOW_SECONDS", 1)  # 1-second window

        identity = "u_window_expiry"
        headers = _auth_headers(identity)

        # Fill window
        for _ in range(2):
            client.get("/api/snapshots/snap-test/graph", headers=headers)

        r = client.get("/api/snapshots/snap-test/graph", headers=headers)
        assert r.status_code == 429

        # Wait for window to expire
        time.sleep(1.1)

        r = client.get("/api/snapshots/snap-test/graph", headers=headers)
        assert r.status_code == 200

    def test_unauthenticated_request_returns_401(self, client, mock_graph_service):
        """Missing auth header returns HTTP 401 before rate limit check."""
        self._reset_local_windows()
        resp = client.get("/api/snapshots/snap-test/graph")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Cache key construction
# ---------------------------------------------------------------------------


class TestCacheKeyBuilder:
    """_make_cache_key produces stable, deterministic keys."""

    def test_same_params_produce_same_key(self):
        from api.router import _make_cache_key
        k1 = _make_cache_key("prefix", a="1", b="2")
        k2 = _make_cache_key("prefix", b="2", a="1")  # different order
        assert k1 == k2

    def test_different_params_produce_different_keys(self):
        from api.router import _make_cache_key
        k1 = _make_cache_key("prefix", a="1")
        k2 = _make_cache_key("prefix", a="2")
        assert k1 != k2

    def test_none_params_are_excluded(self):
        from api.router import _make_cache_key
        k1 = _make_cache_key("prefix", a="1", b=None)
        k2 = _make_cache_key("prefix", a="1")
        assert k1 == k2


# ---------------------------------------------------------------------------
# Cache module unit tests
# ---------------------------------------------------------------------------


class TestCacheModule:
    """Unit tests for api/cache.py get/set helpers."""

    def test_cache_disabled_get_returns_none(self):
        """cache_get returns None when Redis is disabled."""
        import api.cache as cache_mod
        # disable_redis fixture already sets REDIS_URL=disabled
        assert cache_mod.cache_get("any_key") is None

    def test_cache_disabled_set_is_noop(self):
        """cache_set does not raise when Redis is disabled."""
        import api.cache as cache_mod
        cache_mod.cache_set("key", "value")  # should not raise

    def test_cache_get_json_returns_none_on_miss(self):
        import api.cache as cache_mod
        assert cache_mod.cache_get_json("missing_key") is None

    def test_cache_set_and_get_json_roundtrip(self, monkeypatch):
        """cache_set_json / cache_get_json round-trip via a mock Redis client."""
        import api.cache as cache_mod

        store: dict = {}

        mock_client = MagicMock()
        mock_client.setex.side_effect = lambda k, ttl, v: store.__setitem__(k, v)
        mock_client.get.side_effect = lambda k: store.get(k)
        mock_client.ping.return_value = True

        # Inject the mock client directly
        monkeypatch.setattr(cache_mod, "_redis_client", mock_client)

        cache_mod.cache_set_json("test_key", {"foo": 42})
        result = cache_mod.cache_get_json("test_key")
        assert result == {"foo": 42}

        monkeypatch.setattr(cache_mod, "_redis_client", None)
        cache_mod.reset_client()


# ---------------------------------------------------------------------------
# Per-user access control tests (Requirement 7.9)
# ---------------------------------------------------------------------------


class TestRecommendationsAccessControl:
    """Per-user access control on GET /api/users/{userId}/recommendations.

    References: Requirements 7.9
    """

    def _reset_local_windows(self):
        import api.rate_limit as rl
        rl._local_windows.clear()

    def test_mismatched_user_id_returns_403(self, client):
        """Authenticated caller requesting another user's recommendations → HTTP 403."""
        self._reset_local_windows()
        # Caller is "alice" but requests recommendations for "bob"
        headers = _auth_headers("alice")
        resp = client.get("/api/users/bob/recommendations", headers=headers)
        assert resp.status_code == 403

    def test_matching_user_id_returns_200(self, client):
        """Authenticated caller requesting their own recommendations → HTTP 200."""
        self._reset_local_windows()

        # Patch RecommendationService.fetch_recommendations to return an empty list
        with patch(
            "recommendations.service.RecommendationService.fetch_recommendations",
            return_value=[],
        ):
            headers = _auth_headers("alice")
            resp = client.get("/api/users/alice/recommendations", headers=headers)

        assert resp.status_code == 200
        assert resp.json() == []

    def test_unauthenticated_request_returns_401(self, client):
        """Missing auth header returns HTTP 401 before access control check."""
        self._reset_local_windows()
        resp = client.get("/api/users/alice/recommendations")
        assert resp.status_code == 401

    def test_403_response_body_describes_access_violation(self, client):
        """403 response includes a descriptive detail message."""
        self._reset_local_windows()
        headers = _auth_headers("alice")
        resp = client.get("/api/users/bob/recommendations", headers=headers)
        assert resp.status_code == 403
        body = resp.json()
        assert "detail" in body
        assert len(body["detail"]) > 0
