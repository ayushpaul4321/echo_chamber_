"""Tests for MetricsService.persist_metrics and MetricsService.query_metrics.

Covers Requirements 4.6, 4.7, 5.7:
  4.6  Persist PolarizationMetrics per snapshot per datasetSource.
  4.7  Store PolarizationMetrics as a time series (one record per Snapshot).
  5.7  Store UserMetrics per user per snapshot.

Also covers MetricsFilter querying by:
  - snapshotId
  - datasetSource
  - date range (from_date / to_date)
  - communityId
  - min_polarization threshold

Uses an in-memory SQLite database so no external infrastructure is needed.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from graph.db_models import Base, PolarizationMetricRow, UserMetricRow
from graph.models import PolarizationMetrics, UserMetrics
from metrics.service import MetricsFilter, MetricsService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine():
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture()
def db_session(engine):
    """Provide a transactional session that rolls back after each test."""
    with Session(engine) as session:
        yield session
        session.rollback()


@pytest.fixture()
def svc() -> MetricsService:
    return MetricsService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_T2 = datetime(2024, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
_T3 = datetime(2024, 3, 1, 0, 0, 0, tzinfo=timezone.utc)


def _pm(
    snapshot_id: str | None = None,
    dataset_source: str = "reddit_title",
    polarization_index: float = 0.7,
    modularity: float = 0.4,
    community_count: int = 3,
    avg_community_size: float = 10.0,
    inter_community_edge_ratio: float = 0.3,
    computed_at: datetime = _T1,
) -> PolarizationMetrics:
    return PolarizationMetrics(
        snapshotId=snapshot_id or str(uuid.uuid4()),
        polarizationIndex=polarization_index,
        modularity=modularity,
        communityCount=community_count,
        avgCommunitySize=avg_community_size,
        interCommunityEdgeRatio=inter_community_edge_ratio,
        computedAt=computed_at,
        datasetSource=dataset_source,
    )


def _um(
    user_id: str = "user_1",
    community_id: str = "c1",
    diversity_score: float = 0.5,
    intra_edge_count: int = 5,
    inter_edge_count: int = 5,
    betweenness_centrality: float = 0.1,
    snapshot_id: str | None = None,
    computed_at: datetime = _T1,
) -> UserMetrics:
    return UserMetrics(
        userId=user_id,
        communityId=community_id,
        diversityScore=diversity_score,
        intraEdgeCount=intra_edge_count,
        interEdgeCount=inter_edge_count,
        betweennessCentrality=betweenness_centrality,
        snapshotId=snapshot_id or str(uuid.uuid4()),
        computedAt=computed_at,
    )


# ---------------------------------------------------------------------------
# persist_metrics — basic persistence (Requirements 4.6, 5.7)
# ---------------------------------------------------------------------------


class TestPersistMetrics:
    """persist_metrics writes rows to PostgreSQL (SQLite in tests)."""

    def test_persist_polarization_row_written(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """A PolarizationMetricRow is inserted for each call (Req 4.6)."""
        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id)

        svc.persist_metrics(pm, [], db_session)
        db_session.commit()

        rows = db_session.query(PolarizationMetricRow).all()
        assert len(rows) == 1
        assert rows[0].snapshot_id == snap_id

    def test_persist_polarization_row_fields(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """All PolarizationMetrics fields are stored correctly."""
        snap_id = str(uuid.uuid4())
        pm = _pm(
            snapshot_id=snap_id,
            dataset_source="congress",
            polarization_index=0.88,
            modularity=0.45,
            community_count=2,
            avg_community_size=237.5,
            inter_community_edge_ratio=0.12,
            computed_at=_T2,
        )

        svc.persist_metrics(pm, [], db_session)
        db_session.commit()

        row = db_session.query(PolarizationMetricRow).one()
        assert row.snapshot_id == snap_id
        assert row.dataset_source == "congress"
        assert abs(row.polarization_index - 0.88) < 1e-9
        assert abs(row.modularity - 0.45) < 1e-9
        assert row.community_count == 2
        assert abs(row.avg_community_size - 237.5) < 1e-9
        assert abs(row.inter_community_edge_ratio - 0.12) < 1e-9

    def test_persist_multiple_polarization_rows_time_series(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """Two snapshots → two PolarizationMetricRows (Req 4.7 time series)."""
        snap1 = str(uuid.uuid4())
        snap2 = str(uuid.uuid4())
        pm1 = _pm(snapshot_id=snap1, computed_at=_T1)
        pm2 = _pm(snapshot_id=snap2, computed_at=_T2)

        svc.persist_metrics(pm1, [], db_session)
        svc.persist_metrics(pm2, [], db_session)
        db_session.commit()

        rows = db_session.query(PolarizationMetricRow).all()
        assert len(rows) == 2
        snap_ids = {r.snapshot_id for r in rows}
        assert snap1 in snap_ids
        assert snap2 in snap_ids

    def test_persist_user_metrics_row_written(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """UserMetricRow is inserted for each UserMetrics entry (Req 5.7)."""
        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id)
        um = _um(snapshot_id=snap_id, user_id="alice")

        svc.persist_metrics(pm, [um], db_session)
        db_session.commit()

        rows = db_session.query(UserMetricRow).all()
        assert len(rows) == 1
        assert rows[0].user_id == "alice"
        assert rows[0].snapshot_id == snap_id

    def test_persist_user_metrics_fields(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """All UserMetrics fields are stored correctly (Req 5.7)."""
        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id, dataset_source="wiki_rfa")
        um = _um(
            user_id="bob",
            community_id="c2",
            diversity_score=0.75,
            intra_edge_count=10,
            inter_edge_count=3,
            betweenness_centrality=0.22,
            snapshot_id=snap_id,
            computed_at=_T2,
        )

        svc.persist_metrics(pm, [um], db_session)
        db_session.commit()

        row = db_session.query(UserMetricRow).one()
        assert row.user_id == "bob"
        assert row.community_id == "c2"
        assert abs(row.diversity_score - 0.75) < 1e-9
        assert row.intra_edge_count == 10
        assert row.inter_edge_count == 3
        assert abs(row.betweenness_centrality - 0.22) < 1e-9
        assert row.dataset_source == "wiki_rfa"

    def test_persist_multiple_user_metrics(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """Multiple UserMetrics entries are all persisted in one call."""
        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id)
        um_list = [
            _um(user_id=f"user_{i}", snapshot_id=snap_id)
            for i in range(5)
        ]

        svc.persist_metrics(pm, um_list, db_session)
        db_session.commit()

        count = db_session.query(UserMetricRow).count()
        assert count == 5

    def test_persist_empty_user_metrics_list(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """persist_metrics with empty user_metrics_list only writes one polarization row."""
        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id)

        svc.persist_metrics(pm, [], db_session)
        db_session.commit()

        assert db_session.query(PolarizationMetricRow).count() == 1
        assert db_session.query(UserMetricRow).count() == 0

    def test_dataset_source_propagated_to_user_metrics(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """dataset_source from PolarizationMetrics is stored on UserMetricRow."""
        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id, dataset_source="congress")
        um = _um(snapshot_id=snap_id)

        svc.persist_metrics(pm, [um], db_session)
        db_session.commit()

        row = db_session.query(UserMetricRow).one()
        assert row.dataset_source == "congress"


# ---------------------------------------------------------------------------
# persist_metrics — Redis caching
# ---------------------------------------------------------------------------


class TestPersistMetricsRedisCache:
    """Polarization metrics are cached in Redis after persist_metrics."""

    def test_redis_setex_called_with_correct_key(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """persist_metrics calls redis.setex with the polarization_key pattern."""
        from graph.redis_keys import polarization_key

        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id)
        redis_mock = MagicMock()

        svc.persist_metrics(pm, [], db_session, redis_client=redis_mock)

        redis_mock.setex.assert_called_once()
        call_args = redis_mock.setex.call_args
        assert call_args[0][0] == polarization_key(snap_id)

    def test_redis_setex_ttl_is_86400(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """The Redis TTL is DEFAULT_TTL_SECONDS (86400)."""
        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id)
        redis_mock = MagicMock()

        svc.persist_metrics(pm, [], db_session, redis_client=redis_mock)

        call_args = redis_mock.setex.call_args
        assert call_args[0][1] == 86_400

    def test_redis_cached_value_is_valid_json(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """The cached value is valid JSON containing all polarization fields."""
        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id, polarization_index=0.65, dataset_source="reddit_title")
        redis_mock = MagicMock()

        svc.persist_metrics(pm, [], db_session, redis_client=redis_mock)

        call_args = redis_mock.setex.call_args
        cached_json = call_args[0][2]
        payload = json.loads(cached_json)

        assert payload["snapshotId"] == snap_id
        assert payload["datasetSource"] == "reddit_title"
        assert abs(payload["polarizationIndex"] - 0.65) < 1e-9
        assert "modularity" in payload
        assert "communityCount" in payload
        assert "avgCommunitySize" in payload
        assert "interCommunityEdgeRatio" in payload
        assert "computedAt" in payload

    def test_redis_skipped_when_no_client(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """persist_metrics without redis_client writes DB rows without error."""
        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id)

        # Should not raise even without a redis client
        svc.persist_metrics(pm, [], db_session, redis_client=None)
        db_session.commit()

        assert db_session.query(PolarizationMetricRow).count() == 1

    def test_redis_failure_does_not_prevent_db_write(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """A Redis error is caught and does not roll back the DB write."""
        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id)
        redis_mock = MagicMock()
        redis_mock.setex.side_effect = ConnectionError("Redis unavailable")

        # Should not raise
        svc.persist_metrics(pm, [], db_session, redis_client=redis_mock)
        db_session.commit()

        assert db_session.query(PolarizationMetricRow).count() == 1


# ---------------------------------------------------------------------------
# query_metrics — filter by snapshotId
# ---------------------------------------------------------------------------


class TestQueryMetricsBySnapshotId:
    """query_metrics with snapshot_id filter returns matching rows only."""

    def _seed_two_snapshots(
        self, svc: MetricsService, db_session: Session
    ) -> tuple[str, str]:
        snap1 = str(uuid.uuid4())
        snap2 = str(uuid.uuid4())
        svc.persist_metrics(_pm(snapshot_id=snap1), [_um(snapshot_id=snap1)], db_session)
        svc.persist_metrics(_pm(snapshot_id=snap2), [_um(snapshot_id=snap2)], db_session)
        db_session.commit()
        return snap1, snap2

    def test_filter_by_snapshot_id_polarization(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        snap1, snap2 = self._seed_two_snapshots(svc, db_session)
        result = svc.query_metrics(MetricsFilter(snapshot_id=snap1), db_session)
        assert len(result["polarization"]) == 1
        assert result["polarization"][0].snapshot_id == snap1

    def test_filter_by_snapshot_id_user_metrics(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        snap1, snap2 = self._seed_two_snapshots(svc, db_session)
        result = svc.query_metrics(MetricsFilter(snapshot_id=snap1), db_session)
        assert len(result["user_metrics"]) == 1
        assert result["user_metrics"][0].snapshot_id == snap1

    def test_filter_by_nonexistent_snapshot_returns_empty(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        self._seed_two_snapshots(svc, db_session)
        result = svc.query_metrics(MetricsFilter(snapshot_id="does-not-exist"), db_session)
        assert result["polarization"] == []
        assert result["user_metrics"] == []


# ---------------------------------------------------------------------------
# query_metrics — filter by datasetSource
# ---------------------------------------------------------------------------


class TestQueryMetricsByDatasetSource:
    """query_metrics with dataset_source filter returns matching rows only."""

    def test_filter_by_dataset_source_polarization(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        snap1 = str(uuid.uuid4())
        snap2 = str(uuid.uuid4())
        svc.persist_metrics(
            _pm(snapshot_id=snap1, dataset_source="congress"),
            [],
            db_session,
        )
        svc.persist_metrics(
            _pm(snapshot_id=snap2, dataset_source="reddit_title"),
            [],
            db_session,
        )
        db_session.commit()

        result = svc.query_metrics(MetricsFilter(dataset_source="congress"), db_session)
        assert len(result["polarization"]) == 1
        assert result["polarization"][0].dataset_source == "congress"

    def test_filter_by_dataset_source_user_metrics(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        snap1 = str(uuid.uuid4())
        snap2 = str(uuid.uuid4())
        svc.persist_metrics(
            _pm(snapshot_id=snap1, dataset_source="congress"),
            [_um(snapshot_id=snap1)],
            db_session,
        )
        svc.persist_metrics(
            _pm(snapshot_id=snap2, dataset_source="wiki_rfa"),
            [_um(snapshot_id=snap2)],
            db_session,
        )
        db_session.commit()

        result = svc.query_metrics(MetricsFilter(dataset_source="wiki_rfa"), db_session)
        assert len(result["user_metrics"]) == 1
        assert result["user_metrics"][0].dataset_source == "wiki_rfa"

    def test_filter_by_unknown_source_returns_empty(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        snap_id = str(uuid.uuid4())
        svc.persist_metrics(_pm(snapshot_id=snap_id, dataset_source="reddit_title"), [], db_session)
        db_session.commit()

        result = svc.query_metrics(MetricsFilter(dataset_source="twitter"), db_session)
        assert result["polarization"] == []


# ---------------------------------------------------------------------------
# query_metrics — filter by date range
# ---------------------------------------------------------------------------


class TestQueryMetricsByDateRange:
    """query_metrics with from_date / to_date filters."""

    def _seed_three_times(self, svc: MetricsService, db_session: Session) -> list[str]:
        snap_ids = [str(uuid.uuid4()) for _ in range(3)]
        times = [_T1, _T2, _T3]
        for snap_id, t in zip(snap_ids, times):
            svc.persist_metrics(
                _pm(snapshot_id=snap_id, computed_at=t),
                [_um(snapshot_id=snap_id, computed_at=t)],
                db_session,
            )
        db_session.commit()
        return snap_ids

    def test_from_date_filters_out_earlier_rows(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        self._seed_three_times(svc, db_session)
        result = svc.query_metrics(MetricsFilter(from_date=_T2), db_session)
        # _T2 and _T3 should be included
        assert len(result["polarization"]) == 2

    def test_to_date_filters_out_later_rows(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        self._seed_three_times(svc, db_session)
        result = svc.query_metrics(MetricsFilter(to_date=_T2), db_session)
        # _T1 and _T2 should be included
        assert len(result["polarization"]) == 2

    def test_from_and_to_date_narrow_range(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        self._seed_three_times(svc, db_session)
        result = svc.query_metrics(MetricsFilter(from_date=_T2, to_date=_T2), db_session)
        assert len(result["polarization"]) == 1

    def test_date_range_on_user_metrics(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        self._seed_three_times(svc, db_session)
        result = svc.query_metrics(MetricsFilter(from_date=_T3), db_session)
        assert len(result["user_metrics"]) == 1


# ---------------------------------------------------------------------------
# query_metrics — filter by communityId
# ---------------------------------------------------------------------------


class TestQueryMetricsByCommunityId:
    """query_metrics with community_id filter for UserMetrics rows."""

    def test_filter_user_metrics_by_community_id(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id)
        um_list = [
            _um(user_id="u1", community_id="c1", snapshot_id=snap_id),
            _um(user_id="u2", community_id="c2", snapshot_id=snap_id),
            _um(user_id="u3", community_id="c1", snapshot_id=snap_id),
        ]

        svc.persist_metrics(pm, um_list, db_session)
        db_session.commit()

        result = svc.query_metrics(MetricsFilter(community_id="c1"), db_session)
        assert len(result["user_metrics"]) == 2
        for row in result["user_metrics"]:
            assert row.community_id == "c1"

    def test_community_id_filter_does_not_affect_polarization_rows(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """community_id only filters user_metrics; polarization rows are unaffected."""
        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id)
        um = _um(user_id="u1", community_id="c1", snapshot_id=snap_id)

        svc.persist_metrics(pm, [um], db_session)
        db_session.commit()

        result = svc.query_metrics(MetricsFilter(community_id="c1"), db_session)
        assert len(result["polarization"]) == 1

    def test_filter_by_nonexistent_community_returns_empty_user_metrics(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id)
        um = _um(user_id="u1", community_id="c1", snapshot_id=snap_id)

        svc.persist_metrics(pm, [um], db_session)
        db_session.commit()

        result = svc.query_metrics(MetricsFilter(community_id="c999"), db_session)
        assert result["user_metrics"] == []


# ---------------------------------------------------------------------------
# query_metrics — filter by min_polarization
# ---------------------------------------------------------------------------


class TestQueryMetricsByMinPolarization:
    """query_metrics with min_polarization threshold filter."""

    def test_min_polarization_filters_low_pi_rows(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        snap1 = str(uuid.uuid4())
        snap2 = str(uuid.uuid4())
        svc.persist_metrics(
            _pm(snapshot_id=snap1, polarization_index=0.9),
            [],
            db_session,
        )
        svc.persist_metrics(
            _pm(snapshot_id=snap2, polarization_index=0.3),
            [],
            db_session,
        )
        db_session.commit()

        result = svc.query_metrics(MetricsFilter(min_polarization=0.5), db_session)
        assert len(result["polarization"]) == 1
        assert result["polarization"][0].snapshot_id == snap1

    def test_min_polarization_equals_boundary_included(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """Rows at exactly min_polarization should be included (>=)."""
        snap_id = str(uuid.uuid4())
        svc.persist_metrics(
            _pm(snapshot_id=snap_id, polarization_index=0.5),
            [],
            db_session,
        )
        db_session.commit()

        result = svc.query_metrics(MetricsFilter(min_polarization=0.5), db_session)
        assert len(result["polarization"]) == 1

    def test_min_polarization_zero_returns_all(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        """min_polarization=0.0 should return all rows."""
        for _ in range(3):
            svc.persist_metrics(
                _pm(snapshot_id=str(uuid.uuid4()), polarization_index=0.1),
                [],
                db_session,
            )
        db_session.commit()

        result = svc.query_metrics(MetricsFilter(min_polarization=0.0), db_session)
        assert len(result["polarization"]) == 3


# ---------------------------------------------------------------------------
# query_metrics — no filter (returns all)
# ---------------------------------------------------------------------------


class TestQueryMetricsNoFilter:
    """query_metrics with empty MetricsFilter returns all rows."""

    def test_no_filter_returns_all_polarization(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        for _ in range(4):
            svc.persist_metrics(_pm(snapshot_id=str(uuid.uuid4())), [], db_session)
        db_session.commit()

        result = svc.query_metrics(MetricsFilter(), db_session)
        assert len(result["polarization"]) == 4

    def test_no_filter_returns_all_user_metrics(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        snap_id = str(uuid.uuid4())
        pm = _pm(snapshot_id=snap_id)
        um_list = [_um(user_id=f"u{i}", snapshot_id=snap_id) for i in range(6)]
        svc.persist_metrics(pm, um_list, db_session)
        db_session.commit()

        result = svc.query_metrics(MetricsFilter(), db_session)
        assert len(result["user_metrics"]) == 6

    def test_empty_db_returns_empty_lists(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        result = svc.query_metrics(MetricsFilter(), db_session)
        assert result["polarization"] == []
        assert result["user_metrics"] == []


# ---------------------------------------------------------------------------
# query_metrics — combined filters
# ---------------------------------------------------------------------------


class TestQueryMetricsCombinedFilters:
    """query_metrics handles combinations of multiple filters (AND semantics)."""

    def test_snapshot_and_dataset_source(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        snap1 = str(uuid.uuid4())
        snap2 = str(uuid.uuid4())
        svc.persist_metrics(
            _pm(snapshot_id=snap1, dataset_source="congress"),
            [],
            db_session,
        )
        svc.persist_metrics(
            _pm(snapshot_id=snap2, dataset_source="congress"),
            [],
            db_session,
        )
        db_session.commit()

        # Both filters must match — only snap1
        result = svc.query_metrics(
            MetricsFilter(snapshot_id=snap1, dataset_source="congress"),
            db_session,
        )
        assert len(result["polarization"]) == 1
        assert result["polarization"][0].snapshot_id == snap1

    def test_dataset_source_and_min_polarization(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        snap1 = str(uuid.uuid4())
        snap2 = str(uuid.uuid4())
        svc.persist_metrics(
            _pm(snapshot_id=snap1, dataset_source="congress", polarization_index=0.85),
            [],
            db_session,
        )
        svc.persist_metrics(
            _pm(snapshot_id=snap2, dataset_source="congress", polarization_index=0.40),
            [],
            db_session,
        )
        db_session.commit()

        result = svc.query_metrics(
            MetricsFilter(dataset_source="congress", min_polarization=0.80),
            db_session,
        )
        assert len(result["polarization"]) == 1
        assert result["polarization"][0].snapshot_id == snap1

    def test_community_id_and_snapshot_id(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        snap1 = str(uuid.uuid4())
        snap2 = str(uuid.uuid4())
        svc.persist_metrics(
            _pm(snapshot_id=snap1),
            [
                _um(user_id="u1", community_id="c1", snapshot_id=snap1),
                _um(user_id="u2", community_id="c2", snapshot_id=snap1),
            ],
            db_session,
        )
        svc.persist_metrics(
            _pm(snapshot_id=snap2),
            [_um(user_id="u3", community_id="c1", snapshot_id=snap2)],
            db_session,
        )
        db_session.commit()

        # Only snap1 + c1
        result = svc.query_metrics(
            MetricsFilter(snapshot_id=snap1, community_id="c1"),
            db_session,
        )
        assert len(result["user_metrics"]) == 1
        assert result["user_metrics"][0].user_id == "u1"


# ---------------------------------------------------------------------------
# query_metrics — return dict structure
# ---------------------------------------------------------------------------


class TestQueryMetricsReturnStructure:
    """query_metrics always returns the expected dict shape."""

    def test_result_has_polarization_key(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        result = svc.query_metrics(MetricsFilter(), db_session)
        assert "polarization" in result

    def test_result_has_user_metrics_key(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        result = svc.query_metrics(MetricsFilter(), db_session)
        assert "user_metrics" in result

    def test_result_polarization_entries_are_orm_rows(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        snap_id = str(uuid.uuid4())
        svc.persist_metrics(_pm(snapshot_id=snap_id), [], db_session)
        db_session.commit()

        result = svc.query_metrics(MetricsFilter(), db_session)
        for row in result["polarization"]:
            assert isinstance(row, PolarizationMetricRow)

    def test_result_user_metrics_entries_are_orm_rows(
        self, svc: MetricsService, db_session: Session
    ) -> None:
        snap_id = str(uuid.uuid4())
        svc.persist_metrics(
            _pm(snapshot_id=snap_id),
            [_um(snapshot_id=snap_id)],
            db_session,
        )
        db_session.commit()

        result = svc.query_metrics(MetricsFilter(), db_session)
        for row in result["user_metrics"]:
            assert isinstance(row, UserMetricRow)
