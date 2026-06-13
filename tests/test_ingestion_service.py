"""Tests for ingestion/service.py — IngestionService.

Covers:
- Successful ingestion: returns IngestionResult with status="success",
  correct record_count, duplicate_count, snapshot_id UUID, dataset_source.
- Deduplication keys by dataset:
  - reddit_title / reddit_body: (sourceUserId, targetUserId, timestamp)
  - congress: (sourceUserId, targetUserId) only
  - wiki_rfa: (sourceUserId, targetUserId, timestamp) — different timestamps = kept
- File read failure: returns status="failed", preserves previous_snapshot,
  records error string, logs warning.
- Zero records after dedup: returns status="empty", preserves previous_snapshot.
- get_status() returns last IngestionResult.
- IngestionResult fields: record_count == len(records), duplicate_count accurate.

References: Requirements 1.3, 1.4, 1.5, 1.6
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from graph.models import InteractionRecord, InteractionType
from ingestion.adapters import DatasetConfig
from ingestion.service import IngestionResult, IngestionService, IngestionStatus, _dedup_key, _deduplicate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(year: int, month: int = 1, day: int = 1) -> datetime:
    """Return a UTC-aware datetime in the past."""
    return datetime(year, month, day, tzinfo=timezone.utc)


def _record(
    source: str = "userA",
    target: str = "userB",
    dataset: str = "reddit_title",
    timestamp: datetime | None = _ts(2020),
) -> InteractionRecord:
    """Build a minimal valid InteractionRecord."""
    return InteractionRecord(
        id=str(uuid.uuid4()),
        sourceUserId=source,
        targetUserId=target,
        interactionType=InteractionType.HYPERLINK,
        datasetSource=dataset,
        timestamp=timestamp,
    )


class _FakeAdapter:
    """Adapter stub that returns a fixed list of records."""

    DATASET_SOURCE = "reddit_title"

    def __init__(self, records: list[InteractionRecord]) -> None:
        self._records = records

    def fetch(self, config: DatasetConfig) -> list[InteractionRecord]:
        return list(self._records)

    def normalize(self, raw: dict) -> InteractionRecord | None:
        return None  # not used in these tests


class _FailingAdapter:
    """Adapter stub whose fetch() always raises an exception."""

    DATASET_SOURCE = "reddit_title"

    def fetch(self, config: DatasetConfig) -> list[InteractionRecord]:
        raise OSError("file not found")

    def normalize(self, raw: dict) -> InteractionRecord | None:
        return None


# ---------------------------------------------------------------------------
# IngestionResult fields
# ---------------------------------------------------------------------------


def test_ingest_success_returns_success_status() -> None:
    """A batch with unique records produces status='success'."""
    service = IngestionService()
    r1 = _record(source="A", target="B", timestamp=_ts(2020))
    r2 = _record(source="C", target="D", timestamp=_ts(2021))
    adapter = _FakeAdapter([r1, r2])
    config = DatasetConfig(source_type="reddit_title", file_path="/fake.tsv", format="tsv")
    result = service.ingest(adapter, "/fake.tsv", config=config)

    assert result.status == "success"


def test_ingest_success_record_count_matches_records_length() -> None:
    """record_count must equal len(records)."""
    service = IngestionService()
    records = [_record(source="A", target="B", timestamp=_ts(2020)),
               _record(source="C", target="D", timestamp=_ts(2021))]
    adapter = _FakeAdapter(records)
    config = DatasetConfig(source_type="reddit_title", file_path="/fake.tsv", format="tsv")
    result = service.ingest(adapter, "/fake.tsv", config=config)

    assert result.record_count == len(result.records) == 2


def test_ingest_success_snapshot_id_is_uuid() -> None:
    """snapshot_id must be a valid UUID string."""
    service = IngestionService()
    adapter = _FakeAdapter([_record()])
    config = DatasetConfig(source_type="reddit_title", file_path="/fake.tsv", format="tsv")
    result = service.ingest(adapter, "/fake.tsv", config=config)

    # Should not raise
    parsed = uuid.UUID(result.snapshot_id)
    assert str(parsed) == result.snapshot_id


def test_ingest_success_dataset_source_matches_config() -> None:
    """dataset_source in result must match config.source_type."""
    service = IngestionService()
    adapter = _FakeAdapter([_record(dataset="reddit_body")])
    adapter.DATASET_SOURCE = "reddit_body"
    config = DatasetConfig(source_type="reddit_body", file_path="/fake.tsv", format="tsv")
    result = service.ingest(adapter, "/fake.tsv", config=config)

    assert result.dataset_source == "reddit_body"


def test_ingest_success_no_error_field() -> None:
    """On success, error must be None."""
    service = IngestionService()
    adapter = _FakeAdapter([_record()])
    config = DatasetConfig(source_type="reddit_title", file_path="/fake.tsv", format="tsv")
    result = service.ingest(adapter, "/fake.tsv", config=config)

    assert result.error is None


# ---------------------------------------------------------------------------
# Deduplication — Reddit (timestamp-keyed)
# ---------------------------------------------------------------------------


def test_reddit_dedup_removes_exact_duplicate() -> None:
    """Two records with same (src, tgt, ts) for reddit_title → 1 kept, 1 duplicate."""
    service = IngestionService()
    ts = _ts(2020)
    r1 = _record(source="A", target="B", dataset="reddit_title", timestamp=ts)
    r2 = _record(source="A", target="B", dataset="reddit_title", timestamp=ts)
    adapter = _FakeAdapter([r1, r2])
    config = DatasetConfig(source_type="reddit_title", file_path="/fake.tsv", format="tsv")
    result = service.ingest(adapter, "/fake.tsv", config=config)

    assert result.record_count == 1
    assert result.duplicate_count == 1
    assert result.status == "success"


def test_reddit_dedup_keeps_same_pair_different_timestamps() -> None:
    """Same (src, tgt) but different timestamps → both kept for reddit_title."""
    service = IngestionService()
    r1 = _record(source="A", target="B", dataset="reddit_title", timestamp=_ts(2020))
    r2 = _record(source="A", target="B", dataset="reddit_title", timestamp=_ts(2021))
    adapter = _FakeAdapter([r1, r2])
    config = DatasetConfig(source_type="reddit_title", file_path="/fake.tsv", format="tsv")
    result = service.ingest(adapter, "/fake.tsv", config=config)

    assert result.record_count == 2
    assert result.duplicate_count == 0


def test_reddit_body_dedup_same_key_removed() -> None:
    """reddit_body uses same (src, tgt, ts) key — duplicates removed."""
    service = IngestionService()
    ts = _ts(2019, 6, 1)
    r1 = _record(source="X", target="Y", dataset="reddit_body", timestamp=ts)
    r2 = _record(source="X", target="Y", dataset="reddit_body", timestamp=ts)
    r3 = _record(source="X", target="Y", dataset="reddit_body", timestamp=_ts(2019, 7, 1))
    adapter = _FakeAdapter([r1, r2, r3])
    adapter.DATASET_SOURCE = "reddit_body"
    config = DatasetConfig(source_type="reddit_body", file_path="/fake.tsv", format="tsv")
    result = service.ingest(adapter, "/fake.tsv", config=config)

    assert result.record_count == 2
    assert result.duplicate_count == 1


# ---------------------------------------------------------------------------
# Deduplication — Congress (pair-keyed, no timestamp)
# ---------------------------------------------------------------------------


def test_congress_dedup_removes_same_pair_regardless_of_timestamp() -> None:
    """Congress dedup key is (src, tgt) — different None timestamps both match."""
    service = IngestionService()
    # Congress records have timestamp=None
    r1 = _record(source="alice", target="bob", dataset="congress", timestamp=None)
    r2 = _record(source="alice", target="bob", dataset="congress", timestamp=None)
    adapter = _FakeAdapter([r1, r2])
    adapter.DATASET_SOURCE = "congress"
    config = DatasetConfig(source_type="congress", file_path="/fake.edgelist", format="edgelist")
    result = service.ingest(adapter, "/fake.edgelist", config=config)

    assert result.record_count == 1
    assert result.duplicate_count == 1


def test_congress_dedup_keeps_different_pairs() -> None:
    """Congress: (alice→bob) and (alice→carol) are distinct — both kept."""
    service = IngestionService()
    r1 = _record(source="alice", target="bob", dataset="congress", timestamp=None)
    r2 = _record(source="alice", target="carol", dataset="congress", timestamp=None)
    adapter = _FakeAdapter([r1, r2])
    adapter.DATASET_SOURCE = "congress"
    config = DatasetConfig(source_type="congress", file_path="/fake.edgelist", format="edgelist")
    result = service.ingest(adapter, "/fake.edgelist", config=config)

    assert result.record_count == 2
    assert result.duplicate_count == 0


# ---------------------------------------------------------------------------
# Deduplication — Wiki-RfA (timestamp differentiates multi-year votes)
# ---------------------------------------------------------------------------


def test_wiki_rfa_dedup_removes_exact_duplicate() -> None:
    """wiki_rfa: same (src, tgt, ts) → 1 kept, 1 discarded."""
    service = IngestionService()
    ts = _ts(2013)
    r1 = _record(source="voter1", target="candidate1", dataset="wiki_rfa", timestamp=ts)
    r2 = _record(source="voter1", target="candidate1", dataset="wiki_rfa", timestamp=ts)
    adapter = _FakeAdapter([r1, r2])
    adapter.DATASET_SOURCE = "wiki_rfa"
    config = DatasetConfig(source_type="wiki_rfa", file_path="/fake.txt.gz", format="txt_gz")
    result = service.ingest(adapter, "/fake.txt.gz", config=config)

    assert result.record_count == 1
    assert result.duplicate_count == 1


def test_wiki_rfa_dedup_keeps_same_pair_different_years() -> None:
    """wiki_rfa: same voter→candidate across two different years → both kept."""
    service = IngestionService()
    r1 = _record(source="voter1", target="candidate1", dataset="wiki_rfa", timestamp=_ts(2008))
    r2 = _record(source="voter1", target="candidate1", dataset="wiki_rfa", timestamp=_ts(2013))
    adapter = _FakeAdapter([r1, r2])
    adapter.DATASET_SOURCE = "wiki_rfa"
    config = DatasetConfig(source_type="wiki_rfa", file_path="/fake.txt.gz", format="txt_gz")
    result = service.ingest(adapter, "/fake.txt.gz", config=config)

    assert result.record_count == 2
    assert result.duplicate_count == 0


# ---------------------------------------------------------------------------
# File read failure → status="failed", preserve previous snapshot
# ---------------------------------------------------------------------------


def test_file_failure_returns_failed_status() -> None:
    """When adapter.fetch() raises, result status is 'failed'."""
    service = IngestionService()
    adapter = _FailingAdapter()
    config = DatasetConfig(source_type="reddit_title", file_path="/missing.tsv", format="tsv")
    result = service.ingest(adapter, "/missing.tsv", config=config)

    assert result.status == "failed"


def test_file_failure_preserves_previous_snapshot() -> None:
    """On failure, the result records equal previous_snapshot."""
    service = IngestionService()
    adapter = _FailingAdapter()
    config = DatasetConfig(source_type="reddit_title", file_path="/missing.tsv", format="tsv")
    prev = [_record(source="P", target="Q")]
    result = service.ingest(adapter, "/missing.tsv", previous_snapshot=prev, config=config)

    assert result.records == prev
    assert result.record_count == len(prev)


def test_file_failure_with_no_previous_snapshot_returns_empty_records() -> None:
    """On failure with no previous snapshot, records is []."""
    service = IngestionService()
    adapter = _FailingAdapter()
    config = DatasetConfig(source_type="reddit_title", file_path="/missing.tsv", format="tsv")
    result = service.ingest(adapter, "/missing.tsv", config=config)

    assert result.records == []
    assert result.record_count == 0


def test_file_failure_records_error_message() -> None:
    """On failure, error field contains the exception message."""
    service = IngestionService()
    adapter = _FailingAdapter()
    config = DatasetConfig(source_type="reddit_title", file_path="/missing.tsv", format="tsv")
    result = service.ingest(adapter, "/missing.tsv", config=config)

    assert result.error is not None
    assert len(result.error) > 0


def test_file_failure_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A file read error must produce a WARNING-level log entry."""
    import logging

    service = IngestionService()
    adapter = _FailingAdapter()
    config = DatasetConfig(source_type="reddit_title", file_path="/missing.tsv", format="tsv")

    with caplog.at_level(logging.WARNING, logger="ingestion.service"):
        service.ingest(adapter, "/missing.tsv", config=config)

    assert any("warning" in record.levelname.lower() or record.levelno >= logging.WARNING
               for record in caplog.records)


# ---------------------------------------------------------------------------
# Zero valid records → status="empty", preserve previous snapshot
# ---------------------------------------------------------------------------


def test_zero_records_returns_empty_status() -> None:
    """Adapter returning no records → status='empty'."""
    service = IngestionService()
    adapter = _FakeAdapter([])
    config = DatasetConfig(source_type="reddit_title", file_path="/empty.tsv", format="tsv")
    result = service.ingest(adapter, "/empty.tsv", config=config)

    assert result.status == "empty"


def test_zero_records_preserves_previous_snapshot() -> None:
    """Empty adapter output → result.records == previous_snapshot."""
    service = IngestionService()
    adapter = _FakeAdapter([])
    prev = [_record(source="old_A", target="old_B")]
    config = DatasetConfig(source_type="reddit_title", file_path="/empty.tsv", format="tsv")
    result = service.ingest(adapter, "/empty.tsv", previous_snapshot=prev, config=config)

    assert result.records == prev


def test_all_duplicates_treated_as_zero_records() -> None:
    """If adapter returns only duplicates, deduplicated list is size 1, not 0 → success."""
    # Edge case: 3 identical records → dedup leaves 1 → that's not "empty"
    service = IngestionService()
    ts = _ts(2020)
    records = [_record(source="A", target="B", timestamp=ts) for _ in range(3)]
    adapter = _FakeAdapter(records)
    config = DatasetConfig(source_type="reddit_title", file_path="/dup.tsv", format="tsv")
    result = service.ingest(adapter, "/dup.tsv", config=config)

    assert result.status == "success"
    assert result.record_count == 1
    assert result.duplicate_count == 2


def test_zero_records_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Zero valid records must produce a WARNING-level log entry."""
    import logging

    service = IngestionService()
    adapter = _FakeAdapter([])
    config = DatasetConfig(source_type="reddit_title", file_path="/empty.tsv", format="tsv")

    with caplog.at_level(logging.WARNING, logger="ingestion.service"):
        service.ingest(adapter, "/empty.tsv", config=config)

    assert any(record.levelno >= logging.WARNING for record in caplog.records)


# ---------------------------------------------------------------------------
# get_status()
# ---------------------------------------------------------------------------


def test_get_status_before_any_ingest_returns_none_last_result() -> None:
    """Freshly constructed service has last_result=None."""
    service = IngestionService()
    status = service.get_status()
    assert isinstance(status, IngestionStatus)
    assert status.last_result is None


def test_get_status_after_ingest_returns_last_result() -> None:
    """get_status().last_result mirrors the most recent ingest result."""
    service = IngestionService()
    adapter = _FakeAdapter([_record()])
    config = DatasetConfig(source_type="reddit_title", file_path="/f.tsv", format="tsv")
    result = service.ingest(adapter, "/f.tsv", config=config)

    assert service.get_status().last_result is result


def test_get_status_updated_after_second_ingest() -> None:
    """Calling ingest() twice updates last_result to the second call's result."""
    service = IngestionService()
    adapter1 = _FakeAdapter([_record(source="A", target="B")])
    adapter2 = _FakeAdapter([_record(source="C", target="D")])
    config = DatasetConfig(source_type="reddit_title", file_path="/f.tsv", format="tsv")

    result1 = service.ingest(adapter1, "/f.tsv", config=config)
    result2 = service.ingest(adapter2, "/f.tsv", config=config)

    assert service.get_status().last_result is result2
    assert service.get_status().last_result is not result1


# ---------------------------------------------------------------------------
# _dedup_key helper (unit tests)
# ---------------------------------------------------------------------------


def test_dedup_key_reddit_title_includes_timestamp() -> None:
    ts = _ts(2020)
    r = _record(dataset="reddit_title", timestamp=ts)
    key = _dedup_key(r)
    assert len(key) == 3
    assert key[2] == ts


def test_dedup_key_reddit_body_includes_timestamp() -> None:
    ts = _ts(2021)
    r = _record(dataset="reddit_body", timestamp=ts)
    key = _dedup_key(r)
    assert len(key) == 3
    assert key[2] == ts


def test_dedup_key_congress_excludes_timestamp() -> None:
    r = _record(dataset="congress", timestamp=None)
    key = _dedup_key(r)
    assert len(key) == 2


def test_dedup_key_wiki_rfa_includes_timestamp() -> None:
    ts = _ts(2013)
    r = _record(dataset="wiki_rfa", timestamp=ts)
    key = _dedup_key(r)
    assert len(key) == 3
    assert key[2] == ts


def test_dedup_key_unknown_source_falls_back_to_triple() -> None:
    """Unknown dataset source should use the safe 3-tuple fallback."""
    r = _record(dataset="some_future_source", timestamp=_ts(2022))
    key = _dedup_key(r)
    assert len(key) == 3


# ---------------------------------------------------------------------------
# _deduplicate helper (unit tests)
# ---------------------------------------------------------------------------


def test_deduplicate_no_duplicates_returns_all() -> None:
    records = [
        _record(source="A", target="B", timestamp=_ts(2020)),
        _record(source="C", target="D", timestamp=_ts(2021)),
    ]
    unique, dups = _deduplicate(records)
    assert len(unique) == 2
    assert dups == 0


def test_deduplicate_preserves_order() -> None:
    """First occurrence of a key must be kept (insertion order preserved)."""
    ts = _ts(2020)
    r1 = _record(source="A", target="B", timestamp=ts)
    r2 = _record(source="A", target="B", timestamp=ts)
    unique, _ = _deduplicate([r1, r2])
    assert unique[0] is r1


def test_deduplicate_count_is_correct() -> None:
    ts = _ts(2020)
    records = [
        _record(source="A", target="B", timestamp=ts),
        _record(source="A", target="B", timestamp=ts),
        _record(source="A", target="B", timestamp=ts),
        _record(source="C", target="D", timestamp=ts),
    ]
    unique, dups = _deduplicate(records)
    assert len(unique) == 2
    assert dups == 2


def test_deduplicate_empty_input() -> None:
    unique, dups = _deduplicate([])
    assert unique == []
    assert dups == 0
