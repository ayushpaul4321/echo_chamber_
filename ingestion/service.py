"""IngestionService — dataset-aware deduplication and ingestion orchestration.

Wraps any DataSourceAdapter to provide:
  - Input record validation (Requirements 9.1–9.5)
  - Dataset-aware deduplication (Reddit / Congress / Wiki-RfA keying rules)
  - File-read error handling with warning log (no HTTP retry for file adapters)
  - Snapshot preservation when zero valid records are produced
  - Structured IngestionResult / IngestionStatus reporting

References: Requirements 1.3, 1.4, 1.5, 1.6, 9.1–9.5
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from graph.models import InteractionRecord
from ingestion.adapters import DataSourceAdapter, DatasetConfig
from ingestion.validation import validate_record

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result & Status dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Structured outcome of a single ingest() call.

    Attributes:
        records:         Deduplicated valid records produced in this batch.
        snapshot_id:     UUID string identifying this ingestion batch.
        dataset_source:  The datasetSource tag (e.g. ``"reddit_title"``).
        record_count:    Number of records in *records* (len(records)).
        duplicate_count: Number of records discarded as duplicates.
        invalid_count:   Number of records rejected by input validation.
        status:          ``"success"`` | ``"empty"`` | ``"failed"``.
        error:           Human-readable error message on failure; else None.
    """

    records: list[InteractionRecord]
    snapshot_id: str
    dataset_source: str
    record_count: int
    duplicate_count: int
    status: str  # "success" | "empty" | "failed"
    error: Optional[str] = None
    invalid_count: int = 0


@dataclass
class IngestionStatus:
    """Tracks the last completed ingestion result for observability.

    Attributes:
        last_result: The most recent IngestionResult, or None if no
                     ingestion has been run yet.
    """

    last_result: Optional[IngestionResult] = field(default=None)


# ---------------------------------------------------------------------------
# Deduplication key helpers
# ---------------------------------------------------------------------------

# Dataset sources whose dedup key includes a timestamp component.
_TIMESTAMP_KEYED_SOURCES = {"reddit_title", "reddit_body", "wiki_rfa"}

# Dataset sources whose dedup key is (sourceUserId, targetUserId) only.
_PAIR_KEYED_SOURCES = {"congress"}


def _dedup_key(record: InteractionRecord) -> tuple:
    """Return the dataset-appropriate deduplication key for *record*.

    Key rules (per task 2.5 spec):
    - ``reddit_title`` / ``reddit_body``: ``(sourceUserId, targetUserId, timestamp)``
    - ``congress``:                        ``(sourceUserId, targetUserId)``
    - ``wiki_rfa``:                        ``(sourceUserId, targetUserId, timestamp)``
      (Multiple votes by the same user on the same candidate in different
      years are valid; the timestamp component differentiates them.)
    - Any other source:                    falls back to the timestamp-keyed
      triple so that new dataset types are safely handled.

    Args:
        record: An InteractionRecord whose dedup key is needed.

    Returns:
        A hashable tuple suitable for use in a set.
    """
    source = record.datasetSource

    if source in _PAIR_KEYED_SOURCES:
        # Congress: no timestamp in the dataset; key on user pair only.
        return (record.sourceUserId, record.targetUserId)

    # Reddit (title & body) and Wiki-RfA: include timestamp so that the
    # same user pair with different timestamps are kept as distinct records.
    # For Wiki-RfA this correctly allows the same voter to vote on the same
    # candidate across multiple years.
    return (record.sourceUserId, record.targetUserId, record.timestamp)


def _validate_records(
    records: list[InteractionRecord],
) -> tuple[list[InteractionRecord], int]:
    """Validate each record against all ingestion rules (Requirements 9.1–9.5).

    Invalid records are logged at WARNING level with the record ID and
    rejection reason.  Processing continues for all remaining records so that
    a single bad record never halts the overall ingestion job.

    Args:
        records: List of InteractionRecords to validate.

    Returns:
        A 2-tuple of (valid_records, invalid_count).
    """
    valid: list[InteractionRecord] = []
    invalid_count = 0

    for record in records:
        is_valid, reason = validate_record(record)
        if is_valid:
            valid.append(record)
        else:
            invalid_count += 1
            logger.warning(
                "IngestionService: rejected record id='%s' reason='%s'",
                record.id,
                reason,
            )

    return valid, invalid_count


def _deduplicate(
    records: list[InteractionRecord],
) -> tuple[list[InteractionRecord], int]:
    """Remove duplicate records according to dataset-aware key rules.

    The first occurrence of each key is retained; subsequent duplicates are
    discarded.  This preserves insertion order.

    Args:
        records: Raw (possibly duplicate-containing) list of InteractionRecords.

    Returns:
        A 2-tuple of (deduplicated_records, duplicate_count).
    """
    seen: set[tuple] = set()
    unique: list[InteractionRecord] = []
    duplicates = 0

    for record in records:
        key = _dedup_key(record)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
            unique.append(record)

    return unique, duplicates


# ---------------------------------------------------------------------------
# IngestionService
# ---------------------------------------------------------------------------


class IngestionService:
    """Orchestrates ingestion for any DataSourceAdapter with dedup + resilience.

    Usage::

        service = IngestionService()
        config  = DatasetConfig(source_type="reddit_title",
                                file_path="/data/title.tsv",
                                format="tsv")
        result  = service.ingest(RedditTitleAdapter(), config)

    The service is stateless apart from :attr:`_status`, which records the
    last ingestion outcome.
    """

    def __init__(self) -> None:
        self._status = IngestionStatus()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        adapter: DataSourceAdapter,
        filepath: str,
        previous_snapshot: Optional[list[InteractionRecord]] = None,
        *,
        config: Optional[DatasetConfig] = None,
    ) -> IngestionResult:
        """Run ingestion through *adapter* for the file at *filepath*.

        Steps:
        1. Attempt to load records via ``adapter.fetch()``.
        2. On file-read error: log a warning; return a ``"failed"``
           IngestionResult whose ``records`` field preserves
           *previous_snapshot* (if provided) so callers keep the last
           known good dataset.
        3. Validate each record using :func:`~ingestion.validation.validate_record`
           (Requirements 9.1–9.5); invalid records are logged at WARNING level
           with record ID and reason, then discarded without halting the job.
        4. Deduplicate the validated records using dataset-aware key rules.
        5. If zero valid records remain after deduplication: log a warning
           and return an ``"empty"`` IngestionResult whose ``records`` field
           preserves *previous_snapshot*.
        6. On success: log an info message and return a ``"success"``
           IngestionResult containing the deduplicated records.

        Args:
            adapter:           A :class:`DataSourceAdapter` implementation.
            filepath:          Path to the local dataset file.
            previous_snapshot: Optional list of InteractionRecords from the
                               previous successful ingestion.  Used as the
                               fallback ``records`` value on failure or empty
                               result.
            config:            Optional :class:`DatasetConfig` override.  If
                               omitted, one is constructed from *adapter* and
                               *filepath* using sensible defaults.

        Returns:
            :class:`IngestionResult` describing the outcome.
        """
        # Build a DatasetConfig if the caller did not supply one.
        if config is None:
            dataset_source = getattr(adapter, "DATASET_SOURCE", "unknown")
            config = DatasetConfig(
                source_type=dataset_source,
                file_path=filepath,
                format="",
            )

        dataset_source: str = config.source_type
        snapshot_id: str = str(uuid.uuid4())
        preserved: list[InteractionRecord] = previous_snapshot or []

        # ------------------------------------------------------------------
        # Step 1 — fetch records from the adapter
        # ------------------------------------------------------------------
        raw_records: list[InteractionRecord]
        try:
            raw_records = adapter.fetch(config)
        except Exception as exc:  # noqa: BLE001
            # File-based adapters: no HTTP retry.  Log a warning and fall
            # back to the previous snapshot.
            logger.warning(
                "IngestionService: file read error for dataset '%s' "
                "(file: '%s'): %s — preserving previous snapshot (%d records)",
                dataset_source,
                filepath,
                exc,
                len(preserved),
            )
            result = IngestionResult(
                records=preserved,
                snapshot_id=snapshot_id,
                dataset_source=dataset_source,
                record_count=len(preserved),
                duplicate_count=0,
                status="failed",
                error=str(exc),
            )
            self._status.last_result = result
            return result

        # ------------------------------------------------------------------
        # Step 2 — validate records (Requirements 9.1–9.5)
        # ------------------------------------------------------------------
        validated_records, invalid_count = _validate_records(raw_records)

        # ------------------------------------------------------------------
        # Step 3 — deduplicate
        # ------------------------------------------------------------------
        unique_records, duplicate_count = _deduplicate(validated_records)

        # ------------------------------------------------------------------
        # Step 4 — guard: zero valid records
        # ------------------------------------------------------------------
        if not unique_records:
            logger.warning(
                "IngestionService: zero valid records after deduplication "
                "for dataset '%s' (file: '%s') — preserving previous "
                "snapshot (%d records)",
                dataset_source,
                filepath,
                len(preserved),
            )
            result = IngestionResult(
                records=preserved,
                snapshot_id=snapshot_id,
                dataset_source=dataset_source,
                record_count=len(preserved),
                duplicate_count=duplicate_count,
                status="empty",
                error=None,
                invalid_count=invalid_count,
            )
            self._status.last_result = result
            return result

        # ------------------------------------------------------------------
        # Step 5 — success
        # ------------------------------------------------------------------
        logger.info(
            "IngestionService: ingestion complete for dataset '%s' — "
            "%d records loaded, %d duplicates discarded, %d invalid rejected",
            dataset_source,
            len(unique_records),
            duplicate_count,
            invalid_count,
        )
        result = IngestionResult(
            records=unique_records,
            snapshot_id=snapshot_id,
            dataset_source=dataset_source,
            record_count=len(unique_records),
            duplicate_count=duplicate_count,
            status="success",
            error=None,
            invalid_count=invalid_count,
        )
        self._status.last_result = result
        return result

    def get_status(self) -> IngestionStatus:
        """Return the status object tracking the last ingestion result.

        Returns:
            :class:`IngestionStatus` with :attr:`~IngestionStatus.last_result`
            set to the most recent :class:`IngestionResult`, or ``None`` if
            :meth:`ingest` has never been called.
        """
        return self._status
