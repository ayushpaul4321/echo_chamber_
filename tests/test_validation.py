"""Tests for ingestion/validation.py — validate_record().

Covers all general ingestion rules (Requirements 9.1–9.5) and wiki-RfA
specific rules (votePolarity, voteResult bounds).

Unit tests:
- Valid record → (True, "")
- Empty sourceUserId → rejected
- Empty targetUserId → rejected
- Self-loop (source == target) → rejected
- Unrecognised interactionType → rejected
- Future timestamp → rejected
- Past timestamp → accepted
- Absent timestamp (Congress-style) → accepted
- Naive past datetime (no tzinfo) → accepted
- Wiki-RfA: invalid votePolarity (0, 2, -2) → rejected
- Wiki-RfA: valid votePolarity (+1, -1) → accepted
- Wiki-RfA: invalid voteResult (2, -1) → rejected
- Wiki-RfA: valid voteResult (0, 1) → accepted
- Wiki-RfA votePolarity/voteResult checks only for datasetSource == "wiki_rfa"
- Non-wiki_rfa record with votePolarity set → accepted (rule not applied)

Integration tests (IngestionService):
- Invalid records are counted in invalid_count
- Invalid records do not appear in result.records
- Valid records pass through unchanged
- Mix of valid/invalid records: counts are correct
- Log warning emitted for each invalid record with record id and reason

References: Requirements 9.1–9.5
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from graph.models import InteractionRecord, InteractionType
from ingestion.adapters import DatasetConfig
from ingestion.service import IngestionService, IngestionResult
from ingestion.validation import validate_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAST = datetime(2020, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
_FUTURE = datetime.now(timezone.utc) + timedelta(days=365)


def _record(
    source: str = "userA",
    target: str = "userB",
    interaction_type: InteractionType = InteractionType.HYPERLINK,
    dataset: str = "reddit_title",
    timestamp: datetime | None = _PAST,
    vote_polarity: int | None = None,
    vote_result: int | None = None,
) -> InteractionRecord:
    """Build a valid InteractionRecord; caller overrides individual fields."""
    return InteractionRecord(
        id=str(uuid.uuid4()),
        sourceUserId=source,
        targetUserId=target,
        interactionType=interaction_type,
        datasetSource=dataset,
        timestamp=timestamp,
        votePolarity=vote_polarity,
        voteResult=vote_result,
    )


def _wiki_record(
    source: str = "voter1",
    target: str = "candidate1",
    timestamp: datetime | None = _PAST,
    vote_polarity: int | None = 1,
    vote_result: int | None = 1,
) -> InteractionRecord:
    """Build a valid wiki_rfa InteractionRecord."""
    return InteractionRecord(
        id=str(uuid.uuid4()),
        sourceUserId=source,
        targetUserId=target,
        interactionType=InteractionType.VOTE,
        datasetSource="wiki_rfa",
        timestamp=timestamp,
        votePolarity=vote_polarity,
        voteResult=vote_result,
    )


# ---------------------------------------------------------------------------
# validate_record() — valid cases
# ---------------------------------------------------------------------------


class TestValidateRecordValid:
    """Valid records should return (True, "")."""

    def test_valid_reddit_record(self) -> None:
        record = _record()
        is_valid, reason = validate_record(record)
        assert is_valid is True
        assert reason == ""

    def test_valid_congress_record_no_timestamp(self) -> None:
        """Congress records have no timestamp — must be accepted."""
        record = _record(dataset="congress", timestamp=None)
        is_valid, reason = validate_record(record)
        assert is_valid is True
        assert reason == ""

    def test_valid_wiki_rfa_positive_vote(self) -> None:
        record = _wiki_record(vote_polarity=1, vote_result=1)
        is_valid, reason = validate_record(record)
        assert is_valid is True
        assert reason == ""

    def test_valid_wiki_rfa_negative_vote(self) -> None:
        record = _wiki_record(vote_polarity=-1, vote_result=0)
        is_valid, reason = validate_record(record)
        assert is_valid is True
        assert reason == ""

    def test_valid_past_timestamp_utc_aware(self) -> None:
        record = _record(timestamp=datetime(2019, 1, 1, tzinfo=timezone.utc))
        is_valid, _ = validate_record(record)
        assert is_valid is True

    def test_valid_naive_past_timestamp_treated_as_utc(self) -> None:
        """A naive datetime in the past should be accepted (treated as UTC)."""
        naive_past = datetime(2018, 3, 10, 8, 30, 0)  # no tzinfo
        record = InteractionRecord(
            id=str(uuid.uuid4()),
            sourceUserId="A",
            targetUserId="B",
            interactionType=InteractionType.HYPERLINK,
            datasetSource="reddit_title",
            timestamp=naive_past,
        )
        is_valid, reason = validate_record(record)
        assert is_valid is True

    def test_valid_retweet_type(self) -> None:
        record = _record(interaction_type=InteractionType.RETWEET, dataset="congress", timestamp=None)
        is_valid, _ = validate_record(record)
        assert is_valid is True

    def test_valid_vote_type(self) -> None:
        record = _wiki_record()
        is_valid, _ = validate_record(record)
        assert is_valid is True

    def test_reason_is_empty_string_on_valid(self) -> None:
        record = _record()
        _, reason = validate_record(record)
        assert reason == ""


# ---------------------------------------------------------------------------
# Requirement 9.1: empty sourceUserId / targetUserId
# ---------------------------------------------------------------------------


class TestValidateRecordEmptyUserIds:
    """Requirement 9.1: empty userId fields must be rejected."""

    def test_empty_source_user_id_rejected(self) -> None:
        """Empty sourceUserId → (False, reason)."""
        # Build manually to bypass InteractionRecord's own __post_init__ validation,
        # which also raises ValueError — we test the validator separately here.
        # We use object.__setattr__ to skip the dataclass __post_init__.
        record = object.__new__(InteractionRecord)
        object.__setattr__(record, "id", str(uuid.uuid4()))
        object.__setattr__(record, "sourceUserId", "")
        object.__setattr__(record, "targetUserId", "someUser")
        object.__setattr__(record, "interactionType", InteractionType.HYPERLINK)
        object.__setattr__(record, "datasetSource", "reddit_title")
        object.__setattr__(record, "timestamp", _PAST)
        object.__setattr__(record, "contentId", None)
        object.__setattr__(record, "topicTags", [])
        object.__setattr__(record, "sentimentScore", None)
        object.__setattr__(record, "votePolarity", None)
        object.__setattr__(record, "bodyText", None)
        object.__setattr__(record, "voteResult", None)

        is_valid, reason = validate_record(record)
        assert is_valid is False
        assert "sourceUserId" in reason

    def test_empty_target_user_id_rejected(self) -> None:
        """Empty targetUserId → (False, reason)."""
        record = object.__new__(InteractionRecord)
        object.__setattr__(record, "id", str(uuid.uuid4()))
        object.__setattr__(record, "sourceUserId", "someUser")
        object.__setattr__(record, "targetUserId", "")
        object.__setattr__(record, "interactionType", InteractionType.HYPERLINK)
        object.__setattr__(record, "datasetSource", "reddit_title")
        object.__setattr__(record, "timestamp", _PAST)
        object.__setattr__(record, "contentId", None)
        object.__setattr__(record, "topicTags", [])
        object.__setattr__(record, "sentimentScore", None)
        object.__setattr__(record, "votePolarity", None)
        object.__setattr__(record, "bodyText", None)
        object.__setattr__(record, "voteResult", None)

        is_valid, reason = validate_record(record)
        assert is_valid is False
        assert "targetUserId" in reason

    def test_empty_source_returns_false(self) -> None:
        record = object.__new__(InteractionRecord)
        for attr, val in [
            ("id", str(uuid.uuid4())), ("sourceUserId", ""),
            ("targetUserId", "B"), ("interactionType", InteractionType.HYPERLINK),
            ("datasetSource", "reddit_title"), ("timestamp", _PAST),
            ("contentId", None), ("topicTags", []), ("sentimentScore", None),
            ("votePolarity", None), ("bodyText", None), ("voteResult", None),
        ]:
            object.__setattr__(record, attr, val)
        is_valid, _ = validate_record(record)
        assert is_valid is False


# ---------------------------------------------------------------------------
# Self-loop rejection
# ---------------------------------------------------------------------------


class TestValidateRecordSelfLoop:
    """source == target must be rejected."""

    def test_self_loop_rejected(self) -> None:
        record = object.__new__(InteractionRecord)
        for attr, val in [
            ("id", str(uuid.uuid4())), ("sourceUserId", "sameUser"),
            ("targetUserId", "sameUser"), ("interactionType", InteractionType.HYPERLINK),
            ("datasetSource", "reddit_title"), ("timestamp", _PAST),
            ("contentId", None), ("topicTags", []), ("sentimentScore", None),
            ("votePolarity", None), ("bodyText", None), ("voteResult", None),
        ]:
            object.__setattr__(record, attr, val)

        is_valid, reason = validate_record(record)
        assert is_valid is False
        assert "self-loop" in reason or "sourceUserId equals targetUserId" in reason

    def test_distinct_users_not_self_loop(self) -> None:
        record = _record(source="A", target="B")
        is_valid, _ = validate_record(record)
        assert is_valid is True


# ---------------------------------------------------------------------------
# Requirement 9.2: unrecognised interactionType
# ---------------------------------------------------------------------------


class TestValidateRecordInteractionType:
    """Requirement 9.2: only recognised InteractionType enum values accepted."""

    def test_non_enum_interaction_type_rejected(self) -> None:
        """A raw string (not an InteractionType) must be rejected."""
        record = object.__new__(InteractionRecord)
        for attr, val in [
            ("id", str(uuid.uuid4())), ("sourceUserId", "A"),
            ("targetUserId", "B"), ("interactionType", "UNKNOWN_TYPE"),
            ("datasetSource", "reddit_title"), ("timestamp", _PAST),
            ("contentId", None), ("topicTags", []), ("sentimentScore", None),
            ("votePolarity", None), ("bodyText", None), ("voteResult", None),
        ]:
            object.__setattr__(record, attr, val)

        is_valid, reason = validate_record(record)
        assert is_valid is False
        assert "interactionType" in reason

    def test_none_interaction_type_rejected(self) -> None:
        """None is not a valid InteractionType."""
        record = object.__new__(InteractionRecord)
        for attr, val in [
            ("id", str(uuid.uuid4())), ("sourceUserId", "A"),
            ("targetUserId", "B"), ("interactionType", None),
            ("datasetSource", "reddit_title"), ("timestamp", _PAST),
            ("contentId", None), ("topicTags", []), ("sentimentScore", None),
            ("votePolarity", None), ("bodyText", None), ("voteResult", None),
        ]:
            object.__setattr__(record, attr, val)

        is_valid, reason = validate_record(record)
        assert is_valid is False

    def test_hyperlink_accepted(self) -> None:
        record = _record(interaction_type=InteractionType.HYPERLINK)
        is_valid, _ = validate_record(record)
        assert is_valid is True

    def test_retweet_accepted(self) -> None:
        record = _record(
            interaction_type=InteractionType.RETWEET,
            dataset="congress",
            timestamp=None,
        )
        is_valid, _ = validate_record(record)
        assert is_valid is True

    def test_vote_accepted(self) -> None:
        record = _wiki_record()
        is_valid, _ = validate_record(record)
        assert is_valid is True


# ---------------------------------------------------------------------------
# Requirement 9.3: future timestamp rejected
# ---------------------------------------------------------------------------


class TestValidateRecordTimestamp:
    """Requirement 9.3: future timestamps must be rejected; past/absent accepted."""

    def test_future_timestamp_rejected(self) -> None:
        """A timestamp in the future → (False, reason)."""
        record = object.__new__(InteractionRecord)
        for attr, val in [
            ("id", str(uuid.uuid4())), ("sourceUserId", "A"),
            ("targetUserId", "B"), ("interactionType", InteractionType.HYPERLINK),
            ("datasetSource", "reddit_title"), ("timestamp", _FUTURE),
            ("contentId", None), ("topicTags", []), ("sentimentScore", None),
            ("votePolarity", None), ("bodyText", None), ("voteResult", None),
        ]:
            object.__setattr__(record, attr, val)

        is_valid, reason = validate_record(record)
        assert is_valid is False
        assert "timestamp" in reason.lower()

    def test_past_timestamp_accepted(self) -> None:
        record = _record(timestamp=_PAST)
        is_valid, _ = validate_record(record)
        assert is_valid is True

    def test_none_timestamp_accepted(self) -> None:
        """None timestamp (Congress dataset) must be accepted without error."""
        record = _record(dataset="congress", timestamp=None)
        is_valid, _ = validate_record(record)
        assert is_valid is True

    def test_far_past_timestamp_accepted(self) -> None:
        record = _record(timestamp=datetime(2000, 1, 1, tzinfo=timezone.utc))
        is_valid, _ = validate_record(record)
        assert is_valid is True

    def test_near_past_timestamp_accepted(self) -> None:
        """A timestamp 1 second in the past should be accepted."""
        near_past = datetime.now(timezone.utc) - timedelta(seconds=2)
        record = object.__new__(InteractionRecord)
        for attr, val in [
            ("id", str(uuid.uuid4())), ("sourceUserId", "A"),
            ("targetUserId", "B"), ("interactionType", InteractionType.HYPERLINK),
            ("datasetSource", "reddit_title"), ("timestamp", near_past),
            ("contentId", None), ("topicTags", []), ("sentimentScore", None),
            ("votePolarity", None), ("bodyText", None), ("voteResult", None),
        ]:
            object.__setattr__(record, attr, val)
        is_valid, _ = validate_record(record)
        assert is_valid is True


# ---------------------------------------------------------------------------
# Wiki-RfA specific: votePolarity
# ---------------------------------------------------------------------------


class TestValidateRecordWikiVotePolarity:
    """votePolarity must be +1 or -1 for wiki_rfa records."""

    def test_vote_polarity_zero_rejected(self) -> None:
        """votePolarity=0 is not a valid wiki-RfA vote."""
        record = object.__new__(InteractionRecord)
        for attr, val in [
            ("id", str(uuid.uuid4())), ("sourceUserId", "voter"),
            ("targetUserId", "candidate"), ("interactionType", InteractionType.VOTE),
            ("datasetSource", "wiki_rfa"), ("timestamp", _PAST),
            ("contentId", None), ("topicTags", []), ("sentimentScore", None),
            ("votePolarity", 0), ("bodyText", None), ("voteResult", 1),
        ]:
            object.__setattr__(record, attr, val)

        is_valid, reason = validate_record(record)
        assert is_valid is False
        assert "votePolarity" in reason

    def test_vote_polarity_plus_two_rejected(self) -> None:
        """votePolarity=2 is out of range."""
        record = object.__new__(InteractionRecord)
        for attr, val in [
            ("id", str(uuid.uuid4())), ("sourceUserId", "voter"),
            ("targetUserId", "candidate"), ("interactionType", InteractionType.VOTE),
            ("datasetSource", "wiki_rfa"), ("timestamp", _PAST),
            ("contentId", None), ("topicTags", []), ("sentimentScore", None),
            ("votePolarity", 2), ("bodyText", None), ("voteResult", 1),
        ]:
            object.__setattr__(record, attr, val)

        is_valid, reason = validate_record(record)
        assert is_valid is False
        assert "votePolarity" in reason

    def test_vote_polarity_minus_two_rejected(self) -> None:
        """votePolarity=-2 is out of range."""
        record = object.__new__(InteractionRecord)
        for attr, val in [
            ("id", str(uuid.uuid4())), ("sourceUserId", "voter"),
            ("targetUserId", "candidate"), ("interactionType", InteractionType.VOTE),
            ("datasetSource", "wiki_rfa"), ("timestamp", _PAST),
            ("contentId", None), ("topicTags", []), ("sentimentScore", None),
            ("votePolarity", -2), ("bodyText", None), ("voteResult", 1),
        ]:
            object.__setattr__(record, attr, val)

        is_valid, reason = validate_record(record)
        assert is_valid is False

    def test_vote_polarity_plus_one_accepted(self) -> None:
        record = _wiki_record(vote_polarity=1)
        is_valid, _ = validate_record(record)
        assert is_valid is True

    def test_vote_polarity_minus_one_accepted(self) -> None:
        record = _wiki_record(vote_polarity=-1)
        is_valid, _ = validate_record(record)
        assert is_valid is True

    def test_vote_polarity_none_accepted(self) -> None:
        """None votePolarity on a wiki_rfa record is allowed (rule only applies
        when votePolarity is set)."""
        record = _wiki_record(vote_polarity=None)
        is_valid, _ = validate_record(record)
        assert is_valid is True

    def test_vote_polarity_rule_not_applied_to_non_wiki_rfa(self) -> None:
        """votePolarity is not validated for non wiki_rfa records."""
        record = object.__new__(InteractionRecord)
        for attr, val in [
            ("id", str(uuid.uuid4())), ("sourceUserId", "A"),
            ("targetUserId", "B"), ("interactionType", InteractionType.HYPERLINK),
            ("datasetSource", "reddit_title"), ("timestamp", _PAST),
            ("contentId", None), ("topicTags", []), ("sentimentScore", None),
            # votePolarity=5 on a non-wiki_rfa record: validator must NOT reject
            ("votePolarity", 5), ("bodyText", None), ("voteResult", None),
        ]:
            object.__setattr__(record, attr, val)

        is_valid, _ = validate_record(record)
        assert is_valid is True


# ---------------------------------------------------------------------------
# Wiki-RfA specific: voteResult
# ---------------------------------------------------------------------------


class TestValidateRecordWikiVoteResult:
    """voteResult must be 0 or 1 for wiki_rfa records."""

    def test_vote_result_two_rejected(self) -> None:
        """voteResult=2 is out of range."""
        record = object.__new__(InteractionRecord)
        for attr, val in [
            ("id", str(uuid.uuid4())), ("sourceUserId", "voter"),
            ("targetUserId", "candidate"), ("interactionType", InteractionType.VOTE),
            ("datasetSource", "wiki_rfa"), ("timestamp", _PAST),
            ("contentId", None), ("topicTags", []), ("sentimentScore", None),
            ("votePolarity", 1), ("bodyText", None), ("voteResult", 2),
        ]:
            object.__setattr__(record, attr, val)

        is_valid, reason = validate_record(record)
        assert is_valid is False
        assert "voteResult" in reason

    def test_vote_result_minus_one_rejected(self) -> None:
        """voteResult=-1 is not a valid adminship result."""
        record = object.__new__(InteractionRecord)
        for attr, val in [
            ("id", str(uuid.uuid4())), ("sourceUserId", "voter"),
            ("targetUserId", "candidate"), ("interactionType", InteractionType.VOTE),
            ("datasetSource", "wiki_rfa"), ("timestamp", _PAST),
            ("contentId", None), ("topicTags", []), ("sentimentScore", None),
            ("votePolarity", 1), ("bodyText", None), ("voteResult", -1),
        ]:
            object.__setattr__(record, attr, val)

        is_valid, reason = validate_record(record)
        assert is_valid is False
        assert "voteResult" in reason

    def test_vote_result_zero_accepted(self) -> None:
        record = _wiki_record(vote_result=0)
        is_valid, _ = validate_record(record)
        assert is_valid is True

    def test_vote_result_one_accepted(self) -> None:
        record = _wiki_record(vote_result=1)
        is_valid, _ = validate_record(record)
        assert is_valid is True

    def test_vote_result_none_accepted(self) -> None:
        """None voteResult is allowed (rule only applies when set)."""
        record = _wiki_record(vote_result=None)
        is_valid, _ = validate_record(record)
        assert is_valid is True

    def test_vote_result_rule_not_applied_to_non_wiki_rfa(self) -> None:
        """voteResult is not validated for non wiki_rfa records."""
        record = object.__new__(InteractionRecord)
        for attr, val in [
            ("id", str(uuid.uuid4())), ("sourceUserId", "A"),
            ("targetUserId", "B"), ("interactionType", InteractionType.HYPERLINK),
            ("datasetSource", "reddit_title"), ("timestamp", _PAST),
            ("contentId", None), ("topicTags", []), ("sentimentScore", None),
            ("votePolarity", None), ("bodyText", None), ("voteResult", 99),
        ]:
            object.__setattr__(record, attr, val)

        is_valid, _ = validate_record(record)
        assert is_valid is True


# ---------------------------------------------------------------------------
# Requirement 9.5: return values
# ---------------------------------------------------------------------------


class TestValidateRecordReturnValues:
    """validate_record always returns a 2-tuple (bool, str)."""

    def test_valid_returns_true_empty_string(self) -> None:
        is_valid, reason = validate_record(_record())
        assert is_valid is True
        assert isinstance(reason, str)
        assert reason == ""

    def test_invalid_returns_false_non_empty_string(self) -> None:
        record = object.__new__(InteractionRecord)
        for attr, val in [
            ("id", str(uuid.uuid4())), ("sourceUserId", ""),
            ("targetUserId", "B"), ("interactionType", InteractionType.HYPERLINK),
            ("datasetSource", "reddit_title"), ("timestamp", _PAST),
            ("contentId", None), ("topicTags", []), ("sentimentScore", None),
            ("votePolarity", None), ("bodyText", None), ("voteResult", None),
        ]:
            object.__setattr__(record, attr, val)

        is_valid, reason = validate_record(record)
        assert is_valid is False
        assert isinstance(reason, str)
        assert len(reason) > 0


# ---------------------------------------------------------------------------
# IngestionService integration — invalid records counted and logged
# ---------------------------------------------------------------------------


class _FakeAdapter:
    """Stub adapter that returns a fixed list of records without file I/O."""

    DATASET_SOURCE = "reddit_title"

    def __init__(self, records: list[InteractionRecord]) -> None:
        self._records = records

    def fetch(self, config: DatasetConfig) -> list[InteractionRecord]:
        return list(self._records)

    def normalize(self, raw: dict) -> InteractionRecord | None:
        return None


def _make_invalid_record(source: str = "bad") -> InteractionRecord:
    """Craft an invalid record that bypasses __post_init__ using object.__new__."""
    record = object.__new__(InteractionRecord)
    for attr, val in [
        ("id", str(uuid.uuid4())), ("sourceUserId", ""),  # empty — invalid
        ("targetUserId", "B"), ("interactionType", InteractionType.HYPERLINK),
        ("datasetSource", "reddit_title"), ("timestamp", _PAST),
        ("contentId", None), ("topicTags", []), ("sentimentScore", None),
        ("votePolarity", None), ("bodyText", None), ("voteResult", None),
    ]:
        object.__setattr__(record, attr, val)
    return record


class TestIngestionServiceValidation:
    """IngestionService counts and drops invalid records without halting."""

    def _config(self) -> DatasetConfig:
        return DatasetConfig(source_type="reddit_title", file_path="/fake.tsv", format="tsv")

    def test_invalid_record_counted_in_invalid_count(self) -> None:
        """invalid_count reflects the number of records that failed validation."""
        valid = _record()
        invalid = _make_invalid_record()
        service = IngestionService()
        result = service.ingest(_FakeAdapter([valid, invalid]), "/f.tsv", config=self._config())

        assert result.invalid_count == 1

    def test_invalid_record_not_in_result_records(self) -> None:
        """Records failing validation must not appear in result.records."""
        valid = _record()
        invalid = _make_invalid_record()
        service = IngestionService()
        result = service.ingest(_FakeAdapter([valid, invalid]), "/f.tsv", config=self._config())

        ids = {r.id for r in result.records}
        assert invalid.id not in ids

    def test_valid_record_still_included(self) -> None:
        """Valid records must pass through even when invalid ones are present."""
        valid = _record()
        invalid = _make_invalid_record()
        service = IngestionService()
        result = service.ingest(_FakeAdapter([valid, invalid]), "/f.tsv", config=self._config())

        ids = {r.id for r in result.records}
        assert valid.id in ids

    def test_all_invalid_records_produces_empty_status(self) -> None:
        """When all records are invalid, result status is 'empty'."""
        invalid1 = _make_invalid_record()
        invalid2 = _make_invalid_record()
        service = IngestionService()
        result = service.ingest(
            _FakeAdapter([invalid1, invalid2]), "/f.tsv", config=self._config()
        )

        assert result.status == "empty"
        assert result.invalid_count == 2

    def test_multiple_invalid_records_all_counted(self) -> None:
        """invalid_count accumulates across all invalid records in the batch."""
        valid = _record()
        invalids = [_make_invalid_record() for _ in range(5)]
        service = IngestionService()
        result = service.ingest(
            _FakeAdapter([valid] + invalids), "/f.tsv", config=self._config()
        )

        assert result.invalid_count == 5
        assert result.record_count == 1

    def test_warning_logged_per_invalid_record(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Requirement 9.5: each rejected record must generate a WARNING log."""
        valid = _record()
        invalid = _make_invalid_record()
        service = IngestionService()

        with caplog.at_level(logging.WARNING, logger="ingestion.service"):
            service.ingest(_FakeAdapter([valid, invalid]), "/f.tsv", config=self._config())

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        # At least one warning must mention the invalid record's id
        assert any(invalid.id in msg for msg in warning_messages)

    def test_warning_contains_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        """The WARNING log entry must include a rejection reason."""
        invalid = _make_invalid_record()
        service = IngestionService()

        with caplog.at_level(logging.WARNING, logger="ingestion.service"):
            service.ingest(_FakeAdapter([invalid]), "/f.tsv", config=self._config())

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        # At least one warning must contain a non-trivial reason substring
        assert any("sourceUserId" in msg or "reason" in msg for msg in warning_messages)

    def test_ingestion_continues_after_invalid_record(self) -> None:
        """Processing must not halt when an invalid record is encountered."""
        records = [
            _make_invalid_record(),
            _record(source="A", target="B"),
            _make_invalid_record(),
            _record(source="C", target="D"),
        ]
        service = IngestionService()
        result = service.ingest(_FakeAdapter(records), "/f.tsv", config=self._config())

        # Both valid records must be present
        assert result.record_count == 2
        assert result.invalid_count == 2
        assert result.status == "success"

    def test_zero_invalid_records_means_invalid_count_zero(self) -> None:
        """When all records are valid, invalid_count must be 0."""
        records = [_record(source="A", target="B"), _record(source="C", target="D")]
        service = IngestionService()
        result = service.ingest(_FakeAdapter(records), "/f.tsv", config=self._config())

        assert result.invalid_count == 0
