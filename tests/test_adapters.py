"""Tests for ingestion/adapters.py — RedditTitleAdapter and RedditBodyAdapter.

Covers:
- RedditBodyAdapter.normalize(): valid row produces correct InteractionRecord
- RedditBodyAdapter: datasetSource = "reddit_body"
- RedditBodyAdapter: PROPERTIES parsed as JSON (title + body extraction)
- RedditBodyAdapter: PROPERTIES non-JSON raw value stored as bodyText
- RedditBodyAdapter: missing / NaN PROPERTIES → bodyText is None
- RedditBodyAdapter: future-timestamp rejection
- RedditBodyAdapter: unparseable-timestamp rejection
- RedditBodyAdapter: InteractionRecord validation errors (self-loop, empty userId)
  are caught and return None
- RedditBodyAdapter.subreddit_text_corpus populated during fetch()
- RedditBodyAdapter.fetch() resets corpus between calls
- RedditBodyAdapter: bodyText is NOT the empty string when PROPERTIES present

References: Requirements 1.1, 1.2
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from ingestion.adapters import DatasetConfig, RedditBodyAdapter, RedditTitleAdapter
from graph.models import InteractionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAST_TS = "2020-06-15 12:00:00"
_FUTURE_TS = (datetime.now(timezone.utc) + timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S")


def _make_row(
    source: str = "subreddit_a",
    target: str = "subreddit_b",
    post_id: str = "post_001",
    timestamp: str = _PAST_TS,
    sentiment: float = 0.5,
    properties: object = None,
) -> dict:
    """Build a raw row dict matching the TSV column schema."""
    return {
        "SOURCE_SUBREDDIT": source,
        "TARGET_SUBREDDIT": target,
        "POST_ID": post_id,
        "TIMESTAMP": timestamp,
        "LINK_SENTIMENT": sentiment,
        "PROPERTIES": properties,
    }


def _make_tsv_content(rows: list[dict]) -> str:
    """Return a TSV string with header + rows for use in fetch() tests."""
    cols = ["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "POST_ID", "TIMESTAMP", "LINK_SENTIMENT", "PROPERTIES"]
    lines = ["\t".join(cols)]
    for row in rows:
        lines.append("\t".join(str(row.get(c, "")) for c in cols))
    return "\n".join(lines) + "\n"


def _write_tsv(rows: list[dict]) -> str:
    """Write rows to a temp TSV file and return the path."""
    content = _make_tsv_content(rows)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False)
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# RedditBodyAdapter — normalize(): basic valid record
# ---------------------------------------------------------------------------


def test_body_adapter_normalize_basic_valid_row() -> None:
    """normalize() on a valid row returns an InteractionRecord."""
    adapter = RedditBodyAdapter()
    row = _make_row()
    record = adapter.normalize(row)

    assert record is not None
    assert record.sourceUserId == "subreddit_a"
    assert record.targetUserId == "subreddit_b"
    assert record.contentId == "post_001"
    assert record.interactionType == InteractionType.HYPERLINK
    assert record.datasetSource == "reddit_body"
    assert record.sentimentScore == 0.5


def test_body_adapter_dataset_source_is_reddit_body() -> None:
    """datasetSource must be 'reddit_body', not 'reddit_title'."""
    adapter = RedditBodyAdapter()
    record = adapter.normalize(_make_row())
    assert record is not None
    assert record.datasetSource == "reddit_body"


# ---------------------------------------------------------------------------
# PROPERTIES → bodyText extraction
# ---------------------------------------------------------------------------


def test_body_adapter_properties_json_with_title_and_body() -> None:
    """JSON PROPERTIES with both 'title' and 'body' keys → concatenated bodyText."""
    adapter = RedditBodyAdapter()
    props = json.dumps({"title": "My post title", "body": "The body text here."})
    row = _make_row(properties=props)
    record = adapter.normalize(row)

    assert record is not None
    assert record.bodyText == "My post title The body text here."


def test_body_adapter_properties_json_title_only() -> None:
    """JSON PROPERTIES with only 'title' → bodyText = title string."""
    adapter = RedditBodyAdapter()
    props = json.dumps({"title": "Just a title"})
    row = _make_row(properties=props)
    record = adapter.normalize(row)

    assert record is not None
    assert record.bodyText == "Just a title"


def test_body_adapter_properties_json_body_only() -> None:
    """JSON PROPERTIES with only 'body' → bodyText = body string."""
    adapter = RedditBodyAdapter()
    props = json.dumps({"body": "Just body text."})
    row = _make_row(properties=props)
    record = adapter.normalize(row)

    assert record is not None
    assert record.bodyText == "Just body text."


def test_body_adapter_properties_json_empty_keys() -> None:
    """JSON PROPERTIES with empty string values for both keys → bodyText is None."""
    adapter = RedditBodyAdapter()
    props = json.dumps({"title": "", "body": ""})
    row = _make_row(properties=props)
    record = adapter.normalize(row)

    assert record is not None
    assert record.bodyText is None


def test_body_adapter_properties_json_missing_text_keys() -> None:
    """JSON dict with no 'title' or 'body' key → bodyText is None."""
    adapter = RedditBodyAdapter()
    props = json.dumps({"other_key": "some value"})
    row = _make_row(properties=props)
    record = adapter.normalize(row)

    assert record is not None
    assert record.bodyText is None


def test_body_adapter_properties_non_json_numeric_vector() -> None:
    """Non-JSON PROPERTIES (SNAP numeric vector) → raw string stored as bodyText."""
    adapter = RedditBodyAdapter()
    numeric_props = "345.0,298.0,0.756,0.017,0.087,0.150"
    row = _make_row(properties=numeric_props)
    record = adapter.normalize(row)

    assert record is not None
    assert record.bodyText == numeric_props


def test_body_adapter_properties_none_gives_none_body_text() -> None:
    """None PROPERTIES → bodyText is None."""
    adapter = RedditBodyAdapter()
    row = _make_row(properties=None)
    record = adapter.normalize(row)

    assert record is not None
    assert record.bodyText is None


def test_body_adapter_properties_nan_gives_none_body_text() -> None:
    """NaN PROPERTIES (as pandas would produce for missing values) → bodyText is None."""
    adapter = RedditBodyAdapter()
    row = _make_row(properties=float("nan"))
    record = adapter.normalize(row)

    assert record is not None
    assert record.bodyText is None


def test_body_adapter_properties_empty_string_gives_none_body_text() -> None:
    """Empty-string PROPERTIES → bodyText is None."""
    adapter = RedditBodyAdapter()
    row = _make_row(properties="")
    record = adapter.normalize(row)

    assert record is not None
    assert record.bodyText is None


# ---------------------------------------------------------------------------
# Timestamp rejection
# ---------------------------------------------------------------------------


def test_body_adapter_future_timestamp_rejected() -> None:
    """Rows with future timestamps must be rejected (return None)."""
    adapter = RedditBodyAdapter()
    row = _make_row(timestamp=_FUTURE_TS)
    assert adapter.normalize(row) is None


def test_body_adapter_unparseable_timestamp_rejected() -> None:
    """Rows with non-datetime TIMESTAMP values must be rejected."""
    adapter = RedditBodyAdapter()
    row = _make_row(timestamp="not-a-date")
    assert adapter.normalize(row) is None


def test_body_adapter_valid_past_timestamp_accepted() -> None:
    """A clearly past timestamp should yield a valid record."""
    adapter = RedditBodyAdapter()
    row = _make_row(timestamp="2019-03-10 08:30:00")
    record = adapter.normalize(row)
    assert record is not None
    assert record.timestamp is not None


# ---------------------------------------------------------------------------
# Validation errors → None (not raised)
# ---------------------------------------------------------------------------


def test_body_adapter_self_loop_returns_none() -> None:
    """Same source and target subreddit (self-loop) → normalize returns None."""
    adapter = RedditBodyAdapter()
    row = _make_row(source="same_sub", target="same_sub")
    assert adapter.normalize(row) is None


def test_body_adapter_empty_source_returns_none() -> None:
    """Empty SOURCE_SUBREDDIT → normalize returns None."""
    adapter = RedditBodyAdapter()
    row = _make_row(source="")
    assert adapter.normalize(row) is None


def test_body_adapter_empty_target_returns_none() -> None:
    """Empty TARGET_SUBREDDIT → normalize returns None."""
    adapter = RedditBodyAdapter()
    row = _make_row(target="")
    assert adapter.normalize(row) is None


# ---------------------------------------------------------------------------
# Sentiment score handling
# ---------------------------------------------------------------------------


def test_body_adapter_negative_sentiment() -> None:
    """Negative LINK_SENTIMENT is stored correctly."""
    adapter = RedditBodyAdapter()
    row = _make_row(sentiment=-1.0)
    record = adapter.normalize(row)
    assert record is not None
    assert record.sentimentScore == -1.0


def test_body_adapter_zero_sentiment() -> None:
    """Zero LINK_SENTIMENT is stored correctly."""
    adapter = RedditBodyAdapter()
    row = _make_row(sentiment=0.0)
    record = adapter.normalize(row)
    assert record is not None
    assert record.sentimentScore == 0.0


# ---------------------------------------------------------------------------
# subreddit_text_corpus via fetch()
# ---------------------------------------------------------------------------


def test_body_adapter_corpus_populated_after_fetch(tmp_path: Path) -> None:
    """After fetch(), subreddit_text_corpus maps subreddits to body text lists."""
    rows = [
        _make_row(source="politics", target="news",     properties="text A"),
        _make_row(source="politics", target="worldnews", post_id="p2", properties="text B"),
        _make_row(source="gaming",   target="leagueoflegends", post_id="p3", properties="text C"),
    ]
    tsv_path = str(tmp_path / "body.tsv")
    Path(tsv_path).write_text(_make_tsv_content(rows))

    adapter = RedditBodyAdapter()
    config = DatasetConfig(source_type="reddit_body", file_path=tsv_path, format="tsv")
    records = adapter.fetch(config)

    assert len(records) == 3
    assert "politics" in adapter.subreddit_text_corpus
    assert adapter.subreddit_text_corpus["politics"] == ["text A", "text B"]
    assert "gaming" in adapter.subreddit_text_corpus
    assert adapter.subreddit_text_corpus["gaming"] == ["text C"]


def test_body_adapter_corpus_excludes_rows_with_no_body_text(tmp_path: Path) -> None:
    """Rows with None bodyText should not appear in subreddit_text_corpus."""
    rows = [
        _make_row(source="science", target="askscience", properties="valid text"),
        _make_row(source="science", target="news", post_id="p2", properties=None),
    ]
    tsv_path = str(tmp_path / "body.tsv")
    Path(tsv_path).write_text(_make_tsv_content(rows))

    adapter = RedditBodyAdapter()
    config = DatasetConfig(source_type="reddit_body", file_path=tsv_path, format="tsv")
    adapter.fetch(config)

    assert adapter.subreddit_text_corpus["science"] == ["valid text"]
    assert len(adapter.subreddit_text_corpus["science"]) == 1


def test_body_adapter_corpus_reset_between_fetch_calls(tmp_path: Path) -> None:
    """Calling fetch() a second time must reset the corpus, not accumulate."""
    rows = [_make_row(source="gaming", target="pcgaming", properties="first run")]
    tsv_path = str(tmp_path / "body.tsv")
    Path(tsv_path).write_text(_make_tsv_content(rows))

    adapter = RedditBodyAdapter()
    config = DatasetConfig(source_type="reddit_body", file_path=tsv_path, format="tsv")

    adapter.fetch(config)
    first_corpus = dict(adapter.subreddit_text_corpus)

    # Second fetch with same file — corpus should be equivalent, not doubled
    adapter.fetch(config)
    assert adapter.subreddit_text_corpus == first_corpus


def test_body_adapter_corpus_rejects_excluded_rows_not_in_corpus(tmp_path: Path) -> None:
    """Rejected rows (future timestamp) must not appear in subreddit_text_corpus."""
    rows = [
        _make_row(source="valid_sub", target="other_sub", properties="good text"),
        _make_row(source="future_sub", target="other_sub", post_id="p2",
                  timestamp=_FUTURE_TS, properties="future text"),
    ]
    tsv_path = str(tmp_path / "body.tsv")
    Path(tsv_path).write_text(_make_tsv_content(rows))

    adapter = RedditBodyAdapter()
    config = DatasetConfig(source_type="reddit_body", file_path=tsv_path, format="tsv")
    adapter.fetch(config)

    assert "future_sub" not in adapter.subreddit_text_corpus
    assert "valid_sub" in adapter.subreddit_text_corpus


# ---------------------------------------------------------------------------
# fetch() returns correct records
# ---------------------------------------------------------------------------


def test_body_adapter_fetch_returns_list_of_records(tmp_path: Path) -> None:
    """fetch() returns an InteractionRecord for every valid row."""
    rows = [
        _make_row(source="sub_a", target="sub_b"),
        _make_row(source="sub_c", target="sub_d", post_id="p2"),
    ]
    tsv_path = str(tmp_path / "body.tsv")
    Path(tsv_path).write_text(_make_tsv_content(rows))

    adapter = RedditBodyAdapter()
    config = DatasetConfig(source_type="reddit_body", file_path=tsv_path, format="tsv")
    records = adapter.fetch(config)

    assert len(records) == 2
    for r in records:
        assert r.datasetSource == "reddit_body"
        assert r.interactionType == InteractionType.HYPERLINK


def test_body_adapter_fetch_drops_future_timestamp_rows(tmp_path: Path) -> None:
    """fetch() silently drops rows with future timestamps."""
    rows = [
        _make_row(source="valid_a", target="valid_b"),
        _make_row(source="future_a", target="future_b", post_id="p2", timestamp=_FUTURE_TS),
    ]
    tsv_path = str(tmp_path / "body.tsv")
    Path(tsv_path).write_text(_make_tsv_content(rows))

    adapter = RedditBodyAdapter()
    config = DatasetConfig(source_type="reddit_body", file_path=tsv_path, format="tsv")
    records = adapter.fetch(config)

    assert len(records) == 1
    assert records[0].sourceUserId == "valid_a"


def test_body_adapter_initial_corpus_is_empty() -> None:
    """A freshly constructed adapter has an empty subreddit_text_corpus."""
    adapter = RedditBodyAdapter()
    assert adapter.subreddit_text_corpus == {}


# ---------------------------------------------------------------------------
# _extract_body_text — unit tests for the static helper
# ---------------------------------------------------------------------------


def test_extract_body_text_json_dict_with_both_keys() -> None:
    result = RedditBodyAdapter._extract_body_text(json.dumps({"title": "T", "body": "B"}))
    assert result == "T B"


def test_extract_body_text_json_dict_title_only() -> None:
    result = RedditBodyAdapter._extract_body_text(json.dumps({"title": "Only title"}))
    assert result == "Only title"


def test_extract_body_text_json_dict_body_only() -> None:
    result = RedditBodyAdapter._extract_body_text(json.dumps({"body": "Only body"}))
    assert result == "Only body"


def test_extract_body_text_non_json_string() -> None:
    result = RedditBodyAdapter._extract_body_text("1.0,2.0,3.0")
    assert result == "1.0,2.0,3.0"


def test_extract_body_text_none_returns_none() -> None:
    assert RedditBodyAdapter._extract_body_text(None) is None


def test_extract_body_text_nan_returns_none() -> None:
    assert RedditBodyAdapter._extract_body_text(float("nan")) is None


def test_extract_body_text_empty_string_returns_none() -> None:
    assert RedditBodyAdapter._extract_body_text("") is None


def test_extract_body_text_whitespace_returns_none() -> None:
    assert RedditBodyAdapter._extract_body_text("   ") is None


def test_extract_body_text_json_array_fallback_to_raw() -> None:
    """A JSON array (not a dict) falls back to raw string storage."""
    raw = "[1.0, 2.0, 3.0]"
    result = RedditBodyAdapter._extract_body_text(raw)
    assert result == raw


# ===========================================================================
# CongressNetworkAdapter Tests
# ===========================================================================
"""Tests for CongressNetworkAdapter.

Covers:
- normalize(): correct field mapping (nodeA/nodeB → username, weight passthrough)
- normalize(): Congress ID resolution to username via id_to_username lookup
- normalize(): pre_normalized = True flag (PRE_NORMALIZED class attribute)
- normalize(): interactionType = RETWEET, datasetSource = "congress"
- normalize(): timestamp = None (no timestamp in Congress dataset)
- normalize(): deduplication key is (sourceUserId, targetUserId) only
- fetch(): loads JSON + edgelist, resolves IDs to usernames
- fetch(): rejects malformed lines gracefully
- fetch(): unknown node IDs produce no record (returns None)

References: Requirements 1.1, 1.2
"""

import json
import os
import tempfile
from pathlib import Path

from ingestion.adapters import CongressNetworkAdapter, DatasetConfig
from graph.models import InteractionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_congress_files(
    tmp_path: Path,
    username_list: list[str],
    edgelist_lines: list[str],
) -> tuple[str, str]:
    """Write minimal congress JSON + edgelist fixtures.

    Returns (edgelist_path, json_path).
    """
    # congress_network_data.json — wrap usernameList in the expected array/object
    json_data = [{"usernameList": username_list}]
    json_path = tmp_path / "congress_network_data.json"
    json_path.write_text(json.dumps(json_data), encoding="utf-8")

    # congress.edgelist
    edgelist_path = tmp_path / "congress.edgelist"
    edgelist_path.write_text("\n".join(edgelist_lines) + "\n", encoding="utf-8")

    return str(edgelist_path), str(json_path)


# ---------------------------------------------------------------------------
# normalize() — field mapping
# ---------------------------------------------------------------------------


class TestCongressNormalizeFieldMapping:
    """normalize() maps fields correctly when _id_to_username is pre-populated."""

    def setup_method(self) -> None:
        self.adapter = CongressNetworkAdapter()
        # Manually populate the lookup map so we don't need file I/O
        self.adapter._id_to_username = {0: "alice", 1: "bob", 2: "carol"}

    def test_source_user_id_resolved_from_node_a(self) -> None:
        record = self.adapter.normalize({"nodeA": 0, "nodeB": 1, "weight": 0.5})
        assert record is not None
        assert record.sourceUserId == "alice"

    def test_target_user_id_resolved_from_node_b(self) -> None:
        record = self.adapter.normalize({"nodeA": 0, "nodeB": 1, "weight": 0.5})
        assert record is not None
        assert record.targetUserId == "bob"

    def test_weight_stored_as_sentiment_score(self) -> None:
        """Weight (transmission probability) is stored in sentimentScore."""
        record = self.adapter.normalize({"nodeA": 0, "nodeB": 1, "weight": 0.25})
        assert record is not None
        assert record.sentimentScore == 0.25

    def test_interaction_type_is_retweet(self) -> None:
        record = self.adapter.normalize({"nodeA": 0, "nodeB": 1, "weight": 0.5})
        assert record is not None
        assert record.interactionType == InteractionType.RETWEET

    def test_dataset_source_is_congress(self) -> None:
        record = self.adapter.normalize({"nodeA": 0, "nodeB": 1, "weight": 0.5})
        assert record is not None
        assert record.datasetSource == "congress"

    def test_timestamp_is_none(self) -> None:
        """Congress dataset has no timestamp — timestamp must be None."""
        record = self.adapter.normalize({"nodeA": 0, "nodeB": 1, "weight": 0.5})
        assert record is not None
        assert record.timestamp is None

    def test_content_id_is_none(self) -> None:
        record = self.adapter.normalize({"nodeA": 0, "nodeB": 1, "weight": 0.5})
        assert record is not None
        assert record.contentId is None

    def test_topic_tags_is_empty_list(self) -> None:
        record = self.adapter.normalize({"nodeA": 0, "nodeB": 1, "weight": 0.5})
        assert record is not None
        assert record.topicTags == []

    def test_weight_zero_point_one(self) -> None:
        """Boundary: weight = 0.1 passed through unchanged."""
        record = self.adapter.normalize({"nodeA": 1, "nodeB": 2, "weight": 0.1})
        assert record is not None
        assert record.sentimentScore == 0.1

    def test_weight_one_point_zero(self) -> None:
        """Boundary: weight = 1.0 (upper bound of pre-normalized range)."""
        record = self.adapter.normalize({"nodeA": 0, "nodeB": 2, "weight": 1.0})
        assert record is not None
        assert record.sentimentScore == 1.0


# ---------------------------------------------------------------------------
# normalize() — pre_normalized flag
# ---------------------------------------------------------------------------


class TestCongressPreNormalized:
    """CongressNetworkAdapter signals pre-normalized weights."""

    def test_pre_normalized_class_attribute_is_true(self) -> None:
        """PRE_NORMALIZED class attribute must be True."""
        assert CongressNetworkAdapter.PRE_NORMALIZED is True

    def test_dataset_source_is_congress_string(self) -> None:
        """datasetSource = 'congress' is the stable identifier for pre-normalized data."""
        assert CongressNetworkAdapter.DATASET_SOURCE == "congress"


# ---------------------------------------------------------------------------
# normalize() — ID resolution failures
# ---------------------------------------------------------------------------


class TestCongressIdResolution:
    """normalize() handles unknown or missing node IDs correctly."""

    def setup_method(self) -> None:
        self.adapter = CongressNetworkAdapter()
        self.adapter._id_to_username = {0: "alice", 1: "bob"}

    def test_unknown_node_a_returns_none(self) -> None:
        """nodeA not in lookup → normalize returns None."""
        result = self.adapter.normalize({"nodeA": 99, "nodeB": 1, "weight": 0.5})
        assert result is None

    def test_unknown_node_b_returns_none(self) -> None:
        """nodeB not in lookup → normalize returns None."""
        result = self.adapter.normalize({"nodeA": 0, "nodeB": 99, "weight": 0.5})
        assert result is None

    def test_missing_node_a_key_returns_none(self) -> None:
        """Missing 'nodeA' key in raw dict → normalize returns None."""
        result = self.adapter.normalize({"nodeB": 1, "weight": 0.5})
        assert result is None

    def test_missing_node_b_key_returns_none(self) -> None:
        """Missing 'nodeB' key in raw dict → normalize returns None."""
        result = self.adapter.normalize({"nodeA": 0, "weight": 0.5})
        assert result is None

    def test_missing_weight_key_returns_none(self) -> None:
        """Missing 'weight' key in raw dict → normalize returns None."""
        result = self.adapter.normalize({"nodeA": 0, "nodeB": 1})
        assert result is None

    def test_self_loop_returns_none(self) -> None:
        """Same source and target (self-loop) → normalize returns None."""
        result = self.adapter.normalize({"nodeA": 0, "nodeB": 0, "weight": 0.5})
        assert result is None


# ---------------------------------------------------------------------------
# normalize() — deduplication key (sourceUserId, targetUserId) only
# ---------------------------------------------------------------------------


class TestCongressDeduplicationKey:
    """Dedup key is (sourceUserId, targetUserId) — no timestamp component."""

    def setup_method(self) -> None:
        self.adapter = CongressNetworkAdapter()
        self.adapter._id_to_username = {0: "alice", 1: "bob"}

    def test_records_for_same_pair_have_same_source_target(self) -> None:
        """Two normalize calls with the same nodeA/nodeB produce the same
        (sourceUserId, targetUserId) pair — callers can deduplicate on those
        two fields alone (no timestamp needed)."""
        r1 = self.adapter.normalize({"nodeA": 0, "nodeB": 1, "weight": 0.1})
        r2 = self.adapter.normalize({"nodeA": 0, "nodeB": 1, "weight": 0.2})
        assert r1 is not None and r2 is not None
        assert r1.sourceUserId == r2.sourceUserId
        assert r1.targetUserId == r2.targetUserId
        # Timestamps are both None — dedup can use (src, tgt) composite key
        assert r1.timestamp is None
        assert r2.timestamp is None


# ---------------------------------------------------------------------------
# fetch() — integration with fixture files
# ---------------------------------------------------------------------------


class TestCongressFetch:
    """fetch() resolves IDs to usernames from real fixture files."""

    def test_fetch_basic_edge(self, tmp_path: Path) -> None:
        """fetch() returns a record with resolved usernames."""
        edgelist_path, json_path = _write_congress_files(
            tmp_path,
            username_list=["alice", "bob", "carol"],
            edgelist_lines=["0 1 {'weight': 0.5}"],
        )
        adapter = CongressNetworkAdapter()
        config = DatasetConfig(
            source_type="congress",
            file_path=edgelist_path,
            format="edgelist",
            extra={"json_path": json_path},
        )
        records = adapter.fetch(config)

        assert len(records) == 1
        r = records[0]
        assert r.sourceUserId == "alice"
        assert r.targetUserId == "bob"
        assert r.sentimentScore == 0.5
        assert r.interactionType == InteractionType.RETWEET
        assert r.datasetSource == "congress"
        assert r.timestamp is None

    def test_fetch_multiple_edges(self, tmp_path: Path) -> None:
        """fetch() processes multiple edgelist lines."""
        edgelist_path, json_path = _write_congress_files(
            tmp_path,
            username_list=["alice", "bob", "carol"],
            edgelist_lines=[
                "0 1 {'weight': 0.25}",
                "1 2 {'weight': 0.75}",
                "2 0 {'weight': 0.1}",
            ],
        )
        adapter = CongressNetworkAdapter()
        config = DatasetConfig(
            source_type="congress",
            file_path=edgelist_path,
            format="edgelist",
            extra={"json_path": json_path},
        )
        records = adapter.fetch(config)
        assert len(records) == 3

    def test_fetch_resolves_node_ids_to_usernames(self, tmp_path: Path) -> None:
        """Node integer IDs are translated to usernames via usernameList."""
        edgelist_path, json_path = _write_congress_files(
            tmp_path,
            username_list=["user_zero", "user_one", "user_two"],
            edgelist_lines=["2 0 {'weight': 0.33}"],
        )
        adapter = CongressNetworkAdapter()
        config = DatasetConfig(
            source_type="congress",
            file_path=edgelist_path,
            format="edgelist",
            extra={"json_path": json_path},
        )
        records = adapter.fetch(config)
        assert len(records) == 1
        assert records[0].sourceUserId == "user_two"
        assert records[0].targetUserId == "user_zero"

    def test_fetch_skips_malformed_lines(self, tmp_path: Path) -> None:
        """Malformed lines (missing weight, non-integer IDs) are skipped."""
        edgelist_path, json_path = _write_congress_files(
            tmp_path,
            username_list=["alice", "bob"],
            edgelist_lines=[
                "0 1 {'weight': 0.5}",   # valid
                "bad line",              # malformed — no weight dict
                "0 1",                   # malformed — no weight at all
                "",                      # blank line — skipped
            ],
        )
        adapter = CongressNetworkAdapter()
        config = DatasetConfig(
            source_type="congress",
            file_path=edgelist_path,
            format="edgelist",
            extra={"json_path": json_path},
        )
        records = adapter.fetch(config)
        assert len(records) == 1
        assert records[0].sourceUserId == "alice"

    def test_fetch_skips_unknown_node_ids(self, tmp_path: Path) -> None:
        """Edges referencing unknown node IDs are skipped."""
        edgelist_path, json_path = _write_congress_files(
            tmp_path,
            username_list=["alice", "bob"],
            edgelist_lines=[
                "0 1 {'weight': 0.5}",    # valid
                "0 999 {'weight': 0.3}",  # nodeB=999 unknown
            ],
        )
        adapter = CongressNetworkAdapter()
        config = DatasetConfig(
            source_type="congress",
            file_path=edgelist_path,
            format="edgelist",
            extra={"json_path": json_path},
        )
        records = adapter.fetch(config)
        assert len(records) == 1

    def test_fetch_weight_passthrough(self, tmp_path: Path) -> None:
        """Pre-normalized weights are stored unchanged (not modified)."""
        weight_val = 0.002105263157894737
        edgelist_path, json_path = _write_congress_files(
            tmp_path,
            username_list=["alice", "bob"],
            edgelist_lines=[f"0 1 {{'weight': {weight_val}}}"],
        )
        adapter = CongressNetworkAdapter()
        config = DatasetConfig(
            source_type="congress",
            file_path=edgelist_path,
            format="edgelist",
            extra={"json_path": json_path},
        )
        records = adapter.fetch(config)
        assert len(records) == 1
        assert abs(records[0].sentimentScore - weight_val) < 1e-15

    def test_fetch_all_records_have_correct_interaction_type(self, tmp_path: Path) -> None:
        """All congress records use InteractionType.RETWEET."""
        edgelist_path, json_path = _write_congress_files(
            tmp_path,
            username_list=["a", "b", "c"],
            edgelist_lines=["0 1 {'weight': 0.1}", "1 2 {'weight': 0.2}"],
        )
        adapter = CongressNetworkAdapter()
        config = DatasetConfig(
            source_type="congress",
            file_path=edgelist_path,
            format="edgelist",
            extra={"json_path": json_path},
        )
        records = adapter.fetch(config)
        assert all(r.interactionType == InteractionType.RETWEET for r in records)

    def test_fetch_all_records_have_none_timestamp(self, tmp_path: Path) -> None:
        """All congress records must have timestamp = None."""
        edgelist_path, json_path = _write_congress_files(
            tmp_path,
            username_list=["a", "b", "c"],
            edgelist_lines=["0 1 {'weight': 0.1}", "1 2 {'weight': 0.2}"],
        )
        adapter = CongressNetworkAdapter()
        config = DatasetConfig(
            source_type="congress",
            file_path=edgelist_path,
            format="edgelist",
            extra={"json_path": json_path},
        )
        records = adapter.fetch(config)
        assert all(r.timestamp is None for r in records)

    def test_fetch_empty_edgelist_returns_empty_list(self, tmp_path: Path) -> None:
        """An edgelist with no valid lines produces an empty record list."""
        edgelist_path, json_path = _write_congress_files(
            tmp_path,
            username_list=["alice", "bob"],
            edgelist_lines=[],
        )
        adapter = CongressNetworkAdapter()
        config = DatasetConfig(
            source_type="congress",
            file_path=edgelist_path,
            format="edgelist",
            extra={"json_path": json_path},
        )
        records = adapter.fetch(config)
        assert records == []


# ---------------------------------------------------------------------------
# _load_username_map() — unit tests
# ---------------------------------------------------------------------------


class TestLoadUsernameMap:
    """_load_username_map() parses the congress_network_data.json correctly."""

    def test_loads_username_list_from_array_wrapped_json(self, tmp_path: Path) -> None:
        """JSON file wraps data object in an array — usernameList extracted."""
        json_data = [{"usernameList": ["alice", "bob", "carol"]}]
        json_file = tmp_path / "congress_network_data.json"
        json_file.write_text(json.dumps(json_data))

        result = CongressNetworkAdapter._load_username_map(str(json_file))
        assert result == {0: "alice", 1: "bob", 2: "carol"}

    def test_loads_username_list_from_plain_dict(self, tmp_path: Path) -> None:
        """JSON file is a plain dict (not array-wrapped)."""
        json_data = {"usernameList": ["x", "y"]}
        json_file = tmp_path / "congress_network_data.json"
        json_file.write_text(json.dumps(json_data))

        result = CongressNetworkAdapter._load_username_map(str(json_file))
        assert result == {0: "x", 1: "y"}

    def test_missing_file_returns_empty_dict(self) -> None:
        """Non-existent file → returns empty dict (no exception raised)."""
        result = CongressNetworkAdapter._load_username_map("/nonexistent/path.json")
        assert result == {}

    def test_empty_username_list_returns_empty_dict(self, tmp_path: Path) -> None:
        """usernameList = [] → returns empty dict."""
        json_data = [{"usernameList": []}]
        json_file = tmp_path / "congress_network_data.json"
        json_file.write_text(json.dumps(json_data))

        result = CongressNetworkAdapter._load_username_map(str(json_file))
        assert result == {}
