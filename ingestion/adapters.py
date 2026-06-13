"""Ingestion Layer adapters for the Echo Chamber Detector pipeline.

Provides a pluggable adapter interface (DataSourceAdapter) and concrete
implementations for each supported dataset. Task 2.1 implements
RedditTitleAdapter for soc-redditHyperlinks-title.tsv.  Task 2.2 implements
RedditBodyAdapter for soc-redditHyperlinks-body.tsv.

References: Requirements 1.1, 1.2
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd

from graph.models import InteractionRecord, InteractionType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Configuration for a single ingestion source.

    Attributes:
        source_type: Adapter type identifier (e.g. 'reddit_title', 'congress').
        file_path:   Path to the local dataset file.
        format:      File format hint (e.g. 'tsv', 'edgelist', 'txt_gz').
    """

    source_type: str
    file_path: str
    format: str
    extra: dict = field(default_factory=dict)  # adapter-specific overrides


# ---------------------------------------------------------------------------
# Abstract base adapter
# ---------------------------------------------------------------------------


class DataSourceAdapter(ABC):
    """Abstract adapter that every dataset-specific loader must implement.

    Separates retrieval (fetch) from normalization (normalize) so that each
    can be tested and swapped independently.
    """

    @abstractmethod
    def fetch(self, config: DatasetConfig) -> list[InteractionRecord]:
        """Load raw data from *config* and return normalized InteractionRecords.

        Implementations should iterate over all raw rows, call normalize()
        on each, and collect the results. Streaming / chunked reads are
        encouraged for large files — see RedditTitleAdapter for an example.

        Args:
            config: Dataset configuration including file path and format.

        Returns:
            List of InteractionRecord objects ready for downstream processing.
        """

    @abstractmethod
    def normalize(self, raw: dict) -> InteractionRecord | None:
        """Map a single raw row dict to an InteractionRecord.

        Args:
            raw: A dict-like row from the source file (e.g. a pandas Series
                 converted to dict, or a parsed text record).

        Returns:
            InteractionRecord if the row is valid, or None if it should be
            rejected (the caller is responsible for logging the rejection).
        """


# ---------------------------------------------------------------------------
# RedditTitleAdapter — soc-redditHyperlinks-title.tsv
# ---------------------------------------------------------------------------


class RedditTitleAdapter(DataSourceAdapter):
    """Adapter for the SNAP Reddit Hyperlink Title dataset.

    File format (TSV, header row):
        SOURCE_SUBREDDIT  TARGET_SUBREDDIT  POST_ID  TIMESTAMP  LINK_SENTIMENT  PROPERTIES

    Column mapping:
        SOURCE_SUBREDDIT → sourceUserId
        TARGET_SUBREDDIT → targetUserId
        POST_ID          → contentId
        TIMESTAMP        → timestamp   (parsed as UTC datetime; future records rejected)
        LINK_SENTIMENT   → sentimentScore (float)
        interactionType  = InteractionType.HYPERLINK  (fixed)
        datasetSource    = "reddit_title"             (fixed)
        topicTags        = []                         (not available in title TSV)

    Chunked streaming via pandas.read_csv(chunksize=10_000) keeps memory
    footprint bounded even though the file exceeds 50 MB.

    Note: deduplication is intentionally left to IngestionService (task 2.5).
    Full validation (empty userId, unrecognized type, self-loops) is task 2.6.
    This adapter only rejects records with future timestamps as specified in
    task 2.1.
    """

    DATASET_SOURCE: str = "reddit_title"
    CHUNK_SIZE: int = 10_000

    # TSV column names as they appear in the raw file
    _COL_SOURCE = "SOURCE_SUBREDDIT"
    _COL_TARGET = "TARGET_SUBREDDIT"
    _COL_POST_ID = "POST_ID"
    _COL_TIMESTAMP = "TIMESTAMP"
    _COL_SENTIMENT = "LINK_SENTIMENT"

    def fetch(self, config: DatasetConfig) -> list[InteractionRecord]:
        """Stream *config.file_path* in chunks and return all valid records.

        Uses pandas.read_csv with chunksize=10_000 to avoid loading the full
        50 MB+ file into memory at once.

        Args:
            config: DatasetConfig with file_path pointing to the TSV file.

        Returns:
            List of InteractionRecord objects (future-timestamp rows dropped).
        """
        records: list[InteractionRecord] = []
        total_rows = 0
        rejected_rows = 0

        for chunk in self._iter_chunks(config.file_path):
            for _, row in chunk.iterrows():
                total_rows += 1
                record = self._normalize_row(row)
                if record is None:
                    rejected_rows += 1
                else:
                    records.append(record)

        if rejected_rows:
            logger.warning(
                "RedditTitleAdapter: rejected %d / %d rows from '%s'",
                rejected_rows,
                total_rows,
                config.file_path,
            )

        logger.info(
            "RedditTitleAdapter: loaded %d valid records from '%s'",
            len(records),
            config.file_path,
        )
        return records

    def normalize(self, raw: dict) -> InteractionRecord | None:
        """Normalize a single raw row dict to an InteractionRecord.

        Public API that wraps the internal _normalize_row helper, accepting
        a plain dict (e.g. ``pd.Series.to_dict()`` output) for ease of
        testing without a full DataFrame.

        Args:
            raw: Dict with keys matching the TSV column names.

        Returns:
            InteractionRecord, or None if the row should be rejected.
        """
        return self._normalize_row(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_chunks(self, file_path: str) -> Iterator[pd.DataFrame]:
        """Yield successive DataFrame chunks from the TSV file.

        Args:
            file_path: Path to the TSV file.

        Yields:
            DataFrames of up to CHUNK_SIZE rows.
        """
        yield from pd.read_csv(
            file_path,
            sep="\t",
            chunksize=self.CHUNK_SIZE,
            dtype={
                self._COL_SOURCE: str,
                self._COL_TARGET: str,
                self._COL_POST_ID: str,
                self._COL_SENTIMENT: float,
            },
            # TIMESTAMP is left as object here; we parse it manually so we
            # can reject future timestamps before constructing the record.
            parse_dates=False,
        )

    def _normalize_row(self, row: dict | pd.Series) -> InteractionRecord | None:
        """Map one raw row to an InteractionRecord, returning None on rejection.

        Rejection criteria (task 2.1):
        - TIMESTAMP cannot be parsed as a datetime
        - TIMESTAMP is in the future (> UTC now)

        Args:
            row: Mapping with TSV column keys.

        Returns:
            InteractionRecord on success, None if the row is rejected.
        """
        # --- timestamp parsing and future-timestamp rejection ---
        raw_ts = row.get(self._COL_TIMESTAMP) if hasattr(row, "get") else row[self._COL_TIMESTAMP]
        timestamp = self._parse_timestamp(raw_ts)
        if timestamp is None:
            logger.debug(
                "RedditTitleAdapter: rejected row — unparseable timestamp '%s'",
                raw_ts,
            )
            return None

        now_utc = datetime.now(timezone.utc)
        # Ensure timezone-aware comparison
        ts_aware = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
        if ts_aware >= now_utc:
            logger.debug(
                "RedditTitleAdapter: rejected row — future timestamp '%s'",
                timestamp,
            )
            return None

        # --- field extraction ---
        source_subreddit = str(row[self._COL_SOURCE]) if row[self._COL_SOURCE] is not None else ""
        target_subreddit = str(row[self._COL_TARGET]) if row[self._COL_TARGET] is not None else ""
        post_id = str(row[self._COL_POST_ID]) if row[self._COL_POST_ID] is not None else None
        raw_sentiment = row.get(self._COL_SENTIMENT) if hasattr(row, "get") else row[self._COL_SENTIMENT]
        sentiment_score = self._parse_sentiment(raw_sentiment)

        # --- build InteractionRecord ---
        # Note: InteractionRecord.__post_init__ validates non-empty userIds,
        # non-self-loop, and future timestamp — we catch ValueError here and
        # treat those rows as rejected to match task-2.1 future-ts rejection.
        try:
            record = InteractionRecord(
                id=str(uuid.uuid4()),
                sourceUserId=source_subreddit,
                targetUserId=target_subreddit,
                interactionType=InteractionType.HYPERLINK,
                datasetSource=self.DATASET_SOURCE,
                timestamp=timestamp,
                contentId=post_id,
                topicTags=[],
                sentimentScore=sentiment_score,
            )
        except ValueError as exc:
            logger.debug(
                "RedditTitleAdapter: rejected row — validation error: %s", exc
            )
            return None

        return record

    @staticmethod
    def _parse_timestamp(raw: object) -> datetime | None:
        """Parse *raw* into a datetime, returning None on failure.

        The TIMESTAMP column in the Reddit TSV is formatted as
        ``YYYY-MM-DD HH:MM:SS``.  pandas ``to_datetime`` handles this and
        many other ISO-8601 variants.

        Args:
            raw: The raw timestamp value from the TSV cell.

        Returns:
            Parsed datetime (UTC-aware) or None if parsing fails.
        """
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return None
        try:
            ts = pd.to_datetime(raw, utc=True)
            return ts.to_pydatetime()
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_sentiment(raw: object) -> float | None:
        """Coerce *raw* to float, returning None if not possible.

        Args:
            raw: The raw LINK_SENTIMENT value.

        Returns:
            Float sentiment score, or None.
        """
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# RedditBodyAdapter — soc-redditHyperlinks-body.tsv
# ---------------------------------------------------------------------------


class RedditBodyAdapter(DataSourceAdapter):
    """Adapter for the SNAP Reddit Hyperlink Body dataset.

    File format (TSV, header row):
        SOURCE_SUBREDDIT  TARGET_SUBREDDIT  POST_ID  TIMESTAMP  LINK_SENTIMENT  PROPERTIES

    Column mapping (identical to RedditTitleAdapter except datasetSource):
        SOURCE_SUBREDDIT → sourceUserId
        TARGET_SUBREDDIT → targetUserId
        POST_ID          → contentId
        TIMESTAMP        → timestamp   (parsed as UTC datetime; future records rejected)
        LINK_SENTIMENT   → sentimentScore (float)
        interactionType  = InteractionType.HYPERLINK  (fixed)
        datasetSource    = "reddit_body"              (fixed)
        topicTags        = []                         (not available in body TSV)

    Additional body-specific handling:
        PROPERTIES → bodyText
            The PROPERTIES column is attempted to be parsed as JSON first.
            If it contains a JSON object, the "title" key (if present) and the
            "body" key (if present) are concatenated and stored as bodyText.
            If the column is not valid JSON (e.g. the SNAP numeric feature
            vector format), the raw string value is stored directly as bodyText.
            This field is NOT written to the main interaction store; it is
            consumed exclusively by Phase 6 (TF-IDF / sentence embedding).

    Side-effect during ingestion:
        subreddit_text_corpus — dict[str, list[str]]
            Populated during fetch(); maps each SOURCE_SUBREDDIT name to the
            list of bodyText values collected from its rows. Used downstream
            for per-subreddit TF-IDF / sentence embedding vectorization.

    Chunked streaming via pandas.read_csv(chunksize=10_000) keeps memory
    footprint bounded even though the file exceeds 50 MB.

    Note: deduplication is intentionally left to IngestionService (task 2.5).
    Full validation (empty userId, unrecognized type, self-loops) is task 2.6.
    This adapter only rejects records with future timestamps as specified in
    task 2.2.
    """

    DATASET_SOURCE: str = "reddit_body"
    CHUNK_SIZE: int = 10_000

    # TSV column names as they appear in the raw file
    _COL_SOURCE = "SOURCE_SUBREDDIT"
    _COL_TARGET = "TARGET_SUBREDDIT"
    _COL_POST_ID = "POST_ID"
    _COL_TIMESTAMP = "TIMESTAMP"
    _COL_SENTIMENT = "LINK_SENTIMENT"
    _COL_PROPERTIES = "PROPERTIES"

    def __init__(self) -> None:
        # Built incrementally during fetch(); keyed by SOURCE_SUBREDDIT name.
        self.subreddit_text_corpus: dict[str, list[str]] = {}

    def fetch(self, config: DatasetConfig) -> list[InteractionRecord]:
        """Stream *config.file_path* in chunks and return all valid records.

        Also populates ``self.subreddit_text_corpus`` as a side-effect:
        each source subreddit is mapped to the list of bodyText values
        extracted from its rows (for downstream Phase 6 vectorization).

        Uses pandas.read_csv with chunksize=10_000 to avoid loading the full
        50 MB+ file into memory at once.

        Args:
            config: DatasetConfig with file_path pointing to the TSV file.

        Returns:
            List of InteractionRecord objects (future-timestamp rows dropped).
        """
        # Reset corpus on each full fetch so the adapter is reusable.
        self.subreddit_text_corpus = {}

        records: list[InteractionRecord] = []
        total_rows = 0
        rejected_rows = 0

        for chunk in self._iter_chunks(config.file_path):
            for _, row in chunk.iterrows():
                total_rows += 1
                record = self._normalize_row(row)
                if record is None:
                    rejected_rows += 1
                else:
                    records.append(record)
                    # Accumulate body text into subreddit corpus (bodyText
                    # is not stored in the InteractionRecord main store).
                    if record.bodyText:
                        subreddit = record.sourceUserId
                        self.subreddit_text_corpus.setdefault(subreddit, []).append(
                            record.bodyText
                        )

        if rejected_rows:
            logger.warning(
                "RedditBodyAdapter: rejected %d / %d rows from '%s'",
                rejected_rows,
                total_rows,
                config.file_path,
            )

        logger.info(
            "RedditBodyAdapter: loaded %d valid records from '%s'; "
            "corpus covers %d subreddits",
            len(records),
            config.file_path,
            len(self.subreddit_text_corpus),
        )
        return records

    def normalize(self, raw: dict) -> InteractionRecord | None:
        """Normalize a single raw row dict to an InteractionRecord.

        Public API that wraps the internal _normalize_row helper, accepting
        a plain dict (e.g. ``pd.Series.to_dict()`` output) for ease of
        testing without a full DataFrame.

        Note: calling this method directly does NOT update
        ``subreddit_text_corpus`` — use ``fetch()`` for the full pipeline.

        Args:
            raw: Dict with keys matching the TSV column names.

        Returns:
            InteractionRecord, or None if the row should be rejected.
        """
        return self._normalize_row(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_chunks(self, file_path: str) -> Iterator[pd.DataFrame]:
        """Yield successive DataFrame chunks from the TSV file.

        Args:
            file_path: Path to the TSV file.

        Yields:
            DataFrames of up to CHUNK_SIZE rows.
        """
        yield from pd.read_csv(
            file_path,
            sep="\t",
            chunksize=self.CHUNK_SIZE,
            dtype={
                self._COL_SOURCE: str,
                self._COL_TARGET: str,
                self._COL_POST_ID: str,
                self._COL_SENTIMENT: float,
                self._COL_PROPERTIES: str,
            },
            parse_dates=False,
        )

    def _normalize_row(self, row: dict | pd.Series) -> InteractionRecord | None:
        """Map one raw row to an InteractionRecord, returning None on rejection.

        Rejection criteria (task 2.2):
        - TIMESTAMP cannot be parsed as a datetime
        - TIMESTAMP is in the future (> UTC now)

        Args:
            row: Mapping with TSV column keys.

        Returns:
            InteractionRecord on success, None if the row is rejected.
        """
        # --- timestamp parsing and future-timestamp rejection ---
        raw_ts = row.get(self._COL_TIMESTAMP) if hasattr(row, "get") else row[self._COL_TIMESTAMP]
        timestamp = RedditTitleAdapter._parse_timestamp(raw_ts)
        if timestamp is None:
            logger.debug(
                "RedditBodyAdapter: rejected row — unparseable timestamp '%s'",
                raw_ts,
            )
            return None

        now_utc = datetime.now(timezone.utc)
        ts_aware = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
        if ts_aware >= now_utc:
            logger.debug(
                "RedditBodyAdapter: rejected row — future timestamp '%s'",
                timestamp,
            )
            return None

        # --- field extraction ---
        source_subreddit = str(row[self._COL_SOURCE]) if row[self._COL_SOURCE] is not None else ""
        target_subreddit = str(row[self._COL_TARGET]) if row[self._COL_TARGET] is not None else ""
        post_id = str(row[self._COL_POST_ID]) if row[self._COL_POST_ID] is not None else None
        raw_sentiment = row.get(self._COL_SENTIMENT) if hasattr(row, "get") else row[self._COL_SENTIMENT]
        sentiment_score = RedditTitleAdapter._parse_sentiment(raw_sentiment)

        # --- PROPERTIES → bodyText ---
        raw_props = row.get(self._COL_PROPERTIES) if hasattr(row, "get") else row.get(self._COL_PROPERTIES, None)
        body_text = self._extract_body_text(raw_props)

        # --- build InteractionRecord ---
        try:
            record = InteractionRecord(
                id=str(uuid.uuid4()),
                sourceUserId=source_subreddit,
                targetUserId=target_subreddit,
                interactionType=InteractionType.HYPERLINK,
                datasetSource=self.DATASET_SOURCE,
                timestamp=timestamp,
                contentId=post_id,
                topicTags=[],
                sentimentScore=sentiment_score,
                bodyText=body_text,
            )
        except ValueError as exc:
            logger.debug(
                "RedditBodyAdapter: rejected row — validation error: %s", exc
            )
            return None

        return record

    @staticmethod
    def _extract_body_text(raw_props: object) -> str | None:
        """Extract bodyText from the PROPERTIES column value.

        Attempt 1 — JSON parse:
            If *raw_props* is a valid JSON object, concatenate the "title"
            value (if present) and the "body" value (if present) separated
            by a space.  Returns the concatenation, or None if both keys are
            absent.

        Attempt 2 — raw string fallback:
            If *raw_props* is not valid JSON (e.g. the SNAP numeric feature
            vector), return the raw string as-is so that downstream consumers
            (Phase 6) can decide how to interpret it.

        Args:
            raw_props: The raw PROPERTIES cell value from the TSV row.

        Returns:
            A non-empty string representing the body text, or None if the
            PROPERTIES value is absent / NaN.
        """
        if raw_props is None:
            return None
        if isinstance(raw_props, float):
            # pandas represents missing strings as float NaN
            try:
                import math
                if math.isnan(raw_props):
                    return None
            except (TypeError, ValueError):
                pass
            return str(raw_props)

        raw_str = str(raw_props).strip()
        if not raw_str:
            return None

        # Attempt JSON parse
        try:
            parsed = json.loads(raw_str)
            if isinstance(parsed, dict):
                parts: list[str] = []
                if "title" in parsed and parsed["title"]:
                    parts.append(str(parsed["title"]).strip())
                if "body" in parsed and parsed["body"]:
                    parts.append(str(parsed["body"]).strip())
                return " ".join(parts) if parts else None
            # JSON but not a dict (e.g. a JSON array of numbers) — fall through
        except (json.JSONDecodeError, ValueError):
            pass

        # Non-JSON fallback: return the raw string (e.g. numeric feature vector)
        return raw_str


# ---------------------------------------------------------------------------
# CongressNetworkAdapter — congress.edgelist + congress_network_data.json
# ---------------------------------------------------------------------------


class CongressNetworkAdapter(DataSourceAdapter):
    """Adapter for the Congress Network Twitter influence dataset.

    File format:
        congress_network_data.json — JSON object with a ``usernameList`` array
            where index *i* maps to the Twitter username for node ID *i*.
        congress.edgelist — space-separated edge list, one edge per line:
            ``nodeA nodeB {'weight': float}``

    Column mapping:
        nodeA → sourceUserId  (resolved via usernameList[nodeA])
        nodeB → targetUserId  (resolved via usernameList[nodeB])
        weight (float)        → transmission_probability
        interactionType       = InteractionType.RETWEET  (fixed)
        datasetSource         = "congress"               (fixed)
        timestamp             = None  (not present in this dataset)
        pre_normalized        = True  (weights already in [0, 1]; skip
                                       normalization in buildGraph)

    Deduplication key:
        (sourceUserId, targetUserId) — no timestamp field.

    The ``pre_normalized`` flag is communicated to downstream consumers via
    the ``datasetSource`` field as ``"congress"`` and via the class attribute
    ``PRE_NORMALIZED = True``.  It is also stored in ``DatasetConfig.extra``
    when the adapter creates records, allowing ``buildGraph`` to skip the
    normalization step.
    """

    DATASET_SOURCE: str = "congress"
    PRE_NORMALIZED: bool = True

    # Regex to extract the weight value from the dict string, e.g.:
    #   {'weight': 0.00210526315789}
    _WEIGHT_RE = re.compile(r"\{\s*'weight'\s*:\s*([\d.e+\-]+)\s*\}")

    def __init__(self) -> None:
        self._id_to_username: dict[int, str] = {}

    def fetch(self, config: DatasetConfig) -> list[InteractionRecord]:
        """Load *congress_network_data.json* and *congress.edgelist* and return records.

        The ``config.file_path`` should point to the ``.edgelist`` file.
        The JSON file is expected to be in the same directory with the name
        ``congress_network_data.json``.  Override via
        ``config.extra['json_path']``.

        Args:
            config: DatasetConfig with ``file_path`` pointing to the edgelist.

        Returns:
            List of InteractionRecord objects for all parseable edges.
        """
        import os

        edgelist_path = config.file_path
        json_path = config.extra.get(
            "json_path",
            os.path.join(os.path.dirname(edgelist_path), "congress_network_data.json"),
        )

        # Load username lookup map
        self._id_to_username = self._load_username_map(json_path)

        records: list[InteractionRecord] = []
        rejected = 0
        total = 0

        try:
            with open(edgelist_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    record = self._parse_line(line)
                    if record is None:
                        rejected += 1
                    else:
                        records.append(record)
        except OSError as exc:
            logger.error("CongressNetworkAdapter: failed to open '%s': %s", edgelist_path, exc)
            return []

        if rejected:
            logger.warning(
                "CongressNetworkAdapter: rejected %d / %d lines from '%s'",
                rejected,
                total,
                edgelist_path,
            )

        logger.info(
            "CongressNetworkAdapter: loaded %d valid records from '%s'",
            len(records),
            edgelist_path,
        )
        return records

    def normalize(self, raw: dict) -> InteractionRecord | None:
        """Normalize a single raw edge dict to an InteractionRecord.

        Expected keys:
            ``nodeA`` (int), ``nodeB`` (int), ``weight`` (float).
        The username lookup map must have been populated beforehand (either
        via ``fetch()`` or by setting ``_id_to_username`` manually).

        Args:
            raw: Dict with keys ``nodeA``, ``nodeB``, ``weight``.

        Returns:
            InteractionRecord on success, or None on rejection.
        """
        node_a = raw.get("nodeA")
        node_b = raw.get("nodeB")
        weight = raw.get("weight")

        if node_a is None or node_b is None or weight is None:
            logger.debug("CongressNetworkAdapter: rejected row — missing fields: %r", raw)
            return None

        source_username = self._id_to_username.get(int(node_a))
        target_username = self._id_to_username.get(int(node_b))

        if source_username is None:
            logger.debug(
                "CongressNetworkAdapter: rejected row — unknown nodeA ID %r", node_a
            )
            return None
        if target_username is None:
            logger.debug(
                "CongressNetworkAdapter: rejected row — unknown nodeB ID %r", node_b
            )
            return None

        try:
            record = InteractionRecord(
                id=str(uuid.uuid4()),
                sourceUserId=source_username,
                targetUserId=target_username,
                interactionType=InteractionType.RETWEET,
                datasetSource=self.DATASET_SOURCE,
                timestamp=None,
                contentId=None,
                topicTags=[],
                sentimentScore=float(weight),
            )
        except ValueError as exc:
            logger.debug("CongressNetworkAdapter: rejected row — validation error: %s", exc)
            return None

        return record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_username_map(json_path: str) -> dict[int, str]:
        """Load ``usernameList`` from *congress_network_data.json*.

        Returns a dict mapping integer node ID → Twitter username string.

        Args:
            json_path: Path to the JSON file.

        Returns:
            Dict mapping node index to username, or empty dict on failure.
        """
        try:
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "CongressNetworkAdapter: failed to load username map from '%s': %s",
                json_path,
                exc,
            )
            return {}

        if isinstance(data, list):
            # The file contains a JSON array; the first element is the main
            # data object (the README and compute_vc.py confirm this structure).
            root = data[0] if data else {}
        elif isinstance(data, dict):
            root = data
        else:
            logger.error(
                "CongressNetworkAdapter: unexpected JSON structure in '%s'", json_path
            )
            return {}

        username_list = root.get("usernameList", [])
        if not username_list:
            logger.warning(
                "CongressNetworkAdapter: 'usernameList' is missing or empty in '%s'",
                json_path,
            )
            return {}

        return {i: username for i, username in enumerate(username_list)}

    def _parse_line(self, line: str) -> InteractionRecord | None:
        """Parse one ``congress.edgelist`` line into an InteractionRecord.

        Line format: ``nodeA nodeB {'weight': float}``

        Args:
            line: A stripped non-empty line from the edgelist file.

        Returns:
            InteractionRecord on success, or None if the line cannot be parsed.
        """
        # Split off the node IDs from the dict portion
        # Format is: "0 4 {'weight': 0.00210...}"
        parts = line.split(None, 2)  # split on whitespace, max 3 tokens
        if len(parts) < 3:
            logger.debug("CongressNetworkAdapter: skipping malformed line: %r", line)
            return None

        try:
            node_a = int(parts[0])
            node_b = int(parts[1])
        except ValueError:
            logger.debug("CongressNetworkAdapter: non-integer node IDs in line: %r", line)
            return None

        weight_match = self._WEIGHT_RE.search(parts[2])
        if weight_match is None:
            logger.debug("CongressNetworkAdapter: could not extract weight from: %r", line)
            return None

        try:
            weight = float(weight_match.group(1))
        except ValueError:
            logger.debug("CongressNetworkAdapter: non-float weight in line: %r", line)
            return None

        return self.normalize({"nodeA": node_a, "nodeB": node_b, "weight": weight})


# ---------------------------------------------------------------------------
# WikiRfAAdapter — wiki-RfA.txt.gz
# ---------------------------------------------------------------------------


class WikiRfAAdapter(DataSourceAdapter):
    """Adapter for the Wikipedia Requests for Adminship (RfA) signed-vote dataset.

    File format (gzip-compressed plain text, UTF-8):
        Records are separated by blank lines.  Each record contains the
        following prefix-keyed fields (one per line):

            SRC:  <voter username>
            TGT:  <candidate username>
            VOT:  +1 or -1 (support / oppose)
            RES:  0 or 1   (request failed / succeeded)
            YEA:  <year as integer>
            DAT:  <timestamp string, e.g. "23:13, 19 April 2013">
            TXT:  <free-text vote comment>

    Column mapping:
        SRC              → sourceUserId
        TGT              → targetUserId
        VOT (int)        → votePolarity  (+1 or -1)
        RES (int)        → voteResult    (0 or 1)
        DAT (parsed)     → timestamp     (UTC datetime; None if unparseable)
        TXT              → bodyText      (stored for Phase 6 sentiment / topic analysis)
        interactionType  = InteractionType.VOTE    (fixed)
        datasetSource    = "wiki_rfa"              (fixed)
        weight           = 1.0  (binary edge; sign is in votePolarity)
        topicTags        = []

    Deduplication key (per task spec):
        (sourceUserId, targetUserId, timestamp)
        Multiple votes by the same user on the same candidate in different
        years are valid.

    Invalid records (bad VOT value, empty userIds, self-loops, etc.) are
    logged and skipped without raising an exception.

    References: Requirements 1.1, 1.2
    """

    DATASET_SOURCE: str = "wiki_rfa"
    WEIGHT: float = 1.0  # binary edge weight

    def __init__(self) -> None:
        # Built incrementally during fetch(); keyed by SOURCE (voter) userId.
        # Maps each editor userId to the list of TXT comment strings from their
        # vote records.  Used downstream by TopicVectorService (Phase 6).
        self.editor_text_corpus: dict[str, list[str]] = {}

    # Prefix constants
    _PFX_SRC = "SRC:"
    _PFX_TGT = "TGT:"
    _PFX_VOT = "VOT:"
    _PFX_RES = "RES:"
    _PFX_YEA = "YEA:"
    _PFX_DAT = "DAT:"
    _PFX_TXT = "TXT:"

    # Supported datetime formats for the DAT: field
    # Examples: "23:13, 19 April 2013", "01:04, 20 April 2013"
    _DAT_FORMATS = [
        "%H:%M, %d %B %Y",   # "23:13, 19 April 2013"
        "%d %B %Y",           # "19 April 2013" (no time component)
        "%B %Y",              # "April 2013" (month + year only)
    ]

    def fetch(self, config: DatasetConfig) -> list[InteractionRecord]:
        """Decompress and parse *config.file_path*, returning all valid records.

        Uses gzip.open in text mode to decompress on-the-fly.  Records are
        delimited by blank lines; fields within each record use a prefix key.

        Args:
            config: DatasetConfig with file_path pointing to wiki-RfA.txt.gz.

        Returns:
            List of InteractionRecord objects for all parseable vote records.
        """
        import gzip

        # Reset corpus on each full fetch so the adapter is reusable.
        self.editor_text_corpus = {}

        records: list[InteractionRecord] = []
        total = 0
        rejected = 0

        try:
            with gzip.open(config.file_path, "rt", encoding="utf-8", errors="replace") as fh:
                raw_lines = fh.readlines()
        except OSError as exc:
            logger.error("WikiRfAAdapter: failed to open '%s': %s", config.file_path, exc)
            return []

        # Split lines into records on blank lines
        current_block: list[str] = []
        for line in raw_lines:
            stripped = line.rstrip("\n")
            if stripped == "":
                if current_block:
                    total += 1
                    record = self._parse_block(current_block)
                    if record is None:
                        rejected += 1
                    else:
                        records.append(record)
                        # Accumulate TXT comment text into editor corpus
                        # (bodyText is the parsed TXT: field value).
                        if record.bodyText:
                            self.editor_text_corpus.setdefault(
                                record.sourceUserId, []
                            ).append(record.bodyText)
                    current_block = []
            else:
                current_block.append(stripped)

        # Handle final block if file does not end with a blank line
        if current_block:
            total += 1
            record = self._parse_block(current_block)
            if record is None:
                rejected += 1
            else:
                records.append(record)
                if record.bodyText:
                    self.editor_text_corpus.setdefault(
                        record.sourceUserId, []
                    ).append(record.bodyText)

        if rejected:
            logger.warning(
                "WikiRfAAdapter: rejected %d / %d records from '%s'",
                rejected,
                total,
                config.file_path,
            )

        logger.info(
            "WikiRfAAdapter: loaded %d valid records from '%s'; "
            "editor_text_corpus covers %d editors",
            len(records),
            config.file_path,
            len(self.editor_text_corpus),
        )
        return records

    def normalize(self, raw: dict) -> InteractionRecord | None:
        """Normalize a pre-parsed record dict to an InteractionRecord.

        Expected keys (strings): ``SRC``, ``TGT``, ``VOT``, ``RES``,
        ``DAT`` (optional), ``TXT`` (optional).

        Args:
            raw: Dict produced by ``_parse_block`` or provided directly in tests.

        Returns:
            InteractionRecord on success, None if the record is invalid.
        """
        src = (raw.get("SRC") or "").strip()
        tgt = (raw.get("TGT") or "").strip()
        vot_raw = raw.get("VOT")
        res_raw = raw.get("RES")
        dat_raw = raw.get("DAT")
        txt_raw = raw.get("TXT")

        # --- votePolarity ---
        try:
            vote_polarity = int(vot_raw) if vot_raw is not None else None
        except (ValueError, TypeError):
            logger.debug(
                "WikiRfAAdapter: rejected record — unparseable VOT '%s'", vot_raw
            )
            return None

        if vote_polarity not in (1, -1):
            logger.debug(
                "WikiRfAAdapter: rejected record — invalid VOT value %r "
                "(must be +1 or -1)",
                vote_polarity,
            )
            return None

        # --- voteResult ---
        # The wiki-RfA dataset uses RES:1 for "promoted" and RES:-1 for
        # "not promoted".  Normalize -1 → 0 so voteResult is always 0 or 1
        # as the InteractionRecord model requires.
        try:
            vote_result_raw = int(res_raw) if res_raw is not None else None
        except (ValueError, TypeError):
            logger.debug(
                "WikiRfAAdapter: rejected record — unparseable RES '%s'", res_raw
            )
            return None

        if vote_result_raw is None:
            vote_result = None
        elif vote_result_raw == -1:
            vote_result = 0   # dataset uses -1 to mean "not promoted" (= 0)
        elif vote_result_raw in (0, 1):
            vote_result = vote_result_raw
        else:
            logger.debug(
                "WikiRfAAdapter: rejected record — invalid RES value %r "
                "(must be 1, 0, or -1)",
                vote_result_raw,
            )
            return None

        # --- timestamp ---
        timestamp = self._parse_dat(dat_raw) if dat_raw else None

        # --- bodyText ---
        body_text = txt_raw.strip() if txt_raw and txt_raw.strip() else None

        # --- build InteractionRecord (validation in __post_init__) ---
        try:
            record = InteractionRecord(
                id=str(uuid.uuid4()),
                sourceUserId=src,
                targetUserId=tgt,
                interactionType=InteractionType.VOTE,
                datasetSource=self.DATASET_SOURCE,
                timestamp=timestamp,
                contentId=None,
                topicTags=[],
                sentimentScore=self.WEIGHT,   # edge weight = 1.0 (binary)
                votePolarity=vote_polarity,
                voteResult=vote_result,
                bodyText=body_text,
            )
        except ValueError as exc:
            logger.debug("WikiRfAAdapter: rejected record — validation error: %s", exc)
            return None

        return record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_block(self, lines: list[str]) -> InteractionRecord | None:
        """Parse a list of lines (one record block) into an InteractionRecord.

        Each line in the block should have the form ``PREFIX:value``.
        Unrecognised lines are ignored.

        Args:
            lines: Non-empty list of stripped lines for one record.

        Returns:
            InteractionRecord on success, or None if parsing fails.
        """
        fields: dict[str, str] = {}
        for line in lines:
            if line.startswith(self._PFX_SRC):
                fields["SRC"] = line[len(self._PFX_SRC):]
            elif line.startswith(self._PFX_TGT):
                fields["TGT"] = line[len(self._PFX_TGT):]
            elif line.startswith(self._PFX_VOT):
                fields["VOT"] = line[len(self._PFX_VOT):]
            elif line.startswith(self._PFX_RES):
                fields["RES"] = line[len(self._PFX_RES):]
            elif line.startswith(self._PFX_YEA):
                fields["YEA"] = line[len(self._PFX_YEA):]
            elif line.startswith(self._PFX_DAT):
                fields["DAT"] = line[len(self._PFX_DAT):]
            elif line.startswith(self._PFX_TXT):
                fields["TXT"] = line[len(self._PFX_TXT):]

        return self.normalize(fields)

    @staticmethod
    def _parse_dat(dat: str) -> datetime | None:
        """Parse the DAT: field value into a UTC-aware datetime.

        The field format used in the dataset is e.g. "23:13, 19 April 2013".
        A graceful fallback is applied: if the full format fails, a shorter
        format (month + year) is tried.  Returns None if all formats fail.

        Args:
            dat: Raw DAT field value string.

        Returns:
            datetime (UTC, timezone-aware) or None.
        """
        dat = dat.strip()
        # Formats to try in order
        formats = [
            "%H:%M, %d %B %Y",   # "23:13, 19 April 2013"
            "%d %B %Y",           # "19 April 2013"
            "%B %Y",              # "April 2013"
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(dat, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        # Final fallback: extract a four-digit year and return Jan 1 of that year
        import re as _re
        year_match = _re.search(r"\b(\d{4})\b", dat)
        if year_match:
            try:
                year = int(year_match.group(1))
                return datetime(year, 1, 1, tzinfo=timezone.utc)
            except ValueError:
                pass

        logger.debug("WikiRfAAdapter: could not parse DAT field '%s'", dat)
        return None
