"""Input record validation for the Echo Chamber Detector ingestion pipeline.

Provides :func:`validate_record` which checks every :class:`InteractionRecord`
against the general ingestion rules (Requirements 9.1–9.4) and dataset-specific
rules for Wiki-RfA (Requirements 9.1–9.5).

The function returns a ``(valid: bool, reason: str)`` tuple so callers can log
rejections with the record ID and continue processing without halting.

References: Requirements 9.1–9.5
"""

from __future__ import annotations

from datetime import datetime, timezone

from graph.models import InteractionRecord, InteractionType

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_VALID_INTERACTION_TYPES: frozenset[InteractionType] = frozenset(InteractionType)


def validate_record(record: InteractionRecord) -> tuple[bool, str]:
    """Validate *record* against all ingestion rules.

    General rules (applied to every record):
    - Requirement 9.1: ``sourceUserId`` and ``targetUserId`` must be non-empty.
    - Requirement 9.2: ``interactionType`` must be a recognised :class:`InteractionType`
      enum member.
    - Requirement 9.3: ``timestamp``, when present, must be a valid past datetime
      (strictly before UTC now).
    - Self-loop rule: ``sourceUserId`` must not equal ``targetUserId``.

    Wiki-RfA specific rules (applied when ``datasetSource == "wiki_rfa"``):
    - ``votePolarity`` must be ``+1`` or ``-1``.
    - ``voteResult`` must be ``0`` or ``1``.

    Args:
        record: The :class:`InteractionRecord` to validate.

    Returns:
        A 2-tuple ``(is_valid, reason)`` where ``is_valid`` is ``True`` when
        the record passes all applicable rules, and ``reason`` is an empty
        string.  On failure ``is_valid`` is ``False`` and ``reason`` describes
        the first failing rule.
    """
    # --- Requirement 9.1: non-empty user IDs ---
    if not record.sourceUserId:
        return False, "sourceUserId is empty"
    if not record.targetUserId:
        return False, "targetUserId is empty"

    # --- Self-loop rejection ---
    if record.sourceUserId == record.targetUserId:
        return (
            False,
            f"sourceUserId equals targetUserId ('{record.sourceUserId}') — self-loop not allowed",
        )

    # --- Requirement 9.2: recognised interactionType ---
    if not isinstance(record.interactionType, InteractionType):
        return (
            False,
            f"interactionType '{record.interactionType!r}' is not a recognised enum value; "
            f"valid values are {[t.value for t in InteractionType]}",
        )
    if record.interactionType not in _VALID_INTERACTION_TYPES:
        return (
            False,
            f"interactionType '{record.interactionType.value}' is not a recognised enum value; "
            f"valid values are {[t.value for t in InteractionType]}",
        )

    # --- Requirement 9.3: timestamp must be in the past (if present) ---
    if record.timestamp is not None:
        now_utc = datetime.now(timezone.utc)
        ts = record.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= now_utc:
            return (
                False,
                f"timestamp {record.timestamp!r} is not a past datetime (must be before UTC now)",
            )

    # --- Wiki-RfA specific rules ---
    if record.datasetSource == "wiki_rfa":
        if record.votePolarity is not None and record.votePolarity not in (1, -1):
            return (
                False,
                f"votePolarity must be +1 or -1 for wiki_rfa records (got {record.votePolarity!r})",
            )
        if record.voteResult is not None and record.voteResult not in (0, 1):
            return (
                False,
                f"voteResult must be 0 or 1 for wiki_rfa records (got {record.voteResult!r})",
            )

    return True, ""
