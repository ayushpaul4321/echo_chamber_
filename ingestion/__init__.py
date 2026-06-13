# Ingestion Layer — fetches, normalizes, and deduplicates raw interaction records

from ingestion.service import IngestionResult, IngestionService, IngestionStatus

__all__ = ["IngestionService", "IngestionResult", "IngestionStatus"]
