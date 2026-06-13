"""Recommendation persistence and retrieval service for the Echo Chamber Detector.

Implements ``persist_recommendations`` and ``fetch_recommendations`` for storing
and querying cross-community recommendations in PostgreSQL, with optional Redis
caching.

References: Requirements 6.7
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from graph.models import Recommendation
from graph.redis_keys import DEFAULT_TTL_SECONDS, recommendations_key

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for persisting and retrieving Recommendation objects.

    This service is stateless; no ``__init__`` is required.

    Usage::

        service = RecommendationService()
        service.persist_recommendations(recs, snapshot_id, dataset_source, session)
        results = service.fetch_recommendations(user_id, session)
    """

    def persist_recommendations(
        self,
        recommendations: list[Recommendation],
        snapshot_id: str,
        dataset_source: str,
        db_session: Any,
        redis_client: Any = None,
    ) -> None:
        """Persist each Recommendation to the recommendations table.

        Writes one RecommendationRow per Recommendation to the database via
        the supplied SQLAlchemy session. The session is flushed but NOT
        committed — the caller is responsible for committing.

        Optionally caches the list of recommendations for each targetUserId
        in Redis using key ``user:{userId}:recommendations`` with the default
        TTL (:data:`~graph.redis_keys.DEFAULT_TTL_SECONDS`).

        Args:
            recommendations:  List of Recommendation objects to persist.
            snapshot_id:      The snapshot this batch belongs to.
            dataset_source:   The dataset source (e.g. "reddit_title").
            db_session:       Active SQLAlchemy Session.
            redis_client:     Optional Redis client. When None, skip caching.

        Requirements: 6.7
        """
        from graph.db_models import RecommendationRow  # noqa: PLC0415

        if not recommendations:
            logger.info(
                "RecommendationService.persist_recommendations: "
                "no recommendations to persist for snapshot '%s'.",
                snapshot_id,
            )
            return

        # Persist each recommendation row
        for rec in recommendations:
            row = RecommendationRow(
                recommendation_id=rec.recommendationId,
                snapshot_id=snapshot_id,
                dataset_source=dataset_source,
                target_user_id=rec.targetUserId,
                recommended_user_id=rec.recommendedUserId,
                diversity_gain=rec.diversityGain,
                topic_relevance=rec.topicRelevance,
                community_id=rec.communityId,
                reason=rec.reason,
                content_id=rec.contentId,
            )
            db_session.add(row)

        db_session.flush()

        logger.info(
            "RecommendationService.persist_recommendations: "
            "persisted %d recommendations for snapshot '%s' dataset='%s'.",
            len(recommendations),
            snapshot_id,
            dataset_source,
        )

        # Optional Redis caching: group recs by targetUserId
        if redis_client is not None:
            # Group recommendations by targetUserId
            by_user: dict[str, list[Recommendation]] = {}
            for rec in recommendations:
                by_user.setdefault(rec.targetUserId, []).append(rec)

            for user_id, user_recs in by_user.items():
                try:
                    cache_payload = json.dumps([
                        {
                            "recommendationId": r.recommendationId,
                            "targetUserId": r.targetUserId,
                            "recommendedUserId": r.recommendedUserId,
                            "diversityGain": r.diversityGain,
                            "topicRelevance": r.topicRelevance,
                            "communityId": r.communityId,
                            "reason": r.reason,
                            "contentId": r.contentId,
                        }
                        for r in user_recs
                    ])
                    redis_key = recommendations_key(user_id)
                    redis_client.setex(redis_key, DEFAULT_TTL_SECONDS, cache_payload)
                    logger.debug(
                        "RecommendationService.persist_recommendations: "
                        "cached %d recommendations in Redis key='%s' ttl=%ds",
                        len(user_recs),
                        redis_key,
                        DEFAULT_TTL_SECONDS,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "RecommendationService.persist_recommendations: "
                        "Redis caching failed for user '%s' (%s); "
                        "continuing without cache.",
                        user_id,
                        exc,
                    )

    def fetch_recommendations(
        self,
        user_id: str,
        db_session: Any,
        snapshot_id: Optional[str] = None,
        dataset_source: Optional[str] = None,
    ) -> list[Recommendation]:
        """Retrieve persisted recommendations for a user.

        Queries the recommendations table filtered by target_user_id.
        Optionally filters by snapshot_id and/or dataset_source.
        Returns results as Recommendation dataclass objects (not ORM rows),
        ordered by diversity_gain DESC.

        Args:
            user_id:        The targetUserId to look up.
            db_session:     Active SQLAlchemy Session.
            snapshot_id:    Optional snapshot filter.
            dataset_source: Optional dataset source filter.

        Returns:
            List of Recommendation objects, ordered by diversity_gain DESC.

        Requirements: 6.7
        """
        from graph.db_models import RecommendationRow  # noqa: PLC0415
        from sqlalchemy import and_  # noqa: PLC0415

        query = db_session.query(RecommendationRow)
        filters = [RecommendationRow.target_user_id == user_id]

        if snapshot_id is not None:
            filters.append(RecommendationRow.snapshot_id == snapshot_id)

        if dataset_source is not None:
            filters.append(RecommendationRow.dataset_source == dataset_source)

        query = query.filter(and_(*filters))
        query = query.order_by(RecommendationRow.diversity_gain.desc())

        rows = query.all()

        recommendations = [
            Recommendation(
                recommendationId=row.recommendation_id,
                targetUserId=row.target_user_id,
                recommendedUserId=row.recommended_user_id,
                diversityGain=row.diversity_gain,
                topicRelevance=row.topic_relevance,
                communityId=row.community_id,
                reason=row.reason,
                contentId=row.content_id,
            )
            for row in rows
        ]

        logger.info(
            "RecommendationService.fetch_recommendations: "
            "fetched %d recommendations for user '%s' "
            "(snapshot_id=%r, dataset_source=%r).",
            len(recommendations),
            user_id,
            snapshot_id,
            dataset_source,
        )

        return recommendations
