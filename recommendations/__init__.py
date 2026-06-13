# Recommendation Engine — generates balanced viewpoint recommendations for low-diversity users

from recommendations.bridge_nodes import (  # noqa: F401
    BRIDGE_CENTRALITY_THRESHOLD,
    MIN_TOPIC_RELEVANCE_THRESHOLD,
    SPARSE_USER_INTERACTION_THRESHOLD,
    BridgeNodeService,
)
from recommendations.service import RecommendationService  # noqa: F401
from recommendations.topic_vectors import TopicVectorService  # noqa: F401
