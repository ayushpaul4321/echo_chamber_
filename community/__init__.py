# Community Detection Engine — partitions the interaction graph into communities

from community.service import (
    CommunityDetectionService,
    MAX_ITERATIONS,
    compute_modularity,
    detect_communities,
)

__all__ = [
    "CommunityDetectionService",
    "MAX_ITERATIONS",
    "compute_modularity",
    "detect_communities",
]
