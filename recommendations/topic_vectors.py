"""Topic Vector Construction for the Echo Chamber Detector recommendation engine.

Builds per-node topic vectors for each supported dataset source:

- **Reddit body** (``datasetSource == "reddit_body"``):
    Fit a ``TfidfVectorizer(max_features=5000)`` on the
    ``subreddit_text_corpus`` built by ``RedditBodyAdapter`` during Phase 2
    ingestion.  Each subreddit node's TF-IDF vector is stored in
    ``node.topicVector``.  An optional upgrade to ``sentence-transformers``
    (``all-MiniLM-L6-v2``) is available via the ``use_sentence_transformers``
    flag.

- **Wiki-RfA** (``datasetSource == "wiki_rfa"``):
    Fit a separate ``TfidfVectorizer(max_features=5000)`` on the
    ``editor_text_corpus`` built by ``WikiRfAAdapter`` (userId → list of TXT
    comment strings).  Per-editor TF-IDF vectors are stored in
    ``node.topicVector``.

- **Congress** (``datasetSource == "congress"``):
    No body text is available.  Community membership from a Louvain partition
    is used as a proxy:

    - The two dominant communities (by node count) receive categorical vectors:
        * Larger community  → ``[1.0, 0.0]``  (Democrat proxy)
        * Smaller community → ``[0.0, 1.0]``  (Republican proxy)
    - Nodes not in either dominant community get ``[0.5, 0.5]`` (neutral).

- **Other sources** (e.g. ``"reddit_title"``):
    A warning is logged and vectors are left unchanged.

References: Requirements 6.3 (cosine similarity on topic vectors), 6.6
(community centroid fallback for sparse users).
"""

from __future__ import annotations

import logging
from typing import Optional

from graph.models import InteractionGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_FEATURES: int = 5000
_SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"

# Known dataset source identifiers
_REDDIT_BODY_SOURCE: str = "reddit_body"
_WIKI_RFA_SOURCE: str = "wiki_rfa"
_CONGRESS_SOURCE: str = "congress"


# ---------------------------------------------------------------------------
# TopicVectorService
# ---------------------------------------------------------------------------


class TopicVectorService:
    """Builds and assigns topic vectors for every node in an InteractionGraph.

    The service dispatches to dataset-specific methods based on
    ``graph.datasetSource``.  All vectors are stored as plain Python
    ``list[float]`` in ``node.topicVector`` (not numpy arrays), which is the
    format expected by downstream cosine-similarity computations and graph
    serialization.

    Usage::

        service = TopicVectorService()

        # Reddit body
        vectorizer = service.build_reddit_topic_vectors(graph, corpus)

        # Wiki-RfA
        vectorizer = service.build_wiki_rfa_topic_vectors(graph, corpus)

        # Congress (requires Louvain partition dict)
        service.build_congress_topic_vectors(graph, partition)

        # Dispatcher (auto-detects from graph.datasetSource)
        service.build_topic_vectors(graph, subreddit_text_corpus=corpus)
    """

    # ------------------------------------------------------------------
    # Public dispatcher
    # ------------------------------------------------------------------

    def build_topic_vectors(
        self,
        graph: InteractionGraph,
        subreddit_text_corpus: Optional[dict[str, list[str]]] = None,
        editor_text_corpus: Optional[dict[str, list[str]]] = None,
        partition: Optional[dict[str, int]] = None,
    ) -> None:
        """Dispatcher: build topic vectors appropriate for ``graph.datasetSource``.

        Calls the dataset-specific builder and stores vectors in place on
        ``graph.nodes[uid].topicVector``.

        Args:
            graph:                 :class:`InteractionGraph` whose nodes will
                                   have their ``topicVector`` populated.
            subreddit_text_corpus: Required for ``"reddit_body"`` source.
                                   Dict mapping subreddit name → list of body
                                   text strings.
            editor_text_corpus:    Required for ``"wiki_rfa"`` source.
                                   Dict mapping editor userId → list of TXT
                                   comment strings.
            partition:             Required for ``"congress"`` source.
                                   Dict mapping userId → communityId (int).

        Raises:
            Nothing — unsupported dataset sources are logged and skipped.
        """
        source = graph.datasetSource

        if source == _REDDIT_BODY_SOURCE:
            if subreddit_text_corpus is None:
                logger.warning(
                    "TopicVectorService.build_topic_vectors: 'reddit_body' source "
                    "requires subreddit_text_corpus — skipping topic vector build."
                )
                return
            self.build_reddit_topic_vectors(graph, subreddit_text_corpus)

        elif source == _WIKI_RFA_SOURCE:
            if editor_text_corpus is None:
                logger.warning(
                    "TopicVectorService.build_topic_vectors: 'wiki_rfa' source "
                    "requires editor_text_corpus — skipping topic vector build."
                )
                return
            self.build_wiki_rfa_topic_vectors(graph, editor_text_corpus)

        elif source == _CONGRESS_SOURCE:
            if partition is None:
                logger.warning(
                    "TopicVectorService.build_topic_vectors: 'congress' source "
                    "requires a partition dict — skipping topic vector build."
                )
                return
            self.build_congress_topic_vectors(graph, partition)

        else:
            logger.warning(
                "TopicVectorService.build_topic_vectors: dataset source '%s' is "
                "not supported for topic vector construction — no vectors assigned.",
                source,
            )

    # ------------------------------------------------------------------
    # Reddit body — TF-IDF (or sentence-transformers)
    # ------------------------------------------------------------------

    def build_reddit_topic_vectors(
        self,
        graph: InteractionGraph,
        subreddit_text_corpus: dict[str, list[str]],
        *,
        use_sentence_transformers: bool = False,
    ) -> object:
        """Fit topic vectors from the Reddit body subreddit text corpus.

        Each document in the corpus is the concatenation of all body texts for
        a given subreddit.  A single ``TfidfVectorizer`` is fit on all
        subreddit documents; the resulting per-subreddit TF-IDF row vector is
        stored in ``node.topicVector``.

        Nodes whose userId is not present in ``subreddit_text_corpus`` receive
        a zero vector of length ``max_features`` (5000 by default).

        Args:
            graph:                  :class:`InteractionGraph` with
                                    ``datasetSource == "reddit_body"``.
            subreddit_text_corpus:  Dict mapping subreddit name →
                                    list of body text strings.
            use_sentence_transformers: When ``True`` and
                                    ``sentence-transformers`` is installed,
                                    use ``SentenceTransformer`` embeddings
                                    instead of TF-IDF.  Defaults to ``False``.

        Returns:
            The fitted ``TfidfVectorizer`` (or ``SentenceTransformer`` model
            when ``use_sentence_transformers=True``).  Returned for inspection
            and potential reuse.
        """
        if not subreddit_text_corpus:
            logger.warning(
                "TopicVectorService.build_reddit_topic_vectors: "
                "subreddit_text_corpus is empty — all nodes will get zero vectors."
            )

        # Build a per-subreddit document: join all body texts into one string
        subreddit_docs: dict[str, str] = {
            subreddit: " ".join(texts)
            for subreddit, texts in subreddit_text_corpus.items()
            if texts
        }

        if use_sentence_transformers:
            return self._build_reddit_sentence_transformer_vectors(graph, subreddit_docs)

        return self._build_reddit_tfidf_vectors(graph, subreddit_docs)

    def _build_reddit_tfidf_vectors(
        self,
        graph: InteractionGraph,
        subreddit_docs: dict[str, str],
    ) -> object:
        """Internal helper: fit TF-IDF on subreddit documents and populate nodes.

        Args:
            graph:          :class:`InteractionGraph` to update in place.
            subreddit_docs: Dict mapping subreddit name → concatenated doc str.

        Returns:
            Fitted ``TfidfVectorizer``.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

        if not subreddit_docs:
            # No corpus: assign zero vectors to all nodes
            zero_vector: list[float] = [0.0] * _MAX_FEATURES
            for node in graph.nodes.values():
                node.topicVector = list(zero_vector)
            # Return an unfitted vectorizer for interface consistency
            return TfidfVectorizer(max_features=_MAX_FEATURES)

        subreddits = list(subreddit_docs.keys())
        documents = [subreddit_docs[s] for s in subreddits]

        vectorizer = TfidfVectorizer(max_features=_MAX_FEATURES)
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Build a lookup: subreddit → row index in the matrix
        subreddit_to_idx: dict[str, int] = {s: i for i, s in enumerate(subreddits)}

        # Determine zero-vector length from the actual vocabulary (may be
        # smaller than max_features when the corpus is very small)
        vector_length = tfidf_matrix.shape[1]
        zero_vector_fitted: list[float] = [0.0] * vector_length

        for user_id, node in graph.nodes.items():
            if user_id in subreddit_to_idx:
                row_idx = subreddit_to_idx[user_id]
                # Convert sparse row to dense list[float]
                node.topicVector = tfidf_matrix[row_idx].toarray()[0].tolist()
            else:
                # Node absent from corpus → zero vector
                node.topicVector = list(zero_vector_fitted)

        logger.info(
            "TopicVectorService.build_reddit_topic_vectors: assigned TF-IDF "
            "vectors (dim=%d) to %d nodes; %d nodes had no corpus entry.",
            vector_length,
            len(graph.nodes),
            sum(
                1
                for uid in graph.nodes
                if uid not in subreddit_to_idx
            ),
        )

        return vectorizer

    def _build_reddit_sentence_transformer_vectors(
        self,
        graph: InteractionGraph,
        subreddit_docs: dict[str, str],
    ) -> object:
        """Internal helper: use SentenceTransformer embeddings for subreddit docs.

        Falls back to TF-IDF if ``sentence-transformers`` is not installed.

        Args:
            graph:          :class:`InteractionGraph` to update in place.
            subreddit_docs: Dict mapping subreddit name → concatenated doc str.

        Returns:
            Fitted ``SentenceTransformer`` model or ``TfidfVectorizer`` on
            fallback.
        """
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "TopicVectorService: sentence-transformers not installed; "
                "falling back to TF-IDF for Reddit topic vectors."
            )
            return self._build_reddit_tfidf_vectors(graph, subreddit_docs)

        if not subreddit_docs:
            zero_vector: list[float] = [0.0] * 384  # all-MiniLM-L6-v2 dim
            for node in graph.nodes.values():
                node.topicVector = list(zero_vector)
            return SentenceTransformer(_SENTENCE_TRANSFORMER_MODEL)

        model = SentenceTransformer(_SENTENCE_TRANSFORMER_MODEL)

        subreddits = list(subreddit_docs.keys())
        documents = [subreddit_docs[s] for s in subreddits]

        embeddings = model.encode(documents, show_progress_bar=False)

        subreddit_to_idx: dict[str, int] = {s: i for i, s in enumerate(subreddits)}
        embedding_dim = embeddings.shape[1]
        zero_vec: list[float] = [0.0] * embedding_dim

        for user_id, node in graph.nodes.items():
            if user_id in subreddit_to_idx:
                idx = subreddit_to_idx[user_id]
                node.topicVector = embeddings[idx].tolist()
            else:
                node.topicVector = list(zero_vec)

        logger.info(
            "TopicVectorService: assigned sentence-transformer vectors "
            "(dim=%d) to %d nodes.",
            embedding_dim,
            len(graph.nodes),
        )

        return model

    # ------------------------------------------------------------------
    # Wiki-RfA — TF-IDF on editor TXT comments
    # ------------------------------------------------------------------

    def build_wiki_rfa_topic_vectors(
        self,
        graph: InteractionGraph,
        editor_text_corpus: dict[str, list[str]],
    ) -> object:
        """Fit topic vectors from the wiki-RfA editor TXT comment corpus.

        Each document is the concatenation of all TXT comments written by one
        Wikipedia editor.  A single ``TfidfVectorizer`` is fit on all editor
        documents; the resulting per-editor TF-IDF row vector is stored in
        ``node.topicVector``.

        Nodes whose userId is not present in ``editor_text_corpus`` receive a
        zero vector of length equal to the fitted vocabulary size.

        Args:
            graph:               :class:`InteractionGraph` with
                                 ``datasetSource == "wiki_rfa"``.
            editor_text_corpus:  Dict mapping editor userId →
                                 list of TXT comment strings.

        Returns:
            The fitted ``TfidfVectorizer``.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

        if not editor_text_corpus:
            logger.warning(
                "TopicVectorService.build_wiki_rfa_topic_vectors: "
                "editor_text_corpus is empty — all nodes will get zero vectors."
            )
            zero_vector: list[float] = [0.0] * _MAX_FEATURES
            for node in graph.nodes.values():
                node.topicVector = list(zero_vector)
            return TfidfVectorizer(max_features=_MAX_FEATURES)

        # Build per-editor documents
        editor_docs: dict[str, str] = {
            editor: " ".join(texts)
            for editor, texts in editor_text_corpus.items()
            if texts
        }

        if not editor_docs:
            zero_vector = [0.0] * _MAX_FEATURES
            for node in graph.nodes.values():
                node.topicVector = list(zero_vector)
            return TfidfVectorizer(max_features=_MAX_FEATURES)

        editors = list(editor_docs.keys())
        documents = [editor_docs[e] for e in editors]

        vectorizer = TfidfVectorizer(max_features=_MAX_FEATURES)
        tfidf_matrix = vectorizer.fit_transform(documents)

        editor_to_idx: dict[str, int] = {e: i for i, e in enumerate(editors)}
        vector_length = tfidf_matrix.shape[1]
        zero_vec: list[float] = [0.0] * vector_length

        for user_id, node in graph.nodes.items():
            if user_id in editor_to_idx:
                row_idx = editor_to_idx[user_id]
                node.topicVector = tfidf_matrix[row_idx].toarray()[0].tolist()
            else:
                node.topicVector = list(zero_vec)

        logger.info(
            "TopicVectorService.build_wiki_rfa_topic_vectors: assigned TF-IDF "
            "vectors (dim=%d) to %d nodes; %d nodes had no corpus entry.",
            vector_length,
            len(graph.nodes),
            sum(1 for uid in graph.nodes if uid not in editor_to_idx),
        )

        return vectorizer

    # ------------------------------------------------------------------
    # Congress — party affiliation proxy
    # ------------------------------------------------------------------

    def build_congress_topic_vectors(
        self,
        graph: InteractionGraph,
        partition: dict[str, int],
    ) -> None:
        """Assign categorical topic vectors based on Louvain community membership.

        Since the Congress dataset contains no body text, community membership
        (Democrat / Republican) is used as a two-dimensional proxy topic vector:

        - Larger community  → ``[1.0, 0.0]``  (Democrat proxy)
        - Smaller community → ``[0.0, 1.0]``  (Republican proxy)
        - All other nodes   → ``[0.5, 0.5]``  (neutral / third party)

        The two dominant communities are identified by the number of nodes in
        ``graph.nodes`` that belong to each community.

        Args:
            graph:     :class:`InteractionGraph` with
                       ``datasetSource == "congress"``.
            partition: Dict mapping userId → communityId (int).  Should be the
                       raw Louvain output from
                       :class:`~community.service.CommunityDetectionService`.
        """
        if not partition:
            logger.warning(
                "TopicVectorService.build_congress_topic_vectors: "
                "partition is empty — assigning neutral [0.5, 0.5] to all nodes."
            )
            for node in graph.nodes.values():
                node.topicVector = [0.5, 0.5]
            return

        # Count how many graph nodes fall into each community
        community_counts: dict[int, int] = {}
        for user_id in graph.nodes:
            if user_id in partition:
                cid = partition[user_id]
                community_counts[cid] = community_counts.get(cid, 0) + 1

        if not community_counts:
            logger.warning(
                "TopicVectorService.build_congress_topic_vectors: no graph nodes "
                "are present in the partition — assigning neutral [0.5, 0.5] to all."
            )
            for node in graph.nodes.values():
                node.topicVector = [0.5, 0.5]
            return

        # Sort communities by node count (descending); take the two largest
        sorted_communities = sorted(
            community_counts.items(), key=lambda kv: kv[1], reverse=True
        )

        dominant_communities: list[tuple[int, int]] = sorted_communities[:2]

        if len(dominant_communities) < 2:
            # Only one community found — assign [1.0, 0.0] to it, neutral to rest
            only_cid = dominant_communities[0][0]
            for user_id, node in graph.nodes.items():
                cid = partition.get(user_id)
                if cid == only_cid:
                    node.topicVector = [1.0, 0.0]
                else:
                    node.topicVector = [0.5, 0.5]
            logger.info(
                "TopicVectorService.build_congress_topic_vectors: only one "
                "dominant community found (id=%s); using [1.0, 0.0] for it.",
                only_cid,
            )
            return

        larger_cid = dominant_communities[0][0]   # Democrat proxy
        smaller_cid = dominant_communities[1][0]  # Republican proxy

        # Vector assignment
        _LARGER_VEC: list[float] = [1.0, 0.0]
        _SMALLER_VEC: list[float] = [0.0, 1.0]
        _NEUTRAL_VEC: list[float] = [0.5, 0.5]

        for user_id, node in graph.nodes.items():
            cid = partition.get(user_id)
            if cid == larger_cid:
                node.topicVector = list(_LARGER_VEC)
            elif cid == smaller_cid:
                node.topicVector = list(_SMALLER_VEC)
            else:
                node.topicVector = list(_NEUTRAL_VEC)

        logger.info(
            "TopicVectorService.build_congress_topic_vectors: assigned "
            "[1.0, 0.0] to community %s (%d nodes), [0.0, 1.0] to community "
            "%s (%d nodes), [0.5, 0.5] to remaining nodes.",
            larger_cid,
            dominant_communities[0][1],
            smaller_cid,
            dominant_communities[1][1],
        )
