"""
processing/semantic_grouper.py
-------------------------------
Stage 3 of the ML pipeline.
Groups articles about the same real-world event using cosine similarity.
Exactly as specified in doc section 5.2.

Similarity thresholds (from doc):
  0.95+      → near-duplicate, keep only best source
  0.82–0.94  → same event, different wording → group together
  0.60–0.81  → related topic, different events → keep separate
  < 0.60     → unrelated → always keep separate

Grouping cutoff: 0.82 (SIMILARITY_THRESHOLD in settings)
"""

import logging
import numpy as np
from typing import List, Dict

log = logging.getLogger(__name__)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise cosine similarity matrix.
    Embeddings must be L2-normalised (embedder.py does this).
    Normalised vectors: cosine_sim = dot product.
    """
    return np.dot(embeddings, embeddings.T)


def group_articles(
    articles: List[dict],
    grouping_threshold: float = 0.82,
    dedup_threshold: float = 0.95,
) -> List[List[dict]]:
    """
    Group articles into event clusters.

    Args:
        articles:           embedded article dicts (must have 'embedding')
        grouping_threshold: min similarity to group as same event (doc: 0.82)
        dedup_threshold:    similarity above which we keep only best source (doc: 0.95)

    Returns:
        List of groups. Each group is a list of article dicts about the same event.
        Groups are sorted by size descending (largest event cluster first).
    """
    if not articles:
        return []

    embeddings = np.array([a["embedding"] for a in articles], dtype=np.float32)
    n = len(articles)
    sim_matrix = cosine_similarity_matrix(embeddings)

    assigned = [False] * n
    groups: List[List[int]] = []

    # Greedy single-pass grouping — O(n²) but fine for batches up to ~2000
    for i in range(n):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if not assigned[j] and sim_matrix[i, j] >= grouping_threshold:
                group.append(j)
                assigned[j] = True
        groups.append(group)

    # Build article groups; deduplicate within each group
    article_groups = []
    total_deduped = 0
    for idx_list in groups:
        group_articles = [articles[i] for i in idx_list]
        deduped, removed = _dedup_group(group_articles, sim_matrix, idx_list, dedup_threshold)
        total_deduped += removed
        article_groups.append(deduped)

    # Sort: largest groups first (most-covered events surface first)
    article_groups.sort(key=len, reverse=True)

    log.info(
        "Grouper: %d articles → %d event clusters | %d near-dupes removed",
        n, len(article_groups), total_deduped,
    )
    return article_groups


def _dedup_group(
    group: List[dict],
    sim_matrix: np.ndarray,
    idx_list: List[int],
    dedup_threshold: float,
) -> tuple[List[dict], int]:
    """
    Within a group, remove near-duplicates (sim >= 0.95).
    Keep the article from the highest-trust source.
    """
    if len(group) == 1:
        return group, 0

    from config.settings import SOURCE_TRUST

    def trust_score(article: dict) -> float:
        source = (article.get("source") or {}).get("name", "").lower()
        for key, score in SOURCE_TRUST.items():
            if key in source:
                return score
        return SOURCE_TRUST["default"]

    keep = list(range(len(group)))
    removed = set()

    for i in range(len(idx_list)):
        if i in removed:
            continue
        for j in range(i + 1, len(idx_list)):
            if j in removed:
                continue
            if sim_matrix[idx_list[i], idx_list[j]] >= dedup_threshold:
                # Keep higher-trust source
                if trust_score(group[i]) >= trust_score(group[j]):
                    removed.add(j)
                else:
                    removed.add(i)
                    break

    kept = [group[i] for i in range(len(group)) if i not in removed]
    return kept, len(removed)