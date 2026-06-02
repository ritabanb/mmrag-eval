from __future__ import annotations

import math


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    hits = [1 if img in relevant else 0 for img in retrieved[:k]]
    dcg = _dcg(hits)
    ideal = [1] * min(len(relevant), k)
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Recall at K."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def score(retrieved_images: list[str], relevant_images: list[str], k: int = 5) -> dict:
    """Return nDCG@K and Recall@K for a single query."""
    return {
        "ndcg_at_k": ndcg_at_k(retrieved_images, relevant_images, k),
        "recall_at_k": recall_at_k(retrieved_images, relevant_images, k),
        "k": k,
    }
