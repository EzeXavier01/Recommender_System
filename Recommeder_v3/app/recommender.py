"""
Core recommender logic for the tuned Weighted Hybrid model (implicit ALS + content
similarity, alpha=0.7 — see the notebook Section 8 for how this was empirically selected).
Kept separate from the API layer (app/main.py) so it can be unit tested in isolation.
"""
import numpy as np
from scipy.sparse import csr_matrix

ALPHA = 0.7  # ALS weight in the blend; content weight is (1 - ALPHA). Tuned in the notebook.


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def hybrid_score_for_user(user_idx: int, als_model, content_sim: np.ndarray,
                           train_matrix: csr_matrix, alpha: float = ALPHA) -> np.ndarray:
    """Blended score for every item, for a known (non-cold-start) user."""
    als_scores = als_model.user_factors[user_idx] @ als_model.item_factors.T
    content_scores = np.asarray(train_matrix[user_idx] @ content_sim).flatten()
    return alpha * minmax_norm(als_scores) + (1 - alpha) * minmax_norm(content_scores)


def recommend_for_user(user_idx: int, n_users: int, als_model, content_sim: np.ndarray,
                        train_matrix: csr_matrix, popularity_ranking: np.ndarray,
                        k: int = 10, alpha: float = ALPHA) -> list[int]:
    """
    Top-level recommendation function used by the API.
    Falls back to the popularity ranking for cold-start users (unknown/out-of-range index).
    """
    if user_idx < 0 or user_idx >= n_users:
        return [int(i) for i in popularity_ranking[:k]]

    scores = hybrid_score_for_user(user_idx, als_model, content_sim, train_matrix, alpha)
    seen = train_matrix[user_idx].indices
    if len(seen):
        scores[seen] = -np.inf
    top_k = np.argsort(-scores)[:k]
    return [int(i) for i in top_k]


def score_new_item_for_user(user_idx: int, new_item_text: str, tfidf, text_features,
                             als_model, top_n_neighbors: int = 10):
    """
    Cold-start fallback for a BRAND-NEW item with zero purchase history — the hybrid above
    can't help here since there's no learned ALS vector for an item that was never in the
    training matrix. Finds the most content-similar existing items and borrows the user's
    ALS affinity for them, weighted by similarity.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    new_vec = tfidf.transform([new_item_text])
    sims = cosine_similarity(new_vec, text_features).flatten()
    neighbor_idx = np.argsort(-sims)[:top_n_neighbors]
    neighbor_sims = sims[neighbor_idx]
    if neighbor_sims.sum() == 0:
        return 0.0, neighbor_idx, neighbor_sims
    user_vec = als_model.user_factors[user_idx]
    neighbor_scores = als_model.item_factors[neighbor_idx] @ user_vec
    weighted_score = np.average(neighbor_scores, weights=neighbor_sims)
    return weighted_score, neighbor_idx, neighbor_sims


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1 / np.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
