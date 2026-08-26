import numpy as np
from scipy.sparse import csr_matrix
import pytest

from app.recommender import (
    precision_at_k, recall_at_k, ndcg_at_k, minmax_norm,
    hybrid_score_for_user, recommend_for_user,
)


def test_precision_at_k():
    assert precision_at_k([1, 2, 3], {1}, 3) == pytest.approx(1 / 3)


def test_recall_at_k():
    assert recall_at_k([5, 6, 7], {6}, 3) == 1.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0


def test_ndcg_rewards_higher_rank():
    assert ndcg_at_k([1, 2, 3], {1}, 3) > ndcg_at_k([2, 3, 1], {1}, 3)


def test_minmax_norm_scales_to_zero_one():
    arr = np.array([2.0, 4.0, 6.0])
    normed = minmax_norm(arr)
    assert normed.min() == pytest.approx(0.0)
    assert normed.max() == pytest.approx(1.0)


def test_minmax_norm_handles_constant_array():
    arr = np.array([5.0, 5.0, 5.0])
    normed = minmax_norm(arr)
    assert np.all(normed == 0.0)  # must not divide by zero


class DummyALSModel:
    """Deterministic stand-in for a trained implicit ALS model."""
    def __init__(self, n_users, n_items, n_factors=4):
        self.user_factors = np.ones((n_users, n_factors))
        self.item_factors = np.tile(np.arange(n_items).reshape(-1, 1), (1, n_factors)).astype(float)


def test_hybrid_blends_als_and_content_scores():
    n_users, n_items = 3, 5
    als_model = DummyALSModel(n_users, n_items)
    train_matrix = csr_matrix((n_users, n_items))
    content_sim = np.eye(n_items)  # identity: content score = self-similarity only

    scores_pure_als = hybrid_score_for_user(0, als_model, content_sim, train_matrix, alpha=1.0)
    scores_pure_content = hybrid_score_for_user(0, als_model, content_sim, train_matrix, alpha=0.0)
    # alpha=1.0 and alpha=0.0 should generally produce different rankings given different inputs
    assert not np.array_equal(scores_pure_als, scores_pure_content)


def test_recommend_for_user_excludes_seen_items():
    n_users, n_items = 3, 5
    als_model = DummyALSModel(n_users, n_items)
    rows, cols, vals = [0, 0], [4, 3], [1.0, 1.0]
    train_matrix = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    content_sim = np.eye(n_items)
    popularity_ranking = np.arange(n_items)

    recs = recommend_for_user(0, n_users, als_model, content_sim, train_matrix, popularity_ranking, k=3)
    assert 4 not in recs and 3 not in recs


def test_recommend_for_user_cold_start_uses_popularity():
    n_users, n_items = 3, 5
    als_model = DummyALSModel(n_users, n_items)
    train_matrix = csr_matrix((n_users, n_items))
    content_sim = np.eye(n_items)
    popularity_ranking = np.array([2, 4, 0, 1, 3])

    recs = recommend_for_user(999, n_users, als_model, content_sim, train_matrix, popularity_ranking, k=3)
    assert recs == [2, 4, 0]
