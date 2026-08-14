import pytest
from src.evaluation.metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k


def test_precision_and_recall_at_k():
    ranked = ["a", "b", "c", "d"]
    relevance = {"a": 1, "b": 0, "c": 1, "d": 0}
    assert precision_at_k(ranked, relevance, 2) == 0.5
    assert recall_at_k(ranked, relevance, 3) == 1.0  # two relevant, both in top3


def test_mrr_single():
    ranked = ["x", "y", "z"]
    relevance = {"y": 1}
    assert mrr([ranked], [relevance]) == 1.0 / 2


def test_ndcg_at_k_basic():
    ranked = ["a", "b", "c"]
    relevance = {"a": 2, "b": 1, "c": 0}
    ndcg = ndcg_at_k(ranked, relevance, 3)
    assert ndcg > 0


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k([], {}, 0)
