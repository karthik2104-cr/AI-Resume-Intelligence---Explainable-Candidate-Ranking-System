"""Common ranking and retrieval metrics.

Implements Precision@K, Recall@K, MRR, and NDCG@K.
Functions work with a ranked list of candidate ids and a relevance mapping
(candidate_id -> relevance_score, integer where >0 is relevant).
"""
from math import log2
from typing import List, Dict


def precision_at_k(ranked: List[str], relevance: Dict[str, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    topk = ranked[:k]
    if not topk:
        return 0.0
    relevant = sum(1 for c in topk if relevance.get(c, 0) > 0)
    return relevant / len(topk)


def recall_at_k(ranked: List[str], relevance: Dict[str, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    total_relevant = sum(1 for v in relevance.values() if v > 0)
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = sum(1 for c in ranked[:k] if relevance.get(c, 0) > 0)
    return retrieved_relevant / total_relevant


def reciprocal_rank(ranked: List[str], relevance: Dict[str, int]) -> float:
    """Return reciprocal rank (1 / rank_of_first_relevant), or 0 if none."""
    for idx, c in enumerate(ranked, start=1):
        if relevance.get(c, 0) > 0:
            return 1.0 / idx
    return 0.0


def mrr(list_of_ranked: List[List[str]], list_of_relevance: List[Dict[str, int]]) -> float:
    """Mean Reciprocal Rank over multiple queries."""
    if not list_of_ranked:
        return 0.0
    if len(list_of_ranked) != len(list_of_relevance):
        raise ValueError("ranked lists and relevance lists must be same length")
    rr_sum = 0.0
    for ranked, rel in zip(list_of_ranked, list_of_relevance):
        rr_sum += reciprocal_rank(ranked, rel)
    return rr_sum / len(list_of_ranked)


def dcg_at_k(ranked: List[str], relevance: Dict[str, int], k: int) -> float:
    """Discounted Cumulative Gain up to position k (1-based positions)."""
    topk = ranked[:k]
    dcg = 0.0
    for i, c in enumerate(topk, start=1):
        rel = relevance.get(c, 0)
        dcg += (2 ** rel - 1) / log2(i + 1)
    return dcg


def ndcg_at_k(ranked: List[str], relevance: Dict[str, int], k: int) -> float:
    """Normalized DCG up to k. Uses relevance mapping and ideal ordering."""
    if k <= 0:
        raise ValueError("k must be > 0")
    ideal_rels = sorted([v for v in relevance.values() if v > 0], reverse=True)
    if not ideal_rels:
        return 0.0
    ideal_ranked = [str(i) for i, _ in enumerate(ideal_rels)]  # identifiers not used; only values matter
    idcg = 0.0
    for i, rel in enumerate(ideal_rels[:k], start=1):
        idcg += (2 ** rel - 1) / log2(i + 1)
    if idcg == 0.0:
        return 0.0
    dcg = dcg_at_k(ranked, relevance, k)
    return dcg / idcg
