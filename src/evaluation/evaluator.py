"""Basic evaluator harness.

The Evaluator compares different ranking methods on a single controlled
fixture. Methods are callables that accept (job_text, candidate_texts)
and return a ranked list of candidate_ids (highest first).

The module intentionally keeps a minimal dependency surface so unit tests
can run without heavy ML dependencies by mocking method callables.
"""
from typing import Callable, Dict, List, Tuple
from .metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k


class Evaluator:
    def __init__(self, k: int = 5):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = k

    def evaluate_once(
        self,
        job_text: str,
        candidate_texts: Dict[str, str],
        relevance: Dict[str, int],
        methods: Dict[str, Callable[[str, Dict[str, str]], List[str]]],
    ) -> Dict[str, Dict[str, float]]:
        """Run all provided methods on the single fixture and return metrics.

        Returns a mapping method_name -> metrics mapping (precision, recall, mrr, ndcg).
        """
        results: Dict[str, Dict[str, float]] = {}
        ranked_lists: List[List[str]] = []
        relevances: List[Dict[str, int]] = []

        for name, method in methods.items():
            ranked = method(job_text, candidate_texts)
            # ensure deterministic list of candidate ids (filter unknown)
            ranked = [c for c in ranked if c in candidate_texts]
            ranked_lists.append(ranked)
            relevances.append(relevance)

            p = precision_at_k(ranked, relevance, self.k)
            r = recall_at_k(ranked, relevance, self.k)
            rr = mrr([ranked], [relevance])
            ndcg = ndcg_at_k(ranked, relevance, self.k)
            results[name] = {
                "precision_at_k": p,
                "recall_at_k": r,
                "mrr": rr,
                "ndcg": ndcg,
            }

        return results


def pretty_print_results(results: Dict[str, Dict[str, float]]) -> None:
    names = list(results.keys())
    print("Model\tPrecision@K\tRecall@K\tMRR\tNDCG@K")
    print("-----------------------------------------------------------")
    for name in names:
        m = results[name]
        print(f"{name}\t{m['precision_at_k']:.3f}\t{m['recall_at_k']:.3f}\t{m['mrr']:.3f}\t{m['ndcg']:.3f}")


# Small helper to wrap naive methods that rank by simple heuristics

def rank_by_keyword_overlap(job_text: str, candidate_texts: Dict[str, str]) -> List[str]:
    """Naive lexical overlap ranking: counts shared tokens (space-split).

    serves as a minimal TF-IDF-like baseline in case TF-IDF is unavailable.
    """
    job_tokens = set(job_text.lower().split())
    scores = []
    for cid, text in candidate_texts.items():
        tokens = set(text.lower().split())
        overlap = len(job_tokens & tokens)
        scores.append((cid, overlap))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return [cid for cid, _ in scores]
