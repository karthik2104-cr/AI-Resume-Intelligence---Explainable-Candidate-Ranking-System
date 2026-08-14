"""Evaluation script for v2.

Usage: run from v2/ root with python scripts/evaluate.py
This script uses the small controlled fixture. It will attempt to call into
existing matchers if available; when running in CI or locally without heavy
models, consider providing mocked or lightweight callables.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/evaluate.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.datasets import load_small_fixture
from src.evaluation.evaluator import Evaluator, pretty_print_results, rank_by_keyword_overlap


def main():
    jd, candidates, relevance = load_small_fixture()

    # Define simple methods for demonstration. 
    methods = {
        "keyword_overlap": rank_by_keyword_overlap,
    }

    try:
        from src.matching.tfidf_baseline import TfidfBaselineMatcher
        from src.models.job import ParsedJobDescription
        from src.models.resume import ParsedResume

        def tfidf_method(job_text, candidate_texts):
            job = ParsedJobDescription(raw_text=job_text, title="Evaluation Job")
            ids = list(candidate_texts.keys())
            resumes = [
                ParsedResume(raw_text=text, candidate_id=cid)
                for cid, text in candidate_texts.items()
            ]
            matcher = TfidfBaselineMatcher()
            results = matcher.match_batch(resumes, job, ids)
            scored = [
                (r.candidate_id, r.scores.overall or r.scores.baseline_tfidf or 0.0)
                for r in results
            ]
            scored.sort(key=lambda x: (-x[1], x[0] or ""))
            return [cid for cid, _ in scored if cid]

        methods["tfidf_baseline"] = tfidf_method
    except Exception:
        pass

    evaluator = Evaluator(k=5)
    results = evaluator.evaluate_once(jd, candidates, relevance, methods)
    pretty_print_results(results)


if __name__ == "__main__":
    main()
