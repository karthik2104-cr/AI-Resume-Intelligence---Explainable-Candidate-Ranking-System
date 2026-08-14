from __future__ import annotations

import pytest

from src.utils.config import HybridMatchingWeightsConfig
from src.models.job import ParsedJobDescription
from src.models.resume import ParsedResume
from src.ranking.hybrid_ranker import HybridRanker


def test_weight_validation_valid():
    cfg = HybridMatchingWeightsConfig()
    # default should sum to ~1.0 and not raise
    cfg.validate_weights()


def test_weight_validation_invalid_sum():
    cfg = HybridMatchingWeightsConfig(
        required_skill_coverage=0.5,
        preferred_skill_coverage=0.5,
        semantic_similarity=0.5,
        experience_compatibility=0.0,
        education_compatibility=0.0,
        seniority_compatibility=0.0,
    )
    with pytest.raises(ValueError):
        cfg.validate_weights()


def test_renormalization_and_ranking_simple():
    # create a simple job and two resumes where one has closer text
    job = ParsedJobDescription(title="Data Scientist", raw_text="machine learning python pandas")

    r1 = ParsedResume(name="Alice", raw_text="machine learning python pandas", years_experience=3.0)
    r2 = ParsedResume(name="Bob", raw_text="accounting finance bookkeeping", years_experience=5.0)

    ranker = HybridRanker()
    req = type("Req", (), {"job": job, "candidates": [r1, r2], "candidate_ids": ["A", "B"], "top_k": None})
    result = ranker.rank(req)

    assert result.total_candidates == 2
    # Alice should rank above Bob due to semantic similarity
    assert result.entries[0].candidate_name in ("Alice", "Alice")
