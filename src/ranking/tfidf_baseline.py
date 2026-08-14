"""Baseline ranking using TF-IDF matcher."""

from __future__ import annotations

from typing import Optional

from src.matching.tfidf_baseline import TfidfBaselineMatcher
from src.models.ranking import CandidateRankEntry, RankingRequest, RankingResult
from src.ranking.base import RankingEngine
from src.utils.config import get_settings


class TfidfBaselineRanker(RankingEngine):
    """Rank candidates using TF-IDF cosine similarity scores."""

    def __init__(self, matcher: Optional[TfidfBaselineMatcher] = None) -> None:
        self._matcher = matcher or TfidfBaselineMatcher()

    @property
    def name(self) -> str:
        return "tfidf_baseline_ranker"

    def rank(self, request: RankingRequest) -> RankingResult:
        if not request.candidates:
            return RankingResult(
                job_title=request.job.title,
                entries=[],
                total_candidates=0,
                ranker_name=self.name,
                warnings=["No candidates provided."],
            )

        top_k = request.top_k or get_settings().ranking.default_top_k
        match_results = self._matcher.match_batch(
            request.candidates,
            request.job,
            request.candidate_ids or None,
        )

        sorted_results = sorted(
            match_results,
            key=lambda m: m.scores.overall,
            reverse=True,
        )[:top_k]

        entries: list[CandidateRankEntry] = []
        for rank_idx, match in enumerate(sorted_results, start=1):
            entries.append(
                CandidateRankEntry(
                    rank=rank_idx,
                    candidate_id=match.candidate_id,
                    candidate_name=match.candidate_name,
                    overall_score=match.scores.overall,
                    baseline_tfidf_score=match.scores.baseline_tfidf,
                    semantic_score=match.scores.semantic,
                    match_result=match,
                )
            )

        return RankingResult(
            job_title=request.job.title,
            entries=entries,
            total_candidates=len(request.candidates),
            ranker_name=self.name,
        )
