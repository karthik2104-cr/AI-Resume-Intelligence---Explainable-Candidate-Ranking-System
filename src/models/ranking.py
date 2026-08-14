"""Ranking result models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.models.job import ParsedJobDescription
from src.models.matching import MatchResult
from src.models.resume import ParsedResume


class CandidateRankEntry(BaseModel):
    rank: int
    candidate_id: Optional[str] = None
    candidate_name: Optional[str] = None
    overall_score: float
    required_skill_coverage_pct: Optional[float] = None
    experience_score: Optional[float] = None
    semantic_score: Optional[float] = None
    baseline_tfidf_score: Optional[float] = None
    match_result: MatchResult


class RankingRequest(BaseModel):
    job: ParsedJobDescription
    candidates: list[ParsedResume]
    candidate_ids: list[str] = Field(default_factory=list)
    top_k: Optional[int] = None


class RankingResult(BaseModel):
    job_title: Optional[str] = None
    entries: list[CandidateRankEntry] = Field(default_factory=list)
    total_candidates: int = 0
    ranker_name: str = "unknown"
    warnings: list[str] = Field(default_factory=list)
