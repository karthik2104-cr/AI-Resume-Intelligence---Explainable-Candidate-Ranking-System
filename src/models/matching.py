"""Matching and scoring result models."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from src.models.job import ParsedJobDescription
from src.models.resume import ParsedResume


class MatchRequest(BaseModel):
    resume: ParsedResume
    job: ParsedJobDescription
    candidate_id: Optional[str] = None


class ComponentScores(BaseModel):
    """Decomposed scoring components populated by matchers and HybridRanker."""

    overall: float = 0.0
    skill: Optional[float] = None
    semantic: Optional[float] = None
    experience: Optional[float] = None
    education: Optional[float] = None
    project: Optional[float] = None
    baseline_tfidf: Optional[float] = None


class SkillCoverage(BaseModel):
    required_coverage_pct: float = 0.0
    preferred_coverage_pct: float = 0.0
    matched_required: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    matched_preferred: list[str] = Field(default_factory=list)
    missing_preferred: list[str] = Field(default_factory=list)
    partial_matches: list[str] = Field(default_factory=list)


class MatchExplanation(BaseModel):
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    experience_gaps: list[str] = Field(default_factory=list)
    relevant_projects: list[str] = Field(default_factory=list)
    summary: Optional[str] = None
    llm_generated: bool = False


class MatchResult(BaseModel):
    candidate_id: Optional[str] = None
    candidate_name: Optional[str] = None
    scores: ComponentScores
    skill_coverage: Optional[SkillCoverage] = None
    explanation: Optional[MatchExplanation] = None
    matcher_name: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticMatchResult(BaseModel):
    semantic_similarity: float = 0.0
    section_scores: dict[str, float] = Field(default_factory=dict)
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    normalized_embeddings: Optional[bool] = None
    compared_sections: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
