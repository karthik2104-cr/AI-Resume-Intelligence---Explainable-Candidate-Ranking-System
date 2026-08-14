"""Pydantic models for explainability outputs and inputs."""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    source: str
    text: str
    score_component: Optional[str] = None


class ExplanationResult(BaseModel):
    summary: Optional[str] = None
    strengths: List[str] = Field(default_factory=list)
    required_skill_matches: List[str] = Field(default_factory=list)
    preferred_skill_matches: List[str] = Field(default_factory=list)
    skill_gaps: List[str] = Field(default_factory=list)
    experience_alignment: Optional[str] = None
    education_alignment: Optional[str] = None
    seniority_alignment: Optional[str] = None
    concerns: List[str] = Field(default_factory=list)
    interview_focus_areas: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    explanation_source: Literal["deterministic", "llm"]


class ExplanationInput(BaseModel):
    # Deterministic inputs — final scores and components must be provided
    match_result: dict
    component_scores: list[dict]
    hybrid_metadata: Optional[dict] = None
    skill_gap: Optional[dict] = None
    parsed_resume: Optional[dict] = None
    parsed_job: Optional[dict] = None
    ranking_info: Optional[dict] = None
    # free-form evidence list as fallback
    evidence: Optional[List[EvidenceItem]] = None
