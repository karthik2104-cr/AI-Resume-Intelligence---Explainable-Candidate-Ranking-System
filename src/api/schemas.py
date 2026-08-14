"""Pydantic request/response schemas for the screening API.

Keep schemas minimal and privacy-conscious: do not return PII such as
email or phone unless explicitly requested via configuration.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field("ok")


class ScreenRequest(BaseModel):
    job_text: str
    job_title: Optional[str] = None
    top_k: Optional[int] = None
    explain: Optional[bool] = False


class CandidateScore(BaseModel):
    candidate_id: str
    overall_score: float
    component_scores: Dict[str, float]
    matched_skills: List[str] = []
    missing_required_skills: List[str] = []
    missing_preferred_skills: List[str] = []
    parsing_quality: Optional[str] = None
    retrieval_similarity: Optional[float] = None


class ScreenResponse(BaseModel):
    job_title: Optional[str]
    top_k: int
    candidates: List[CandidateScore]
    explanations: Optional[Dict[str, Any]] = None
