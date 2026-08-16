from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievalMetadata(BaseModel):
    parsing_quality: Optional[str] = None
    normalized_skills: list[str] = Field(default_factory=list)
    years_experience: Optional[float] = None
    education_levels: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class CandidateRecord(BaseModel):
    candidate_id: str
    embedding: Optional[list[float]] = None
    metadata: RetrievalMetadata = Field(default_factory=RetrievalMetadata)

    @field_validator("candidate_id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("candidate_id must be non-empty")
        return v


class RetrievalResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    candidate_id: str
    similarity: float
    rank: int
    metadata: RetrievalMetadata
