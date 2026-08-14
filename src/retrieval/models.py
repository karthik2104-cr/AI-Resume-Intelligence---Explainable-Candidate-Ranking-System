from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, Field, validator


class RetrievalMetadata(BaseModel):
    parsing_quality: Optional[str] = None
    normalized_skills: list[str] = Field(default_factory=list)
    years_experience: Optional[float] = None
    education_levels: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    # allow arbitrary extra metadata
    extra: Dict[str, Any] = Field(default_factory=dict)


class CandidateRecord(BaseModel):
    candidate_id: str
    embedding: Optional[list[float]] = None
    metadata: RetrievalMetadata = Field(default_factory=RetrievalMetadata)

    @validator("candidate_id")
    def id_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("candidate_id must be non-empty")
        return v


class RetrievalResult(BaseModel):
    candidate_id: str
    similarity: float
    rank: int
    metadata: RetrievalMetadata

    class Config:
        arbitrary_types_allowed = True
