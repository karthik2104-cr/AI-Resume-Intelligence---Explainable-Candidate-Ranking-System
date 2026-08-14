from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ComponentScore(BaseModel):
    name: str
    score: Optional[float] = None
    weight: Optional[float] = None
    applied_weight: Optional[float] = None
    weighted_contribution: Optional[float] = None
    evidence: list[str] = Field(default_factory=list)
    availability: str = "available"  # available | unavailable


class HybridMatchResult(BaseModel):
    overall_score: float
    overall_score_pct: float
    component_scores: list[ComponentScore] = Field(default_factory=list)
    skill_gap: dict[str, Any] | None = None
    semantic_result: dict[str, Any] | None = None
    explanation: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

