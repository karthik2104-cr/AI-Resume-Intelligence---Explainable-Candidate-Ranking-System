"""Explainability abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.job import ParsedJobDescription
from src.models.matching import MatchExplanation, MatchResult
from src.models.resume import ParsedResume


class ExplainabilityEngine(ABC):
    """Abstract base for feature-based match explanations."""

    @abstractmethod
    def explain(
        self,
        resume: ParsedResume,
        job: ParsedJobDescription,
        match_result: MatchResult,
    ) -> MatchExplanation:
        """Generate explanation from calculated features — no fabrication."""
