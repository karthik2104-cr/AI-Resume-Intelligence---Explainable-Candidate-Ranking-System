"""Scoring engine abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.job import ParsedJobDescription
from src.models.matching import ComponentScores
from src.models.resume import ParsedResume


class ScoringEngine(ABC):
    """Abstract base for hybrid multi-signal scoring (Phase 8+)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable scorer identifier."""

    @abstractmethod
    def score(
        self,
        resume: ParsedResume,
        job: ParsedJobDescription,
    ) -> ComponentScores:
        """Compute decomposed component scores."""
