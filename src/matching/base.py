"""Matching engine abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.job import ParsedJobDescription
from src.models.matching import MatchResult
from src.models.resume import ParsedResume


class MatchingEngine(ABC):
    """Abstract base for resume–job matching strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable matcher identifier."""

    @abstractmethod
    def match(
        self,
        resume: ParsedResume,
        job: ParsedJobDescription,
        candidate_id: str | None = None,
    ) -> MatchResult:
        """Compute match between one resume and one job."""

    @abstractmethod
    def match_batch(
        self,
        resumes: list[ParsedResume],
        job: ParsedJobDescription,
        candidate_ids: list[str] | None = None,
    ) -> list[MatchResult]:
        """Compute matches for multiple resumes against one job."""
