"""Persistence abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.models.job import ParsedJobDescription
from src.models.matching import MatchResult
from src.models.resume import ParsedResume


class PersistenceStore(ABC):
    """Abstract base for storing candidates, jobs, and match results."""

    @abstractmethod
    def save_resume(self, resume: ParsedResume, candidate_id: str) -> None:
        """Persist parsed resume metadata."""

    @abstractmethod
    def get_resume(self, candidate_id: str) -> Optional[ParsedResume]:
        """Retrieve parsed resume by ID."""

    @abstractmethod
    def save_job(self, job: ParsedJobDescription, job_id: str) -> None:
        """Persist parsed job description."""

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[ParsedJobDescription]:
        """Retrieve parsed job by ID."""

    @abstractmethod
    def save_match_result(
        self,
        match_result: MatchResult,
        candidate_id: str,
        job_id: str,
    ) -> None:
        """Persist match result."""

    @abstractmethod
    def list_jobs(self) -> list[dict[str, Any]]:
        """List stored job descriptions."""
