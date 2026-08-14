"""Job description parsing abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.job import JobDescription, ParsedJobDescription


class JobDescriptionParser(ABC):
    """Abstract base for job description analysis."""

    @abstractmethod
    def parse(self, job: JobDescription) -> ParsedJobDescription:
        """Parse raw job description into structured form."""

    def parse_text(self, text: str, title: str | None = None) -> ParsedJobDescription:
        """Convenience method for plain-text job descriptions."""
        return self.parse(JobDescription(title=title, raw_text=text))
