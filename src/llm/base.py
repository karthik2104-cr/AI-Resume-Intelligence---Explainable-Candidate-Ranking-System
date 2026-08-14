"""LLM service abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.models.matching import MatchExplanation, MatchResult


class LLMServiceError(Exception):
    """Raised when LLM service fails."""


class LLMService(ABC):
    """
    Abstract LLM provider.

    LLM must NOT independently decide numerical scores.
    It consumes structured match results for natural-language output.
    """

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the LLM provider is configured and reachable."""

    @abstractmethod
    def summarize_match(
        self,
        match_result: MatchResult,
        context: Optional[dict[str, Any]] = None,
    ) -> MatchExplanation:
        """Generate natural-language summary from structured match data."""

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Request structured JSON output from the LLM."""
