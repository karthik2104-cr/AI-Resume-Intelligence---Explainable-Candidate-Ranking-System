"""Ranking engine abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.ranking import RankingRequest, RankingResult


class RankingEngine(ABC):
    """Abstract base for candidate ranking."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable ranker identifier."""

    @abstractmethod
    def rank(self, request: RankingRequest) -> RankingResult:
        """Rank candidates for a job description."""
