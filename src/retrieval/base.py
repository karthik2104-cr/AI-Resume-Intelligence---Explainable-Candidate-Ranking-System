from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.retrieval.models import CandidateRecord, RetrievalResult


class CandidateRetriever(ABC):
    """Abstract interface for candidate retrieval implementations."""

    @abstractmethod
    def index(self, record: CandidateRecord) -> None:
        """Index a candidate record containing embedding and metadata.

        May raise domain-specific RetrievalError subclasses on invalid input.
        """

    @abstractmethod
    def retrieve(self, query_embedding, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve top-K candidates for a given query embedding."""

    @abstractmethod
    def get_candidate(self, candidate_id: str) -> CandidateRecord:
        """Return candidate record or raise CandidateNotFoundError."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed candidates."""

    @abstractmethod
    def count(self) -> int:
        """Return number of indexed candidates."""
