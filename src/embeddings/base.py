"""Embedding engine abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingEngine(ABC):
    """Abstract base for semantic embedding generation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the embedding model identifier."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns shape (n, dim)."""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text. Returns shape (dim,)."""

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts."""
        vec_a = self.embed_text(text_a)
        vec_b = self.embed_text(text_b)
        return float(self._cosine_similarity(vec_a, vec_b))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
