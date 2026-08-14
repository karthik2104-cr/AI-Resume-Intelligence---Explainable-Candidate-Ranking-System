"""Sentence-Transformers embedding engine implementation.

Implements the EmbeddingEngine interface defined in src.embeddings.base.

This implementation loads the model once and reuses it. It supports optional
L2-normalization of embeddings and an optional in-memory cache.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from src.embeddings.base import EmbeddingEngine
from src.utils.config import get_settings


class SentenceTransformerEmbedding(EmbeddingEngine):
    """Embedding engine backed by sentence-transformers.

    - Loads model once (cached by class-level lru_cache)
    - Supports batch encoding
    - Optional normalization
    - Optional simple in-memory cache
    """

    def __init__(self):
        self._settings = get_settings().embeddings
        if self._settings.provider not in ("sentence_transformers", "sentence-transformers"):
            # still allow constructing but warn in metadata consumers
            pass
        self._model = self._load_model(self._settings.model_name)
        self._dim = None
        # simple in-memory cache mapping text -> tuple(dim array)
        self._cache_enabled = bool(self._settings.cache_embeddings)
        self._cache: dict[str, np.ndarray] = {}
        self._normalize = bool(get_settings().embeddings.normalize_embeddings)

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model(model_name: str):
        from sentence_transformers import SentenceTransformer

        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer model '{model_name}': {e}")
        return model

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        # Use the underlying model's encode which returns numpy arrays
        # We'll request show_progress_bar=False, convert_to_numpy=True
        model = self._model
        # sentence_transformers may return list[list[float]] or np.ndarray
        encoded = model.encode(texts, batch_size=self._settings.batch_size, show_progress_bar=False, convert_to_numpy=True)
        arr = np.asarray(encoded, dtype=float)
        if self._dim is None and arr.size:
            self._dim = arr.shape[1]
        if self._normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=float)

        if self._cache_enabled:
            results = []
            to_encode = []
            to_encode_idx = []
            for i, t in enumerate(texts):
                key = t.strip()
                if not key:
                    results.append(None)
                elif key in self._cache:
                    results.append(self._cache[key])
                else:
                    results.append(None)
                    to_encode.append(key)
                    to_encode_idx.append(i)
            if to_encode:
                encoded = self._encode_texts(to_encode)
                # store back in the same order
                for j, idx in enumerate(to_encode_idx):
                    vec = encoded[j]
                    key = to_encode[j]
                    self._cache[key] = vec
                    results[idx] = vec
            # replace None with zero vectors
            final = []
            for r in results:
                if r is None:
                    final.append(np.zeros((self._dim or 0,), dtype=float))
                else:
                    final.append(r)
            return np.vstack(final)
        else:
            return self._encode_texts([t or "" for t in texts])

    def embed_text(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros((self._dim or 0,), dtype=float)
        key = text.strip()
        if self._cache_enabled and key in self._cache:
            return self._cache[key]
        arr = self.embed_texts([text])[0]
        if self._cache_enabled:
            self._cache[key] = arr
        return arr

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<SentenceTransformerEmbedding model={self.model_name} dim={self._dim} norm={self._normalize} cache={self._cache_enabled}>"
