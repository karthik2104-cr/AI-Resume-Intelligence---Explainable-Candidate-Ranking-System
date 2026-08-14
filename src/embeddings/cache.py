from __future__ import annotations

from typing import Any
import numpy as np


class EmbeddingCache:
    """Simple in-memory cache for embeddings keyed by a deterministic key.

    This is intentionally simple and meant for local development/testing.
    If the underlying embedding engine exposes a cache, this wrapper can
    be pointed at it by passing the backing dict to the constructor.
    """

    def __init__(self, backing: dict[str, Any] | None = None):
        self._store: dict[str, Any] = backing if backing is not None else {}

    def get(self, key: str) -> np.ndarray | None:
        return self._store.get(key)

    def set(self, key: str, value: np.ndarray) -> None:
        # store a copy to avoid external mutation
        self._store[key] = value.copy() if hasattr(value, "copy") else value

    def contains(self, key: str) -> bool:
        return key in self._store

    def clear(self) -> None:
        self._store.clear()

    def size(self) -> int:
        return len(self._store)
