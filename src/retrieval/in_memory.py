from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.retrieval.base import CandidateRetriever
from src.retrieval.models import CandidateRecord, RetrievalResult
from src.retrieval.errors import (
    InvalidEmbeddingError,
    CandidateNotFoundError,
    DuplicateCandidateError,
    EmbeddingDimensionMismatchError,
)


class InMemoryCandidateRetriever(CandidateRetriever):
    """In-memory candidate retriever storing embeddings and lightweight metadata.

    - Deterministic behavior: sorting ties by candidate_id
    - Validates embeddings for NaN/inf and consistent dimensionality
    """

    def __init__(self):
        self._records: Dict[str, CandidateRecord] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._dim: Optional[int] = None

    def index(self, record: CandidateRecord) -> None:
        cid = record.candidate_id
        if cid in self._records:
            raise DuplicateCandidateError(f"Candidate already indexed: {cid}")

        if record.embedding is None:
            raise InvalidEmbeddingError("Candidate embedding is required for indexing")

        vec = np.asarray(record.embedding, dtype=float)

        # validate
        if vec.size == 0:
            raise InvalidEmbeddingError("Embedding must be non-empty")
        if not np.isfinite(vec).all():
            raise InvalidEmbeddingError("Embedding contains NaN or infinite values")

        if self._dim is None:
            self._dim = vec.size
        elif vec.size != self._dim:
            raise EmbeddingDimensionMismatchError(
                f"Embedding dimension mismatch: expected {self._dim}, got {vec.size}"
            )

        # store a defensive copy
        self._records[cid] = CandidateRecord(candidate_id=cid, embedding=vec.tolist(), metadata=record.metadata)
        self._embeddings[cid] = vec.copy()

    def retrieve(self, query_embedding, top_k: int = 10) -> List[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        if not self._records:
            return []

        q = np.asarray(query_embedding, dtype=float)

        if q.size == 0:
            raise InvalidEmbeddingError("Query embedding must be non-empty")
        if not np.isfinite(q).all():
            raise InvalidEmbeddingError("Query embedding contains NaN or infinite values")
        if self._dim is not None and q.size != self._dim:
            raise EmbeddingDimensionMismatchError(f"Query dim {q.size} != index dim {self._dim}")

        # compute cosine similarities
        results: List[tuple[str, float]] = []
        q_norm = np.linalg.norm(q)
        for cid, vec in self._embeddings.items():
            v = vec
            v_norm = np.linalg.norm(v)
            if q_norm == 0 or v_norm == 0:
                sim = 0.0
            else:
                sim = float(np.dot(q, v) / (q_norm * v_norm))
            results.append((cid, sim))

        # sort by similarity desc, then candidate_id asc for deterministic tie-breaking
        results.sort(key=lambda t: (-t[1], t[0]))

        # top_k may be larger than available
        top = results[: min(top_k, len(results))]

        retrieval_results: List[RetrievalResult] = []
        for rank, (cid, sim) in enumerate(top, start=1):
            rec = self._records[cid]
            retrieval_results.append(
                RetrievalResult(candidate_id=cid, similarity=float(sim), rank=rank, metadata=rec.metadata)
            )
        return retrieval_results

    def get_candidate(self, candidate_id: str) -> CandidateRecord:
        if candidate_id not in self._records:
            raise CandidateNotFoundError(f"Candidate not found: {candidate_id}")
        # return a copy to avoid external mutation
        rec = self._records[candidate_id]
        return CandidateRecord(candidate_id=rec.candidate_id, embedding=list(rec.embedding), metadata=rec.metadata)

    def clear(self) -> None:
        self._records.clear()
        self._embeddings.clear()
        self._dim = None

    def count(self) -> int:
        return len(self._records)
