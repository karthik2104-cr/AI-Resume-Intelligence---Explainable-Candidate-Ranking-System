import numpy as np
import pytest

from src.retrieval.in_memory import InMemoryCandidateRetriever
from src.retrieval.models import CandidateRecord, RetrievalMetadata
from src.retrieval.errors import (
    InvalidEmbeddingError,
    DuplicateCandidateError,
    EmbeddingDimensionMismatchError,
)


def make_vec(vals):
    return list(np.array(vals, dtype=float))


def test_index_and_retrieve_single():
    r = InMemoryCandidateRetriever()
    rec = CandidateRecord(candidate_id="c1", embedding=make_vec([1.0, 0.0]), metadata=RetrievalMetadata())
    r.index(rec)
    assert r.count() == 1
    q = [1.0, 0.0]
    results = r.retrieve(q, top_k=5)
    assert len(results) == 1
    assert results[0].candidate_id == "c1"
    assert pytest.approx(results[0].similarity, rel=1e-6) == 1.0


def test_top_k_and_ordering():
    r = InMemoryCandidateRetriever()
    r.index(CandidateRecord(candidate_id="a", embedding=make_vec([1.0, 0.0]), metadata=RetrievalMetadata()))
    r.index(CandidateRecord(candidate_id="b", embedding=make_vec([0.0, 1.0]), metadata=RetrievalMetadata()))
    r.index(CandidateRecord(candidate_id="c", embedding=make_vec([0.7071, 0.7071]), metadata=RetrievalMetadata()))
    q = [0.7, 0.7]
    results = r.retrieve(q, top_k=2)
    assert len(results) == 2
    # top should be c then a or b depending similarity; deterministic tie-breaking uses id
    assert results[0].candidate_id == "c"


def test_empty_index_returns_empty():
    r = InMemoryCandidateRetriever()
    results = r.retrieve([1.0, 0.0], top_k=10)
    assert results == []


def test_duplicate_ids_raise():
    r = InMemoryCandidateRetriever()
    r.index(CandidateRecord(candidate_id="dup", embedding=make_vec([1, 0]), metadata=RetrievalMetadata()))
    with pytest.raises(DuplicateCandidateError):
        r.index(CandidateRecord(candidate_id="dup", embedding=make_vec([0, 1]), metadata=RetrievalMetadata()))


def test_invalid_embedding_nan():
    r = InMemoryCandidateRetriever()
    with pytest.raises(InvalidEmbeddingError):
        r.index(CandidateRecord(candidate_id="bad", embedding=[float('nan'), 0.0], metadata=RetrievalMetadata()))


def test_dimension_mismatch():
    r = InMemoryCandidateRetriever()
    r.index(CandidateRecord(candidate_id="one", embedding=make_vec([1, 0, 0]), metadata=RetrievalMetadata()))
    with pytest.raises(EmbeddingDimensionMismatchError):
        r.index(CandidateRecord(candidate_id="two", embedding=make_vec([1, 0]), metadata=RetrievalMetadata()))


def test_zero_vector_handling():
    r = InMemoryCandidateRetriever()
    r.index(CandidateRecord(candidate_id="z", embedding=make_vec([0.0, 0.0, 0.0]), metadata=RetrievalMetadata()))
    r.index(CandidateRecord(candidate_id="u", embedding=make_vec([1.0, 0.0, 0.0]), metadata=RetrievalMetadata()))
    results = r.retrieve([1.0, 0.0, 0.0], top_k=10)
    # zero vector should have similarity 0 and rank after positive sims
    assert any(res.candidate_id == "u" for res in results)
    assert any(res.candidate_id == "z" for res in results)


def test_get_candidate_and_clear():
    r = InMemoryCandidateRetriever()
    r.index(CandidateRecord(candidate_id="c1", embedding=make_vec([1.0, 2.0]), metadata=RetrievalMetadata()))
    rec = r.get_candidate("c1")
    assert rec.candidate_id == "c1"
    r.clear()
    assert r.count() == 0
    with pytest.raises(Exception):
        r.get_candidate("c1")
