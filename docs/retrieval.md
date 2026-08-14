Retrieval Layer (Phase 10)

Overview
--------
The retrieval layer provides a lightweight, in-memory candidate retriever that
produces a top-K shortlist of potentially relevant candidates for a given job
embedding. Retrieval is intentionally separate from detailed ranking — it is a
fast filtering step that reduces the candidate set that the hybrid ranker will
score in detail.

Key principles
--------------
- Retrieval is NOT final ranking. The hybrid ranker remains authoritative.
- Retrieval uses semantic embeddings (via the existing EmbeddingEngine).
- Retrieval stores only lightweight metadata (no PII such as email or phone).
- The implementation is behind an interface so a future vector DB (FAISS, Pinecone)
  can be substituted without changing the application logic.

Files added
-----------
- src/retrieval/base.py — CandidateRetriever interface
- src/retrieval/models.py — Pydantic models for CandidateRecord and RetrievalResult
- src/retrieval/in_memory.py — InMemoryCandidateRetriever implementation
- src/retrieval/errors.py — Domain exceptions
- src/embeddings/cache.py — Simple embedding cache wrapper
- docs/retrieval.md — this document

How it integrates
-----------------
1. The application obtains a JD embedding from the existing embedding engine.
2. The retriever is given the query embedding and returns top-K candidate ids
   with retrieval similarities and lightweight metadata.
3. The application passes the shortlisted ParsedResume objects into the
   existing matching/ranking pipeline (semantic matcher and hybrid ranker).
4. Retrieval similarity is preserved in RetrievalResult.metadata and does not
   overwrite or replace MatchResult scores.

Limitations
-----------
- This is an in-memory retriever intended for development, testing, and small
  scale. It is not a production vector DB.
- Persistence, sharding, and large-scale performance are out of scope for Phase 10.
- The retriever requires embeddings to be provided at indexing time.

Next steps
----------
- Add integration wiring that produces JD embedding and calls the retriever.
- Add optional serialization for the retriever index (if persistence is desired).
- Implement a FAISS-backed retriever behind the same interface for larger datasets.
