# Retrieval Layer

## Overview

The retrieval layer provides a lightweight, in-memory candidate retriever that
produces a top-K shortlist of potentially relevant candidates for a given job
embedding. Retrieval is intentionally separate from detailed ranking — it is a
fast filtering step that reduces the candidate set the hybrid ranker scores
in detail.

## Key principles

- Retrieval is **not** final ranking. `HybridRanker` remains authoritative.
- Retrieval uses semantic embeddings via the existing `EmbeddingEngine`.
- Retrieval stores only lightweight metadata (no email or phone).
- The implementation is behind an interface so a future vector store can be
  substituted without changing application logic.

## Files

- `src/retrieval/base.py` — `CandidateRetriever` interface
- `src/retrieval/models.py` — `CandidateRecord`, `RetrievalResult`
- `src/retrieval/in_memory.py` — `InMemoryCandidateRetriever`
- `src/retrieval/errors.py` — domain exceptions

## How it integrates

1. `ScreeningService` embeds the JD via the embedding engine.
2. The retriever returns top-K candidate IDs with cosine similarities.
3. Shortlisted resumes enter TF-IDF + semantic matching and hybrid ranking.
4. Retrieval similarity is attached to match metadata for evidence only and
   never overwrites hybrid scores.

## Limitations

- In-memory only — suitable for demos, tests, and small candidate pools.
- Embeddings must be provided at indexing time.
- Not a production vector database (no persistence or sharding).
