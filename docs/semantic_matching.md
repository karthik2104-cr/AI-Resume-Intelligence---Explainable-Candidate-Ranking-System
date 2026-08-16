# Semantic Matching

## Overview

Semantic matching complements TF-IDF lexical overlap with embedding cosine
similarity so paraphrases and related skills can still score highly.

## Why TF-IDF alone is insufficient

TF-IDF measures lexical overlap and can miss semantic equivalence when wording
differs (for example "built predictive models" vs "developed ML pipelines").

## Model

Default: `sentence-transformers/all-MiniLM-L6-v2` — lightweight and CPU-friendly.

Configured in `configs/config.yaml` under `embeddings`.

## How it works

1. Build section texts from the parsed resume and JD.
2. Embed sections with the shared singleton embedding engine.
3. Compare mapped resume↔JD sections with cosine similarity.
4. Aggregate with configurable section weights (`semantic_matching.weights`).
5. Missing sections redistribute weight over available sections only.

The result populates `MatchResult.scores.semantic` and metadata; it does **not**
set the final ranking score. `HybridRanker` consumes the semantic signal as one
component among skill coverage, experience, education, and seniority.

## Caching / lifecycle

- `get_embedding_engine()` returns a process-level singleton per provider.
- `SentenceTransformerEmbedding` optionally caches identical input texts in memory
  (`embeddings.cache_embeddings`).

## Limitations

- First run downloads the model.
- In-memory text cache is not shared across processes.
- Section heuristics depend on parsing quality.

See `src/matching/semantic_matcher.py` and `src/embeddings/sentence_transformers.py`.
