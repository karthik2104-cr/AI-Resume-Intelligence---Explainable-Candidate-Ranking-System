Embedding cache strategy

This project uses two related caching concepts:

1. Model instance caching (`src/embeddings/__init__.py`): `get_embedding_engine()`
   returns a singleton instance per configured provider for the process. This
   avoids repeated model loads on each request.

2. Optional per-engine embedding caches: `SentenceTransformerEmbedding` may cache
   identical input texts in memory when `embeddings.cache_embeddings` is true.

Prefer keeping the singleton engine instance. The per-text cache is optional and
can be disabled via configuration.
