Embedding cache strategy

This project uses two related caching concepts:

1. Model instance caching (in src/embeddings/__init__.py): the get_embedding_engine()
   factory returns a singleton instance per configured provider for the application
   process. This avoids repeated model loads on each request and is intended for
   long-lived service processes.

2. Optional per-engine embedding caches: individual embedding engine
   implementations (for example SentenceTransformerEmbedding) may implement
   internal caches for small repeated inputs or use disk-backed caches. These
   caches are implementation details of the engine and may be kept or removed
   independently.

Why both exist

- Model instance caching prevents the heavy model loading cost on each request.
- Per-engine embedding caches can accelerate repeat encodings for identical
  input texts across requests. They are optional and implementation-specific.

If you find duplicate or redundant caching behavior for your workload, prefer
keeping the singleton engine instance and simplify or remove redundant per-engine
caches to reduce complexity.
