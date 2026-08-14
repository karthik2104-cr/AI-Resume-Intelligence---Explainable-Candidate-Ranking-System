Semantic Matching (Phase 7)

Overview
--------
This document describes the semantic embedding matching layer added in Phase 7.

Why TF-IDF is insufficient
--------------------------
TF-IDF measures lexical overlap and can fail to capture semantic similarity when
phrasing differs. For example, "scikit-learn" vs "sklearn" or "built models"
vs "developed predictive models" may present low lexical overlap but high
semantic similarity.

What embeddings provide
-----------------------
Dense vector representations capture semantic meaning. Cosine similarity between
vectors reflects semantic relatedness beyond exact token overlap.

Selected model
--------------
sentence-transformers/all-MiniLM-L6-v2 (default)

This model is lightweight, CPU-friendly, and practical for local development.

Architecture
------------
Resume/JD parsed -> Entity Intelligence -> Semantic Text Builder -> Embedding
Model -> Semantic Representation -> Semantic Matcher

Section-aware representation
----------------------------
We build embeddings for meaningful sections rather than a single document
embedding. Resume sections: summary, skills, experience, projects, education.
JD sections: title, requirements (required+preferred), responsibilities,
experience, education.

Cosine similarity
-----------------
Cosine similarity (implemented defensively) is used for comparisons. When
enabled, embeddings are L2-normalized which lets dot-product equal cosine.

Weighting and missing sections
------------------------------
Semantic section weights are configurable in config.yaml and in the
EmbeddingsConfig/SemanticMatchingConfig objects. Missing sections are handled by
normalizing weights over only the available sections so missing data doesn't
bias results.

Caching
-------
A simple optional in-memory cache prevents recomputing embeddings for identical
texts within a process. This is configurable via the embeddings.cache_embeddings
setting.

Evaluation methodology
----------------------
A small controlled fixture (tests/fixtures/matching_pairs.py) is provided to
compare TF-IDF baseline vs semantic embeddings using simple metrics (mean
similarity by class, Spearman where applicable). This is a demonstration, not a
production benchmark.

Limitations
-----------
- Local model download is required on first run. Tests avoid mandatory downloads
  by mocking the encoder where possible.
- The cache is in-memory and not persistent across processes.
- No hybrid scoring is performed in Phase 7; semantic scores remain an
  independent signal.

See src/matching/semantic_matcher.py and src/embeddings/sentence_transformers.py
for implementation details.
