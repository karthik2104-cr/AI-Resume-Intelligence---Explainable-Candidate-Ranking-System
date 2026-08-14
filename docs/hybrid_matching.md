# Hybrid Matching (Phase 8)

This document describes the explainable hybrid matching engine implemented in Phase 8.

Key points:

- Signals: required skill coverage, preferred skill coverage, semantic similarity, experience compatibility, education compatibility, seniority compatibility.
- Configurable weights: set in `configs/config.yaml` under `matching.hybrid_matching.weights`.
- Missing data: components that are unavailable are marked `unavailable` and their configured weight is redistributed across available components (dynamic renormalization).
- Components produce deterministic scores in [0,1]. The overall hybrid score is the weighted sum of normalized component scores and exposed as 0–1 and 0–100 values.
- Every hybrid result contains component-level breakdown, applied vs configured weights, and preserved evidence pulled from skill-gap analysis.

See `src/ranking/hybrid_ranker.py` and `src/models/hybrid_matching.py` for implementation details.
