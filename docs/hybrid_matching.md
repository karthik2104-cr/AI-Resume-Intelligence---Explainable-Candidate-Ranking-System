# Hybrid Ranking

`HybridRanker` is the authoritative final scoring component.

## Signals

| Component | Source |
|-----------|--------|
| Required skill coverage | `compute_skill_gap()` |
| Preferred skill coverage | `compute_skill_gap()` |
| Semantic similarity | `SemanticMatcher` via `CombinedMatcher` |
| Experience compatibility | Heuristic vs JD experience requirements |
| Education compatibility | Heuristic degree matching |
| Seniority compatibility | Years-of-experience vs JD seniority |

Weights live in `configs/config.yaml` under `matching.hybrid_matching.weights`
and must sum to 1.0.

## Missing data

Unavailable components are marked `unavailable` and their configured weight is
redistributed across available components (dynamic renormalization).

## Outputs

- Overall score in `[0, 1]`
- Per-component breakdown with applied vs configured weights
- Skill-gap payload stored under `match_result.metadata["hybrid"]`
- Presentation fields also written to top-level metadata:
  `matched_skills`, `missing_required_skills`, `missing_preferred_skills`
- Component values mirrored onto `MatchResult.scores` (`skill`, `semantic`,
  `experience`, `education`, `overall`)

Retrieval similarity is **never** used as the overall score.

See `src/ranking/hybrid_ranker.py` and `src/models/hybrid_matching.py`.
