from __future__ import annotations

from typing import List, Optional

from src.matching.skill_gap import compute_skill_gap, SkillGapResult
from src.matching.tfidf_baseline import TfidfBaselineMatcher
from src.models.job import ParsedJobDescription
from src.models.matching import MatchResult
from src.models.resume import ParsedResume
from src.models.hybrid_matching import ComponentScore, HybridMatchResult
from src.models.ranking import CandidateRankEntry, RankingRequest, RankingResult
from src.ranking.base import RankingEngine
from src.utils.config import get_settings


def _compute_required_score(skill_gap: SkillGapResult) -> Optional[float]:
    if skill_gap.required_total == 0:
        return None
    return skill_gap.required_skill_coverage / 100.0


def _compute_preferred_score(skill_gap: SkillGapResult) -> Optional[float]:
    if skill_gap.preferred_total == 0:
        return None
    return skill_gap.preferred_skill_coverage / 100.0


def _compute_experience_score(job: ParsedJobDescription, resume: ParsedResume) -> Optional[float]:
    # conservative deterministic approach
    if not job.experience_requirements:
        return None
    req = job.experience_requirements[0]
    if resume.years_experience is None:
        return None
    cand = resume.years_experience
    if req.min_years is not None:
        if cand >= req.min_years:
            return 1.0
        # partial credit proportional to minimum
        return max(0.0, cand / req.min_years)
    if req.max_years is not None:
        # favor within max
        return 1.0 if cand <= req.max_years else 1.0
    return None


def _compute_education_score(job: ParsedJobDescription, resume: ParsedResume) -> Optional[float]:
    if not job.education_requirements:
        return None
    if not resume.education:
        return None
    # simple string-based matching
    for ed_req in job.education_requirements:
        if not ed_req.degree_level:
            continue
        req_lvl = ed_req.degree_level.lower()
        for cand in resume.education:
            if not cand.degree:
                continue
            deg = cand.degree.lower()
            if req_lvl in deg or deg in req_lvl:
                return 1.0
        # not found for this requirement -> low
        return 0.0
    return None


def _compute_seniority_score(job: ParsedJobDescription, resume: ParsedResume) -> Optional[float]:
    if not job.seniority_level:
        return None
    if resume.years_experience is None:
        return None
    lvl = job.seniority_level.lower()
    years = resume.years_experience
    # heuristic mapping
    if years < 2:
        cand_lvl = "junior"
    elif years < 5:
        cand_lvl = "mid"
    elif years < 10:
        cand_lvl = "senior"
    else:
        cand_lvl = "lead"

    return 1.0 if lvl.startswith(cand_lvl[0:3]) else 0.8 if cand_lvl in lvl or lvl in cand_lvl else 0.5


class HybridRanker(RankingEngine):
    def __init__(self, matcher: Optional[TfidfBaselineMatcher] = None) -> None:
        self._matcher = matcher or TfidfBaselineMatcher()
        self._settings = get_settings()

    @property
    def name(self) -> str:
        return "hybrid_ranker"

    def rank(self, request: RankingRequest) -> RankingResult:
        if not request.candidates:
            return RankingResult(job_title=request.job.title, entries=[], total_candidates=0, ranker_name=self.name, warnings=["No candidates provided."])

        top_k = request.top_k or self._settings.ranking.default_top_k

        # compute skill gaps
        skill_gaps: List[SkillGapResult] = [compute_skill_gap(r, request.job) for r in request.candidates]

        # get semantic/baseline matches
        match_results = self._matcher.match_batch(request.candidates, request.job, request.candidate_ids or None)

        # component weight source
        cfg = self._settings.matching.hybrid_matching.weights
        weights = cfg.model_dump()

        entries: List[CandidateRankEntry] = []
        hybrid_results: List[HybridMatchResult] = []

        for resume, skill_gap, match in zip(request.candidates, skill_gaps, match_results):
            comps: List[ComponentScore] = []

            required_score = _compute_required_score(skill_gap)
            prefs_score = _compute_preferred_score(skill_gap)
            # Use semantic similarity only if provided by the matcher. Do NOT fall back to TF-IDF.
            semantic_score = match.scores.semantic if getattr(match.scores, "semantic", None) is not None else None
            experience_score = _compute_experience_score(request.job, resume)
            education_score = _compute_education_score(request.job, resume)
            seniority_score = _compute_seniority_score(request.job, resume)

            comp_map = {
                "required_skill_coverage": (required_score, weights.get("required_skill_coverage")),
                "preferred_skill_coverage": (prefs_score, weights.get("preferred_skill_coverage")),
                "semantic_similarity": (semantic_score, weights.get("semantic_similarity")),
                "experience_compatibility": (experience_score, weights.get("experience_compatibility")),
                "education_compatibility": (education_score, weights.get("education_compatibility")),
                "seniority_compatibility": (seniority_score, weights.get("seniority_compatibility")),
            }

            # identify available components
            available = {k: v for k, v in comp_map.items() if v[0] is not None}
            total_config_weight = sum(v for (_, v) in comp_map.values() if v is not None)
            # renormalize: redistribute weights of missing components proportionally
            if available:
                available_weight = sum(w for (_, w) in ((val[0], val[1]) for val in available.values()) )
                # compute applied weight per available component proportional to its configured weight
                applied_weights = {k: (v[1] / available_weight if available_weight > 0 else 0.0) for k, v in available.items()}
            else:
                applied_weights = {k: 0.0 for k in comp_map.keys()}

            overall = 0.0
            for name, (score, cfg_w) in comp_map.items():
                avail = "available" if score is not None else "unavailable"
                applied_w = applied_weights.get(name) if name in applied_weights else 0.0
                contrib = (score or 0.0) * applied_w
                comps.append(
                    ComponentScore(
                        name=name,
                        score=score,
                        weight=cfg_w,
                        applied_weight=applied_w,
                        weighted_contribution=contrib,
                        evidence=[],
                        availability=avail,
                    )
                )
                overall += contrib

            overall = max(0.0, min(1.0, overall))

            hybrid = HybridMatchResult(
                overall_score=overall,
                overall_score_pct=overall * 100.0,
                component_scores=comps,
                skill_gap=skill_gap.model_dump() if hasattr(skill_gap, "model_dump") else None,
                semantic_result={"score": semantic_score} if semantic_score is not None else None,
                explanation=None,
                metadata={"applied_weights": applied_weights, "configured_weights": weights},
            )
            # attach hybrid result into match metadata for downstream inspection
            match.metadata.setdefault("hybrid", {})
            match.metadata["hybrid"] = hybrid.model_dump()

            hybrid_results.append(hybrid)

            entries.append(
                CandidateRankEntry(
                    rank=0,
                    candidate_id=match.candidate_id,
                    candidate_name=match.candidate_name,
                    overall_score=hybrid.overall_score,
                    required_skill_coverage_pct=skill_gap.required_skill_coverage,
                    experience_score=experience_score,
                    semantic_score=semantic_score,
                    baseline_tfidf_score=match.scores.baseline_tfidf,
                    match_result=match,
                )
            )

        # sort with deterministic tie-breaking
        entries_sorted = sorted(
            entries,
            key=lambda e: (
                -e.overall_score,
                -(e.required_skill_coverage_pct or 0.0),
                -(e.semantic_score or 0.0),
                -(e.experience_score or 0.0),
                e.candidate_id or "",
            ),
        )[:top_k]

        # assign ranks
        for idx, ent in enumerate(entries_sorted, start=1):
            ent.rank = idx

        return RankingResult(job_title=request.job.title, entries=entries_sorted, total_candidates=len(request.candidates), ranker_name=self.name)
