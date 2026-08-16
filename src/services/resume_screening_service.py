from __future__ import annotations

from typing import List, Optional, Dict

import numpy as np

from src.retrieval.in_memory import InMemoryCandidateRetriever
from src.retrieval.models import CandidateRecord
from src.models.job import ParsedJobDescription
from src.models.resume import ParsedResume
from src.models.ranking import RankingRequest
from src.ranking.hybrid_ranker import HybridRanker
from src.matching.tfidf_baseline import TfidfBaselineMatcher
from src.matching.semantic_matcher import SemanticMatcher
from src.explainability.service import ExplanationService
from src.explainability.models import ExplanationInput
from src import embeddings as embeddings_mod
from src.retrieval.errors import DuplicateCandidateError
from src.utils.config import get_settings


class CombinedMatcher(TfidfBaselineMatcher):
    """Composite matcher: TF-IDF baseline + semantic matcher merged into MatchResult.

    This allows HybridRanker to receive baseline MatchResult objects that also
    carry a semantic score when available.
    """

    def __init__(self):
        super().__init__()
        self._semantic = SemanticMatcher()

    def match_batch(self, resumes: list[ParsedResume], job: ParsedJobDescription, candidate_ids: list[str] | None = None):
        base_results = super().match_batch(resumes, job, candidate_ids)
        sem_results = self._semantic.match_batch(resumes, job, candidate_ids)
        for b, s in zip(base_results, sem_results):
            try:
                if getattr(s, "scores", None) and getattr(s.scores, "semantic", None) is not None:
                    b.scores.semantic = s.scores.semantic
                if s.metadata:
                    b.metadata.setdefault("semantic", s.metadata.get("semantic"))
            except Exception:
                continue
        return base_results


class ScreeningService:
    """Orchestrates JD embedding → retrieval → top-K → hybrid ranking → explanation."""

    def __init__(self, retriever: Optional[InMemoryCandidateRetriever] = None):
        self._settings = get_settings()
        self._engine = embeddings_mod.get_embedding_engine()
        self._retriever = retriever or InMemoryCandidateRetriever()
        self._explain = ExplanationService()
        self._matcher = CombinedMatcher()
        self._ranker = HybridRanker(matcher=self._matcher)

    def index_candidates(self, resumes: List[ParsedResume], candidate_ids: Optional[List[str]] = None) -> None:
        """Index candidate embeddings into the retriever. Duplicates raise DuplicateCandidateError."""
        ids = candidate_ids or [f"cand_{i}" for i in range(len(resumes))]
        texts = [r.full_text_for_matching or r.raw_text for r in resumes]
        if hasattr(self._engine, "embed_texts"):
            vecs = self._engine.embed_texts(texts)
        else:
            vecs = np.stack([self._engine.embed_text(t) for t in texts])
        for cid, resume, vec in zip(ids, resumes, vecs):
            meta = {
                "parsing_quality": resume.parsing_quality,
                "normalized_skills": resume.skills_normalized,
                "years_experience": resume.years_experience,
                "domains": getattr(resume, "domains", []),
            }
            rec = CandidateRecord(
                candidate_id=cid,
                embedding=list(np.asarray(vec, dtype=float)),
                metadata=meta,
            )
            self._retriever.index(rec)

    def screen(
        self,
        job: ParsedJobDescription,
        resumes: List[ParsedResume],
        candidate_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict:
        """Run full screening: index → retrieve top-K → hybrid rank → explain.

        Returns a dict with keys: ranking_result, retrieval_results, explanations.

        ``top_k`` overrides retrieval shortlist size for this call only (does not
        mutate global settings). Ranking top_k still comes from ranking config
        unless fewer candidates are shortlisted.
        """
        try:
            self.index_candidates(resumes, candidate_ids)
        except DuplicateCandidateError:
            pass

        jd_text = job.full_text_for_matching or job.raw_text
        q_vec = self._engine.embed_text(jd_text)

        retrieval_top_k = top_k if top_k is not None else self._settings.retrieval.top_k
        ranking_top_k = top_k if top_k is not None else self._settings.ranking.default_top_k
        retrieval_results = self._retriever.retrieve(q_vec, top_k=retrieval_top_k)

        if not candidate_ids:
            id_to_resume = {f"cand_{i}": r for i, r in enumerate(resumes)}
        else:
            id_to_resume = {cid: r for cid, r in zip(candidate_ids, resumes)}

        shortlisted_ids = [res.candidate_id for res in retrieval_results]
        shortlisted_pairs = [
            (cid, id_to_resume[cid]) for cid in shortlisted_ids if cid in id_to_resume
        ]
        if not shortlisted_pairs:
            return {"ranking_result": None, "retrieval_results": [], "explanations": {}}

        s_ids, s_resumes = zip(*shortlisted_pairs)

        ranking_request = RankingRequest(
            job=job,
            candidates=list(s_resumes),
            candidate_ids=list(s_ids),
            top_k=ranking_top_k,
        )
        ranking_result = self._ranker.rank(ranking_request)

        # Attach retrieval similarity for evidence only — never overwrites final scores.
        retrieval_map = {r.candidate_id: r for r in retrieval_results}
        for entry in ranking_result.entries:
            cid = entry.candidate_id
            if cid in retrieval_map:
                entry.match_result.metadata.setdefault("retrieval", {})
                entry.match_result.metadata["retrieval"]["similarity"] = retrieval_map[cid].similarity
                entry.match_result.metadata["retrieval"]["rank"] = retrieval_map[cid].rank

        shortlist_resume_map = {cid: r for cid, r in shortlisted_pairs}
        explanations = {}
        for entry in ranking_result.entries:
            cid = entry.candidate_id
            resume_for_explanation = shortlist_resume_map.get(cid)
            hybrid_meta = (entry.match_result.metadata or {}).get("hybrid") or {}
            component_scores = hybrid_meta.get("component_scores") or []
            skill_gap = hybrid_meta.get("skill_gap")
            inp = ExplanationInput(
                match_result=entry.match_result.model_dump(),
                component_scores=component_scores,
                hybrid_metadata=hybrid_meta or None,
                skill_gap=skill_gap,
                parsed_resume=resume_for_explanation.model_dump() if resume_for_explanation is not None else None,
                parsed_job=job.model_dump() if hasattr(job, "model_dump") else {},
                ranking_info={"retrieval_rank": retrieval_map[cid].rank} if cid in retrieval_map else {},
            )
            explanations[cid] = self._explain.explain(inp).model_dump()

        return {
            "ranking_result": ranking_result,
            "retrieval_results": [r.model_dump() for r in retrieval_results],
            "explanations": explanations,
        }
