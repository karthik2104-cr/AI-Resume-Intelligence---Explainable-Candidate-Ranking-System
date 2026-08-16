"""Semantic matching engine using embeddings.

Implements the MatchingEngine interface and produces semantic match signals
without modifying the existing TF-IDF baseline or SkillGap engines.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.matching.base import MatchingEngine
from src.models.matching import MatchResult, ComponentScores, SemanticMatchResult
from src.models.resume import ParsedResume
from src.models.job import ParsedJobDescription
from src.utils.config import get_settings
from src import embeddings as embeddings_mod


def _build_resume_sections(resume: ParsedResume) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    if resume.summary:
        sections["summary"] = resume.summary.strip()
    if resume.skills:
        sections["skills"] = ", ".join(resume.skills)
    if resume.experience:
        exp_texts = []
        for e in resume.experience:
            txt = " ".join(filter(None, [e.title or "", e.organization or "", e.description or e.raw_text or ""]))
            exp_texts.append(txt)
        sections["experience"] = "\n".join(exp_texts)
    if resume.projects:
        proj_texts = []
        for p in resume.projects:
            proj_texts.append(" ".join([p.name or "", p.description or ""]))
        sections["projects"] = "\n".join(proj_texts)
    if resume.education:
        edu_texts = [e.raw_text for e in resume.education if e.raw_text]
        if edu_texts:
            sections["education"] = "\n".join(edu_texts)
    return sections


def _build_job_sections(job: ParsedJobDescription) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    if job.title:
        sections["title"] = job.title
    # requirements: required + preferred
    reqs = []
    for s in job.required_skills:
        reqs.append(s.raw_skill)
    for s in job.preferred_skills:
        reqs.append(s.raw_skill)
    if reqs:
        sections["requirements"] = ", ".join(reqs)
    if job.responsibilities:
        sections["responsibilities"] = "\n".join(r.text for r in job.responsibilities)
    if job.experience_requirements:
        sections["experience"] = "\n".join(e.raw_text for e in job.experience_requirements if e.raw_text)
    if job.education_requirements:
        sections["education"] = "\n".join(e.raw_text for e in job.education_requirements if e.raw_text)
    return sections


# mapping of resume -> job sections for similarity
_DEFAULT_SECTION_MAP = {
    "summary": ["title", "responsibilities", "requirements"],
    "skills": ["requirements"],
    "experience": ["experience", "responsibilities"],
    "projects": ["responsibilities"],
    "education": ["education"],
}


class SemanticMatcher(MatchingEngine):
    def __init__(self):
        self._settings = get_settings()
        if not self._settings.embeddings.enabled:
            raise RuntimeError("Embeddings are disabled in configuration")
        self._engine = embeddings_mod.get_embedding_engine()
        # load section weights from configuration; provide sensible defaults if missing
        if hasattr(self._settings, "semantic_matching") and getattr(self._settings, "semantic_matching") is not None:
            self._weights = self._settings.semantic_matching.weights.model_dump()
        else:
            # fallback defaults (should be set via config)
            self._weights = {"summary": 0.1, "skills": 0.25, "experience": 0.35, "projects": 0.2, "education": 0.1}

    @property
    def name(self) -> str:
        return "semantic_matcher"

    def _safe_cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        try:
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            if a.size == 0 or b.size == 0:
                return 0.0
            if a.shape != b.shape:
                min_len = min(a.size, b.size)
                a = a[:min_len]
                b = b[:min_len]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)
        except Exception:
            return 0.0

    def match(self, resume: ParsedResume, job: ParsedJobDescription, candidate_id: str | None = None) -> MatchResult:
        resume_sections = _build_resume_sections(resume)
        job_sections = _build_job_sections(job)

        # Prepare texts to embed
        # We'll embed all resume sections and job sections that are used in mapping
        resume_texts = []
        resume_keys = []
        for k, v in resume_sections.items():
            resume_keys.append(k)
            resume_texts.append(v)

        job_texts = []
        job_keys = []
        for k, v in job_sections.items():
            job_keys.append(k)
            job_texts.append(v)

        # compute embeddings
        resume_embeds = self._engine.embed_texts(resume_texts) if resume_texts else np.zeros((0, 0))
        job_embeds = self._engine.embed_texts(job_texts) if job_texts else np.zeros((0, 0))

        # build quick lookup by key
        resume_map = {k: resume_embeds[i] for i, k in enumerate(resume_keys)}
        job_map = {k: job_embeds[i] for i, k in enumerate(job_keys)}

        # compute section similarities
        section_scores: Dict[str, float] = {}
        available_weight_total = 0.0
        for sec, weight in self._weights.items():
            # skip sections not present in resume
            if sec not in resume_map:
                continue
            # find best matching job section from map
            mapped = _DEFAULT_SECTION_MAP.get(sec, [])
            best_score = 0.0
            for target in mapped:
                if target in job_map:
                    score = self._safe_cosine(resume_map[sec], job_map[target])
                    if score > best_score:
                        best_score = score
            section_scores[sec] = best_score
            available_weight_total += float(weight)

        # normalize weights over available sections
        if available_weight_total <= 0:
            overall = 0.0
        else:
            overall = 0.0
            for sec, raw_w in self._weights.items():
                if sec not in section_scores:
                    continue
                norm_w = float(raw_w) / available_weight_total
                overall += section_scores[sec] * norm_w

        # build SemanticMatchResult
        sem = SemanticMatchResult(
            semantic_similarity=float(overall),
            section_scores=section_scores,
            embedding_model=getattr(self._engine, "model_name", None),
            embedding_dimension=getattr(self._engine, "_dim", None),
            normalized_embeddings=getattr(self._engine, "_normalize", None),
            compared_sections=list(section_scores.keys()),
            metadata={"resume_sections": list(resume_sections.keys()), "job_sections": list(job_sections.keys())},
        )

        scores = ComponentScores(overall=0.0, semantic=float(overall))
        result = MatchResult(
            candidate_id=candidate_id,
            candidate_name=resume.name,
            scores=scores,
            matcher_name=self.name,
            metadata={"semantic": sem.model_dump()},
        )
        return result

    def match_batch(self, resumes: List[ParsedResume], job: ParsedJobDescription, candidate_ids: List[str] | None = None) -> List[MatchResult]:
        results = []
        for i, r in enumerate(resumes):
            cid = candidate_ids[i] if candidate_ids and i < len(candidate_ids) else None
            results.append(self.match(r, job, candidate_id=cid))
        return results
