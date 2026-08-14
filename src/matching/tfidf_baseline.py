"""TF-IDF + cosine similarity baseline matching engine."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.matching.base import MatchingEngine
from src.models.job import ParsedJobDescription
from src.models.matching import ComponentScores, MatchResult
from src.models.resume import ParsedResume
from src.preprocessing.text_cleaner import clean_text
from src.utils.config import BaselineMatchingConfig, get_settings


class TfidfBaselineMatcher(MatchingEngine):
    """
    Reproducible TF-IDF + cosine similarity baseline.

    Vectorizes the job description and resume(s) together so vocabulary
    is shared within each match batch — mirroring the legacy notebook
    `rank_resumes` approach.
    """

    def __init__(self, config: Optional[BaselineMatchingConfig] = None) -> None:
        self._config = config or get_settings().matching.baseline
        self._vectorizer: Optional[TfidfVectorizer] = None

    @property
    def name(self) -> str:
        return "tfidf_baseline"

    def _build_vectorizer(self) -> TfidfVectorizer:
        vcfg = self._config.vectorizer
        ngram = tuple(vcfg.ngram_range)
        return TfidfVectorizer(
            max_features=vcfg.max_features,
            sublinear_tf=vcfg.sublinear_tf,
            stop_words=vcfg.stop_words,
            ngram_range=ngram,  # type: ignore[arg-type]
        )

    def _prepare_texts(
        self,
        job: ParsedJobDescription,
        resumes: list[ParsedResume],
    ) -> tuple[str, list[str]]:
        job_text = clean_text(job.full_text_for_matching or job.raw_text)
        resume_texts = [
            clean_text(r.full_text_for_matching or r.raw_text) for r in resumes
        ]
        return job_text, resume_texts

    def _compute_similarities(
        self,
        job_text: str,
        resume_texts: list[str],
    ) -> np.ndarray:
        if not job_text.strip():
            raise ValueError("Job description text is empty after preprocessing.")

        if not resume_texts:
            return np.array([])

        if all(not t.strip() for t in resume_texts):
            raise ValueError("All resume texts are empty after preprocessing.")

        corpus = [job_text] + resume_texts
        vectorizer = self._build_vectorizer()
        matrix = vectorizer.fit_transform(corpus)

        jd_vector = matrix[0:1]
        resume_matrix = matrix[1:]
        similarities = cosine_similarity(jd_vector, resume_matrix).flatten()
        return np.clip(similarities, 0.0, 1.0)

    def match(
        self,
        resume: ParsedResume,
        job: ParsedJobDescription,
        candidate_id: str | None = None,
    ) -> MatchResult:
        results = self.match_batch([resume], job, [candidate_id] if candidate_id else None)
        return results[0]

    def match_batch(
        self,
        resumes: list[ParsedResume],
        job: ParsedJobDescription,
        candidate_ids: list[str] | None = None,
    ) -> list[MatchResult]:
        if not resumes:
            return []

        ids = candidate_ids or [None] * len(resumes)  # type: ignore[list-item]
        if len(ids) != len(resumes):
            raise ValueError("candidate_ids length must match resumes length.")

        job_text, resume_texts = self._prepare_texts(job, resumes)
        similarities = self._compute_similarities(job_text, resume_texts)

        results: list[MatchResult] = []
        for idx, (resume, cid) in enumerate(zip(resumes, ids)):
            score = float(similarities[idx])
            results.append(
                MatchResult(
                    candidate_id=cid,
                    candidate_name=resume.name,
                    scores=ComponentScores(
                        overall=score,
                        baseline_tfidf=score,
                    ),
                    matcher_name=self.name,
                    metadata={
                        "similarity_metric": self._config.similarity_metric,
                        "vectorizer_max_features": self._config.vectorizer.max_features,
                    },
                )
            )
        return results

    def rank_by_similarity(
        self,
        resumes: list[ParsedResume],
        job: ParsedJobDescription,
        candidate_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[int, MatchResult]]:
        """
        Rank resumes by TF-IDF cosine similarity.

        Returns list of (original_index, MatchResult) sorted by score descending.
        """
        match_results = self.match_batch(resumes, job, candidate_ids)
        indexed = list(enumerate(match_results))
        indexed.sort(key=lambda x: x[1].scores.overall, reverse=True)
        if top_k is not None:
            indexed = indexed[:top_k]
        return indexed
