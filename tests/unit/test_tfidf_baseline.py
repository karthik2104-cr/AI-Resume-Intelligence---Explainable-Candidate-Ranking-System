"""Unit tests for TF-IDF baseline matching engine."""

import pytest

from src.matching.tfidf_baseline import TfidfBaselineMatcher
from src.models.job import ParsedJobDescription
from src.models.resume import ParsedResume
from src.ranking.tfidf_baseline import TfidfBaselineRanker
from src.models.ranking import RankingRequest


PYTHON_RESUME = ParsedResume(
    name="Alice",
    raw_text=(
        "Python developer with 5 years experience in Django, Flask, "
        "PostgreSQL, REST APIs, and machine learning pipelines."
    ),
    skills=["Python", "Django", "PostgreSQL", "Machine Learning"],
)

JAVA_RESUME = ParsedResume(
    name="Bob",
    raw_text=(
        "Java developer with Spring Boot, Hibernate, SQL, and microservices "
        "architecture experience in enterprise applications."
    ),
    skills=["Java", "Spring Boot", "SQL"],
)

PYTHON_JD = ParsedJobDescription(
    title="Python Developer",
    raw_text=(
        "Looking for a Python developer with Django, PostgreSQL, REST APIs, "
        "and machine learning experience. 3+ years required."
    ),
)

JAVA_JD = ParsedJobDescription(
    title="Java Developer",
    raw_text=(
        "Java developer with Spring Boot, SQL, RESTful APIs, and "
        "microservices. 5+ years experience required."
    ),
)


@pytest.fixture
def matcher() -> TfidfBaselineMatcher:
    return TfidfBaselineMatcher()


class TestTfidfBaselineMatcher:
    def test_matcher_name(self, matcher: TfidfBaselineMatcher):
        assert matcher.name == "tfidf_baseline"

    def test_identical_text_high_similarity(self, matcher: TfidfBaselineMatcher):
        text = "Python Django PostgreSQL machine learning REST APIs"
        resume = ParsedResume(raw_text=text)
        job = ParsedJobDescription(raw_text=text)
        result = matcher.match(resume, job, candidate_id="c1")
        assert result.scores.overall > 0.99
        assert result.scores.baseline_tfidf == result.scores.overall
        assert result.matcher_name == "tfidf_baseline"

    def test_python_resume_matches_python_jd_better_than_java(
        self, matcher: TfidfBaselineMatcher
    ):
        py_result = matcher.match(PYTHON_RESUME, PYTHON_JD)
        java_result = matcher.match(JAVA_RESUME, PYTHON_JD)
        assert py_result.scores.overall > java_result.scores.overall

    def test_java_resume_matches_java_jd_better_than_python(
        self, matcher: TfidfBaselineMatcher
    ):
        java_result = matcher.match(JAVA_RESUME, JAVA_JD)
        py_result = matcher.match(PYTHON_RESUME, JAVA_JD)
        assert java_result.scores.overall > py_result.scores.overall

    def test_batch_match_returns_all_results(self, matcher: TfidfBaselineMatcher):
        results = matcher.match_batch(
            [PYTHON_RESUME, JAVA_RESUME],
            PYTHON_JD,
            ["c1", "c2"],
        )
        assert len(results) == 2
        assert results[0].candidate_id == "c1"
        assert results[1].candidate_id == "c2"
        assert all(0.0 <= r.scores.overall <= 1.0 for r in results)

    def test_batch_match_empty_resumes(self, matcher: TfidfBaselineMatcher):
        assert matcher.match_batch([], PYTHON_JD) == []

    def test_empty_job_raises(self, matcher: TfidfBaselineMatcher):
        job = ParsedJobDescription(raw_text="   ")
        with pytest.raises(ValueError, match="Job description text is empty"):
            matcher.match(PYTHON_RESUME, job)

    def test_empty_resumes_raises(self, matcher: TfidfBaselineMatcher):
        resume = ParsedResume(raw_text="   ")
        with pytest.raises(ValueError, match="All resume texts are empty"):
            matcher.match(resume, PYTHON_JD)

    def test_candidate_ids_length_mismatch(self, matcher: TfidfBaselineMatcher):
        with pytest.raises(ValueError, match="candidate_ids length"):
            matcher.match_batch([PYTHON_RESUME], PYTHON_JD, ["a", "b"])

    def test_rank_by_similarity_ordering(self, matcher: TfidfBaselineMatcher):
        ranked = matcher.rank_by_similarity(
            [JAVA_RESUME, PYTHON_RESUME],
            PYTHON_JD,
            top_k=2,
        )
        assert ranked[0][1].candidate_name == "Alice"
        assert ranked[0][1].scores.overall >= ranked[1][1].scores.overall


class TestTfidfBaselineRanker:
    def test_rank_candidates(self):
        ranker = TfidfBaselineRanker()
        request = RankingRequest(
            job=PYTHON_JD,
            candidates=[JAVA_RESUME, PYTHON_RESUME],
            candidate_ids=["bob", "alice"],
            top_k=2,
        )
        result = ranker.rank(request)
        assert result.total_candidates == 2
        assert len(result.entries) == 2
        assert result.entries[0].rank == 1
        assert result.entries[0].candidate_name == "Alice"
        assert result.entries[0].overall_score >= result.entries[1].overall_score

    def test_rank_empty_candidates(self):
        ranker = TfidfBaselineRanker()
        result = ranker.rank(RankingRequest(job=PYTHON_JD, candidates=[]))
        assert result.entries == []
        assert "No candidates" in result.warnings[0]
