from src.ranking.hybrid_ranker import HybridRanker
from src.models.resume import ParsedResume
from src.models.job import ParsedJobDescription, SkillRequirement
from src.models.matching import MatchResult, ComponentScores
from src.models.ranking import RankingRequest
from typing import List


class DummyMatcher:
    def __init__(self, semantic_scores: List[float] | None):
        self.semantic_scores = semantic_scores or []

    def match_batch(self, resumes, job, candidate_ids=None):
        results = []
        ids = candidate_ids or [str(i) for i in range(len(resumes))]
        for i, r in enumerate(resumes):
            sem = self.semantic_scores[i] if i < len(self.semantic_scores) else None
            scores = ComponentScores(overall=0.0, semantic=sem, baseline_tfidf=0.5)
            results.append(
                MatchResult(
                    candidate_id=ids[i],
                    candidate_name=r.name,
                    scores=scores,
                    matcher_name="dummy",
                )
            )
        return results


def test_semantic_unavailable_does_not_fallback():
    resumes = [ParsedResume(name="A", raw_text="text")]
    job = ParsedJobDescription(title="Job", raw_text="text")
    matcher = DummyMatcher([None])
    ranker = HybridRanker(matcher)
    req = RankingRequest(job=job, candidates=resumes, candidate_ids=["A"])
    res = ranker.rank(req)
    for ent in res.entries:
        assert ent.semantic_score is None


def test_semantic_used_when_available():
    resumes = [ParsedResume(name="A", raw_text="text", years_experience=3.0)]
    job = ParsedJobDescription(title="Job", raw_text="text", experience_requirements=[])
    matcher = DummyMatcher([0.9])
    ranker = HybridRanker(matcher)
    req = RankingRequest(job=job, candidates=resumes, candidate_ids=["A"])
    res = ranker.rank(req)
    for ent in res.entries:
        assert ent.semantic_score is not None


def test_hybrid_populates_presentation_metadata():
    resumes = [
        ParsedResume(
            name="Alice",
            raw_text="Python developer with SQL experience",
            skills=["Python", "SQL"],
            skills_normalized=["Python", "SQL"],
            years_experience=4.0,
        )
    ]
    job = ParsedJobDescription(
        title="Backend Engineer",
        raw_text="Need Python and AWS",
        required_skills=[
            SkillRequirement(raw_skill="Python", normalized_skill="Python"),
            SkillRequirement(raw_skill="AWS", normalized_skill="AWS"),
        ],
        experience_requirements=[],
    )
    matcher = DummyMatcher([0.75])
    ranker = HybridRanker(matcher)
    res = ranker.rank(RankingRequest(job=job, candidates=resumes, candidate_ids=["a1"]))
    assert res.entries
    mr = res.entries[0].match_result
    assert "Python" in mr.metadata.get("matched_skills", [])
    assert "AWS" in mr.metadata.get("missing_required_skills", [])
    assert mr.scores.skill is not None
    assert mr.scores.overall == res.entries[0].overall_score
    assert mr.scores.semantic == 0.75
