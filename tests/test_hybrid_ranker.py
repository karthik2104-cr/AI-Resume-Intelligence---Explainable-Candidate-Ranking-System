from src.ranking.hybrid_ranker import HybridRanker
from src.models.resume import ParsedResume
from src.models.job import ParsedJobDescription
from src.models.matching import MatchResult, ComponentScores
from src.models.hybrid_matching import HybridMatchResult, ComponentScore
from src.models.ranking import RankingRequest
from typing import List


class DummyMatcher:
    def __init__(self, semantic_scores: List[float] | None):
        self.semantic_scores = semantic_scores or []

    def match_batch(self, resumes, job, candidate_ids=None):
        results = []
        for i, r in enumerate(resumes):
            sem = self.semantic_scores[i] if i < len(self.semantic_scores) else None
            scores = ComponentScores(overall=0.0, semantic=sem, baseline_tfidf=0.5)
            results.append(MatchResult(candidate_id=str(i), candidate_name=r.name, scores=scores, matcher_name="dummy"))
        return results


def test_semantic_unavailable_does_not_fallback():
    resumes = [ParsedResume(name="A", raw_text="text")]
    job = ParsedJobDescription(title="Job", raw_text="text")
    # matcher that has semantic unavailable
    matcher = DummyMatcher([None])
    ranker = HybridRanker(matcher)
    req = RankingRequest(job=job, candidates=resumes, candidate_ids=["A"])
    res = ranker.rank(req)
    # ensure semantic component in applied weights is absent or marked unavailable
    for ent in res.entries:
        # semantic_score should be None
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
