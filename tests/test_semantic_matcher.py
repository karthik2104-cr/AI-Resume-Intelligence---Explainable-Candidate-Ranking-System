from src.matching.semantic_matcher import SemanticMatcher
from src.models.resume import ParsedResume
from src.models.job import ParsedJobDescription
from unittest.mock import MagicMock
import numpy as np


class FakeEngine:
    def __init__(self):
        # simple 3-d vectors to emulate semantics
        self.model_name = "fake-model"
        self._dim = 3
        self._normalize = True

    def embed_texts(self, texts):
        # Orthogonal vectors so frontend and ML profiles do not overlap by accident
        mapping = {
            "ml_resume": np.array([1.0, 0.0, 0.0]),
            "ml_jd": np.array([1.0, 0.0, 0.0]),
            "frontend_resume": np.array([0.0, 1.0, 0.0]),
            "ml_unrelated": np.array([0.0, 0.0, 0.0]),
        }
        out = []
        for t in texts:
            lower = (t or "").lower()
            if "react" in lower or "frontend" in lower or "typescript" in lower:
                out.append(mapping["frontend_resume"])
            elif (
                "engineer" in lower
                or "predict" in lower
                or "sklearn" in lower
                or lower.strip() in {"sklearn", "machine learning"}
            ):
                out.append(mapping["ml_jd"])
            elif "machine" in lower or "scikit" in lower:
                out.append(mapping["ml_resume"])
            else:
                out.append(mapping["ml_unrelated"])
        return np.vstack(out)

    def embed_text(self, text):
        return self.embed_texts([text])[0]


def test_semantic_matcher_basic(monkeypatch):
    # monkeypatch the engine factory to return fake engine
    monkeypatch.setattr("src.matching.semantic_matcher.get_embedding_engine", lambda: FakeEngine())

    matcher = SemanticMatcher()

    from src.models.job import SkillRequirement
    resume = ParsedResume(summary="Built machine learning models using Python and scikit-learn.", skills=["Python", "scikit-learn"], experience=[])
    job = ParsedJobDescription(
        title="Machine learning engineer",
        responsibilities=[],
        required_skills=[SkillRequirement(raw_skill="sklearn"), SkillRequirement(raw_skill="machine learning")],
        preferred_skills=[],
        raw_text="Predictive ML systems using sklearn",
    )

    res = matcher.match(resume, job)
    assert res.scores.semantic > 0.5, "Expected high semantic similarity for ML texts"

    resume2 = ParsedResume(summary="Built React frontends and TypeScript apps.", skills=["React", "TypeScript"], experience=[])
    res2 = matcher.match(resume2, job)
    assert res2.scores.semantic < 0.01, "Expected low semantic similarity between frontend and ML"
