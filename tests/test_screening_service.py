import pytest
import numpy as np

from src.services.resume_screening_service import ScreeningService
from src.models.resume import ParsedResume
from src.models.job import ParsedJobDescription


class MockEmbeddingEngine:
    def __init__(self):
        self._dim = 3

    @property
    def model_name(self):
        return "mock"

    def embed_text(self, text: str):
        # deterministic simple encoding based on token presence
        t = text.lower()
        return np.array([
            1.0 if "python" in t else 0.0,
            1.0 if "java" in t else 0.0,
            float(len(t) % 5),
        ], dtype=float)

    def embed_texts(self, texts):
        return np.vstack([self.embed_text(t) for t in texts])


@pytest.fixture(autouse=True)
def patch_embedding(monkeypatch):
    import src.embeddings as emb

    monkeypatch.setattr(emb, "get_embedding_engine", lambda: MockEmbeddingEngine())


def make_resume(name: str, text: str) -> ParsedResume:
    return ParsedResume(name=name, raw_text=text, skills=["Python"] if "Python" in text else [], skills_normalized=["Python"] if "Python" in text else [])


def test_screening_happy_path():
    resumes = [
        make_resume("Alice", "Experienced Python developer with data science projects."),
        make_resume("Bob", "Java backend engineer experienced with microservices."),
        make_resume("Carol", "Full stack engineer with Python and Java skills."),
    ]
    job = ParsedJobDescription(title="Senior Python Engineer", raw_text="Looking for Python developer with experience in ML.")
    service = ScreeningService()
    result = service.screen(job, resumes, candidate_ids=["a", "b", "c"])

    assert result["ranking_result"] is not None
    # retrieval results should be present
    assert isinstance(result["retrieval_results"], list)
    # explanations should exist for shortlisted candidates
    assert isinstance(result["explanations"], dict)
