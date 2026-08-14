import pytest

from src.api.main import create_app
from fastapi.testclient import TestClient
import io


class MockEmbeddingEngine:
    def embed_text(self, text):
        # deterministic simple vector based on length
        return [len(text), sum(1 for c in text.lower() if c in 'aeiou')]

    def embed_texts(self, texts):
        return [self.embed_text(t) for t in texts]


@pytest.fixture(autouse=True)
def patch_embedding(monkeypatch):
    import src.embeddings as emb
    monkeypatch.setattr(emb, "get_embedding_engine", lambda: MockEmbeddingEngine())
    yield


def test_repeated_runs_are_deterministic():
    app = create_app()
    client = TestClient(app)

    job_text = "Machine learning engineer with scikit-learn"
    resume_bytes = io.BytesIO(b"Experienced ML engineer using scikit-learn and PyTorch.")
    files = {"resumes": ("r1.txt", resume_bytes, "text/plain")}
    data = {"job_text": job_text, "job_title": "ML"}

    r1 = client.post('/api/screen', data=data, files=files)
    assert r1.status_code == 200
    body1 = r1.json()

    # Re-post same payload
    resume_bytes2 = io.BytesIO(b"Experienced ML engineer using scikit-learn and PyTorch.")
    files2 = {"resumes": ("r1.txt", resume_bytes2, "text/plain")}
    r2 = client.post('/api/screen', data=data, files=files2)
    assert r2.status_code == 200
    body2 = r2.json()

    assert body1 == body2
