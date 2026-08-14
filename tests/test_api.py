from fastapi.testclient import TestClient
import io
import pytest

from src.api.main import create_app

# monkeypatch the embedding engine to avoid heavy model downloads
class MockEmbeddingEngine:
    def embed_text(self, text):
        # simple deterministic embedding: length and vowel fraction
        return [len(text), sum(1 for c in text.lower() if c in 'aeiou') / max(1, len(text))]

    def embed_texts(self, texts):
        return [self.embed_text(t) for t in texts]


@pytest.fixture(autouse=True)
def patch_embedding(monkeypatch):
    import src.embeddings as emb

    monkeypatch.setattr(emb, "get_embedding_engine", lambda: MockEmbeddingEngine())
    yield


def test_health():
    app = create_app()
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_screen_minimal():
    app = create_app()
    client = TestClient(app)

    job_text = "Machine learning engineer skilled in scikit-learn and PyTorch"
    # create a small text resume as UploadFile replacement
    resume_bytes = io.BytesIO(b"Experienced ML engineer using scikit-learn and PyTorch. Deployed models.")

    files = {
        'resumes': ('resume1.txt', resume_bytes, 'text/plain')
    }
    data = {
        'job_text': job_text,
        'job_title': 'ML Engineer'
    }

    r = client.post('/api/screen', data=data, files=files)
    assert r.status_code == 200
    body = r.json()
    assert 'candidates' in body
    assert isinstance(body['candidates'], list)
    # since only one resume, expect one candidate
    assert len(body['candidates']) == 1
    cand = body['candidates'][0]
    assert 'candidate_id' in cand
    assert 'overall_score' in cand
