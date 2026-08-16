import io
import pytest

from fastapi.testclient import TestClient
from src.api.main import create_app


@pytest.fixture(autouse=True)
def patch_embedding(monkeypatch):
    # reuse lightweight deterministic embedding to avoid transformer downloads
    class MockEmbeddingEngine:
        def embed_text(self, text):
            return [len(text), sum(1 for c in text.lower() if c in 'aeiou')]

        def embed_texts(self, texts):
            return [self.embed_text(t) for t in texts]

        @property
        def model_name(self):
            return "mock"

    import src.embeddings as emb

    monkeypatch.setattr(emb, "get_embedding_engine", lambda: MockEmbeddingEngine())
    yield


def test_api_does_not_return_pii():
    app = create_app()
    client = TestClient(app)

    job_text = "Looking for data analyst"
    # resume contains PII
    resume_text = b"John Doe\nemail@example.com\n+91 98765 43210\nExperienced analyst."

    files = {"resumes": ("resume1.txt", io.BytesIO(resume_text), "text/plain")}
    data = {"job_text": job_text, "job_title": "Analyst"}

    r = client.post('/api/screen', data=data, files=files)
    assert r.status_code == 200
    body = r.json()
    # Ensure PII values do not appear in response
    assert "email@example.com" not in str(body)
    assert "+91 98765 43210" not in str(body)
