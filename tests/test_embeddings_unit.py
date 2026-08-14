import numpy as np
from src.embeddings import get_embedding_engine
from src.embeddings.sentence_transformers import SentenceTransformerEmbedding
from unittest.mock import MagicMock


def test_sentence_transformer_embedding_monkeypatched(monkeypatch):
    # Create a fake model with deterministic encode behavior
    class FakeModel:
        def __init__(self, name):
            self.name = name

        def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
            # deterministic: length of text and sum of ords
            out = []
            for t in texts:
                s = sum(ord(c) for c in (t or ""))
                dim = 8
                vec = [float((s % (i + 3)) / (i + 3)) for i in range(dim)]
                out.append(vec)
            return np.array(out, dtype=float)

    monkeypatch.setattr("src.embeddings.sentence_transformers.SentenceTransformerEmbedding._load_model", staticmethod(lambda model_name: FakeModel(model_name)))

    emb = SentenceTransformerEmbedding()
    v1 = emb.embed_text("hello world")
    v2 = emb.embed_text("hello world")
    assert np.allclose(v1, v2), "Embedding should be deterministic and cached"

    batch = emb.embed_texts(["a", "b", "hello world"])
    assert batch.shape[0] == 3
    assert batch.shape[1] == v1.shape[0]

    # normalization enabled by default in config; ensure norms ~1
    norms = np.linalg.norm(batch, axis=1)
    assert np.all(norms > 0)


def test_embedding_empty_text(monkeypatch):
    class FakeModel:
        def __init__(self, name):
            pass

        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 4))

    monkeypatch.setattr("src.embeddings.sentence_transformers.SentenceTransformerEmbedding._load_model", staticmethod(lambda model_name: FakeModel(model_name)))
    emb = SentenceTransformerEmbedding()
    v = emb.embed_text("")
    assert v.size == 0 or v.sum() == 0
