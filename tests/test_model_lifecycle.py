import pytest

import src.embeddings as emb


def test_get_embedding_engine_singleton(monkeypatch):
    # Create a fake provider class and patch provider map
    class FakeEngine:
        def __init__(self):
            self._id = id(self)

        def embed_text(self, t):
            return [1.0]

        def embed_texts(self, ts):
            return [[1.0] for _ in ts]

    # Patch provider map to point to a local fake class
    monkeypatch.setitem(emb._provider_map, "fake_provider", "src.tests._fake.FakeEngine")

    # Make a module for the fake class so import works
    import types, sys

    mod = types.ModuleType("src.tests._fake")
    mod.FakeEngine = FakeEngine
    sys.modules["src.tests._fake"] = mod

    # Temporarily patch settings to use fake_provider
    class Settings:
        class _Emb:
            provider = "fake_provider"

        embeddings = _Emb()

    monkeypatch.setattr("src.utils.config.get_settings", lambda: Settings)

    e1 = emb.get_embedding_engine()
    e2 = emb.get_embedding_engine()
    assert e1 is e2
