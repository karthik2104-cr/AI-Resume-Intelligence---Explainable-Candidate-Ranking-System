from src.embeddings.base import EmbeddingEngine
from src.utils.config import get_settings

# Lazy import map for providers
_provider_map = {
    "sentence_transformers": "src.embeddings.sentence_transformers.SentenceTransformerEmbedding",
    "sentence-transformers": "src.embeddings.sentence_transformers.SentenceTransformerEmbedding",
}


_ENGINE_INSTANCES: dict[str, EmbeddingEngine] = {}


def get_embedding_engine() -> EmbeddingEngine:
    """Factory to return configured embedding engine instance.

    Reads provider + model_name from settings and returns a singleton instance per
    configured provider for the application process. This avoids repeated model
    loading across requests while preserving the interface-driven design.
    """
    settings = get_settings().embeddings
    provider = settings.provider if hasattr(settings, "provider") else "sentence_transformers"
    impl_path = _provider_map.get(provider)
    if not impl_path:
        raise RuntimeError(f"Unknown embedding provider configured: {provider}")
    # Return cached instance when available
    if provider in _ENGINE_INSTANCES:
        return _ENGINE_INSTANCES[provider]

    module_path, class_name = impl_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    inst = cls()
    _ENGINE_INSTANCES[provider] = inst
    return inst


__all__ = ["EmbeddingEngine", "get_embedding_engine"]
