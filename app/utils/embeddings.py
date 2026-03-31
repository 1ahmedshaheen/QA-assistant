# app/utils/embeddings.py — embedding factory
from config.settings import settings
from loguru import logger

_embedder_instance = None


def get_embedder():
    """Singleton embedder — loads once, reuses forever."""
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance

    provider = settings.embedding_provider
    model = settings.embedding_model

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        logger.info(f"[Embedder] OpenAI: {model}")
        _embedder_instance = OpenAIEmbeddings(model=model, openai_api_key=settings.openai_api_key)
        return _embedder_instance

    elif provider == "sentence-transformers":
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info(f"[Embedder] SentenceTransformers: {model}")
        _embedder_instance = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return _embedder_instance

    raise ValueError(f"Unknown embedding provider: {provider}")
