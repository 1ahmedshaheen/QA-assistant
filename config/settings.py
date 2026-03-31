# config/settings.py
# ═══════════════════════════════════════════════════════
#  Central settings — loaded from .env via pydantic
# ═══════════════════════════════════════════════════════
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────
    llm_provider: str = Field("openai", env="LLM_PROVIDER")
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    nvidia_api_key: str = Field("", env="NVIDIA_API_KEY")
    llm_model: str = Field("gpt-3.5-turbo", env="LLM_MODEL")
    llm_temperature: float = Field(0.2, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(1024, env="LLM_MAX_TOKENS")

    # ── Embeddings ───────────────────────────────────
    embedding_provider: str = Field("sentence-transformers", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")

    # ── Vector Store ─────────────────────────────────
    vector_store: str = Field("chroma", env="VECTOR_STORE")
    chroma_persist_dir: str = Field("./vectorstore/chroma_db", env="CHROMA_PERSIST_DIR")
    faiss_index_path: str = Field("./vectorstore/faiss_index", env="FAISS_INDEX_PATH")

    # ── Chunking ─────────────────────────────────────
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")

    # ── Server ───────────────────────────────────────
    langserve_port: int = Field(9012, env="LANGSERVE_PORT")
    gradio_port: int = Field(8090, env="GRADIO_PORT")

    # ── Guardrails ───────────────────────────────────
    guardrail_similarity_threshold: float = Field(0.75, env="GUARDRAIL_SIMILARITY_THRESHOLD")
    max_context_docs: int = Field(5, env="MAX_CONTEXT_DOCS")

    # ── Logging ──────────────────────────────────────
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # ── Model Mode ───────────────────────────────────
    model_mode: str = Field("local", env="MODEL_MODE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton — import this everywhere
settings = Settings()
