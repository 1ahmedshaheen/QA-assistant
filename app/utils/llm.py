# app/utils/llm.py — LLM factory
# Optimized for Real-Time Local Inference via Ollama
from config.settings import settings
from loguru import logger

_llm_instance = None


def get_llm():
    """Singleton LLM — switches based on MODEL_MODE (local vs api)."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    mode = settings.model_mode.lower()
    
    if mode == "api":
        # Use OpenAI for API mode
        from langchain_openai import ChatOpenAI
        logger.info(f"[LLM] Mode: API | Provider: OpenAI | Model: {settings.llm_model}")
        
        if not settings.openai_api_key:
            logger.error("OPENAI_API_KEY is missing in .env but MODEL_MODE=api is set!")
            raise ValueError("OPENAI_API_KEY must be provided when MODEL_MODE=api")
            
        _llm_instance = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            openai_api_key=settings.openai_api_key,
        )
    else:
        # Default to local Ollama
        from langchain_ollama import ChatOllama
        logger.info(f"[LLM] Mode: Local | Provider: Ollama | Model: {settings.llm_model}")
        
        _llm_instance = ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
            base_url="http://localhost:11434",
        )

    return _llm_instance
