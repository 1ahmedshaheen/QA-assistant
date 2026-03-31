# app/pipelines/guardrails.py
# Topic relevance + hallucination grounding checks
import numpy as np
from langchain_core.runnables import RunnableLambda
from loguru import logger
from app.utils.embeddings import get_embedder
from config.settings import settings

CONTRACT_TOPICS = [
    "contract clause", "agreement terms", "legal obligations",
    "payment terms", "termination", "liability", "indemnification",
    "intellectual property", "confidentiality", "governing law",
]


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def is_on_topic(query: str) -> bool:
    embedder = get_embedder()
    q_emb = embedder.embed_query(query)
    topic_embs = embedder.embed_documents(CONTRACT_TOPICS)
    max_sim = max(cosine_similarity(q_emb, t) for t in topic_embs)
    logger.debug(f"[Guardrail] Topic similarity: {max_sim:.3f}")
    return max_sim >= settings.guardrail_similarity_threshold


def guardrail_chain(query: str) -> dict:
    if not is_on_topic(query):
        return {
            "blocked": True,
            "reason": "off_topic",
            "message": "I can only answer questions about the uploaded contract documents.",
        }
    return {"blocked": False}


def build_guardrail_chain():
    return RunnableLambda(guardrail_chain)
