# app/utils/vector_store.py
# Using FAISS instead of Chroma for local stability on Windows
import os
from langchain_community.vectorstores import FAISS
from loguru import logger
from config.settings import settings

_vectorstore_instance = None

def get_or_create_vectorstore(embeddings):
    """Singleton VectorStore — loads FAISS or creates new one."""
    global _vectorstore_instance
    if _vectorstore_instance is not None:
        return _vectorstore_instance

    path = settings.faiss_index_path
    if os.path.exists(os.path.join(path, "index.faiss")):
        logger.info(f"[VectorStore] Loading FAISS index from {path}")
        _vectorstore_instance = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    else:
        logger.info("[VectorStore] Creating fresh FAISS index with placeholder")
        _vectorstore_instance = FAISS.from_texts(["Initializing..."], embeddings)
    
    return _vectorstore_instance

def save_vectorstore(vectorstore):
    path = settings.faiss_index_path
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)
    logger.success(f"[VectorStore] Saved FAISS index to {path}")
