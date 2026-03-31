# app/pipelines/ingestion.py
# Ingestion pipeline: PDF/DOCX -> Chunks -> Vector Store
import os
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.utils.embeddings import get_embedder
from app.utils.vector_store import get_or_create_vectorstore
from config.settings import settings

def is_useful_text(text: str, alpha_threshold: float = 0.5) -> bool:
    """Filters out chunks that are mostly numbers/symbols (likely garbage)."""
    if not text.strip(): return False
    alpha_chars = sum(c.isalpha() for c in text)
    return (alpha_chars / len(text)) >= alpha_threshold

def ingest_document(file_path: str):
    logger.info(f"[Ingestion] Processing: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    
    # ── Nvidia-Style Splitting ──────────────────────────────
    import time
    t_start = time.time()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""]
    )
    
    raw_chunks = splitter.split_documents(docs)
    logger.info(f"[Ingestion] Document splitting took {time.time() - t_start:.2f}s")
    
    # ── Quality Filtering ──────────────────────────────────
    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        if is_useful_text(chunk.page_content):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["source"] = os.path.basename(file_path)
            final_chunks.append(chunk)

    if not final_chunks:
        logger.warning(f"[Ingestion] No useful text found in {file_path}. Skipping vector storage.")
        raise ValueError("No readable or high-quality text found in document.")

    logger.info(f"[Ingestion] Created {len(final_chunks)} high-quality chunks (Filtered out {len(raw_chunks)-len(final_chunks)}).")

    # ── Vector Store Update ────────────────────────────────
    t_v_start = time.time()
    embedder = get_embedder()
    vectorstore = get_or_create_vectorstore(embedder)
    
    vectorstore.add_documents(final_chunks)
    vectorstore.save_local(settings.faiss_index_path)
    logger.info(f"[Ingestion] Vector storage update took {time.time() - t_v_start:.2f}s")
    
    logger.success(f"[Ingestion] Added {len(final_chunks)} chunks from {os.path.basename(file_path)}")
    return {
        "status": "success",
        "file": os.path.basename(file_path),
        "chunks": len(final_chunks),
        "chars": sum(len(c.page_content) for c in final_chunks)
    }
