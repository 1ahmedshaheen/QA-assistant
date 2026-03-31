# app/api/server.py
# FastAPI + LangServe backend server
# Optimized for Real-Time Local Inference via Ollama & Streaming
import os
import sys
import threading
from typing import Any, AsyncIterator, Dict, List, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.runnables import Runnable, RunnableConfig
from loguru import logger

from config.settings import settings

app = FastAPI(
    title="Smart Contract Q&A API",
    version="1.3.1",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── LazyRunnable Wrapper ───────────────────────────────────
# This class ensures that model loading is deferred until the 
# very first call (invoke, stream, etc.), while preserving 
# all LangChain runnable methods for LangServe.

class LazyRunnable(Runnable):
    def __init__(self, loader_fn):
        self.loader_fn = loader_fn
        self._chain = None
        self._lock = threading.Lock()

    @property
    def chain(self):
        with self._lock:
            if self._chain is None:
                logger.info(f"[Server] Lazy load triggered for {self.loader_fn.__name__}")
                self._chain = self.loader_fn()
            return self._chain

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        return self.chain.invoke(input, config)

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        return await self.chain.ainvoke(input, config)

    def stream(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        return self.chain.stream(input, config)

    async def astream(self, input: Any, config: Optional[RunnableConfig] = None) -> AsyncIterator[Any]:
        async for chunk in self.chain.astream(input, config):
            yield chunk

# ── Route Builders ─────────────────────────────────────────

def get_basic_chat():
    from app.utils.llm import get_llm
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer the user's question.\n\nQuestion: {input}\nAnswer:")
    return prompt | get_llm()

def get_retriever():
    from app.pipelines.rag_chain import build_retriever_chain
    return build_retriever_chain()

def get_generator():
    from app.pipelines.rag_chain import build_generator_chain
    return build_generator_chain()

def get_guardrail():
    from app.pipelines.guardrails import build_guardrail_chain
    return build_guardrail_chain()

# ── Route Registration ─────────────────────────────────────

logger.info("[Server] Registering streaming-capable routes (Lazy)...")
add_routes(app, LazyRunnable(get_basic_chat), path="/basic_chat")
add_routes(app, LazyRunnable(get_retriever), path="/retriever")
add_routes(app, LazyRunnable(get_generator), path="/generator")
add_routes(app, LazyRunnable(get_guardrail), path="/guardrail")

@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    from app.pipelines.ingestion import ingest_document
    
    # ── Security: Extension Validation ─────────────────────
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".docx", ".txt"]:
        logger.warning(f"[Security] Blocked upload of {file.filename} (Invalid extension: {ext})")
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Please upload PDF, DOCX, or TXT.")

    upload_dir = "./data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    save_path = os.path.join(upload_dir, os.path.basename(file.filename))
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
        
    logger.info(f"[Upload] Saved {file.filename}")
    try:
        return ingest_document(save_path)
    except ValueError as e:
        logger.warning(f"[Ingestion] Validation failed for {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[Ingestion] Unexpected error for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/summarize")
async def summarize_endpoint(file_name: str):
    from app.pipelines.summarizer import summarize_documents
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    file_path = os.path.join("./data/uploads", os.path.basename(file_name))
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        logger.info(f"[Summarize] Loading {file_name}...")
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        
        # Split into manageable chunks for the iterative summarizer
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        # Limit to first few chunks if on slow CPU to avoid timeouts, or just run full
        # For now, full run
        summary_obj = summarize_documents(chunks)
        
        return {
            "status": "success",
            "file": file_name,
            "summary": summary_obj.running_summary,
            "main_ideas": summary_obj.main_ideas,
            "loose_ends": summary_obj.loose_ends
        }
    except Exception as e:
        logger.error(f"[Summarize] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate")
async def evaluate_endpoint(num_questions: int = 3):
    """Triggers synthetic Q&A generation and evaluation."""
    from app.pipelines.evaluation import generate_synthetic_qa, evaluate_rag_answer
    from app.pipelines.rag_chain import build_retriever_chain, build_generator_chain
    
    logger.info(f"[Server] Starting evaluation with {num_questions} questions...")
    qa_pairs = generate_synthetic_qa(num_questions)
    
    retriever = build_retriever_chain()
    generator = build_generator_chain()
    
    results = []
    for pair in qa_pairs:
        # Simulate RAG
        context = retriever.invoke(pair["question"])
        rag_answer = generator.invoke({"input": pair["question"], "context": context})
        
        # Evaluate
        score_text = evaluate_rag_answer(pair["question"], pair["ground_truth"], rag_answer)
        results.append({
            "question": pair["question"],
            "ground_truth": pair["ground_truth"],
            "rag_answer": rag_answer,
            "evaluation": score_text
        })
        
    return {"results": results}

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "mode": settings.model_mode,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "version": "1.3.1"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"[Server] Starting on port {settings.langserve_port}")
    # Use app object directly to avoid redundant module imports in the same process
    uvicorn.run(app, host="0.0.0.0", port=settings.langserve_port)
