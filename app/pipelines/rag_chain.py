# app/pipelines/rag_chain.py
# RAG pipeline: retriever + generator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.document_transformers import LongContextReorder
from loguru import logger

from app.utils.embeddings import get_embedder
from app.utils.vector_store import get_or_create_vectorstore
from app.utils.llm import get_llm
from config.settings import settings


def build_retriever_chain():
    embedder = get_embedder()
    vectorstore = get_or_create_vectorstore(embedder)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.max_context_docs},
    )

    def retrieve_and_reorder(query: str):
        import time
        t_start = time.time()
        logger.info(f"[Retriever] Searching for: {query[:50]}...")
        docs = retriever.invoke(query)
        logger.info(f"[Retriever] Found {len(docs)} docs in {time.time() - t_start:.2f}s")
        # Reorder to handle "lost-in-the-middle" problem
        reorder = LongContextReorder()
        reordered_docs = reorder.transform_documents(docs)
        
        # Return serializable dicts
        return [
            {
                "page_content": d.page_content,
                "metadata": d.metadata,
                "title": d.metadata.get("source", "Unknown Document")
            }
            for d in reordered_docs
        ]

    return RunnableLambda(retrieve_and_reorder)


# ── Enhanced Prompt — from Nvidia reference & User Request
RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a professional document analysis assistant. Provide a highly detailed and comprehensive answer based ON-ONLY the provided context.\n\n"
    "If the answer is not found in the context say: 'I cannot find this information in the document.'\n\n"
    "Be precise and cite relevant clauses if available.\n\n"
    "RETRIEVED CONTEXT:\n{context}\n\n"
    "USER QUESTION: {input}\n\n"
    "INSTRUCTIONS:\n"
    "1. Be extremely thorough. Explain complex points in detail.\n"
    "2. Cite the document source title for every major fact or claim.\n"
    "3. Use professional, clear language.\n"
    "4. If the information is missing, state it clearly.\n"
    "5. MANDATORY: End your response with a section titled '### Summary' followed by a 2-3 sentence wrap-up.\n\n"
    "DETAILED ANALYSIS:"
)


def format_docs_nvidia_style(docs) -> str:
    """Format documents with clear labels. Handles both list of docs and raw strings."""
    if not docs:
        return "No relevant context found."
    
    if isinstance(docs, str):
        return docs
        
    formatted_parts = []
    for doc in docs:
        # Handle both LangChain Document objects and dictionaries
        if hasattr(doc, "metadata"): # LangChain Document
            meta = doc.metadata
            content = doc.page_content
        elif isinstance(doc, dict): # Dictionary (from LangServe JSON)
            meta = doc.get("metadata", {})
            content = doc.get("page_content", str(doc))
        else:
            meta = {}
            content = str(doc)
            
        title = meta.get("source", "Document")
        formatted_parts.append(f"[Quote from {title}]\n{content}")
    
    return "\n\n".join(formatted_parts)


def build_generator_chain():
    import time
    llm = get_llm()
    # Input: {"input": str, "context": list[dict]}
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs_nvidia_style(x.get("context", []))
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    
    def chain_with_logging(input_data):
        t0 = time.time()
        logger.info(f"[RAG] Generating answer for: {input_data.get('input', 'Unknown query')[:50]}...")
        result = chain.invoke(input_data)
        logger.info(f"[RAG] Generation took {time.time() - t0:.2f}s")
        return result

    return RunnableLambda(chain_with_logging)
