# app/pipelines/evaluation.py
# Synthetic Q&A Generation & RAG Evaluation
# Adapted from Nvidia reference code for local Ollama usage
import random
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.utils.llm import get_llm
from app.utils.embeddings import get_embedder
from app.utils.vector_store import get_or_create_vectorstore
from loguru import logger

def generate_synthetic_qa(num_questions: int = 3) -> List[Dict]:
    """Generates synthetic Q&A pairs from the current vector store."""
    llm = get_llm()
    embedder = get_embedder()
    vectorstore = get_or_create_vectorstore(embedder)
    
    if not vectorstore:
        logger.error("[Eval] No vector store found for evaluation.")
        return []

    # Get some random chunks
    # Note: FAISS doesn't directly expose all docs easily without knowing IDs, 
    # but we can do a broad search or just grab from the docstore dict if using FAISS.
    docs = list(vectorstore.docstore._dict.values())
    if len(docs) < 2:
        logger.warning("[Eval] Not enough documents to generate quality pairs.")
        return []

    synth_data = []
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Use the documents provided by the user to generate an interesting question-answer pair.\n"
         "Try to use both documents if possible.\n"
         "Use the format:\nQuestion: (detailed question)\nAnswer: (answer derived from docs)\n"
         "DO NOT say 'Here is a pair'. FOLLOW FORMAT!"),
        ("human", "Document 1: {doc1}\n\nDocument 2: {doc2}")
    ])

    chain = qa_prompt | llm | StrOutputParser()

    for i in range(num_questions):
        d1, d2 = random.sample(docs, 2)
        logger.info(f"[Eval] Generating pair {i+1}/{num_questions}...")
        
        try:
            raw_output = chain.invoke({
                "doc1": d1.page_content,
                "doc2": d2.page_content
            })
            
            # Simple parsing
            if "Question:" in raw_output and "Answer:" in raw_output:
                parts = raw_output.split("Answer:")
                question = parts[0].replace("Question:", "").strip()
                answer = parts[1].strip()
                synth_data.append({
                    "question": question,
                    "ground_truth": answer,
                    "sources": [d1.metadata.get("source"), d2.metadata.get("source")]
                })
        except Exception as e:
            logger.error(f"[Eval] Error generating pair: {e}")

    return synth_data

def evaluate_rag_answer(question: str, ground_truth: str, rag_answer: str) -> str:
    """Evaluates a RAG answer against a ground truth using LLM-as-a-judge."""
    llm = get_llm()
    
    eval_prompt = ChatPromptTemplate.from_template(
        "INSTRUCTION:\n"
        "Evaluate the following Question-Answer pair for preference and consistency.\n"
        "Assume the Ground Truth is correct. Assume the RAG Answer may or may not be.\n\n"
        "Question: {question}\n"
        "Ground Truth: {ground_truth}\n"
        "RAG Answer: {rag_answer}\n\n"
        "EVALUATION CRITERIA:\n"
        "[1] The RAG answer is incorrect, hallucinations, or fails to answer.\n"
        "[2] The RAG answer is consistent with ground truth and helpful.\n\n"
        "Output ONLY the [Score] and a short Justification."
    )
    
    eval_chain = eval_prompt | llm | StrOutputParser()
    return eval_chain.invoke({
        "question": question,
        "ground_truth": ground_truth,
        "rag_answer": rag_answer
    })
