import pytest
from unittest.mock import MagicMock, patch
from app.pipelines.ingestion import ingest_document, is_useful_text
from app.utils.llm import get_llm
from app.utils.embeddings import get_embedder
from app.pipelines.rag_chain import build_retriever_chain, build_generator_chain

def test_is_useful_text():
    assert is_useful_text("This is a normal sentence.") is True
    assert is_useful_text("123 456 !!! ???") is False

@patch("app.pipelines.ingestion.PyMuPDFLoader")
@patch("app.pipelines.ingestion.get_or_create_vectorstore")
def test_ingest_document(mock_vs_func, mock_loader):
    # Mock loader
    mock_instance = mock_loader.return_value
    mock_instance.load.return_value = [MagicMock(page_content="test content", metadata={})]
    
    # Mock vectorstore
    mock_vs = MagicMock()
    mock_vs_func.return_value = mock_vs
    
    result = ingest_document("test.pdf")
    
    assert result["status"] == "success"
    assert mock_vs.add_documents.called

def test_singleton_llm():
    llm1 = get_llm()
    llm2 = get_llm()
    assert llm1 is llm2

def test_singleton_embedder():
    emb1 = get_embedder()
    emb2 = get_embedder()
    assert emb1 is emb2

@patch("app.pipelines.rag_chain.get_or_create_vectorstore")
def test_retriever_chain(mock_vs_func):
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [MagicMock(page_content="test doc", metadata={"source": "test.pdf"})]
    mock_vs.as_retriever.return_value = mock_retriever
    mock_vs_func.return_value = mock_vs

    chain = build_retriever_chain()
    results = chain.invoke("test query")

    assert len(results) == 1
    assert results[0]["page_content"] == "test doc"

@patch("app.pipelines.rag_chain.get_llm")
def test_generator_chain(mock_llm_func):
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    
    # Use a RunnableLambda to simulate the LLM in the chain
    mock_llm = RunnableLambda(lambda x: AIMessage(content="Generated answer."))
    mock_llm_func.return_value = mock_llm
    
    chain = build_generator_chain()
    answer = chain.invoke({"input": "What is test?", "context": "Test context."})
    
    assert "Generated answer" in answer
