# Smart Contract Summary & Q&A Assistant

A local RAG-based web app for analyzing contracts and legal documents via conversational AI.

---

## Project Structure

```
contract_assistant/
├── app/
│   ├── api/
│   │   └── server.py          # FastAPI + LangServe backend
│   ├── pipelines/
│   │   ├── ingestion.py       # Parse → Chunk → Embed → Store
│   │   ├── rag_chain.py       # Retriever + Generator chains
│   │   ├── guardrails.py      # Topic & grounding safety checks
│   │   └── summarizer.py      # Map-reduce document summarization
│   └── utils/
│       ├── embeddings.py      # Embedding model factory
│       ├── vector_store.py    # Chroma / FAISS factory
│       └── llm.py             # LLM factory (OpenAI / Ollama)
├── config/
│   └── settings.py            # Pydantic settings from .env
├── frontend/
│   └── gradio_app.py          # Gradio UI (Upload + Chat tabs)
├── evaluation/
│   └── eval_pipeline.py       # RAGAS + ROUGE evaluation
├── vectorstore/               # Auto-created: persisted vector data
├── data/uploads/              # Auto-created: uploaded documents
├── tests/
│   └── test_pipelines.py      # Automated pipeline tests
├── requirements.txt
├── .env                       # Configuration (not in version control)
├── run_project.py             # All-in-one startup script
└── env.example                # Template for .env
```

---

## Quick Start

### 1. Set up Python environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp env.example .env
# Edit .env — set OPENAI_API_KEY at minimum
```

### 3. Run Project (All-in-one)

```bash
python run_project.py
```
This script launches both the **FastAPI Backend (9012)** and **Gradio Frontend (8090)** concurrently.

---

## Multi-Model Support

The project supports switching between local and API-based models via the `MODEL_MODE` variable in `.env`.

### Local Mode (Ollama)
Requires [Ollama](https://ollama.com/) installed and running.
```env
MODEL_MODE=local
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:1b
```

### API Mode (OpenAI)
```env
MODEL_MODE=api
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o  # or gpt-3.5-turbo
OPENAI_API_KEY=your_key_here
```

---

## Testing

Run the automated pipeline tests:
```bash
python -m pytest tests/test_pipelines.py
```

Run end-to-end smoke tests (requires backend running):
```bash
python evaluation/test_app.py
```

---

## RAG Optimization
RAG quality is tuned via the following variables in `.env`:
- `CHUNK_SIZE`: 400 (Optimized for contract clauses)
- `CHUNK_OVERLAP`: 80 (Ensures continuity between chunks)
- `MAX_CONTEXT_DOCS`: 5 (Balanced top-k retrieval)

---

## GitHub Push Instructions

To push your project to a new GitHub repository:

1. **Initialize Git** (if not already):
   ```bash
   git init
   ```
2. **Add Remote**:
   ```bash
   git remote add origin https://github.com/your-username/contract-assistant.git
   ```
3. **Commit and Push**:
   ```bash
   git add .
   # Ensure .env is ignored (check .gitignore)
   git commit -m "Initial commit: Enhanced RAG with Multi-Model support"
   git branch -M main
   git push -u origin main
   ```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /basic_chat` | Direct LLM (no retrieval) |
| `POST /retriever` | Semantic search → Documents |
| `POST /generator` | Grounded answer from {input, context} |
| `POST /guardrail` | Topic relevance check |
| `POST /ingest` | Upload + ingest PDF/DOCX |
| `POST /summarize` | Summarize ingested document |
| `GET  /health` | Health check |

---

## Evaluation

```python
from evaluation.eval_pipeline import run_full_eval

results = run_full_eval([{
    "question":     "What is the termination notice period?",
    "answer":       "30 days written notice.",
    "contexts":     ["...contract chunk..."],
    "ground_truth": "The contract requires 30 days written notice.",
}])
```

---

## Tech Stack

LangChain · LangServe · FastAPI · Gradio · Chroma · FAISS · SentenceTransformers · Ollama · OpenAI · PyMuPDF · RAGAS