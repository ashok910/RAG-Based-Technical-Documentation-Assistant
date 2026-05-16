# 🧠 DocuMind — RAG-Based Technical Documentation Assistant

A **production-grade, self-corrective Retrieval-Augmented Generation (RAG)** system built with LangGraph, FastAPI, and a stunning modern UI. Answers questions about technical documentation with citations, hallucination checking, and adaptive query rewriting.

---

## 🏗️ Architecture Overview

```
User Question
      │
      ▼
┌─────────────────┐
│  Query Analysis │  → Classify type + rewrite query for better retrieval
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Retrieval    │  → ChromaDB/FAISS vector similarity search (top-k chunks)
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Document Grading   │  → LLM grades each chunk as relevant/irrelevant
│  (self-corrective)  │
└──────────┬──────────┘
           │
    ┌──────┴──────────────────────────┐
    │              │                  │
    ▼              ▼                  ▼
Generate     Query Rewrite        Web Search
(relevant    (retry with new     (Tavily fallback,
 docs found)  query, max 2x)     bonus feature)
    │              │                  │
    └──────┬───────┘                  │
           │◄─────────────────────────┘
           ▼
┌──────────────────────┐
│  Hallucination Check │  → Verify answer is grounded in context (bonus)
└──────────┬───────────┘
           │
           ▼
      Final Answer
   (with citations)
```

### LangGraph State Schema

The `RAGState` TypedDict flows through all nodes:

| Field | Type | Purpose |
|-------|------|---------|
| `original_question` | str | Raw user input |
| `rewritten_query` | str | Query after analysis/rewrite |
| `query_type` | str | conceptual/how-to/troubleshooting/api-reference |
| `retrieved_documents` | List[Document] | Raw retrieval results |
| `relevant_documents` | List[Document] | After grading filter |
| `retry_count` | int | Self-correction loop counter |
| `route_decision` | str | generate/rewrite/web_search/fallback |
| `answer` | str | Generated response |
| `citations` | List[dict] | Source references |
| `hallucination_score` | float | Groundedness score 0–1 |
| `processing_steps` | List[str] | Full audit trail |

---

## ⚡ Quick Start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd rag-assistant
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your LLM API key
```

**Recommended: Groq (free, fast)**
```env
LLM_PROVIDER=groq
LLM_MODEL=llama3-8b-8192
GROQ_API_KEY=your_key_from_console.groq.com
```

Other options: OpenAI (`gpt-4o-mini`), Google Gemini (`gemini-1.5-flash`), Anthropic (`claude-3-haiku`).

### 3. Pre-ingest corpus (optional)

The server auto-ingests `docs/` on startup if the vector store is empty. To do it manually:

```bash
python scripts/ingest.py                    # Ingest all docs/ files
python scripts/ingest.py --url <url>        # Ingest from URL
python scripts/ingest.py --reset            # Clear and re-ingest
```

### 4. Run the server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open the UI

Navigate to **http://localhost:8000** — the full frontend is served automatically.

API docs available at **http://localhost:8000/docs** (Swagger UI).

---

## 📡 API Reference

### POST `/api/query`
Submit a question to the RAG pipeline.

**Request:**
```json
{
  "question": "How do I use dependency injection in FastAPI?",
  "session_id": "session_abc123",
  "chat_history": [
    {"role": "user", "content": "Tell me about FastAPI"},
    {"role": "assistant", "content": "FastAPI is a modern web framework..."}
  ]
}
```

**Response:**
```json
{
  "answer": "FastAPI's dependency injection system uses `Depends()`...",
  "citations": [
    {
      "title": "FastAPI Official Docs",
      "source": "docs/fastapi_guide.md",
      "snippet": "FastAPI has a powerful but simple Dependency Injection..."
    }
  ],
  "query_type": "how-to",
  "rewritten_query": "FastAPI dependency injection Depends() usage example",
  "retry_count": 0,
  "web_search_used": false,
  "hallucination_score": 0.95,
  "hallucination_flag": false,
  "processing_steps": [
    "query_analysis: type=how-to, rewritten='FastAPI dependency injection...'",
    "retrieval: fetched 5 chunks",
    "document_grading: 4/5 relevant → generate",
    "generation: generated from 4 docs",
    "hallucination_check: score=0.95, flagged=False"
  ],
  "error": null
}
```

---

### POST `/api/ingest`
Upload a file (PDF, MD, TXT, HTML).

```bash
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@my_docs.pdf" \
  -F "title=My Documentation"
```

### POST `/api/ingest/url`
Ingest from a URL.

```bash
curl -X POST http://localhost:8000/api/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://fastapi.tiangolo.com/tutorial/"}'
```

### GET `/api/documents`
List all indexed documents.

```bash
curl http://localhost:8000/api/documents
```

### POST `/api/feedback`
Submit thumbs up/down feedback.

```bash
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?", "answer": "FastAPI is...", "rating": "up"}'
```

### GET `/api/health`
Check system status.

```bash
curl http://localhost:8000/api/health
```

---

## 🗂️ Document Corpus

The default corpus (in `docs/`) includes:
- **FastAPI Guide** — Installation, path/query params, dependency injection, middleware, CORS
- **LangChain & LangGraph Guide** — Chains, retrievers, vector stores, StateGraph, RAG patterns
- **Pydantic v2 Guide** — Models, field validation, settings, serialization

Add your own docs via the UI "Ingest" panel or the API.

---

## 🧩 Chunking & Embedding Strategy

### Chunking
- **Splitter**: `RecursiveCharacterTextSplitter`
- **Chunk size**: 800 characters (~200 tokens)
- **Overlap**: 100 characters
- **Separators**: `["\n\n", "\n", "```", "---", ". ", " ", ""]`

**Rationale**: The recursive splitter respects document structure (paragraphs → sentences → words). Technical docs often have code blocks and structured sections — the triple-backtick separator ensures code blocks aren't split mid-block. 800 chars provides enough context per chunk without excessive noise. 100-char overlap prevents losing context at chunk boundaries.

### Embeddings
- **Model**: `all-MiniLM-L6-v2` (sentence-transformers, local, free)
- **Dimensions**: 384
- **Normalization**: L2-normalized for cosine similarity

**Rationale**: MiniLM offers a strong quality/speed tradeoff. It runs locally with no API key and handles technical text well. The sentence-transformers library is battle-tested.

---

## ⚙️ Design Decisions & Tradeoffs

| Decision | Choice | Tradeoff |
|----------|--------|----------|
| LLM grading per-chunk | Individual LLM calls | More accurate but slower; mitigated by small chunks |
| Retry on irrelevant | Query rewrite + loop | Risk of infinite loop → bounded by `MAX_RETRIES` |
| Local embeddings | sentence-transformers | No cost, but lower quality than OpenAI embeddings |
| ChromaDB default | Persistent, simple | FAISS faster at scale; swappable via config |
| Stateless API | No server-side sessions | Client sends history; easier to scale horizontally |
| TypedDict state | Full schema | Verbose but caught many bugs at development time |

### What I Would Improve With More Time

1. **Streaming responses** — Stream tokens to the frontend as they're generated
2. **Re-ranking** — Add cross-encoder re-ranking after retrieval for better precision
3. **Hybrid search** — Combine BM25 keyword search with vector search
4. **Persistent sessions** — Store chat history server-side with Redis
5. **Async nodes** — Make LangGraph nodes async for concurrent LLM calls
6. **Observability** — Add LangSmith tracing for debugging retrieval quality
7. **Multi-vector retrieval** — Store both chunk and document-level embeddings

---

## 🌲 Project Structure

```
rag-assistant/
├── main.py                        # FastAPI app entry point
├── requirements.txt
├── .env.example                   # Environment config template
├── docs/                          # Default document corpus
│   ├── fastapi_guide.md
│   ├── langchain_langgraph_guide.md
│   └── pydantic_guide.md
├── frontend/
│   └── index.html                 # Single-file SPA frontend
├── scripts/
│   └── ingest.py                  # Standalone ingestion CLI
├── data/                          # Auto-created at runtime
│   ├── chroma_db/                 # Persisted vector store
│   └── feedback.jsonl             # Feedback log
└── backend/
    ├── config.py                  # Pydantic settings
    ├── llm_factory.py             # LLM provider abstraction
    ├── vector_store.py            # ChromaDB/FAISS manager
    ├── ingestion.py               # Document loading pipeline
    ├── graph.py                   # LangGraph workflow
    ├── models.py                  # API Pydantic schemas
    ├── nodes/
    │   ├── state.py               # RAGState TypedDict
    │   ├── query_analysis.py      # Node 1
    │   ├── retrieval.py           # Node 2
    │   ├── document_grading.py    # Node 3 (self-corrective)
    │   ├── generation.py          # Node 4
    │   └── bonus_nodes.py         # Rewrite + web search + hallucination check
    ├── routers/
    │   └── api.py                 # All FastAPI endpoints
    └── utils/
        └── feedback_store.py      # Feedback persistence
```

---

## 🔑 Getting API Keys

| Provider | URL | Notes |
|----------|-----|-------|
| Groq | https://console.groq.com | Recommended — generous free tier |
| OpenAI | https://platform.openai.com | Pay-as-you-go |
| Google | https://aistudio.google.com | Free tier available |
| Anthropic | https://console.anthropic.com | Pay-as-you-go |
| Tavily (web search) | https://tavily.com | 1000 free searches/month |

---

## 📄 License

MIT License — feel free to use and modify.
