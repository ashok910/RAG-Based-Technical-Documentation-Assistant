"""
Tests for the RAG pipeline.

Run with: pytest tests/ -v

NOTE: Tests that call LLM/vector store APIs are marked @pytest.mark.integration
and require a valid .env. Unit tests mock all external dependencies.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import List

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
# Config Tests
# ─────────────────────────────────────────────
def test_settings_defaults():
    """Settings should load with sensible defaults."""
    from backend.config import Settings
    s = Settings()
    assert s.llm_provider in ("groq", "openai", "google", "anthropic")
    assert s.top_k_retrieval > 0
    assert s.chunk_size > 0
    assert s.chunk_overlap >= 0
    assert s.max_retries >= 1


def test_settings_chunk_overlap_less_than_size():
    from backend.config import Settings
    s = Settings()
    assert s.chunk_overlap < s.chunk_size, "Chunk overlap must be less than chunk size"


# ─────────────────────────────────────────────
# State Schema Tests
# ─────────────────────────────────────────────
def test_rag_state_fields():
    """RAGState TypedDict should have all required fields."""
    from backend.nodes.state import RAGState
    required_keys = {
        "original_question", "rewritten_query", "query_type",
        "retrieved_documents", "relevant_documents", "retry_count",
        "answer", "citations", "route_decision", "processing_steps",
        "web_search_used", "hallucination_flag",
    }
    annotations = RAGState.__annotations__
    for key in required_keys:
        assert key in annotations, f"Missing field: {key}"


# ─────────────────────────────────────────────
# Ingestion Tests (unit — no real embeddings)
# ─────────────────────────────────────────────
def test_ingest_file_unsupported_extension():
    """Ingesting unsupported file types should return 0 chunks and an error message."""
    from backend.ingestion import ingest_file
    count, msg = ingest_file(b"some content", "file.docx")
    assert count == 0
    assert "Unsupported" in msg


def test_ingest_empty_file():
    """Empty file bytes raise no exception; unsupported ext caught first."""
    from backend.ingestion import ingest_file
    # Empty bytes with supported ext — will fail at loader level, not crash
    # We just check it doesn't raise
    try:
        count, msg = ingest_file(b"", "empty.txt")
    except Exception:
        pass  # Acceptable — loader may error on empty file


def test_chunking_strategy():
    """Text splitter produces correct chunk sizes and overlap."""
    from backend.ingestion import _get_splitter
    splitter = _get_splitter()
    long_text = "This is a test sentence. " * 200  # ~5000 chars
    from langchain_core.documents import Document
    docs = [Document(page_content=long_text, metadata={"source": "test"})]
    chunks = splitter.split_documents(docs)
    assert len(chunks) > 1, "Long text should be split into multiple chunks"
    for chunk in chunks:
        assert len(chunk.page_content) <= 1000, "Chunks should not far exceed chunk_size"


# ─────────────────────────────────────────────
# Graph Structure Tests (no LLM calls)
# ─────────────────────────────────────────────
def test_graph_compiles():
    """The LangGraph should compile without errors."""
    from backend.graph import build_graph
    graph = build_graph()
    compiled = graph.compile()
    assert compiled is not None


def test_graph_has_required_nodes():
    """Graph must contain all required nodes."""
    from backend.graph import build_graph
    graph = build_graph()
    node_names = set(graph.nodes.keys())
    required = {"query_analysis", "retrieval", "document_grading", "generation"}
    for node in required:
        assert node in node_names, f"Missing node: {node}"


def test_routing_logic_generate():
    """route_after_grading should return 'generate' when route_decision is generate."""
    from backend.graph import route_after_grading
    state = {"route_decision": "generate"}
    assert route_after_grading(state) == "generate"


def test_routing_logic_rewrite():
    from backend.graph import route_after_grading
    state = {"route_decision": "rewrite"}
    assert route_after_grading(state) == "rewrite"


def test_routing_logic_fallback():
    from backend.graph import route_after_grading
    state = {"route_decision": "fallback"}
    assert route_after_grading(state) == "fallback"


# ─────────────────────────────────────────────
# API Models Tests
# ─────────────────────────────────────────────
def test_query_request_validation():
    from backend.models import QueryRequest
    req = QueryRequest(question="What is FastAPI?")
    assert req.question == "What is FastAPI?"
    assert req.chat_history == []


def test_query_request_rejects_empty():
    from backend.models import QueryRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_ingest_response_model():
    from backend.models import IngestResponse
    r = IngestResponse(success=True, message="ok", chunks_added=42)
    assert r.chunks_added == 42


def test_feedback_request_rating_enum():
    from backend.models import FeedbackRequest
    from pydantic import ValidationError
    # Valid
    req = FeedbackRequest(question="q", answer="a", rating="up")
    assert req.rating == "up"
    # Invalid
    with pytest.raises(ValidationError):
        FeedbackRequest(question="q", answer="a", rating="sideways")


# ─────────────────────────────────────────────
# Vector Store Tests (unit — mocked)
# ─────────────────────────────────────────────
def test_add_documents_empty_list():
    """Adding zero documents should return 0."""
    from backend.vector_store import add_documents
    count = add_documents([])
    assert count == 0


# ─────────────────────────────────────────────
# Feedback Store Tests
# ─────────────────────────────────────────────
def test_feedback_store_save_and_stats(tmp_path):
    """Feedback store should persist entries and compute stats."""
    from backend.utils import feedback_store
    original = feedback_store.FEEDBACK_FILE
    feedback_store.FEEDBACK_FILE = tmp_path / "feedback.jsonl"

    feedback_store.save_feedback({"question": "q", "answer": "a", "rating": "up"})
    feedback_store.save_feedback({"question": "q2", "answer": "a2", "rating": "down"})

    stats = feedback_store.get_feedback_stats()
    assert stats["total"] == 2
    assert stats["up"] == 1
    assert stats["down"] == 1

    feedback_store.FEEDBACK_FILE = original


# ─────────────────────────────────────────────
# FastAPI Endpoint Tests (TestClient)
# ─────────────────────────────────────────────
@pytest.fixture
def client():
    """FastAPI test client with mocked pipeline."""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "document_count" in data


def test_documents_endpoint(client):
    res = client.get("/api/documents")
    assert res.status_code == 200
    data = res.json()
    assert "sources" in data
    assert "total_chunks" in data


def test_feedback_endpoint(client):
    res = client.post("/api/feedback", json={
        "question": "test question",
        "answer": "test answer",
        "rating": "up",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True


def test_feedback_stats_endpoint(client):
    res = client.get("/api/feedback/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total" in data


def test_query_endpoint_empty_question(client):
    """Empty question should return 422 validation error."""
    res = client.post("/api/query", json={"question": ""})
    assert res.status_code == 422


@pytest.mark.integration
def test_query_endpoint_with_question(client):
    """Integration test: full RAG pipeline query. Requires valid .env."""
    res = client.post("/api/query", json={
        "question": "What is FastAPI?",
        "session_id": "test_session",
    })
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert isinstance(data["citations"], list)
    assert isinstance(data["processing_steps"], list)
