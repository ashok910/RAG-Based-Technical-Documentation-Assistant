"""
FastAPI routers — all API endpoints.
"""
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from typing import Optional

from backend.models import (
    QueryRequest, QueryResponse, Citation,
    IngestURLRequest, IngestResponse,
    DocumentsListResponse, DocumentInfo,
    FeedbackRequest, FeedbackResponse,
    HealthResponse,
)
from backend.graph import run_rag_pipeline
from backend.ingestion import ingest_file, ingest_url
from backend.vector_store import list_document_sources, get_document_count
from backend.utils.feedback_store import save_feedback, get_feedback_stats
from backend.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        vector_store=settings.vector_store_type,
        llm_provider=settings.llm_provider,
        document_count=get_document_count(),
    )


# ── Query ─────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_endpoint(request: QueryRequest):
    """
    Submit a natural language question to the RAG pipeline.
    Returns an answer with citations and metadata.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = run_rag_pipeline(
        question=request.question,
        session_id=request.session_id,
        chat_history=request.chat_history or [],
    )

    if result.get("error") and not result.get("answer"):
        raise HTTPException(status_code=500, detail=result["error"])

    citations = [Citation(**c) for c in result.get("citations", [])]

    return QueryResponse(
        answer=result["answer"],
        citations=citations,
        query_type=result["query_type"],
        rewritten_query=result["rewritten_query"],
        retry_count=result["retry_count"],
        web_search_used=result["web_search_used"],
        hallucination_score=result["hallucination_score"],
        hallucination_flag=result["hallucination_flag"],
        processing_steps=result["processing_steps"],
        error=result.get("error"),
    )


# ── Ingest ────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
):
    """
    Upload a document (PDF, TXT, MD, HTML) to be ingested into the vector store.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    chunks_added, message = ingest_file(contents, file.filename, title=title)
    return IngestResponse(
        success=chunks_added > 0,
        message=message,
        chunks_added=chunks_added,
    )


@router.post("/ingest/url", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_url_endpoint(request: IngestURLRequest):
    """
    Fetch and ingest a document from a URL.
    """
    chunks_added, message = ingest_url(request.url)
    return IngestResponse(
        success=chunks_added > 0,
        message=message,
        chunks_added=chunks_added,
    )


# ── Documents ─────────────────────────────────────────────────────

@router.get("/documents", response_model=DocumentsListResponse, tags=["Documents"])
async def list_documents():
    """List all indexed documents and their chunk counts."""
    sources = list_document_sources()
    return DocumentsListResponse(
        total_chunks=get_document_count(),
        sources=[DocumentInfo(**s) for s in sources],
    )


# ── Feedback ──────────────────────────────────────────────────────

@router.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """Submit thumbs-up/down feedback for an answer."""
    ok = save_feedback(request.model_dump())
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save feedback.")
    stats = get_feedback_stats()
    return FeedbackResponse(
        success=True,
        message=f"Feedback saved. Total: {stats['total']} (👍 {stats['up']} / 👎 {stats['down']})",
    )


@router.get("/feedback/stats", tags=["Feedback"])
async def feedback_stats():
    """Get aggregated feedback statistics."""
    return get_feedback_stats()
