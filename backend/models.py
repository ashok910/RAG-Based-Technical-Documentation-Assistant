"""
API Pydantic models — request and response schemas.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ── Query ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The user's question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    chat_history: Optional[List[dict]] = Field(
        default=[], description="Previous messages [{'role': 'user'|'assistant', 'content': str}]"
    )

class Citation(BaseModel):
    title: str
    source: str
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    query_type: str
    rewritten_query: str
    retry_count: int
    web_search_used: bool
    hallucination_score: Optional[float]
    hallucination_flag: bool
    processing_steps: List[str]
    error: Optional[str]


# ── Ingest ─────────────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    url: str = Field(..., description="URL to fetch and ingest")
    title: Optional[str] = Field(None, description="Optional display title")

class IngestResponse(BaseModel):
    success: bool
    message: str
    chunks_added: int


# ── Documents ──────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    title: str
    source: str
    chunks: int

class DocumentsListResponse(BaseModel):
    total_chunks: int
    sources: List[DocumentInfo]


# ── Feedback ──────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    answer: str
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(None, max_length=1000)

class FeedbackResponse(BaseModel):
    success: bool
    message: str


# ── Health ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    vector_store: str
    llm_provider: str
    document_count: int
    version: str = "1.0.0"
