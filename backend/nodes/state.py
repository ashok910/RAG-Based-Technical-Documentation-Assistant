"""
LangGraph state schema for the RAG workflow.

All nodes read from and write to this typed dict.
Using TypedDict ensures LangGraph can validate state transitions.
"""
from typing import List, Optional, Literal
from typing_extensions import TypedDict
from langchain_core.documents import Document


class RAGState(TypedDict):
    # Input
    original_question: str
    session_id: Optional[str]
    chat_history: List[dict]          # [{"role": "user"|"assistant", "content": str}]

    # Query analysis
    rewritten_query: str
    query_type: Literal["conceptual", "how-to", "troubleshooting", "api-reference", "general"]

    # Retrieval
    retrieved_documents: List[Document]

    # Grading
    relevant_documents: List[Document]
    irrelevant_count: int
    retry_count: int

    # Generation
    answer: str
    citations: List[dict]             # [{"title": str, "source": str, "snippet": str}]

    # Hallucination check (bonus)
    hallucination_score: Optional[float]
    hallucination_flag: bool

    # Web search fallback (bonus)
    web_search_used: bool
    web_search_results: List[str]

    # Routing
    route_decision: Literal["generate", "rewrite", "fallback", "web_search"]

    # Meta
    error: Optional[str]
    processing_steps: List[str]       # audit trail of steps taken
