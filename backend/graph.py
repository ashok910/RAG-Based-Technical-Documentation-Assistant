"""
LangGraph StateGraph — the core RAG workflow.

Graph topology:
  query_analysis
       │
  retrieval
       │
  document_grading ──→ [relevant?] ──→ generation ──→ hallucination_check
       │
       ├─ [rewrite] ──→ query_rewrite ──→ retrieval (loop, max_retries times)
       │
       ├─ [web_search] ──→ web_search ──→ generation ──→ hallucination_check
       │
       └─ [fallback] ──→ generation (returns graceful no-answer)
"""
import logging
from langgraph.graph import StateGraph, END

from backend.nodes.state import RAGState
from backend.nodes.query_analysis import query_analysis_node
from backend.nodes.retrieval import retrieval_node
from backend.nodes.document_grading import document_grading_node
from backend.nodes.generation import generation_node
from backend.nodes.bonus_nodes import (
    query_rewrite_node,
    web_search_node,
    hallucination_check_node,
)

logger = logging.getLogger(__name__)


def route_after_grading(state: RAGState) -> str:
    """Conditional edge: decides next node based on document grading outcome."""
    decision = state.get("route_decision", "fallback")
    logger.info(f"Routing decision: {decision}")
    return decision


def build_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    # ── Add nodes ────────────────────────────────────────────
    graph.add_node("query_analysis", query_analysis_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("document_grading", document_grading_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generation", generation_node)
    graph.add_node("hallucination_check", hallucination_check_node)

    # ── Entry point ───────────────────────────────────────────
    graph.set_entry_point("query_analysis")

    # ── Linear edges ─────────────────────────────────────────
    graph.add_edge("query_analysis", "retrieval")
    graph.add_edge("retrieval", "document_grading")

    # ── Conditional edge from document_grading ───────────────
    graph.add_conditional_edges(
        "document_grading",
        route_after_grading,
        {
            "generate": "generation",
            "rewrite": "query_rewrite",
            "web_search": "web_search",
            "fallback": "generation",
        },
    )

    # ── Rewrite loop ─────────────────────────────────────────
    graph.add_edge("query_rewrite", "retrieval")

    # ── Web search → generation ───────────────────────────────
    graph.add_edge("web_search", "generation")

    # ── Generation → hallucination check → END ────────────────
    graph.add_edge("generation", "hallucination_check")
    graph.add_edge("hallucination_check", END)

    return graph


# Compile once and reuse
_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
        logger.info("LangGraph compiled successfully.")
    return _compiled_graph


def run_rag_pipeline(
    question: str,
    session_id: str = None,
    chat_history: list = None,
) -> dict:
    """
    Entry point: run the full RAG pipeline for a question.
    Returns a dict with answer, citations, and metadata.
    """
    graph = get_compiled_graph()
    initial_state: RAGState = {
        "original_question": question,
        "session_id": session_id,
        "chat_history": chat_history or [],
        "rewritten_query": question,
        "query_type": "general",
        "retrieved_documents": [],
        "relevant_documents": [],
        "irrelevant_count": 0,
        "retry_count": 0,
        "answer": "",
        "citations": [],
        "hallucination_score": None,
        "hallucination_flag": False,
        "web_search_used": False,
        "web_search_results": [],
        "route_decision": "generate",
        "error": None,
        "processing_steps": [],
    }

    try:
        final_state = graph.invoke(initial_state)
        return {
            "answer": final_state.get("answer", ""),
            "citations": final_state.get("citations", []),
            "query_type": final_state.get("query_type", "general"),
            "rewritten_query": final_state.get("rewritten_query", question),
            "retry_count": final_state.get("retry_count", 0),
            "web_search_used": final_state.get("web_search_used", False),
            "hallucination_score": final_state.get("hallucination_score"),
            "hallucination_flag": final_state.get("hallucination_flag", False),
            "processing_steps": final_state.get("processing_steps", []),
            "error": None,
        }
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}", exc_info=True)
        return {
            "answer": "An error occurred while processing your question. Please try again.",
            "citations": [],
            "query_type": "general",
            "rewritten_query": question,
            "retry_count": 0,
            "web_search_used": False,
            "hallucination_score": None,
            "hallucination_flag": False,
            "processing_steps": [],
            "error": str(e),
        }
