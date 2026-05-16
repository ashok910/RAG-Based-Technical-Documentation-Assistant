"""
Node 2 — Retrieval

Uses the vector store to fetch the top-k most relevant document chunks
for the (rewritten) query.
"""
import logging
from backend.nodes.state import RAGState
from backend.vector_store import similarity_search
from backend.config import get_settings

logger = logging.getLogger(__name__)


def retrieval_node(state: RAGState) -> RAGState:
    """Retrieve top-k relevant chunks from the vector store."""
    logger.info("=== Node: Retrieval ===")
    settings = get_settings()
    query = state.get("rewritten_query") or state["original_question"]

    docs = similarity_search(query, k=settings.top_k_retrieval)
    logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")

    steps = list(state.get("processing_steps", []))
    steps.append(f"retrieval: fetched {len(docs)} chunks")

    return {
        **state,
        "retrieved_documents": docs,
        "processing_steps": steps,
    }
