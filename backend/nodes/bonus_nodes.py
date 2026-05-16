"""
Bonus Nodes
  - query_rewrite_node: rewrites the query after failed grading (self-correction)
  - web_search_node: Tavily web search fallback
  - hallucination_check_node: verifies answer is grounded in context
"""
import json
import logging
from langchain_core.prompts import ChatPromptTemplate

from backend.llm_factory import get_llm
from backend.nodes.state import RAGState
from backend.config import get_settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Query Rewrite Node
# ──────────────────────────────────────────────
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriter for a technical documentation search system.
The initial retrieval returned no relevant results. Your job is to rephrase the query
to try a different angle — use synonyms, split compound questions, or simplify.

Respond ONLY with the new query string, nothing else.
"""),
    ("human", """Original question: {original}
Previous query used: {previous}
Attempt number: {attempt}

New query:"""),
])


def query_rewrite_node(state: RAGState) -> RAGState:
    logger.info("=== Node: Query Rewrite ===")
    llm = get_llm()
    chain = REWRITE_PROMPT | llm

    try:
        response = chain.invoke({
            "original": state["original_question"],
            "previous": state.get("rewritten_query", state["original_question"]),
            "attempt": state.get("retry_count", 0) + 1,
        })
        new_query = response.content.strip().strip('"')
    except Exception as e:
        logger.error(f"Rewrite failed: {e}")
        new_query = state["original_question"]

    retry_count = state.get("retry_count", 0) + 1
    steps = list(state.get("processing_steps", []))
    steps.append(f"query_rewrite (attempt {retry_count}): '{new_query}'")

    return {
        **state,
        "rewritten_query": new_query,
        "retry_count": retry_count,
        "retrieved_documents": [],
        "relevant_documents": [],
        "processing_steps": steps,
    }


# ──────────────────────────────────────────────
# Web Search Fallback Node (Bonus)
# ──────────────────────────────────────────────
def web_search_node(state: RAGState) -> RAGState:
    logger.info("=== Node: Web Search Fallback ===")
    settings = get_settings()
    query = state.get("rewritten_query") or state["original_question"]

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        results = client.search(query=query, max_results=3)
        snippets = [
            f"[{r.get('title', 'Web Result')}]({r.get('url', '')})\n{r.get('content', '')}"
            for r in results.get("results", [])
        ]
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        snippets = []

    steps = list(state.get("processing_steps", []))
    steps.append(f"web_search: got {len(snippets)} results")

    return {
        **state,
        "web_search_used": True,
        "web_search_results": snippets,
        "processing_steps": steps,
    }


# ──────────────────────────────────────────────
# Hallucination Check Node (Bonus)
# ──────────────────────────────────────────────
HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a hallucination detector for a RAG system.
Check if the generated answer is fully supported by the provided context documents.

Score from 0.0 to 1.0:
  1.0 = fully grounded (every claim is in the context)
  0.5 = partially grounded (some claims may be inferred)
  0.0 = hallucinated (answer contains claims not in context)

Respond ONLY with valid JSON:
{{"score": 0.0-1.0, "hallucinated": true/false, "issues": "<brief description or none>"}}
"""),
    ("human", """Context:
{context}

Generated answer:
{answer}
"""),
])


def hallucination_check_node(state: RAGState) -> RAGState:
    logger.info("=== Node: Hallucination Check ===")
    llm = get_llm()
    docs = state.get("relevant_documents", [])
    answer = state.get("answer", "")

    if not docs or not answer:
        steps = list(state.get("processing_steps", []))
        steps.append("hallucination_check: skipped (no docs or answer)")
        return {**state, "hallucination_score": 1.0, "hallucination_flag": False, "processing_steps": steps}

    context = "\n\n".join(d.page_content[:600] for d in docs[:3])
    try:
        chain = HALLUCINATION_PROMPT | llm
        response = chain.invoke({"context": context, "answer": answer})
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        score = float(result.get("score", 1.0))
        flagged = result.get("hallucinated", False)
    except Exception as e:
        logger.warning(f"Hallucination check failed: {e}")
        score = 1.0
        flagged = False

    steps = list(state.get("processing_steps", []))
    steps.append(f"hallucination_check: score={score:.2f}, flagged={flagged}")

    # If flagged, append a warning to the answer
    answer_out = state["answer"]
    if flagged:
        answer_out += (
            "\n\n> ⚠️ **Confidence Note:** Parts of this answer may not be fully "
            "supported by the retrieved documents. Please verify with the source material."
        )

    return {
        **state,
        "hallucination_score": score,
        "hallucination_flag": flagged,
        "answer": answer_out,
        "processing_steps": steps,
    }
