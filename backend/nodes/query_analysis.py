"""
Node 1 — Query Analysis

Responsibilities:
  1. Classify the query type (conceptual / how-to / troubleshooting / api-reference / general)
  2. Rewrite/expand the query to improve retrieval quality
  3. Incorporate chat history for follow-up questions (conversation memory bonus)
"""
import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from backend.llm_factory import get_llm
from backend.nodes.state import RAGState

logger = logging.getLogger(__name__)

QUERY_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert query analyst for a technical documentation assistant.
Your job is to:
1. Classify the user's question into one of: conceptual, how-to, troubleshooting, api-reference, general
2. Rewrite the query to be more specific and retrieval-friendly (add synonyms, clarify ambiguity,
   expand abbreviations). Keep rewrites under 60 words.
3. If there is chat history, incorporate the context to resolve pronouns / references.

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{
  "query_type": "<type>",
  "rewritten_query": "<improved query>",
  "reasoning": "<one sentence why>"
}}
"""),
    ("human", """Chat history (if any):
{chat_history}

Current question: {question}"""),
])


def query_analysis_node(state: RAGState) -> RAGState:
    """Classify and rewrite the incoming query."""
    logger.info("=== Node: Query Analysis ===")
    llm = get_llm()
    chain = QUERY_ANALYSIS_PROMPT | llm

    # Format chat history
    history_str = ""
    if state.get("chat_history"):
        history_str = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in state["chat_history"][-6:]  # last 3 turns
        )

    try:
        response = chain.invoke({
            "question": state["original_question"],
            "chat_history": history_str or "None",
        })
        raw = response.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)

        rewritten = result.get("rewritten_query", state["original_question"])
        qtype = result.get("query_type", "general")

        logger.info(f"Query type: {qtype}")
        logger.info(f"Rewritten: {rewritten}")

        steps = list(state.get("processing_steps", []))
        steps.append(f"query_analysis: type={qtype}, rewritten='{rewritten}'")

        return {
            **state,
            "rewritten_query": rewritten,
            "query_type": qtype,
            "processing_steps": steps,
            "retry_count": state.get("retry_count", 0),
            "web_search_used": state.get("web_search_used", False),
            "hallucination_flag": state.get("hallucination_flag", False),
        }
    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        steps = list(state.get("processing_steps", []))
        steps.append(f"query_analysis: fallback (error: {e})")
        return {
            **state,
            "rewritten_query": state["original_question"],
            "query_type": "general",
            "processing_steps": steps,
            "retry_count": state.get("retry_count", 0),
            "web_search_used": state.get("web_search_used", False),
            "hallucination_flag": state.get("hallucination_flag", False),
        }
