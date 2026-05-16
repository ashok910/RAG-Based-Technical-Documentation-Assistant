"""
Node 3 — Document Grading  (self-corrective component)

For each retrieved chunk:
  - Ask the LLM: is this document relevant to the query?
  - Filter out irrelevant chunks
  - Set route_decision:
      * "generate"    → at least one relevant doc found
      * "rewrite"     → no relevant docs, retry budget remaining
      * "web_search"  → no relevant docs, web search enabled
      * "fallback"    → no relevant docs, retries exhausted
"""
import json
import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from backend.llm_factory import get_llm
from backend.nodes.state import RAGState
from backend.config import get_settings

logger = logging.getLogger(__name__)

GRADING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a relevance grader for a technical documentation retrieval system.
Given a user question and a document chunk, decide if the chunk is relevant.

A chunk is RELEVANT if it contains information that could help answer the question,
even partially. Be generous — technical synonyms and related concepts count.

Respond ONLY with valid JSON (no markdown):
{{"relevant": true/false, "reason": "<one short sentence>"}}
"""),
    ("human", """Question: {question}

Document chunk:
---
{document}
---
"""),
])


def _grade_single(llm, question: str, doc: Document) -> bool:
    """Return True if the document is relevant."""
    try:
        chain = GRADING_PROMPT | llm
        response = chain.invoke({
            "question": question,
            "document": doc.page_content[:1500],  # cap to avoid token blowout
        })
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        relevant = result.get("relevant", False)
        logger.debug(f"  chunk grade={relevant} | {result.get('reason', '')[:80]}")
        return relevant
    except Exception as e:
        logger.warning(f"Grading failed for chunk; defaulting to relevant. Error: {e}")
        return True  # be lenient on grading errors


def document_grading_node(state: RAGState) -> RAGState:
    """Grade each retrieved document and decide routing."""
    logger.info("=== Node: Document Grading ===")
    settings = get_settings()
    llm = get_llm()
    question = state.get("rewritten_query") or state["original_question"]
    retrieved: List[Document] = state.get("retrieved_documents", [])

    if not retrieved:
        logger.info("No documents retrieved — routing to fallback/rewrite")
        route = _decide_empty_route(state, settings)
        steps = list(state.get("processing_steps", []))
        steps.append(f"document_grading: 0 docs → {route}")
        return {**state, "relevant_documents": [], "route_decision": route, "processing_steps": steps}

    relevant_docs = []
    irrelevant_count = 0
    for doc in retrieved:
        if _grade_single(llm, question, doc):
            relevant_docs.append(doc)
        else:
            irrelevant_count += 1

    logger.info(f"Grading: {len(relevant_docs)} relevant, {irrelevant_count} irrelevant")

    if relevant_docs:
        route = "generate"
    else:
        route = _decide_empty_route(state, settings)

    steps = list(state.get("processing_steps", []))
    steps.append(
        f"document_grading: {len(relevant_docs)}/{len(retrieved)} relevant → {route}"
    )

    return {
        **state,
        "relevant_documents": relevant_docs,
        "irrelevant_count": irrelevant_count,
        "route_decision": route,
        "processing_steps": steps,
    }


def _decide_empty_route(state: RAGState, settings) -> str:
    retry_count = state.get("retry_count", 0)
    if retry_count < settings.max_retries:
        return "rewrite"
    if settings.enable_web_search and state.get("tavily_api_key", ""):
        return "web_search"
    return "fallback"
