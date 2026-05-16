"""
Node 4 — Generation

Generates a grounded, cited answer using the relevant document chunks.
"""
import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from backend.llm_factory import get_llm
from backend.nodes.state import RAGState

logger = logging.getLogger(__name__)

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert technical documentation assistant.
Your task is to answer the user's question using ONLY the provided context documents.

Rules:
1. Answer clearly and accurately, grounded strictly in the context.
2. After your answer, include a "Sources" section listing which documents you used.
3. If the context only partially answers the question, say so honestly.
4. Do NOT make up information not present in the context.
5. Use markdown formatting for clarity (code blocks, bullet points where appropriate).
6. Be concise but complete.

Format:
<your detailed answer here>

**Sources:**
- [Document Title](source_url_or_path) — brief note on what it contributed
"""),
    ("human", """Context documents:
{context}

---
Question: {question}

Query type: {query_type}
"""),
])

WEB_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert technical assistant.
Answer the question using the web search results provided.
Clearly indicate that the information comes from web search.
Include source URLs in your answer.
"""),
    ("human", """Web search results:
{context}

Question: {question}
"""),
])


def _format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "Unknown")
        source = doc.metadata.get("source", "")
        parts.append(
            f"[Doc {i}] {title}\nSource: {source}\n\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_citations(docs: List[Document]) -> List[dict]:
    seen = set()
    citations = []
    for doc in docs:
        src = doc.metadata.get("source", "")
        if src not in seen:
            seen.add(src)
            citations.append({
                "title": doc.metadata.get("title", src),
                "source": src,
                "snippet": doc.page_content[:200].strip() + "...",
            })
    return citations


def generation_node(state: RAGState) -> RAGState:
    """Generate final answer from relevant documents."""
    logger.info("=== Node: Generation ===")
    llm = get_llm()
    question = state.get("rewritten_query") or state["original_question"]
    relevant_docs: List[Document] = state.get("relevant_documents", [])
    web_results: List[str] = state.get("web_search_results", [])

    steps = list(state.get("processing_steps", []))

    if web_results:
        context = "\n\n".join(web_results)
        chain = WEB_GENERATION_PROMPT | llm
        response = chain.invoke({"context": context, "question": question})
        answer = response.content
        citations = [{"title": "Web Search", "source": "web", "snippet": ""}]
        steps.append("generation: used web search results")
    elif relevant_docs:
        context = _format_context(relevant_docs)
        chain = GENERATION_PROMPT | llm
        response = chain.invoke({
            "context": context,
            "question": question,
            "query_type": state.get("query_type", "general"),
        })
        answer = response.content
        citations = _extract_citations(relevant_docs)
        steps.append(f"generation: generated from {len(relevant_docs)} docs")
    else:
        answer = (
            "I'm sorry, I couldn't find relevant information in the documentation "
            "to answer your question. Please try rephrasing, or check if the relevant "
            "documents have been ingested."
        )
        citations = []
        steps.append("generation: fallback no-answer")

    logger.info(f"Generated answer ({len(answer)} chars)")

    return {
        **state,
        "answer": answer,
        "citations": citations,
        "processing_steps": steps,
    }
