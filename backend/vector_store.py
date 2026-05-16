"""
Vector store manager — handles persistence and retrieval
for both ChromaDB and FAISS backends.
"""
import os
import pickle
import logging
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS

from backend.config import get_settings
from backend.llm_factory import get_embeddings

logger = logging.getLogger(__name__)

_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = _load_or_create_store()
    return _vector_store


def reset_vector_store():
    global _vector_store
    _vector_store = None


def _load_or_create_store():
    settings = get_settings()
    embeddings = get_embeddings()

    if settings.vector_store_type == "chroma":
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        store = Chroma(
            collection_name="rag_docs",
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        logger.info(f"ChromaDB loaded from {settings.chroma_persist_dir}")
        return store

    elif settings.vector_store_type == "faiss":
        index_path = Path(settings.faiss_index_path)
        if index_path.exists():
            store = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"FAISS index loaded from {index_path}")
            return store
        else:
            # Create empty FAISS — needs at least one doc to initialise
            logger.info("No FAISS index found; will create on first ingest.")
            return None

    raise ValueError(f"Unknown vector store type: {settings.vector_store_type}")


def add_documents(documents: List[Document]) -> int:
    """Add documents to the vector store. Returns count added."""
    global _vector_store
    settings = get_settings()
    embeddings = get_embeddings()

    if not documents:
        return 0

    if settings.vector_store_type == "chroma":
        if _vector_store is None:
            _vector_store = _load_or_create_store()
        _vector_store.add_documents(documents)
        _vector_store.persist()

    elif settings.vector_store_type == "faiss":
        if _vector_store is None:
            _vector_store = FAISS.from_documents(documents, embeddings)
        else:
            _vector_store.add_documents(documents)
        Path(settings.faiss_index_path).mkdir(parents=True, exist_ok=True)
        _vector_store.save_local(settings.faiss_index_path)

    logger.info(f"Added {len(documents)} document chunks to vector store.")
    return len(documents)


def similarity_search(query: str, k: int = 5) -> List[Document]:
    """Retrieve top-k documents for a query."""
    store = get_vector_store()
    if store is None:
        return []
    try:
        return store.similarity_search(query, k=k)
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return []


def list_document_sources() -> List[dict]:
    """Return a summary of all indexed sources."""
    store = get_vector_store()
    if store is None:
        return []

    settings = get_settings()
    try:
        if settings.vector_store_type == "chroma":
            collection = store._collection
            results = collection.get(include=["metadatas"])
            sources = {}
            for meta in results["metadatas"]:
                src = meta.get("source", "unknown")
                title = meta.get("title", src)
                if src not in sources:
                    sources[src] = {"source": src, "title": title, "chunks": 0}
                sources[src]["chunks"] += 1
            return list(sources.values())
        else:
            # FAISS — iterate docstore
            docs = list(store.docstore._dict.values())
            sources = {}
            for doc in docs:
                src = doc.metadata.get("source", "unknown")
                title = doc.metadata.get("title", src)
                if src not in sources:
                    sources[src] = {"source": src, "title": title, "chunks": 0}
                sources[src]["chunks"] += 1
            return list(sources.values())
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return []


def get_document_count() -> int:
    store = get_vector_store()
    if store is None:
        return 0
    try:
        settings = get_settings()
        if settings.vector_store_type == "chroma":
            return store._collection.count()
        else:
            return len(store.docstore._dict)
    except Exception:
        return 0
