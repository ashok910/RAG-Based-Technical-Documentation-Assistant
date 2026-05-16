"""
Document ingestion pipeline.

Strategy:
  - Chunk size 800 tokens with 100-token overlap.
  - RecursiveCharacterTextSplitter respects paragraph/sentence boundaries,
    crucial for technical docs where a mid-sentence split loses context.
  - Each chunk carries source metadata for citation.
"""
import os
import logging
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
from urllib.parse import urlparse

import requests
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    WebBaseLoader,
)

from backend.config import get_settings
from backend.vector_store import add_documents

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}


def _get_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "```", "---", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )


def _load_file(file_path: str, title: Optional[str] = None) -> List[Document]:
    """Load a single file and return raw Documents."""
    ext = Path(file_path).suffix.lower()
    fname = title or Path(file_path).name

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in {".md"}:
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext in {".html", ".htm"}:
        loader = UnstructuredHTMLLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", file_path)
        doc.metadata.setdefault("title", fname)
    return docs


def _load_url(url: str) -> List[Document]:
    """Fetch a URL and return raw Documents."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        parsed = urlparse(url)
        title = parsed.path.split("/")[-1] or parsed.netloc
        for doc in docs:
            doc.metadata["source"] = url
            doc.metadata.setdefault("title", title)
        return docs
    except Exception as e:
        logger.error(f"Failed to load URL {url}: {e}")
        return []


def ingest_file(
    file_bytes: bytes, filename: str, title: Optional[str] = None
) -> Tuple[int, str]:
    """
    Ingest an uploaded file (bytes).
    Returns (chunks_added, message).
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return 0, f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = _load_file(tmp_path, title=title or filename)
        splitter = _get_splitter()
        chunks = splitter.split_documents(docs)
        # Enrich metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["title"] = title or filename
        count = add_documents(chunks)
        return count, f"Successfully ingested '{filename}' → {count} chunks"
    finally:
        os.unlink(tmp_path)


def ingest_url(url: str) -> Tuple[int, str]:
    """Ingest content from a URL. Returns (chunks_added, message)."""
    docs = _load_url(url)
    if not docs:
        return 0, f"Could not load content from URL: {url}"

    splitter = _get_splitter()
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    count = add_documents(chunks)
    return count, f"Successfully ingested URL '{url}' → {count} chunks"


def ingest_default_corpus():
    """
    Ingest the default technical documentation corpus shipped with the repo.
    Called on startup if the vector store is empty.
    """
    docs_dir = Path("docs")
    if not docs_dir.exists():
        logger.warning("No 'docs/' directory found. Skipping default corpus ingest.")
        return

    total = 0
    for f in docs_dir.glob("**/*"):
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                raw_docs = _load_file(str(f), title=f.stem.replace("_", " ").title())
                splitter = _get_splitter()
                chunks = splitter.split_documents(raw_docs)
                for i, chunk in enumerate(chunks):
                    chunk.metadata["chunk_index"] = i
                count = add_documents(chunks)
                total += count
                logger.info(f"Ingested {f.name} → {count} chunks")
            except Exception as e:
                logger.error(f"Failed to ingest {f}: {e}")

    logger.info(f"Default corpus ingest complete. Total chunks: {total}")
    return total
