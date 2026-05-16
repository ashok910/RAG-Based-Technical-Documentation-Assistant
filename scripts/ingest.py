#!/usr/bin/env python3
"""
Standalone ingestion script.
Run this to pre-populate the vector store before starting the API server.

Usage:
    python scripts/ingest.py                    # Ingest all docs/ files
    python scripts/ingest.py --url <url>        # Ingest from URL
    python scripts/ingest.py --file <path>      # Ingest single file
    python scripts/ingest.py --reset            # Clear and re-ingest
"""
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store")
    parser.add_argument("--url", help="URL to fetch and ingest")
    parser.add_argument("--file", help="File path to ingest")
    parser.add_argument("--title", help="Optional title for the document")
    parser.add_argument("--reset", action="store_true", help="Clear vector store before ingesting")
    args = parser.parse_args()

    from backend.config import get_settings
    settings = get_settings()
    logger.info(f"Vector store: {settings.vector_store_type}")
    logger.info(f"Embedding model: {settings.embedding_model}")

    if args.reset:
        logger.info("Resetting vector store…")
        import shutil
        if Path(settings.chroma_persist_dir).exists():
            shutil.rmtree(settings.chroma_persist_dir)
        if Path(settings.faiss_index_path).exists():
            shutil.rmtree(settings.faiss_index_path)
        logger.info("Vector store cleared.")

    from backend.ingestion import ingest_file, ingest_url, ingest_default_corpus

    if args.url:
        logger.info(f"Ingesting URL: {args.url}")
        count, msg = ingest_url(args.url)
        logger.info(msg)

    elif args.file:
        fp = Path(args.file)
        if not fp.exists():
            logger.error(f"File not found: {fp}")
            sys.exit(1)
        with open(fp, "rb") as f:
            data = f.read()
        count, msg = ingest_file(data, fp.name, title=args.title or fp.stem)
        logger.info(msg)

    else:
        logger.info("Ingesting default corpus from docs/")
        total = ingest_default_corpus()
        logger.info(f"Done. Total chunks ingested: {total}")


if __name__ == "__main__":
    main()
