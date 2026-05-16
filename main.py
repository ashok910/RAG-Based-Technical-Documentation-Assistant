"""
FastAPI application entry point.
"""
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.config import get_settings
from backend.routers.api import router
from backend.vector_store import get_document_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("  RAG Technical Documentation Assistant")
    logger.info("=" * 60)
    logger.info(f"  LLM Provider  : {settings.llm_provider} / {settings.llm_model}")
    logger.info(f"  Vector Store  : {settings.vector_store_type}")
    logger.info(f"  Embeddings    : {settings.embedding_model}")
    logger.info(f"  Web Search    : {'enabled' if settings.enable_web_search else 'disabled'}")

    # Auto-ingest default corpus if vector store is empty
    count = get_document_count()
    if count == 0:
        logger.info("Vector store is empty — ingesting default corpus...")
        try:
            from backend.ingestion import ingest_default_corpus
            ingest_default_corpus()
        except Exception as e:
            logger.warning(f"Default corpus ingest failed: {e}")
    else:
        logger.info(f"Vector store: {count} chunks already indexed")

    logger.info("Application ready.")
    yield
    logger.info("Application shutdown.")


app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description=(
        "A self-corrective RAG system built with LangGraph. "
        "Answers questions about technical documentation with citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api")

# Serve frontend static files
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))

    @app.get("/{path:path}", include_in_schema=False)
    async def serve_spa(path: str):
        file = frontend_dir / path
        if file.exists() and file.is_file():
            return FileResponse(str(file))
        return FileResponse(str(frontend_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
