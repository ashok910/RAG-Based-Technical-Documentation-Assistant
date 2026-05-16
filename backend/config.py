"""
Application configuration using pydantic-settings.
Reads from environment variables / .env file.
"""
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM
    llm_provider: str = Field(default="groq", env="LLM_PROVIDER")
    llm_model: str = Field(default="llama3-8b-8192", env="LLM_MODEL")

    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")

    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")

    # Vector store
    vector_store_type: str = Field(default="chroma", env="VECTOR_STORE_TYPE")
    chroma_persist_dir: str = Field(default="./data/chroma_db", env="CHROMA_PERSIST_DIR")
    faiss_index_path: str = Field(default="./data/faiss_index", env="FAISS_INDEX_PATH")

    # Retrieval
    top_k_retrieval: int = Field(default=5, env="TOP_K_RETRIEVAL")
    chunk_size: int = Field(default=800, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    max_retries: int = Field(default=2, env="MAX_RETRIES")

    # Web search
    enable_web_search: bool = Field(default=False, env="ENABLE_WEB_SEARCH")
    tavily_api_key: str = Field(default="", env="TAVILY_API_KEY")

    # App
    app_host: str = Field(default="0.0.0.0", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:8000"],
        env="CORS_ORIGINS"
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
