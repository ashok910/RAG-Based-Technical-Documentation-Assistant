"""
LLM factory — returns the correct LangChain chat model
based on the configured provider.
"""
from functools import lru_cache
from langchain_core.language_models.chat_models import BaseChatModel
from backend.config import get_settings


@lru_cache()
def get_llm() -> BaseChatModel:
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=0,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=0,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key,
            model=settings.llm_model,
            temperature=0,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
            temperature=0,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


@lru_cache()
def get_embeddings():
    """Returns sentence-transformer embeddings (free, local)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
