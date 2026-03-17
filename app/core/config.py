"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # Database
    database_url: str = ""

    # LLM
    gemini_model: str = ""
    gemini_api_key: str = ""

    # Embedding Model
    embedding_model: str = ""
    embedding_model_url: str = ""

    # Tools
    tavily_api_key: str = ""

    # VectorDB
    qdrant_url: str = ""

    # Observability
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
