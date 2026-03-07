"""Application configuration."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # Database
    database_url: str = ""
    
    # LLM
    ollama_base_url: str = ""
    ollama_model: str = ""
    
    # Tools
    tavily_api_key: str = ""

    # VectorDB
    qdrant_url: str = ""
    
    # Observability
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()