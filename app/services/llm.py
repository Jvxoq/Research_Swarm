"""
Create a singleton class for the LLM client
"""
from typing import Optional, cast
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from app.core.config import settings
from app.core.logging import logger


class LLMService:
    _instance: Optional["LLMService"] = None
    _client: Optional[BaseChatModel] = None

    def __new__(cls) -> "LLMService":
        # Create an instance of this class if not already created
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def client(self) -> BaseChatModel:
        # Define the Client as class attribute
        if self._client is None:
            logger.info(
                "Initialising_llm_client",
                model=settings.ollama_model,
            )

            LLMService._client = ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                num_predict=2000,
                num_ctx=1024,
                keep_alive=-1,
                temperature=0,
                format="json",
            )

            logger.info(
                "Initialised_llm_client",
                model=settings.ollama_model,
            )
        return self._client


llm_service = LLMService()
