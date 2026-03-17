"""
Create a singleton class for the LLM client
"""

from typing import Optional, cast
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from app.core.logging import logger


class LLMService:
    _instance: Optional["LLMService"] = None
    _gemini_client: Optional[BaseChatModel] = None

    def __new__(cls) -> "LLMService":
        # Create an instance of this class if not already created
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def gemini_client(self) -> BaseChatModel:
        if self._gemini_client is None:
            logger.info(
                "Initialising_gemini_client",
                model=settings.gemini_model,
            )

            LLMService._gemini_client = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.gemini_api_key,
                temperature=0,
            )

            logger.info(
                "Initialised_gemini_client",
                model=settings.gemini_model,
            )

        return self._gemini_client


llm_service = LLMService()
