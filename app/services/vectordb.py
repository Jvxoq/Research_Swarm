from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from ollama import Client
from app.core.config import settings
from app.core.logging import logger
from typing import List, Optional
import uuid


class VectorDB:
    """Singleton VectorDB service using Qdrant."""

    _instance: Optional["VectorDB"] = None

    def __new__(cls) -> "VectorDB":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._client = None
            self._ollama_client = None
            self._embed_dimension = None
            self._initialized = True

    @property
    def client(self) -> QdrantClient:
        """Lazy-load Qdrant client."""
        if self._client is None:
            logger.info("initializing_qdrant_client")
            self._client = QdrantClient(url=settings.qdrant_url, prefer_grpc=False)
        return self._client

    @property
    def ollama(self) -> Client:
        """Lazy-load Ollama client."""
        if self._ollama_client is None:
            logger.info("initializing_ollama_client")
            self._ollama_client = Client(host=settings.ollama_base_url)
        return self._ollama_client

    @property
    def embed_dimension(self) -> int:
        """Get embedding dimension from model (cached)."""
        if self._embed_dimension is None:
            sample_embedding = self.embed_text("dimension check")
            self._embed_dimension = len(sample_embedding)
            logger.info("embedding_dimension_detected", dimension=self._embed_dimension)
        return self._embed_dimension

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using Ollama."""
        try:
            response = self.ollama.embeddings(
                model=settings.embedding_model, prompt=text
            )
            return response["embedding"]
        except Exception as e:
            logger.error("embedding_failed", error=str(e))
            raise

    def create_collection(self, name: str) -> None:
        """Create a collection if it doesn't exist."""
        try:
            exists = self.client.collection_exists(name)
            if not exists:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self.embed_dimension, distance=Distance.COSINE
                    ),
                )
                logger.info("collection_created", name=name)
        except Exception as e:
            logger.error("collection_creation_failed", error=str(e))
            raise

    def store_fact(
        self, collection: str, fact_id: str, claim: str, metadata: dict
    ) -> None:
        """Store a fact in the vector database."""
        try:
            self.create_collection(collection)
            embedding = self.embed_text(claim)
            point = PointStruct(
                id=fact_id, vector=embedding, payload={"claim": claim, **metadata}
            )
            self.client.upsert(collection_name=collection, points=[point])
            logger.debug("fact_stored", fact_id=fact_id, collection=collection)
        except Exception as e:
            logger.error("store_fact_failed", error=str(e))
            raise

    def find_similar(
        self, collection: str, claim: str, limit: int = 10,
    ) -> List[dict]:
        """Find similar facts in the vector database."""
        try:
            embedding = self.embed_text(claim)
            results = self.client.query_points(
                collection_name=collection,
                query=embedding,
                limit=limit,
                with_payload=True,
            )
            return results
        except Exception as e:
            logger.error("find_similar_failed", error=str(e))
            raise

    def clear_collection(self, collection: str) -> None:
        """Clear a collection."""
        try:
            self.client.delete_collection(collection)
            logger.info("collection_cleared", name=collection)
        except Exception as e:
            logger.error("clear_collection_failed", error=str(e))
            raise


def get_vectordb() -> VectorDB:
    """Get singleton VectorDB instance."""
    return VectorDB()
