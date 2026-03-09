from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama
from app.core.config import settings
from app.core.logging import logger
from typing import List, Optional


class VectorDB:
    """Singleton VectorDB service using Qdrant."""

    _instance: Optional["VectorDB"] = None
    EMBED_DIMENSION = 1024

    def __new__(cls) -> "VectorDB":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
            cls._instance._ollama_client = None
        return cls._instance

    @property
    def client(self) -> QdrantClient:
        """Lazy-load Qdrant client."""
        if self._client is None:
            logger.info("initializing_qdrant_client")
            self._client = QdrantClient(url=settings.qdrant_url)
        return self._client

    @property
    def ollama(self) -> "ollama.Client":
        """Lazy-load Ollama client."""
        if self._ollama_client is None:
            logger.info("initializing_ollama_client")
            self._ollama_client = ollama.Client(host=settings.ollama_base_url)
        return self._ollama_client

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using Ollama."""
        response = self.ollama.embeddings(model=settings.embedding_model, prompt=text)
        return response["embedding"]

    def create_collection(self, name: str) -> None:
        """Create a collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == name for c in collections)

        if not exists:
            logger.info("creating_collection", name=name)
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.EMBED_DIMENSION, distance=Distance.COSINE
                ),
            )

    def store_fact(
        self, collection: str, fact_id: str, claim: str, metadata: dict
    ) -> None:
        """Store a fact in the vector database."""
        self.create_collection(collection)

        embedding = self.embed_text(claim)

        point = PointStruct(
            id=fact_id, vector=embedding, payload={"claim": claim, **metadata}
        )

        self.client.upsert(collection_name=collection, points=[point])

        logger.debug("fact_stored", fact_id=fact_id, collection=collection)

    def find_similar(
        self, collection: str, claim: str, limit: int = 10, threshold: float = 0.85
    ) -> List[dict]:
        """Find similar facts in the vector database."""
        embedding = self.embed_text(claim)

        results = self.client.search(
            collection_name=collection,
            query_vector=embedding,
            limit=limit,
            score_threshold=threshold,
        )

        return [{"id": hit.id, "score": hit.score, **hit.payload} for hit in results]

    def clear_collection(self, collection: str) -> None:
        """Clear a collection."""
        self.client.delete_collection(collection)
        logger.info("collection_cleared", name=collection)


def get_vectordb() -> VectorDB:
    """Get singleton VectorDB instance."""
    return VectorDB()

