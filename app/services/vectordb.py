"""Vector database service for semantic clustering."""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logging import logger
from typing import List
import uuid


class VectorDBService:
    """Manages Qdrant vector database for fact clustering."""
    
    _client = None
    _embedder = None
    COLLECTION_NAME = "verified_facts"
    
    @classmethod
    def get_client(cls) -> QdrantClient:
        """Get or create Qdrant client."""
        if cls._client is None:
            logger.info("initializing_qdrant_client")
            cls._client = QdrantClient(
                url=settings.qdrant_url,
            )
            cls._ensure_collection()
        return cls._client
    
    @classmethod
    def get_embedder(cls) -> SentenceTransformer:
        """Get or create embedding model."""
        if cls._embedder is None:
            logger.info("loading_embedding_model")
            cls._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._embedder
    
    @classmethod
    def _ensure_collection(cls):
        """Create collection if it doesn't exist."""
        client = cls._client
        
        collections = client.get_collections().collections
        exists = any(c.name == cls.COLLECTION_NAME for c in collections)
        
        if not exists:
            logger.info("creating_qdrant_collection", name=cls.COLLECTION_NAME)
            client.create_collection(
                collection_name=cls.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=384,  # all-MiniLM-L6-v2 embedding size
                    distance=Distance.COSINE
                )
            )
    
    @classmethod
    def embed_text(cls, text: str) -> List[float]:
        """Generate embedding for text."""
        embedder = cls.get_embedder()
        embedding = embedder.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    @classmethod
    def store_fact(cls, fact_id: str, claim: str, metadata: dict):
        """
        Store a verified fact in vector DB.
        
        Args:
            fact_id: Unique ID for the fact
            claim: The claim text to embed
            metadata: Additional data (task, source_url, confidence, etc.)
        """
        client = cls.get_client()
        embedding = cls.embed_text(claim)
        
        point = PointStruct(
            id=fact_id,
            vector=embedding,
            payload={
                "claim": claim,
                **metadata
            }
        )
        
        client.upsert(
            collection_name=cls.COLLECTION_NAME,
            points=[point]
        )
        
        logger.debug("fact_stored", fact_id=fact_id)
    
    @classmethod
    def find_similar(cls, claim: str, limit: int = 10, threshold: float = 0.85) -> List[dict]:
        """
        Find similar facts in vector DB.
        
        Args:
            claim: The claim to search for
            limit: Max number of results
            threshold: Minimum similarity score (0.85 = 85% similar)
        
        Returns:
            List of similar facts with metadata
        """
        client = cls.get_client()
        embedding = cls.embed_text(claim)
        
        results = client.search(
            collection_name=cls.COLLECTION_NAME,
            query_vector=embedding,
            limit=limit,
            score_threshold=threshold
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "claim": hit.payload["claim"],
                "task": hit.payload["task"],
                "source_url": hit.payload["source_url"],
                "confidence": hit.payload["confidence"]
            }
            for hit in results
        ]
    
    @classmethod
    def clear_collection(cls):
        """Clear all facts (useful for testing or per-job isolation)."""
        client = cls.get_client()
        client.delete_collection(cls.COLLECTION_NAME)
        cls._ensure_collection()
        logger.info("collection_cleared")


# Convenience functions
def get_vectordb() -> VectorDBService:
    """Get VectorDB service."""
    return VectorDBService