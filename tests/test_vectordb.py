"""Test VectorDB service."""

from app.services.vectordb import VectorDB, get_vectordb
from app.core.config import settings


def test_singleton():
    """Test singleton pattern."""
    db1 = get_vectordb()
    db2 = get_vectordb()
    assert db1 is db2, "Should return same instance"
    print("Singleton works")


def test_embed_text():
    """Test text embedding."""
    db = get_vectordb()
    embedding = db.embed_text("The sky is blue")
    assert isinstance(embedding, list), "Should return list"
    assert len(embedding) == 384, "MiniLM-L6-v2 has 384 dimensions"
    print(f"✓ Embedding shape: {len(embedding)}")


def test_create_collection():
    """Test collection creation."""
    db = get_vectordb()
    db.create_collection("test_collection")
    print("✓ Collection created")


def test_store_and_find():
    """Test storing and finding facts."""
    db = get_vectordb()
    collection = "test_facts"

    db.create_collection(collection)

    db.store_fact(
        collection=collection,
        fact_id="fact_1",
        claim="The sky is blue due to Rayleigh scattering",
        metadata={
            "task": "test",
            "source_url": "https://example.com",
            "confidence": 0.9,
        },
    )

    db.store_fact(
        collection=collection,
        fact_id="fact_2",
        claim="Blue light is scattered in all directions by the atmosphere",
        metadata={
            "task": "test",
            "source_url": "https://example2.com",
            "confidence": 0.85,
        },
    )

    results = db.find_similar(
        collection=collection, claim="Why is the sky blue?", limit=10, threshold=0.5
    )

    assert len(results) >= 1, "Should find similar facts"
    print(f"✓ Found {len(results)} similar fact(s)")
    print(f"  Top match score: {results[0]['score']:.3f}")

    db.client.delete_collection(collection)
    print("✓ Collection cleaned up")


def test_lazy_loading():
    """Test lazy initialization."""
    db = get_vectordb()

    assert db._client is None, "Client should be None before use"
    _ = db.client
    assert db._client is not None, "Client should be initialized after access"

    assert db._embedder is None, "Embedder should be None before use"
    _ = db.embedder
    assert db._embedder is not None, "Embedder should be initialized after access"
    print("✓ Lazy loading works")


if __name__ == "__main__":
    print("Testing VectorDB...\n")
    test_singleton()
    test_lazy_loading()
    test_embed_text()
    test_create_collection()
    test_store_and_find()
    print("\n✓ All tests passed!")
