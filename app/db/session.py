"""Database session management."""
from sqlmodel import create_engine, Session
from sqlalchemy.pool import QueuePool
from app.core.config import settings
from app.core.logging import logger


# Database Engine (singleton)
engine = create_engine(
    settings.database_url,
    echo=False,  # Used to log SQL queries
    poolclass=QueuePool,
    pool_size=5,  # Keep 5 connections open
    max_overflow=10,  # Allow 10 more connections if needed
    pool_pre_ping=True,  # Test connection before using
    pool_recycle=3600,  # Recycle connections after 1 hour
)

logger.info(
    "database_engine_created",
    url=settings.database_url.split("@")[-1]  # Log without credentials
)

def get_db():
    """
    FastAPI dependency that yields a database session.

    How it works:
    - Creates a new session from the engine
    - Yields it to the endpoint
    - Automatically commits on success
    - Automatically rolls back on error
    - Closes connection after request
    """
    with Session(engine) as session:
        yield session