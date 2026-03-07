"""FastAPI application entry point."""
from fastapi import FastAPI, Request
from sqlmodel import SQLModel
from app.db.session import engine
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.models.research import ResearchJob, Report
from app.core.logging import setup_logging, logger
from app.core.langgraph.graph import get_graph
from app.api.research import router as research_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    
    Startup:
    - Initialize logging
    - Preload LangGraph (warm start)
    - Log app start
    
    Shutdown:
    - Log graceful shutdown
    """
    # Initialize structured logging
    setup_logging()

    logger.info(
        "application_startup",
        app_name="Research-Agent",

    )
    # Creates the tables in the db
    SQLModel.metadata.create_all(engine)
    # Preload graph with singleton initialization
    logger.info("preloading_langgraph")
    get_graph()
    logger.info("langgraph_ready")
    
    logger.info("application_ready")
    
    yield  # App runs here

    logger.info("application_shutdown")

app = FastAPI(
    title="Research Swarm API",
    description="Multi-agent research system with consensus verification",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return 500."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

app.include_router(
    research_router,
    prefix="/api",
    tags=["research"]
)

@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Research Swarm API",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "create_job": "POST /api/research",
            "stream_job": "GET /api/research/{job_id}/stream"
        }
    }