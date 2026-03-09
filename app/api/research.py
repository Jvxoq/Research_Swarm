from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from fastapi.responses import StreamingResponse

from app.schemas.research import (
    ResearchRequest,
    ResearchResponse,
)
from sqlmodel import Session
from app.core.langgraph.graph import get_graph
from app.api.auth import get_current_session
from app.models.session import UserSession
from app.services.database import DatabaseService
from app.db.session import get_db
from app.core.logging import logger
from app.services.stream import StreamingService


router = APIRouter()

@router.post("/research", response_model=ResearchResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_research_job(
    research_request: ResearchRequest,
    current_session: UserSession = Depends(get_current_session), # Auth Session Pending
    db: Session = Depends(get_db) # DB Session Pending
):
    """Accept a research task and return a job_id

    Args:
        research_request: The Research request containing Topic.
        current_session: The current session from the auth token.
        db: The database session of the User

    Returns:
        ResearchResponse: The status of the Research request

    Raise:
        HTTPException: If there's an error processing the request.
    """
    try:
        logger.info(
            "research_request_received",
            session_id=str(current_session.id),
            research_topic=research_request.topic,
        )
        # Delegate to Database Service
        db_service = DatabaseService(db)
        job = db_service.create_job(current_session.id, research_request.topic)
        logger.info("research_job_created", job_id=str(job.id))
        return ResearchResponse(job_id=job.id, status=job.status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("research_request_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create research job",
        )

@router.get("/research/{job_id}/stream", response_model=StreamingResponse)
async def stream_research_job_status(
    job_id: str,
    current_session: UserSession = Depends(get_current_session),
    db: Session = Depends(get_db)
):
    """Stream research job progress via Server-Sent Events (SSE) and return final report.

    Args:
        job_id: UUID of the research job
        current_session: Authenticated UserSession
        db: Database Session

    Returns:
        streaming_response: Stream the Agent Progress

    Raise:
        HTTPException: Error in streaming progress or returning final report
    """
    try:
        logger.info(
            "stream_request_received",
            job_id=job_id,
            user_id=str(current_session.id)
        )
        
        db_service = DatabaseService(db)
        graph = get_graph()
        stream_service = StreamingService(db_service, graph)

        return await stream_service.stream_progress(
            job_id=job_id,
            user_id=current_session.id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("stream_request_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stream research report",
        )




        
    


        

    
