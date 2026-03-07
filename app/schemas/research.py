"""The Research Schema for the Research Endpoint"""
import uuid
from pydantic import (
    BaseModel,
    Field,
)

class ResearchRequest(BaseModel):
    """Request Model for Research Query from the user.

    Attributes:
        topic: Research request topic
    """
    topic: str = Field(..., min_length=3, max_length=1000)
    
class ResearchResponse:
    """Response model for research request.

    Attributes:
        job_id: Job ID for research topic
        job_status: Status of Research Topic
    """
    job_id: uuid.UUID
    job_status: str

