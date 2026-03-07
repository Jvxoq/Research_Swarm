"""This file contains the Research Job Model for the application"""
from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional
import uuid

class ResearchJob(SQLModel, table=True):
    __tablename__ = "research_jobs"
    
    job_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: str
    topic: str
    status: str  # "pending", "running", "completed", "failed"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class Report(SQLModel, table=True):
    __tablename__ = "research_reports"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    job_id: uuid.UUID
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


