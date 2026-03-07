"""Database service for research job operations."""

from sqlmodel import Session, select
from sqlalchemy.exc import SQLAlchemyError
from app.core.logging import logger
from app.models.research import ResearchJob, Report
from datetime import datetime
import uuid


# Custom Exceptions for Research Job Operations
class DatabaseError(Exception):
    """Database operation failed."""
    pass

class JobNotFoundError(Exception):
    """Job doesn't exist in database."""
    pass


# Database Service for Research Job Operations
class DatabaseService:
    """Handles all database operations for research jobs."""

    def __init__(self, db: Session):
        """
        Initialize database service.

        Args:
            db: Active SQLModel session from connection pool
        """
        self.db = db

    def create_job(self, user_id: str, topic: str) -> ResearchJob:
        """
        Create a new research job.

        Args:
            user_id: ID of user creating the job
            topic: Research topic string

        Returns:
            ResearchJob: Created job with generated job_id

        Raises:
            DatabaseError: If database operation fails
        """
        try:
            job = ResearchJob(user_id=user_id, topic=topic, status="pending")
            self.db.add(job)
            self.db.commit()
            self.db.refresh(job)

            logger.info("job_created", job_id=str(job.job_id))
            return job

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error("create_job_failed", error=str(e), exc_info=True)
            raise DatabaseError(f"Failed to create job: {str(e)}")

    def get_job(self, job_id: uuid.UUID) -> ResearchJob:
        """
        Get job by ID.

        Args:
            job_id: UUID of the job

        Returns:
            ResearchJob: The job object

        Raises:
            JobNotFoundError: If job doesn't exist
            DatabaseError: If database query fails
        """
        try:
            statement = select(ResearchJob).where(ResearchJob.job_id == job_id)
            job = self.db.exec(statement).first()

            if not job:
                raise JobNotFoundError(f"Job {job_id} not found")

            return job

        except SQLAlchemyError as e:
            logger.error("get_job_failed", job_id=job_id, error=str(e), exc_info=True)
            raise DatabaseError(f"Failed to get job: {str(e)}")

    def update_job_status(self, job_id: uuid.UUID, status: str) -> None:
        """
        Update job status.

        Args:
            job_id: UUID of the job
            status: New status (pending/running/completed/failed)

        Raises:
            JobNotFoundError: If job doesn't exist
            DatabaseError: If database operation fails
        """
        try:
            job = self.get_job(job_id)

            job.status = status

            # Update timestamps
            if status == "running" and not job.started_at:
                job.started_at = datetime.utcnow()
            elif status in ["completed", "failed"] and not job.completed_at:
                job.completed_at = datetime.utcnow()

            self.db.add(job)
            self.db.commit()
            self.db.refresh(job)

            logger.info("job_status_updated", job_id=job_id, status=status)

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(
                "update_status_failed", job_id=job_id, error=str(e), exc_info=True
            )
            raise DatabaseError(f"Failed to update status: {str(e)}")

    def save_report(
        self,
        job_id: uuid.UUID,
        report_content: str,
    ) -> Report:
        """
        Save final report.

        Args:
            job_id: UUID of the job
            report_content: Markdown report text

        Returns:
            Report: Created report object

        Raises:
            DatabaseError: If database operation fails
        """
        try:
            report = Report(
                job_id=job_id,
                content=report_content,
            )
            self.db.add(report)
            self.db.commit()
            self.db.refresh(report)

            logger.info("report_saved", job_id=job_id)
            return report

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(
                "save_report_failed", job_id=job_id, error=str(e), exc_info=True
            )
            raise DatabaseError(f"Failed to save report: {str(e)}")

    def get_report(self, job_id: uuid.UUID) -> Report | None:
        """
        Get report by job ID.

        Args:
            job_id: UUID of the job

        Returns:
            Report if exists, None otherwise

        Raises:
            DatabaseError: If database query fails
        """
        try:
            statement = select(Report).where(Report.job_id == job_id)
            return self.db.exec(statement).first()

        except SQLAlchemyError as e:
            logger.error(
                "get_report_failed", job_id=job_id, error=str(e), exc_info=True
            )
            raise DatabaseError(f"Failed to get report: {str(e)}") 