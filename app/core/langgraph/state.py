"""LangGraph state definition."""
from typing import TypedDict, Annotated, List
import operator
from pydantic import BaseModel


class SubTasksOutput(BaseModel):
    sub_tasks: List[str]

class WorkerResult(BaseModel):
    """Worker output."""
    task: str
    claim: str
    summary: str
    source_url: str


class VerifiedFact(BaseModel):
    """Critic output."""
    task: str
    claim: str
    summary: str
    source_url: str
    confidence: float


class ApprovedFact(BaseModel):
    """Consensus output."""
    task: str
    claim: str
    summary: List[str]
    sources: List[str]
    confidence: float


class FinalReport(BaseModel):
    """Writer output."""
    title: str
    report: str
    generated_at: str


# Graph state definition
class SwarmState(TypedDict):
    """Shared state passed between all nodes."""
    thread_id: str
    user_id: str
    topic: str
    sub_tasks: List[str]
    failed_tasks: List[str]
    worker_results: Annotated[List[WorkerResult], operator.add]  # Append mode
    verified_facts: List[VerifiedFact]
    approved_facts: List[ApprovedFact]
    rejected_facts: List[VerifiedFact]
    final_report: FinalReport
    retry_count: int