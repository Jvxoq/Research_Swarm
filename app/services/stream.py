"""Streaming service for LangGraph node progress."""

import json
import uuid
from datetime import datetime
from typing import AsyncIterator

from app.services.database import DatabaseService
from app.core.langgraph.graph import ResearchGraph
from app.api.constants import NODE_PROGRESS, NODE_MESSAGES


class StreamingService:
    """
    Streams LangGraph node progress to client via SSE.

    Injected dependencies:
        - db_service: Database operations (save report, update status)
        - graph: LangGraph instance to execute
    """

    def __init__(self, db_service: DatabaseService, graph: ResearchGraph):
        self.db_service = db_service
        self.graph = graph

    def _convert_event_to_sse(self, event: dict) -> str | None:
        """Convert LangGraph event to SSE message."""
        if event.get("event") != "on_chain_end":
            return None

        node_name = event.get("name")
        if node_name not in NODE_PROGRESS:
            return None

        return self._build_sse_message(
            "node_complete",
            {
                "node": node_name,
                "progress": NODE_PROGRESS.get(node_name, 0),
                "message": NODE_MESSAGES.get(node_name, f"Completed {node_name}"),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def _build_sse_message(self, event_type: str, data: dict) -> str:
        """Build SSE-formatted message."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    async def stream_progress(
        self,
        job_id: str,
        user_id: str,
        topic: str,
    ) -> AsyncIterator[str]:
        """
        Stream graph execution progress to client.

        Event structure:
        {
            "event": "on_chain_end",
            "name": "orchestrator_node",
            "run_id": "abc-123",
            "data": {...},
            "metadata": {...}
        }
        """
        job_uuid = uuid.UUID(job_id)

        config = {
            "configurable": {
                "thread_id": job_id,
            }
        }
        
        input_data = {
            "topic": topic,
            "retry_count": 0,
            "user_id": user_id,
        }

        yield self._build_sse_message(
            "started",
            {
                "status": "running",
                "job_id": job_id,
                "topic": topic,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        self.db_service.update_job_status(job_id, "running")

        async for event in self.graph.astream_events(input_data, config, version="v2"):
            sse_message = self._convert_event_to_sse(event)
            if sse_message:
                yield sse_message

        final_state = await self.graph.aget_state(config)
        final_report = final_state.get("final_report", "No report generated")
        approved_facts = final_state.get("approved_facts", [])
        rejected_facts = final_state.get("rejected_facts", [])

        self.db_service.save_report(
            job_id=job_uuid,
            report_content=final_report,
        )

        self.db_service.update_job_status(job_uuid, "completed")

        yield self._build_sse_message(
            "complete",
            {
                "status": "completed",
                "report": final_report,
                "progress": 100,
                "stats": {
                    "approved_facts": len(approved_facts),
                    "rejected_facts": len(rejected_facts),
                },
                "completed_at": datetime.utcnow().isoformat(),
            },
        )
