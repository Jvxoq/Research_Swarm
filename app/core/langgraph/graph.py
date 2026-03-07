"""Research Swarm LangGraph."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from app.core.langgraph.state import SwarmState
from app.core.langgraph.nodes import (
    orchestrator_node,
    router_node,
    worker_node,
    critic_node,
    consensus_node,
    writer_node,
)
from app.core.config import settings
from app.core.logging import logger


class ResearchGraph:
    """
    Research Swarm multi-agent graph.
    
    Orchestrates: orchestrator → router → workers → critic → consensus → writer
    """
    
    def __init__(self):
        """Initialize graph with checkpointer."""
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        logger.info("building_research_graph")
        
        # Create graph
        builder = StateGraph(SwarmState)
        
        # Add nodes
        builder.add_node("orchestrator", orchestrator_node)
        builder.add_node("router", router_node)
        builder.add_node("worker_node", worker_node)
        builder.add_node("critic", critic_node)
        builder.add_node("consensus", consensus_node)
        # builder.add_node("writer", writer_node)        # TODO
        
        # Define edges
        builder.set_entry_point("orchestrator")
        builder.add_edge("orchestrator", "router")
        builder.add_edge("router", "worker_node")  # Send() handles fan-out
        builder.add_edge("worker_node", "critic")
        
        # Conditional: retry or proceed
        builder.add_conditional_edges(
            "critic",
            self._should_retry,
            {
                "retry": "router",
                "proceed": END  # TODO: Change to "consensus" when ready
            }
        )
        
        # Compile with checkpointer
        checkpointer = PostgresSaver.from_conn_string(settings.database_url)
        compiled = builder.compile(checkpointer=checkpointer)
        
        logger.info("graph_built")
        return compiled
    
    def _should_retry(self, state: SwarmState) -> str:
        """Decide whether to retry failed tasks."""
        MAX_RETRIES = 3
        
        if state["failed_tasks"] and state["retry_count"] < MAX_RETRIES:
            logger.info("retrying_failed_tasks", retry_count=state["retry_count"])
            return "retry"
        
        logger.info("proceeding_to_next_stage")
        return "proceed"
    
    async def ainvoke(self, input_data: dict, config: dict):
        """Run graph asynchronously."""
        return await self.graph.ainvoke(input_data, config)
    
    def astream_events(self, input_data: dict, config: dict, version: str):
        """Stream graph events."""
        return self.graph.astream_events(input_data, config, version=version)
    
    async def aget_state(self, config: dict):
        """Get final state from checkpointer.
        
        Args:
            config: {"configurable": {"thread_id": "job-123"}}
        
        Returns:
            SwarmState: The final state dict"""
        return await self.graph.aget_state(config)


# Singleton instance
_graph_instance = None

def get_graph() -> ResearchGraph:
    """Get or create graph instance."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = ResearchGraph()
    return _graph_instance