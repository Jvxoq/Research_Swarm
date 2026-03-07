# app/core/langgraph/nodes.py
"""Agent nodes for research graph."""
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.types import Send
from app.core.langgraph.state import SwarmState, WorkerResult, VerifiedFact, SubTasksOutput, FinalReport
from app.core.langgraph.tools import get_web_search_tool, fetch_url
from app.services.llm import llm_service
from app.services.vectordb import get_vectordb
from app.core.logging import logger
from pydantic import BaseModel
from app.core.langgraph.state import SwarmState, VerifiedFact, ApprovedFact
from collections import defaultdict
import uuid




# Orchestrator Node
ORCHESTRATOR_PROMPT = """
You are a research planner. Generate EXACTLY 3 focused research sub-tasks.

Rules:
- Each sub-task must be a specific research question
- No overlapping questions
- Return ONLY valid JSON

Output format:
{{"sub_tasks": ["question1", "question2", "question3"]}}

Topic: {topic}"""


def orchestrator_node(state: SwarmState) -> dict:
    """Break down research topic into sub-tasks."""
    logger.info("orchestrator_started", topic=state["topic"])
    
    llm_client = llm_service.client()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research planning assistant."),
        ("human", ORCHESTRATOR_PROMPT)
    ])
    
    chain = prompt | llm_client | JsonOutputParser()
    result = chain.invoke({"topic": state["topic"]})
    
    # Validate
    validated = SubTasksOutput.model_validate(result)
    
    logger.info("orchestrator_complete", sub_tasks=validated.sub_tasks)
    return {"sub_tasks": validated.sub_tasks}

# Router Node
def router_node(state: SwarmState):
    """Route tasks to workers in parallel."""
    tasks = state.get("failed_tasks") or state["sub_tasks"]
    
    logger.info("routing_tasks", task_count=len(tasks))
    
    return [
        Send("worker_node", {"task": task})
        for task in tasks
    ]


# Worker Node
WORKER_PROMPT = """You are a research assistant.

Task: {task}

Use the web_search tool to find factual information.

Extract:
1. ONE key claim (one sentence)
2. A 1-2 paragraph summary
3. Source URL
4. Confidence (0.7-0.9 based on source quality)

Return JSON:
{{"task": "{task}", "claim": "...", "summary": "...", "source_url": "...", "confidence": 0.8}}"""


def worker_node(state: dict) -> dict:
    """Search web and extract findings."""
    task = state["task"]
    logger.info("worker_started", task=task)
    
    # Get tools
    web_search = get_web_search_tool()
    llm_client = llm_service.client()
    
    # Search web
    search_results = web_search.invoke(task)
    
    # Extract claim from results
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant."),
        ("human", WORKER_PROMPT + f"\n\nSearch results:\n{search_results}")
    ])
    
    chain = prompt | llm_client | JsonOutputParser()
    result = chain.invoke({"task": task})
    
    # Validate
    validated = WorkerResult.model_validate(result)
    
    logger.info("worker_complete", task=task, confidence=validated.confidence)
    return {"worker_results": [validated]}


# Critic Node
CRITIC_PROMPT = """You are a fact-checker.

Claim: {claim}
Source URL: {source_url}
Page content: {content}

Check if the claim is supported by the content.

Score confidence:
- 0.9-1.0: Directly stated
- 0.75-0.89: Implied/paraphrased
- 0.4-0.74: Partially supported
- 0.0-0.39: Not supported

Return JSON:
{{"task": "{task}", "claim": "{claim}", "source_url": "{source_url}", "confidence": 0.x, "status": "success"}}"""


def critic_node(state: SwarmState) -> dict:
    """Verify worker results."""
    logger.info("critic_started", result_count=len(state["worker_results"]))
    
    llm_client = llm_service.client()
    verified = []
    failed_tasks = []
    
    for result in state["worker_results"]:
        # Fetch URL content
        content = fetch_url.invoke(result.source_url)
        
        # Verify claim
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a fact-checking assistant."),
            ("human", CRITIC_PROMPT)
        ])
        
        chain = prompt | llm_client | JsonOutputParser()
        fact = chain.invoke({
            "task": result.task,
            "claim": result.claim,
            "source_url": result.source_url,
            "content": content
        })
        
        verified_fact = VerifiedFact.model_validate(fact)
        
        if verified_fact.confidence >= 0.75:
            verified.append(verified_fact)
        else:
            failed_tasks.append(verified_fact.task)
    
    logger.info("critic_complete", verified=len(verified), failed=len(failed_tasks))
    return {
        "verified_facts": verified,
        "failed_tasks": failed_tasks,
        "retry_count": state["retry_count"] + (1 if failed_tasks else 0)
    }


def consensus_node(state: SwarmState) -> dict:
    """
    Cluster similar facts and apply voting.
    
    Flow:
    1. Store all verified facts in VectorDB
    2. For each fact, find similar facts (≥0.85 similarity)
    3. Group into clusters
    4. Apply voting: cluster size ≥2 = approved
    5. Return approved and rejected lists
    """
    logger.info("consensus_started", fact_count=len(state["verified_facts"]))
    
    vectordb = get_vectordb()
    
    # Clear collection for this job (isolate per-job)
    vectordb.clear_collection()
    
    # STEP 1: Store all facts in VectorDB
    fact_map = {}  # fact_id -> VerifiedFact
    
    for fact in state["verified_facts"]:
        fact_id = str(uuid.uuid4())
        fact_map[fact_id] = fact
        
        vectordb.store_fact(
            fact_id=fact_id,
            claim=fact.claim,
            metadata={
                "task": fact.task,
                "source_url": fact.source_url,
                "confidence": fact.confidence
            }
        )
    
    # Find Clusters
    processed = set()
    clusters = []
    
    for fact_id, fact in fact_map.items():
        if fact_id in processed:
            continue
        
        # Find similar facts
        similar = vectordb.find_similar(
            claim=fact.claim,
            limit=10,
            threshold=0.85
        )
        
        # Build cluster
        cluster = []
        for hit in similar:
            hit_id = hit["id"]
            if hit_id not in processed:
                cluster.append(fact_map[hit_id])
                processed.add(hit_id)
        
        if cluster:
            clusters.append(cluster)
    
    logger.info("clustering_complete", cluster_count=len(clusters))
    

    MIN_SOURCES = 2
    approved = []
    rejected = []
    
    for cluster in clusters:
        if len(cluster) >= MIN_SOURCES:
            # Approved: corroborated by multiple sources
            approved_fact = ApprovedFact(
                claim=cluster[0].claim,  # Use first claim as canonical
                sources=[f.source_url for f in cluster],
                confidence=sum(f.confidence for f in cluster) / len(cluster),  # Average
                task=cluster[0].task
            )
            approved.append(approved_fact)
            
            logger.debug(
                "fact_approved",
                claim=approved_fact.claim[:50],
                source_count=len(cluster)
            )
        else:
            # Rejected: isolated claim
            rejected.extend(cluster)
            
            logger.debug(
                "fact_rejected",
                claim=cluster[0].claim[:50],
                reason="isolated"
            )
    
    logger.info(
        "consensus_complete",
        approved=len(approved),
        rejected=len(rejected)
    )
    
    return {
        "approved_facts": approved,
        "rejected_facts": rejected
    }

# app/core/langgraph/nodes.py

from datetime import datetime
from langchain_core.output_parsers import JsonOutputParser

WRITER_PROMPT = """You are a professional research report writer.

Topic: {topic}

Approved Facts:
{facts}

Instructions:
1. Create a markdown report with:
   - Title (# heading)
   - Executive Summary (2-3 paragraphs)
   - One section (## heading) per approved fact
   - Cite sources as [1], [2], etc. and list URLs at the end
2. Use professional, objective tone
3. Only use information from approved facts

Return ONLY valid JSON (no explanation):
{{
  "title": "Research Report: [Topic]",
  "report": "# Title\\n\\n## Executive Summary\\n\\n...full markdown..."
}}
"""


def writer_node(state: SwarmState) -> dict:
    """Compile approved facts into final markdown report."""
    logger.info(
        "writer_started",
        approved_count=len(state["approved_facts"])
    )

    facts_text = []
    for i, fact in enumerate(state["approved_facts"], 1):
        sources_list = "\n".join(f"  - {url}" for url in fact.sources)
        fact_block = f"""
Fact {i}:
  Claim: {fact.claim}
  Task: {fact.task}
  Confidence: {fact.confidence:.2f}
  Sources: {sources_list}
"""
        facts_text.append(fact_block)
    
    formatted_facts = "\n".join(facts_text)

    llm_client = llm_service.client()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional research report writer."),
        ("human", WRITER_PROMPT)
    ])
    
    chain = prompt | llm_client | JsonOutputParser()
    
    result = chain.invoke({
        "topic": state["topic"],
        "facts": formatted_facts,
    })
    
    final_report = FinalReport(
        title=result["title"],
        report=result["report"],
        generated_at=datetime.utcnow().isoformat()
    )
    
    logger.info("writer_complete", title=final_report.title)
    
    return {"final_report": final_report}