from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from google.genai._interactions.types.thought_content import Summary
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.types import Send
from app.core.langgraph.state import (
    SwarmState,
    WorkerResult,
    VerifiedFact,
    SubTasksOutput,
    FinalReport,
)
from app.core.langgraph.tools import web_search_tool, fetch_content
from app.services.llm import llm_service
from app.services.vectordb import get_vectordb
from app.core.logging import logger
from app.core.langgraph.state import SwarmState, VerifiedFact, ApprovedFact
import uuid


# Orchestrator Node
ORCHESTRATOR_PROMPT = """
CRITICAL: 
Follow INSTRUCTIONS & RULES.

INSTRUCTION:
You are a Research Orchestrator.
Given the TOPIC & CURRENT INFORMATION, generate EXACTLY 3 deep question as research sub-tasks
about the TOPIC.

TOPIC: {topic}
CURRENT INFORMATION: {current_info}

RULES:
- Each sub-task must be a specific research question
- No overlapping questions
- Return ONLY valid JSON with key: sub_tasks & value: the list of sub-tasks.
"""


def orchestrator_node(state: SwarmState) -> dict:
    """Break down research topic into sub-tasks."""
    logger.info("orchestrator_started", topic=state["topic"])

    search_result = web_search_tool.invoke(
        f"Latest Information Regarding - {state['topic']}"
    )

    current_info = ""
    if search_result:
        for result in search_result['results']:
            current_info = current_info + result['content']

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Return ONLY valid JSON. No explanation, no markdown, no extra text."),
            ("human", ORCHESTRATOR_PROMPT),
        ]
    )

    chain = prompt | llm_service.gemini_client | JsonOutputParser()
    result = chain.invoke({"topic": state["topic"], "current_info": current_info})

    # Validate
    validated = SubTasksOutput.model_validate(result)
    logger.info("orchestrator_complete", sub_tasks=validated.sub_tasks)
    return {"sub_tasks": validated.sub_tasks}


# Router Edge
def router_edge(state: SwarmState):
    """Route tasks to workers in parallel.
    
    Input from Orchestrator:
    
    ['What is the current ...', 'How do upgrades ...', "What is the total ..."]
    """
    tasks = state.get("failed_tasks") or state["sub_tasks"]

    logger.info("routing_tasks", task_count=len(tasks))

    return [Send("worker_node", {"task": task}) for task in tasks]


# Worker Node
WORKER_PROMPT = """
You are a Research Assistant.
Analyze the SEARCH RESULTS below and generate a structured research output for the TASK.

CRITICAL: 
Follow INSTRUCTIONS & RULES.

INSTRUCTIONS:
- Generate ONE concise claim (single sentence) from the search results
- Write a short summary paragraph from the search results
- Keep the source URL from search results

TASK: {task}

SEARCH RESULTS: {search_results}

RULES:
- Use ONLY information from SEARCH RESULTS

OUTPUT FORMAT EXAMPLE:
{{"task": "{task}", "claim": "...", "summary": "...", "source_url": "..."}}
"""


def worker_node(input: dict) -> dict:
    """Search web and extract findings.
    
    Args:
        task: Sub-task from Router Edge
    
    Returns:
        State update with worker_result"""
    logger.info("worker_started", task=input['task'])

    # Search web
    search_results = web_search_tool.invoke(str(input['task']))

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Return ONLY valid JSON. No explanation, no markdown, no extra text."),
            ("human", WORKER_PROMPT),
        ]
    )

    chain = prompt | llm_service.gemini_client | JsonOutputParser()
    result = chain.invoke({"task": input["task"], "search_results": search_results})

    # Validate
    validated = WorkerResult.model_validate(result)
    logger.info("worker_complete", task=input['task'])

    return {"worker_results": [validated]}


# Critic Node
CRITIC_PROMPT = """
You are a fact-checker.

INSTRUCTIONS
-Check if the CLAIM & SUMMARY is supported by the PAGE CONTENT.
-Score Confindence accordingly

Score confidence:
- 0.9-1.0: Directly stated
- 0.75-0.89: Implied/paraphrased
- 0.4-0.74: Partially supported
- 0.0-0.39: Not supported

CLAIM: {claim}
SUMMARY: {summary}
SOURCE URL: {source_url}
PAGE CONTENT: {content}

Return JSON:
{{"task": "{task}", "claim": "{claim}", "source_url": "{source_url}", "summary": {summary}, "confidence": 0.x}}
"""


def critic_node(state: SwarmState) -> dict:
    """Verify worker results."""
    logger.info("critic_started", result_count=len(state["worker_results"]))

    verified = []
    failed_tasks = []

    for result in state["worker_results"]:
        # Fetch URL content
        content = fetch_content.invoke(result.source_url)

        # Verify claim
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Return ONLY valid JSON. No explanation, no markdown, no extra text."), 
                ("human", CRITIC_PROMPT)
            ]
        )

        chain = prompt | llm_service.gemini_client | JsonOutputParser()
        fact = chain.invoke(
            {
                "task": result.task,
                "claim": result.claim,
                "source_url": result.source_url,
                "summary": result.summary,
                "content": content,
            }
        )

        critic_ouput = VerifiedFact.model_validate(fact)

        if critic_ouput.confidence >= 0.75:
            verified.append(critic_ouput)
        else:
            failed_tasks.append(critic_ouput)

    logger.info("critic_complete", verified=len(verified), failed=len(failed_tasks))
    return {
        "verified_facts": verified,
        "failed_tasks": failed_tasks,
        "retry_count": state["retry_count"] + (1 if failed_tasks else 0),
    }


def consensus_node(state: SwarmState) -> dict:
    """
    Cluster similar facts and apply voting.

    Args:
        state: verified_facts from the graph state
    
    Returns:
        approved_facts: approved facts from verified_facts
        rejected_facts: rejected facts from verified facts
    """
    logger.info("consensus_started", fact_count=len(state["verified_facts"]))

    # Create collection per job
    vectordb = get_vectordb()
    vectordb.create_collection(state['thread_id'])

    # Store all facts in VectorDB
    fact_map = {}  # fact_id -> VerifiedFact

    for fact in state["verified_facts"]:
        fact_id = str(uuid.uuid4())
        fact_map[fact_id] = fact

        vectordb.store_fact(
            collection=state['thread_id'],
            fact_id=fact_id,
            claim=fact.claim,
            metadata={
                "task": fact.task,
                "source_url": fact.source_url,
                "confidence": fact.confidence,
            },
        )

    # Find Clusters
    processed = set()
    clusters = []
    print(f"\n\nFact_map -> {fact_map}\n\n")
    for fact_id, fact in fact_map.items():
        if fact_id in processed:
            continue
        
        print(f"\n\nFact -> {fact}")
        # Find similar facts
        similar_facts = vectordb.find_similar(
            collection=state['thread_id'], claim=fact.claim,
        )
        print(f"\n\nSimilar_facts -> {similar_facts}")

        # Build cluster
        cluster = []
        for hit in similar_facts:
            hit_id = hit.id
            if hit_id not in processed:
                cluster.append(fact_map[hit_id])
                processed.add(hit_id)

        if cluster:
            clusters.append(cluster)
        

    logger.info("clustering_complete", cluster_count=len(clusters))

    MIN_SOURCES = 2
    approved = []
    rejected = []
    print(f"\n\nClusters -> {clusters}\n\n")
    for cluster in clusters:
        print(f"\n\nCluster -> {cluster}\n\n")
        if len(cluster) >= MIN_SOURCES:
            # Approved: corroborated by multiple sources
            approved_fact = ApprovedFact(
                claim=cluster[0].claim,  # Use first claim as canonical
                summary=[fact.summary for fact in clusters[0]],
                sources=[f.source_url for f in cluster],
                confidence=sum(f.confidence for f in cluster) / len(cluster),  # Average
                task=cluster[0].task,
            )
            approved.append(approved_fact)

            logger.debug(
                "fact_approved",
                claim=approved_fact.claim[:50],
                source_count=len(cluster),
            )
        else:
            # Rejected: isolated claim
            rejected.extend(cluster)

            logger.debug(
                "fact_rejected", claim=cluster[0].claim[:50], reason="isolated"
            )

    logger.info("consensus_complete", approved=len(approved), rejected=len(rejected))

    return {"approved_facts": approved, "rejected_facts": rejected}


WRITER_PROMPT = """
You are a professional Markdown Report Writer.

INSTRUCTIONS:
1. Create a markdown research report about the TOPIC using the SUMMARY with:
   - Title (# heading)
   - Executive Summary (1 paragraphs)
   - One section (## heading) per summary
   - Cite sources with URL as [1], [2], etc. at the end
2. Use professional, objective tone

TOPIC: {topic}
SUMMARY: {summary}

RETURN JASON:
{{
  "title": "Title...",
  "report": "Your Generated markdown report..."
}}
"""


def writer_node(state: SwarmState) -> dict:
    """Compile approved facts into final markdown report."""
    logger.info("writer_started", approved_count=len(state["approved_facts"]))

    print(f"\n\nCurrrent State -> {state}")
    approved_fact = state['approved_facts'][0]

    summaries = "\n".join(approved_fact.summary)
    final_summary = f"""
{approved_fact.claim}

{summaries}
""" 

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Return ONLY valid JSON."),
            ("human", WRITER_PROMPT),
        ]
    )

    chain = prompt | llm_service.gemini_client| JsonOutputParser()
    result = chain.invoke(
        {
            "topic": state["topic"],
            "summary": final_summary,
        }
    )

    final_report = FinalReport(
        title=result["title"],
        report=result["report"],
        generated_at=datetime.utcnow().isoformat(),
    )

    logger.info("writer_complete", title=final_report.title)

    return {"final_report": final_report}
