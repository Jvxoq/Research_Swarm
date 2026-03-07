"""Configure node names to progress % and user messages."""

NODE_PROGRESS = {
    "orchestrator": 15,
    "router_node": 20,
    "worker_node": 50,
    "critic": 70,
    "consensus": 85,
    "writer": 95,
}

NODE_MESSAGES = {
    "orchestrator": "Breaking down research topic into sub-tasks...",
    "router_node": "Routing tasks to worker agents...",
    "worker_node": "Searching web sources and gathering information...",
    "critic": "Verifying sources and fact-checking claims...",
    "consensus": "Applying consensus voting on verified facts...",
    "writer": "Compiling final research report...",
}