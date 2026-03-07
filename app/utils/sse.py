"""Build SSE Compliant Message"""
import json


def build_sse_message(event_type: str, data: dict) -> str:
    """
    Build a Server-Sent Events Message.

    Format:
        event: <type>
        data: <json>
        <blank line>

    Args:
        event_type: The type of the event (node_complete, complete, error).
        data: The data to be sent in JSON format
    
    Returns:
       SSE-formated string with double newline terminator.
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
