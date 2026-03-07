from app.utils.sse import build_sse_message

def test_build_sse_message():

    msg = build_sse_message(
        event_type="node_complete",
        data={
            "node": "search",
            "status": "ok"
        }
    )

    # SSE format assertion
    assert msg.startswith("event: node_complete")
    assert "data: " in msg
    assert msg.endswith("\n\n")