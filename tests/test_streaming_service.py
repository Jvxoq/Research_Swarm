import pytest
from unittest.mock import Mock, AsyncMock

# Mock problematic imports before importing app modules
import sys

sys.modules["sentence_transformers"] = Mock()
sys.modules["transformers"] = Mock()
sys.modules["huggingface_hub"] = Mock()

from app.services.stream import StreamingService


class TestConvertEventToSse:
    def test_non_on_chain_end_returns_none(self):
        mock_db = Mock()
        mock_graph = Mock()
        service = StreamingService(mock_db, mock_graph)

        event = {"event": "on_chain_start", "name": "orchestrator"}
        result = service._convert_event_to_sse(event)

        assert result is None

    def test_node_not_in_progress_returns_none(self):
        mock_db = Mock()
        mock_graph = Mock()
        service = StreamingService(mock_db, mock_graph)

        event = {"event": "on_chain_end", "name": "unknown_node"}
        result = service._convert_event_to_sse(event)

        assert result is None

    def test_valid_event_returns_sse_message(self):
        mock_db = Mock()
        mock_graph = Mock()
        service = StreamingService(mock_db, mock_graph)

        event = {"event": "on_chain_end", "name": "orchestrator"}

        result = service._convert_event_to_sse(event)

        assert result is not None
        assert "event: node_complete" in result
        assert "orchestrator" in result


class TestBuildSseMessage:
    def test_builds_correct_format(self):
        mock_db = Mock()
        mock_graph = Mock()
        service = StreamingService(mock_db, mock_graph)

        result = service._build_sse_message("complete", {"status": "done"})

        assert result == 'event: complete\ndata: {"status": "done"}\n\n'
