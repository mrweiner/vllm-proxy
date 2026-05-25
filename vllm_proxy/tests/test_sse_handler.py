#!/usr/bin/env python3
"""Integration tests for Anthropic tool-call repair in the SSE handler."""

import json
import unittest
from vllm_proxy.sse_handler import handle_sse_stream


def _make_sse_line(data: dict) -> bytes:
    return b"data: " + json.dumps(data, separators=(",", ":")).encode() + b"\n"


class TestAnthropicStreamRepair(unittest.TestCase):
    """Test the full SSE streaming path for Anthropic format."""

    def test_repaired_tool_input_is_emitted_at_finish(self):
        """Truncated JSON in tool input is repaired and emitted as valid JSON."""
        upstream = [
            # New message
            _make_sse_line({"type": "message_start", "message": {"id": "msg_1", "content": []}}),
            # content_block_start: tool_use, index 0
            _make_sse_line({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "tool_abc", "name": "read", "input": {}},
            }),
            # partial json — truncated, missing closing }
            _make_sse_line({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"filePath": "'},
            }),
            _make_sse_line({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": 'x.py"'},
            }),
            # content_block_stop
            _make_sse_line({"type": "content_block_stop", "index": 0}),
            # message_delta with stop_reason = tool_use (the "finish")
            _make_sse_line({"type": "message_delta", "stop_reason": "tool_use", "delta": {}}),
        ]
        upstream_data = b"".join(upstream)

        emitted = []
        def fake_write(data: bytes):
            for line in data.split(b"\n"):
                if line.startswith(b"data: "):
                    try:
                        emitted.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass

        handle_sse_stream(
            raw=upstream_data,
            fmt="anthropic",
            write=fake_write,
        )

        # Find the content_block_start in emitted output — its input should be valid JSON
        block_starts = [e for e in emitted if e.get("type") == "content_block_start"]
        self.assertEqual(len(block_starts), 1)
        cb = block_starts[0]["content_block"]
        self.assertEqual(cb["type"], "tool_use")
        self.assertEqual(cb["name"], "read")
        # The input should be valid JSON (repaired)
        input_obj = cb["input"]
        self.assertIsInstance(input_obj, dict)
        self.assertEqual(input_obj["filePath"], "x.py")

    def test_text_blocks_pass_through_immediately(self):
        """Text and reasoning blocks are emitted immediately, not buffered."""
        upstream = b"".join([
            _make_sse_line({"type": "message_start", "message": {"id": "msg_1", "content": []}}),
            _make_sse_line({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "Thinking..."},
            }),
            _make_sse_line({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " Let me check."},
            }),
            _make_sse_line({"type": "content_block_stop", "index": 0}),
            # Tool use follows
            _make_sse_line({
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "tool_x", "name": "bash", "input": {}},
            }),
            _make_sse_line({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls"}'},
            }),
            _make_sse_line({"type": "content_block_stop", "index": 1}),
            _make_sse_line({"type": "message_delta", "stop_reason": "tool_use", "delta": {}}),
        ])

        emitted = []
        def collect(data):
            for line in data.split(b"\n"):
                if line.startswith(b"data: "):
                    try:
                        emitted.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
        handle_sse_stream(raw=upstream, fmt="anthropic", write=collect)

        # Should have text block + tool_use block
        types = [e.get("type") for e in emitted]
        self.assertIn("content_block_start", types)

    def test_reasoning_before_tool_call_passes_through(self):
        """Reasoning blocks before tool calls are emitted immediately, not buffered."""
        upstream = b"".join([
            _make_sse_line({"type": "message_start", "message": {"id": "msg_1", "content": []}}),
            # Reasoning block — should pass through
            _make_sse_line({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            }),
            _make_sse_line({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me think about this."},
            }),
            _make_sse_line({"type": "content_block_stop", "index": 0}),
            # Tool use block — should be buffered until finish
            _make_sse_line({
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "tool_abc", "name": "read", "input": {}},
            }),
            _make_sse_line({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"filePath": "'},
            }),
            _make_sse_line({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": 'src/main.py"}'},
            }),
            _make_sse_line({"type": "content_block_stop", "index": 1}),
            _make_sse_line({"type": "message_delta", "stop_reason": "tool_use", "delta": {}}),
        ])

        emitted = []
        def collect(data):
            for line in data.split(b"\n"):
                if line.startswith(b"data: "):
                    try:
                        emitted.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
        handle_sse_stream(raw=upstream, fmt="anthropic", write=collect)

        # Reasoning block should be emitted
        reasoning_starts = [e for e in emitted if e.get("type") == "content_block_start" and e.get("content_block", {}).get("type") == "thinking"]
        self.assertEqual(len(reasoning_starts), 1)

        # Tool block should be emitted with valid JSON
        tool_starts = [e for e in emitted if e.get("type") == "content_block_start" and e.get("content_block", {}).get("type") == "tool_use"]
        self.assertEqual(len(tool_starts), 1)
        self.assertEqual(tool_starts[0]["content_block"]["input"]["filePath"], "src/main.py")

    def test_message_delta_without_stop_reason_emits_tools(self):
        """vLLM sends message_delta without stop_reason — tools should still be emitted."""
        upstream = b"".join([
            _make_sse_line({"type": "message_start", "message": {"id": "msg_1", "content": []}}),
            _make_sse_line({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "tool_abc", "name": "read", "input": {}},
            }),
            _make_sse_line({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"filePath": "'},
            }),
            _make_sse_line({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": 'x.py"'},
            }),
            _make_sse_line({"type": "content_block_stop", "index": 0}),
            # message_delta WITHOUT stop_reason (vLLM behavior)
            _make_sse_line({"type": "message_delta", "delta": {}}),
        ])

        emitted = []
        def collect(data):
            for line in data.split(b"\n"):
                if line.startswith(b"data: "):
                    try:
                        emitted.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
        handle_sse_stream(raw=upstream, fmt="anthropic", write=collect)

        # Tool block should still be emitted with repaired JSON
        tool_starts = [e for e in emitted if e.get("type") == "content_block_start" and e.get("content_block", {}).get("type") == "tool_use"]
        self.assertEqual(len(tool_starts), 1)
        self.assertEqual(tool_starts[0]["content_block"]["input"]["filePath"], "x.py")

        # message_delta should pass through
        message_deltas = [e for e in emitted if e.get("type") == "message_delta"]
        self.assertEqual(len(message_deltas), 1)

    def test_valid_json_is_not_modified(self):
        """When tool input JSON is already valid, pass through without repair."""
        upstream = b"".join([
            _make_sse_line({"type": "message_start", "message": {"id": "msg_1", "content": []}}),
            _make_sse_line({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "tool_abc", "name": "bash", "input": {}},
            }),
            _make_sse_line({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls", "description": "list"}'},
            }),
            _make_sse_line({"type": "content_block_stop", "index": 0}),
            _make_sse_line({"type": "message_delta", "stop_reason": "tool_use", "delta": {}}),
        ])

        emitted = []
        def collect(data):
            for line in data.split(b"\n"):
                if line.startswith(b"data: "):
                    try:
                        emitted.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
        handle_sse_stream(raw=upstream, fmt="anthropic", write=collect)

        block_starts = [e for e in emitted if e.get("type") == "content_block_start"]
        cb = block_starts[0]["content_block"]
        self.assertEqual(cb["input"], {"command": "ls", "description": "list"})


if __name__ == "__main__":
    unittest.main()
