#!/usr/bin/env python3
"""Test for duplicate-chunk: vLLM re-sends complete JSON in final args chunk,
causing concatenation with prior partial."""

import json
import sys
import unittest

sys.path.insert(0, "/home/weiner/repos/local-llm/proxy")
from vllm_proxy.sse_handler import handle_sse_stream


def _make_sse_line(data: dict) -> bytes:
    return b"data: " + json.dumps(data, separators=(",", ":")).encode() + b"\n"


def _collect_downstream(raw: bytes) -> tuple[dict, list]:
    """Simulate what opencode does: accumulate all args fragments, then parse.
    Returns (final_parsed_json, list_of_emitted_chunks)."""
    emitted = []
    def fake_write(data: bytes):
        for line in data.split(b"\n"):
            line = line.strip()
            if line.startswith(b"data: "):
                try:
                    emitted.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

    handle_sse_stream(raw=raw, fmt="openai", write=fake_write)

    # Accumulate args fragments like opencode does
    all_fragments = []
    for chunk in emitted:
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        for tc in delta.get("tool_calls", []):
            args = tc.get("function", {}).get("arguments", "")
            if args:
                all_fragments.append(args)

    combined = "".join(all_fragments)
    return combined, emitted


class TestDuplicateChunkNoLongerOccurs(unittest.TestCase):

    def test_vllm_resends_complete_json_in_final_chunk(self):
        """vLLM sends partial args in chunk 2, then COMPLETE JSON (same keys + extras)
        in chunk 3 with finish_reason. Accumulator should NOT duplicate."""
        upstream = b"".join([
            _make_sse_line({
                "id": "1",
                "choices": [{"index": 0, "delta": {
                    "role": "assistant",
                    "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                     "function": {"name": "bash", "arguments": ""}}],
                }}],
            }),
            # vLLM sends initial args
            _make_sse_line({
                "id": "1",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": 0, "function": {
                        "arguments": '{"command": "ls"}'},
                    }],
                }}],
            }),
            # vLLM re-sends COMPLETE args in the finish chunk (not just delta)
            _make_sse_line({
                "id": "1",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": 0, "function": {
                        "arguments": '{"command": "ls", "description": "list"}'
                    }},
                    ],
                }, "finish_reason": "stop"}],
            }),
        ])

        combined, _ = _collect_downstream(upstream)
        # Must NOT contain duplication like '{"command":"ls"}{"command":"ls","description":"list"}'
        parsed = json.loads(combined)
        self.assertEqual(parsed["command"], "ls")
        self.assertEqual(parsed["description"], "list")

    def test_instrumental_trace_emitted_chunks(self):
        """Verify emitted chunks don't carry duplicated content in their arguments."""
        upstream = b"".join([
            _make_sse_line({
                "id": "1",
                "choices": [{"index": 0, "delta": {
                    "role": "assistant",
                    "tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                     "function": {"name": "glob", "arguments": ""}}],
                }}],
            }),
            _make_sse_line({
                "id": "1",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": 0, "function": {
                        "arguments": '{"pattern": "foo"'},
                    }],
                }}],
            }),
            # Final: complete JSON + finish
            _make_sse_line({
                "id": "1",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": 0, "function": {
                        "arguments": '{"pattern": "foo/**/*.md"}'},
                    }],
                }, "finish_reason": "stop"}],
            }),
        ])

        _, emitted = _collect_downstream(upstream)
        for chunk in emitted:
            for tc in chunk.get("choices", [{}])[0].get("delta", {}).get("tool_calls", []):
                args = tc.get("function", {}).get("arguments", "")
                if args:
                    # Each individual chunk's args should be parseable
                    json.loads(args)  # Should not raise


if __name__ == "__main__":
    unittest.main()
