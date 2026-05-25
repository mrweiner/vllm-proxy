#!/usr/bin/env python3
"""Test that proxy preserves model's original finish_reason after tool calls.

The proxy was unconditionally rewriting finish_reason to "tool_calls" whenever
tool args were accumulated, even when the model sent "stop". This caused
opencode's agentic loop to fire an extra generation turn after tool execution
completed, producing self-referential hallucinated output.

References:
  - Traced from session ses_1a99a150fffef4B8zNU1BNYZav (agent re-answered
    original question after writing 4 files)
  - Opencode hasToolCalls loop condition:
    https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/session/prompt.ts#L1265
  - Proxy rewrite was added in commit f149d2a (proxy v1) with no rationale,
    and Anthropic guard added in 301b6d7 — the OpenAI side was never guarded.
  - Related opencode loop issues:
    https://github.com/anomalyco/opencode/issues/28986 (self-replies on non-monotonic IDs)
    https://github.com/anomalyco/opencode/issues/28618 (runLoop clock-skew extra call)

Removal: sse_handler.py:319-322 (OpenAI) and sse_handler.py:191-201 (Anthropic).
"""

import json
import unittest
from vllm_proxy.sse_handler import handle_sse_stream


def _make_sse_line(data: dict) -> bytes:
    return b"data: " + json.dumps(data, separators=(",", ":")).encode() + b"\n"


class TestOpenAIStopPreservation(unittest.TestCase):
    """Ensure model's finish_reason passes through after tool calls."""

    def test_stop_not_rewritten_when_tool_calls_present(self):
        """Model sends tool calls then finish_reason='stop' — stop should pass through.

        Previously the proxy unconditionally rewrote finish_reason to 'tool_calls',
        causing opencode's agentic loop to continue one extra turn with no new
        user input, resulting in the model hallucinating it needed to answer
        the original question again.
        """
        write_args = '{"filePath": "/tmp/x"}'
        upstream = b"".join([
            # Initial chunk: role + tool call id/name
            _make_sse_line({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"index": 0, "id": "call_abc", "type": "function", "function": {"name": "write"}}],
                    },
                }],
            }),
            # Args chunk (will be buffered)
            _make_sse_line({
                "id": "chatcmpl-2",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": write_args}}],
                    },
                }],
            }),
            # Finish chunk — model says "stop" (it's done, tools were emitted)
            _make_sse_line({
                "id": "chatcmpl-3",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }),
        ])

        emitted = []
        def collect(data):
            for line in data.split(b"\n"):
                if line.startswith(b"data: "):
                    try:
                        emitted.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass

        handle_sse_stream(raw=upstream, fmt="openai", write=collect)

        # Find the finish chunk
        finish_chunks = [e for e in emitted if e.get("choices", [{}])[0].get("finish_reason")]
        self.assertEqual(len(finish_chunks), 1)

        # finish_reason should be "stop", NOT rewritten to "tool_calls"
        fr = finish_chunks[0]["choices"][0]["finish_reason"]
        self.assertEqual(fr, "stop")

    def test_tool_calls_already_set_passthrough(self):
        """When model already sends finish_reason='tool_calls', pass it through unchanged."""
        bash_args = '{"command": "ls"}'
        upstream = b"".join([
            _make_sse_line({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"index": 0, "id": "call_x", "type": "function", "function": {"name": "bash"}}],
                    },
                }],
            }),
            _make_sse_line({
                "id": "chatcmpl-2",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": bash_args}}],
                    },
                }],
            }),
            _make_sse_line({
                "id": "chatcmpl-3",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }],
            }),
        ])

        emitted = []
        def collect(data):
            for line in data.split(b"\n"):
                if line.startswith(b"data: "):
                    try:
                        emitted.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass

        handle_sse_stream(raw=upstream, fmt="openai", write=collect)

        finish_chunks = [e for e in emitted if e.get("choices", [{}])[0].get("finish_reason")]
        self.assertEqual(len(finish_chunks), 1)
        self.assertEqual(finish_chunks[0]["choices"][0]["finish_reason"], "tool_calls")

    def test_repaired_args_still_injected_on_stop(self):
        """Even though we don't rewrite finish_reason, repaired args should still be injected into the finish chunk."""
        truncated_args = '{"filePath": "x.py"}'  # valid but would normally be truncated in practice
        upstream = b"".join([
            _make_sse_line({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"index": 0, "id": "call_z", "type": "function", "function": {"name": "read"}}],
                    },
                }],
            }),
            _make_sse_line({
                "id": "chatcmpl-2",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": truncated_args}}],
                    },
                }],
            }),
            _make_sse_line({
                "id": "chatcmpl-3",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }),
        ])

        emitted = []
        def collect(data):
            for line in data.split(b"\n"):
                if line.startswith(b"data: "):
                    try:
                        emitted.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass

        handle_sse_stream(raw=upstream, fmt="openai", write=collect)

        finish_chunks = [e for e in emitted if e.get("choices", [{}])[0].get("finish_reason")]
        self.assertEqual(len(finish_chunks), 1)
        fr = finish_chunks[0]["choices"][0]["finish_reason"]
        self.assertEqual(fr, "stop")

        # The repaired args should be injected into the finish chunk
        delta = finish_chunks[0]["choices"][0].get("delta", {})
        tool_calls = delta.get("tool_calls", [])
        self.assertTrue(len(tool_calls) > 0)
        args = tool_calls[0].get("function", {}).get("arguments", "")
        repaired = json.loads(args)
        self.assertEqual(repaired["filePath"], "x.py")


if __name__ == "__main__":
    unittest.main()
