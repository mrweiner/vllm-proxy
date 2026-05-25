#!/usr/bin/env python3
"""Test to reproduce the duplicate-chunk and double-brace bugs in OpenAI SSE streaming.

Pattern 1 (Double-brace):     {{"pattern": "..."}}   — LLM emits extra { wrapping
Pattern 2 (Duplicate-chunk):   {"cmd": "x"{"cmd": "x", "desc": "y"} — partial + full concatenated
"""

import json
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, "/home/weiner/repos/local-llm/proxy")
from vllm_proxy.sse_handler import handle_sse_stream


def _make_sse_line(data: dict) -> bytes:
    return b"data: " + json.dumps(data, separators=(",", ":")).encode() + b"\n"


def _collect_emitted(raw: bytes, fmt: str = "openai"):
    """Run handle_sse_stream and return list of emitted JSON objects."""
    emitted = []
    def fake_write(data: bytes):
        for line in data.split(b"\n"):
            line = line.strip()
            if line.startswith(b"data: "):
                try:
                    emitted.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
    handle_sse_stream(raw=raw, fmt=fmt, write=fake_write)
    return emitted


class TestOpenAIDuplicateChunk(unittest.TestCase):
    """Reproduce Pattern 2: duplicate concatenated args in final output.

    The error was: args like {"command": "x"{"command": "x", "description": "y"}
    This happens because the proxy both:
    (a) passes through the raw last-args chunk (with partial args)
    (b) injects full repaired args into the finish chunk
    Downstream consumer accumulates both -> partial + full = duplication
    """

    def test_duplicate_when_last_args_chunk_also_has_finish(self):
        """vLLM sends final args AND finish_reason in one chunk -> duplication."""
        upstream = b"".join([
            # Chunk 1: role + tool call start
            _make_sse_line({
                "id": "chatcmpl-1",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "bash", "arguments": ""},
                        }],
                    },
                }],
            }),
            # Chunk 2: partial args (no finish_reason)
            _make_sse_line({
                "id": "chatcmpl-1",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": '{"command": "python3 tmp/fix.py"'},
                        }],
                    },
                }],
            }),
            # Chunk 3: COMPLETE args + finish_reason in SAME chunk
            # This is the problematic case — the args flow through AND get repaired/injected
            _make_sse_line({
                "id": "chatcmpl-1",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": ',"description": "Fix overviews"'}}
                        ],
                    },
                    "finish_reason": "stop",
                }],
            }),
        ])

        emitted = _collect_emitted(upstream)

        # Find all arguments fragments emitted to downstream
        all_args_frags = []
        for chunk in emitted:
            for tc in chunk.get("choices", [{}])[0].get("delta", {}).get("tool_calls", []):
                args = tc.get("function", {}).get("arguments", "")
                if args:
                    all_args_frags.append(args)

        # Simulate what downstream does: concatenate all fragments
        downstream_accumulated = "".join(all_args_frags)
        # Debug: print what was emitted per-chunk and the combined result
        print(f"  Fragments emitted: {all_args_frags}")
        print(f"  Combined: {downstream_accumulated}")
        # Should be valid JSON, NOT duplicated like '{"command": "...","description": "..."},{"command": "...","description": "..."}'
        parsed = json.loads(downstream_accumulated)
        self.assertIn("command", parsed)
        self.assertIn("description", parsed)

    def test_no_duplication_when_separate_finish_chunk(self):
        """vLLM sends separate finish chunk = no duplication risk."""
        upstream = b"".join([
            # Chunk 1: role + tool call start
            _make_sse_line({
                "id": "chatcmpl-1",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "bash", "arguments": ""},
                        }],
                    },
                }],
            }),
            # Chunk 2: truncated args (no finish_reason)
            _make_sse_line({
                "id": "chatcmpl-1",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": '{"command": "echo hi"'},
                        }],
                    },
                }],
            }),
            # Chunk 3: finish chunk (NO args, just finish_reason) — separate from args
            _make_sse_line({
                "id": "chatcmpl-1",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }),
        ])

        emitted = _collect_emitted(upstream)

        all_args_frags = []
        for chunk in emitted:
            for tc in chunk.get("choices", [{}])[0].get("delta", {}).get("tool_calls", []):
                args = tc.get("function", {}).get("arguments", "")
                if args:
                    all_args_frags.append(args)

        downstream_accumulated = "".join(all_args_frags)
        # This should be valid JSON with the repaired args
        parsed = json.loads(downstream_accumulated)
        self.assertEqual(parsed["command"], "echo hi")

    def test_truncated_no_finish_only_args_chunk(self):
        """vLLM sends args but no separate finish_reason chunk -> truncation not repaired."""
        upstream = b"".join([
            # Chunk 1: role + tool call start
            _make_sse_line({
                "id": "chatcmpl-1",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "edit", "arguments": ""},
                        }],
                    },
                }],
            }),
            # Chunk 2: truncated args, no finish_reason anywhere
            _make_sse_line({
                "id": "chatcmpl-1",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": '{"filePath": "x.py", "oldString": "'},
                        }],
                    },
                }],
            }),
        ])

        emitted = _collect_emitted(upstream)

        all_args_frags = []
        for chunk in emitted:
            for tc in chunk.get("choices", [{}])[0].get("delta", {}).get("tool_calls", []):
                args = tc.get("function", {}).get("arguments", "")
                if args:
                    all_args_frags.append(args)

        downstream_accumulated = "".join(all_args_frags)
        # Currently this will be truncated and invalid. The proxy should still emit
        # something repairable, or at least not broken.
        # For now, just verify it passes through without crashing.
        self.assertTrue(downstream_accumulated)


class TestOpenAIDoubleBrace(unittest.TestCase):
    """Reproduce Pattern 1: {{"key": "val"}} double-brace args from LLM output."""

    def test_double_brace_in_tool_args(self):
        """LLM emits {{"pattern": "foo"}} — upstream vLLM passes this through,
        but our repair should handle the double brace."""
        upstream = b"".join([
            _make_sse_line({
                "id": "chatcmpl-1",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "glob", "arguments": ""},
                        }],
                    },
                }],
            }),
            # Args contain double-brace (qwen3 parser artifact)
            _make_sse_line({
                "id": "chatcmpl-1",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": '{{"pattern": "plans/**/*.md"}'},
                        }],
                    },
                    "finish_reason": "stop",
                }],
            }),
        ])

        emitted = _collect_emitted(upstream)

        # Find the tool call args that downstream would see
        all_args = []
        for chunk in emitted:
            for tc in chunk.get("choices", [{}])[0].get("delta", {}).get("tool_calls", []):
                args = tc.get("function", {}).get("arguments", "")
                if args:
                    all_args.append(args)

        combined = "".join(all_args)
        # Should be parseable as valid JSON (the proxy should strip the extra brace)
        try:
            parsed = json.loads(combined)
            self.assertEqual(parsed["pattern"], "plans/**/*.md")
        except json.JSONDecodeError as e:
            self.fail(f"Proxy emitted unparseable args: {combined!r}. Error: {e}")


if __name__ == "__main__":
    unittest.main()
