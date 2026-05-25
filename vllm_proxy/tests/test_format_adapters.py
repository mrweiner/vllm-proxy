#!/usr/bin/env python3
"""Tests for vllm_proxy.format_adapters — format-agnostic tool call handling."""

import unittest
from vllm_proxy.format_adapters import (
    detect_format,
    is_chat_endpoint,
    openai_extract,
    openai_repair_inplace,
    anthropic_extract,
    build_anthropic_tool_use_chunk,
)


class TestFormatDetection(unittest.TestCase):
    def test_openai_chat_completions_path(self):
        self.assertEqual(detect_format("/v1/chat/completions"), "openai")

    def test_anthropic_messages_path(self):
        self.assertEqual(detect_format("/v1/messages"), "anthropic")

    def test_unknown_path(self):
        self.assertEqual(detect_format("/v1/completions"), "unknown")

    def test_is_chat_openai(self):
        self.assertTrue(is_chat_endpoint("/v1/chat/completions"))

    def test_is_chat_anthropic(self):
        self.assertTrue(is_chat_endpoint("/v1/messages"))

    def test_is_chat_unknown(self):
        self.assertFalse(is_chat_endpoint("/v1/completions"))


class TestOpenAIExtract(unittest.TestCase):
    def test_extracts_tool_call_from_delta(self):
        chunk = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"name": "read", "arguments": '{"filePath": "x.py"}'},
                    }]
                },
            }],
        }
        deltas, finish = openai_extract(chunk)
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0].idx, 0)
        self.assertEqual(deltas[0].name, "read")
        self.assertEqual(deltas[0].args_fragment, '{"filePath": "x.py"}')
        self.assertIsNone(finish)

    def test_extracts_finish_reason(self):
        chunk = {
            "choices": [{
                "delta": {"content": "done"},
                "finish_reason": "tool_calls",
            }],
        }
        deltas, finish = openai_extract(chunk)
        self.assertEqual(len(deltas), 0)
        self.assertEqual(finish.finish_reason, "tool_calls")


class TestOpenAIRepairInplace(unittest.TestCase):
    def test_writes_repaired_args_into_chunk(self):
        chunk = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '{"filePath": "x.py"'},
                    }],
                },
                "finish_reason": "stop",
            }],
        }
        openai_repair_inplace(chunk, 0, '{"filePath": "x.py"}')
        args = chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(args, '{"filePath": "x.py"}')

    def test_skips_non_matching_index(self):
        chunk = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 1,
                        "function": {"arguments": '{"old": "value"}'},
                    }],
                },
            }],
        }
        openai_repair_inplace(chunk, 0, '{"new": "value"}')
        args = chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(args, '{"old": "value"}')


class TestAnthropicExtract(unittest.TestCase):
    def test_extracts_tool_use_from_content_block_start(self):
        chunk = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "tool_abc",
                "name": "read",
                "input": {},
            },
        }
        deltas, finish = anthropic_extract(chunk)
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0].idx, 0)
        self.assertEqual(deltas[0].name, "read")
        self.assertIsNone(finish)

    def test_extracts_partial_json_from_delta(self):
        chunk = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"filePath": "',
            },
        }
        deltas, finish = anthropic_extract(chunk)
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0].idx, 0)
        self.assertEqual(deltas[0].args_fragment, '{"filePath": "')
        self.assertIsNone(finish)

    def test_extracts_stop_reason_from_message_delta(self):
        chunk = {
            "type": "message_delta",
            "stop_reason": "tool_use",
            "delta": {},
        }
        deltas, finish = anthropic_extract(chunk)
        self.assertEqual(len(deltas), 0)
        self.assertEqual(finish.finish_reason, "tool_use")

    def test_ignores_text_content_block_start(self):
        chunk = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "Hello"},
        }
        deltas, finish = anthropic_extract(chunk)
        self.assertEqual(len(deltas), 0)
        self.assertIsNone(finish)





class TestBuildAnthropicToolUseChunk(unittest.TestCase):
    def test_builds_content_block_start_with_repaired_input(self):
        chunk = build_anthropic_tool_use_chunk(
            idx=0, tool_id="tool_abc", tool_name="read",
            repaired_input='{"filePath": "x.py"}',
        )
        self.assertEqual(chunk["type"], "content_block_start")
        self.assertEqual(chunk["index"], 0)
        cb = chunk["content_block"]
        self.assertEqual(cb["type"], "tool_use")
        self.assertEqual(cb["id"], "tool_abc")
        self.assertEqual(cb["name"], "read")
        self.assertEqual(cb["input"], {"filePath": "x.py"})


if __name__ == "__main__":
    unittest.main()
