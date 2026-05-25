#!/usr/bin/env python3
"""Tests for the shared repair loop extracted from both Anthropic and OpenAI SSE handlers."""

import json
import unittest
from vllm_proxy.sse_handler import repair_all_tools


class TestRepairAllTools(unittest.TestCase):
    """repair_all_tools should independently run repair_json + apply_tool_patches
    for each accumulated tool and return the final args strings."""

    def test_single_truncated_tool_is_repaired(self):
        """Truncated JSON (missing closing brace) gets repaired."""
        state = {
            "args": {0: '{"filePath": "x.py"'},
            "names": {0: "read"},
            "tool_ids": {0: "tool_abc"},
        }
        result = repair_all_tools(state, "anthropic")

        self.assertIn(0, result)
        args, patch_status = result[0]
        parsed = json.loads(args)
        self.assertEqual(parsed["filePath"], "x.py")
        self.assertEqual(patch_status, "ok")

    def test_valid_json_passes_through(self):
        """Already-valid JSON is returned unchanged."""
        state = {
            "args": {0: '{"command": "ls", "description": "list"}'},
            "names": {0: "bash"},
            "tool_ids": {},
        }
        result = repair_all_tools(state, "openai")

        args, patch_status = result[0]
        self.assertEqual(args, '{"command": "ls", "description": "list"}')
        self.assertEqual(patch_status, "ok")

    def test_bash_tool_missing_description_is_patched(self):
        """bash tool missing 'description' gets auto-patched."""
        state = {
            "args": {0: '{"command": "ls"}'},
            "names": {0: "bash"},
            "tool_ids": {},
        }
        result = repair_all_tools(state, "anthropic")

        args, patch_status = result[0]
        parsed = json.loads(args)
        self.assertEqual(parsed["command"], "ls")
        self.assertEqual(parsed["description"], "[auto]")
        self.assertEqual(patch_status, "patched")

    def test_multiple_tools_repaired_independently(self):
        """Multiple tool indices are repaired independently."""
        state = {
            "args": {
                0: '{"filePath": "a.py"}',
                1: '{"command": "grep foo bar.py"}'
            },
            "names": {
                0: "read",
                1: "bash"
            },
            "tool_ids": {0: "tool_a", 1: "tool_b"}
        }
        result = repair_all_tools(state, "openai")

        self.assertEqual(len(result), 2)
        args0, _ = result[0]
        self.assertEqual(json.loads(args0), {"filePath": "a.py"})
        args1, status1 = result[1]
        self.assertEqual(status1, "patched")
        parsed1 = json.loads(args1)
        self.assertEqual(parsed1["command"], "grep foo bar.py")
        self.assertEqual(parsed1["description"], "[auto]")

    def test_trailing_braces_stripped(self):
        """Trailing {} artifacts (qwen3_coder parser quirk) are stripped."""
        state = {
            "args": {0: '{"content": "hello"}{}'},
            "names": {0: "write"},
            "tool_ids": {},
        }
        result = repair_all_tools(state, "anthropic")

        args, _ = result[0]
        parsed = json.loads(args)
        self.assertEqual(parsed, {"content": "hello"})

    def test_unknown_tool_name_skips_patch(self):
        """Tool name not in TOOL_PATCHES passes through with 'ok' status."""
        state = {
            "args": {0: '{"anything": "goes"}'},
            "names": {0: "unknown_tool"},
            "tool_ids": {},
        }
        result = repair_all_tools(state, "openai")

        args, status = result[0]
        self.assertEqual(args, '{"anything": "goes"}')
        self.assertEqual(status, "ok")

    def test_empty_args_state_returns_empty(self):
        """No accumulated args produces empty result."""
        state = {
            "args": {},
            "names": {},
            "tool_ids": {},
        }
        result = repair_all_tools(state, "anthropic")
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
