#!/usr/bin/env python3
"""Tests for policy injection into chat completion requests."""

import unittest
from vllm_proxy.policies import inject_policies, build_policy_block, SYSTEM_POLICIES


class TestBuildPolicyBlock(unittest.TestCase):
    def test_returns_concatenated_policies(self):
        block = build_policy_block()
        self.assertIsNotNone(block)
        self.assertIn("<plan-mode-policy>", block)
        self.assertIn("<tdd-policy>", block)

    def test_returns_none_when_no_policies(self):
        original = SYSTEM_POLICIES[:]
        SYSTEM_POLICIES.clear()
        try:
            self.assertIsNone(build_policy_block())
        finally:
            SYSTEM_POLICIES[:] = original


class TestInjectPolicies(unittest.TestCase):
    def test_appends_to_existing_system_message(self):
        req = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
        }
        result = inject_policies(req)
        self.assertIs(result, req)
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertIn("You are helpful.", result["messages"][0]["content"])
        self.assertIn("<plan-mode-policy>", result["messages"][0]["content"])

    def test_prepends_system_message_when_none_exists(self):
        req = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        }
        result = inject_policies(req)
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertIn("<plan-mode-policy>", result["messages"][0]["content"])
        self.assertEqual(result["messages"][1]["role"], "user")

    def test_returns_unchanged_when_no_messages(self):
        req = {"model": "test"}
        result = inject_policies(req)
        self.assertIs(result, req)
        self.assertNotIn("messages", result)

    def test_returns_unchanged_when_empty_policies(self):
        original = SYSTEM_POLICIES[:]
        SYSTEM_POLICIES.clear()
        try:
            req = {"messages": [{"role": "user", "content": "hi"}]}
            result = inject_policies(req)
            self.assertIs(result, req)
            self.assertEqual(len(result["messages"]), 1)
        finally:
            SYSTEM_POLICIES[:] = original

    # Anthropic: system as string
    def test_anthropic_sets_system_field(self):
        req = {
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = inject_policies(req, fmt="anthropic")
        self.assertIn("system", result)
        self.assertIn("<plan-mode-policy>", result["system"])
        self.assertNotIn("system", [m.get("role") for m in result["messages"]])

    def test_anthropic_appends_to_existing_system_string(self):
        req = {
            "system": "Be concise.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = inject_policies(req, fmt="anthropic")
        self.assertIn("Be concise.", result["system"])
        self.assertIn("<plan-mode-policy>", result["system"])

    # Anthropic: system as list of content blocks (what opencode actually sends)
    def test_anthropic_appends_to_system_content_block_list(self):
        req = {
            "system": [{"type": "text", "text": "Be concise."}],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = inject_policies(req, fmt="anthropic")
        # Should still be a list with an appended text block
        self.assertIsInstance(result["system"], list)
        texts = [b.get("text", "") for b in result["system"] if b.get("type") == "text"]
        joined = "\n".join(texts)
        self.assertIn("Be concise.", joined)
        self.assertIn("<plan-mode-policy>", joined)

    def test_anthropic_creates_system_from_message_system_block(self):
        req = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "Existing system."}]},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = inject_policies(req, fmt="anthropic")
        # System should be created as content block list (preserving original format)
        self.assertIn("system", result)
        self.assertIsInstance(result["system"], list)
        self.assertIn("<plan-mode-policy>", result["system"][-1].get("text", ""))


if __name__ == "__main__":
    unittest.main()
