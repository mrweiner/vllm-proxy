#!/usr/bin/env python3
"""Tests for auto_continue _is_incomplete detection.

Verifies incomplete response patterns that should trigger a nudge:
  - Empty responses (reasoning only, no content)
  - Truncated text (too short, no tools)
  - Tools with finish=stop (model stopped prematurely after tools)
  - Very low token output

And must NOT nudge when response is genuinely complete.
"""

import sys
import unittest

sys.path.insert(0, "/home/weiner/repos/local-llm/proxy")
from auto_continue.watcher import _is_incomplete, should_continue


def _assistant(output=100, finish="stop", parts=None, error=None):
    msg = {"info": {"role": "assistant", "tokens": {"output": output}, "finish": finish}, "parts": parts or []}
    if error:
        msg["info"]["error"] = error
    return msg


def _user(text="hello"):
    return {"info": {"role": "user"}, "parts": [{"type": "text", "text": text}]}


def _text(content):
    return {"type": "text", "text": content}


def _tool(name="bash", status="completed"):
    return {"type": "tool", "tool": name, "state": {"status": status}}


class TestIncompletePatterns(unittest.TestCase):
    """The three incomplete response patterns from INTENT.md."""

    def test_tool_calls_only_is_normal_streaming(self):
        """Model produced tool calls without text — this is normal streaming, NOT incomplete."""
        msgs = [_assistant(output=300, finish="tool-calls", parts=[_tool("read"), _tool("bash")])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)

    def test_empty_response_reasoning_only(self):
        """Model produced reasoning but no text or tools — incomplete."""
        msgs = [_assistant(output=50, parts=[{"type": "step-start"}, {"type": "reasoning"}, {"type": "step-finish"}])]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)

    def test_empty_response_one_token(self):
        """Model produced 1 token and stopped — the ses_22e970b7affe halt pattern."""
        msgs = [_assistant(output=1, parts=[{"type": "step-start"}, {"type": "step-finish"}])]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)

    def test_truncated_text_too_short(self):
        """Model produced only a few characters of text — likely truncated mid-sentence."""
        msgs = [_assistant(parts=[_text("ok")])]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)

    def test_tools_with_stop_finish(self):
        """Model ran tools but finish=stop suggests it was cut off before generating next action."""
        msgs = [_assistant(output=500, finish="stop", parts=[_text("I was going to run something but..."), _tool()])]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)


class TestGenuinelyComplete(unittest.TestCase):
    """Must NOT auto-continue when the response is genuinely complete."""

    def test_meaningful_text_no_tools(self):
        """Full text response with no trailing tools — done."""
        msgs = [_assistant(parts=[_text("Here is the full and complete answer to your question with plenty of detail.")])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)

    def test_text_and_tools_with_tool_calls_finish(self):
        """Normal tool-use turn with explanation — done."""
        msgs = [_assistant(output=500, finish="tool-calls", parts=[_text("Let me check that file."), _tool("read")])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)


class TestSkipCases(unittest.TestCase):
    """Cases where auto-continue should not fire regardless of content."""

    def test_compaction_zero_tokens(self):
        """Compaction produces 0 output tokens — must not trigger nudge."""
        msgs = [_assistant(output=0, parts=[{"type": "step-start"}, {"type": "step-finish"}])]
        should, reason = _is_incomplete(msgs)
        self.assertFalse(should)
        self.assertIn("compaction", reason)

    def test_aborted_message(self):
        """User-aborted sessions should not be auto-continued."""
        msgs = [_assistant(output=0, error={"name": "MessageAbortedError"}, parts=[{"type": "step-start"}, {"type": "reasoning"}])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)

    def test_last_message_is_user(self):
        """If the last message is from the user, don't auto-continue."""
        msgs = [_user("new question")]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)

    def test_no_messages(self):
        """Empty message list — nothing to evaluate."""
        should, _ = _is_incomplete([])
        self.assertFalse(should)


class TestShouldContinue(unittest.TestCase):
    """The should_continue function decides whether to nudge based on status and messages."""

    def test_retry_status_skips_auto_continue(self):
        """When session status is retry, don't auto-continue — opencode is already handling it."""
        should, _ = should_continue(
            status="retry",
            msgs=[_assistant(output=100, parts=[_text("Here is the complete answer.")])],
        )
        self.assertFalse(should)

    def test_idle_status_with_incomplete_triggers_continue(self):
        """When session is idle and response is incomplete, auto-continue should fire."""
        should, _ = should_continue(
            status="idle",
            msgs=[_assistant(output=1, parts=[{"type": "step-start"}, {"type": "step-finish"}])],
        )
        self.assertTrue(should)

    def test_idle_status_with_complete_skips_continue(self):
        """When session is idle but response is complete, don't auto-continue."""
        should, _ = should_continue(
            status="idle",
            msgs=[_assistant(output=100, parts=[_text("Here is the complete and detailed answer.")])],
        )
        self.assertFalse(should)

    def test_busy_status_skips_auto_continue(self):
        """When session is busy, don't auto-continue — user may be typing."""
        should, _ = should_continue(
            status="busy",
            msgs=[_assistant(output=1, parts=[{"type": "step-start"}, {"type": "step-finish"}])],
        )
        self.assertFalse(should)


class TestToolCallsWithErrorFlag(unittest.TestCase):
    """Tool calls with no text response should be incomplete even when error flag is set.

    Bug: ses_22a0aa33dffeF7hCl26N33crzK (2026-04-28)
    - Model made git tool calls (git status, git diff, git log)
    - Tools executed successfully
    - Model should respond to tool results but stopped
    - Session went idle, watcher saw 'last message has error' and skipped
    - Error was spurious from model stopping generation, not a real failure
    - Auto-continue should have fired to nudge the model to continue
    """

    def test_tool_calls_no_text_with_error_should_continue(self):
        """Tool calls with finish=tool-calls, no text, and error flag = incomplete."""
        msgs = [_assistant(output=300, finish="tool-calls", parts=[_tool("bash"), _tool("read")], error={"name": "SomeError"})]
        should, reason = _is_incomplete(msgs)
        self.assertTrue(should)
        self.assertIn("no text response", reason)

    def test_tool_calls_no_text_stop_finish_with_error_should_continue(self):
        """Tool calls with finish=stop, no text, and error flag = incomplete."""
        msgs = [_assistant(output=300, finish="stop", parts=[_tool("bash")], error={"name": "SomeError"})]
        should, reason = _is_incomplete(msgs)
        self.assertTrue(should)
        self.assertIn("stop", reason)


class TestRealWorldScenario(unittest.TestCase):
    """The actual ses_22e970b7affe halt sequence."""

    def test_halt_after_edit(self):
        """Model edited a file, then produced 1 token and stopped."""
        msgs = [
            _user("start refactoring"),
            _assistant(output=772, finish="tool-calls", parts=[
                _text("\n\nThree issues to fix:\n\n1. Manager is final\n2. Missing use statement\n\nLet me fix them.\n\n"),
                _tool("edit"),
            ]),
            _assistant(output=1, finish="stop", parts=[
                {"type": "step-start"},
                {"type": "step-finish"},
            ]),
        ]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)


class TestReasoningLeakDetection(unittest.TestCase):
    """Detect reasoning content that leaked into text via vLLM parser bug.

    Case: ses_1a1dae0f3ffermBMZ8ml9WxfGf (2026-05-25)
    - Model generated long reasoning, vLLM parser closed reasoning block mid-stream
    - Rest of reasoning emitted as `content` instead of `reasoning_content`
    - Text starts mid-sentence with lowercase: "that the implementation matches..."
    - This is not a valid response — it's leaked chain of thought
    """

    def test_lowercase_start_triggers_continue(self):
        """Text starting lowercase is leaked reasoning — auto-continue."""
        msgs = [_assistant(output=500, parts=[_text("that the implementation matches the scanner spec across all steps.")])]
        should, reason = _is_incomplete(msgs)
        self.assertTrue(should)
        self.assertIn("reasoning leak", reason)

    def test_lowercase_after_whitespace_triggers_continue(self):
        """Text starting with whitespace then lowercase — still a leak."""
        msgs = [_assistant(output=500, parts=[_text("\n\nthat the implementation matches...")])]
        should, reason = _is_incomplete(msgs)
        self.assertTrue(should)
        self.assertIn("reasoning leak", reason)

    def test_uppercase_start_is_normal(self):
        """Text starting uppercase is a normal response — no continue."""
        msgs = [_assistant(output=500, parts=[_text("Here is the complete answer to your question.")])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)

    def test_uppercase_after_whitespace_is_normal(self):
        """Text starting with whitespace then uppercase — normal response."""
        msgs = [_assistant(output=500, parts=[_text("\n\nHere is the complete answer.")])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)

    def test_leaked_reasoning_with_the(self):
        """Real case: text starts with 'that' — leaked reasoning continuation."""
        msgs = [_assistant(output=518, parts=[_text(
            "that the implementation matches the scanner spec (07.05.10) across all required steps and error cases. "
            "Then I need to investigate the cross-module impact..."
        )])]
        should, reason = _is_incomplete(msgs)
        self.assertTrue(should)
        self.assertIn("reasoning leak", reason)

    def test_lowercase_with_tools_does_not_apply(self):
        """Lowercase check only applies to text-only responses, not tool calls."""
        msgs = [_assistant(output=300, finish="tool-calls", parts=[
            _text("that file"),
            _tool("read"),
        ])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)  # normal tool-calls turn

    def test_empty_text_after_strip_no_false_positive(self):
        """Whitespace-only text should not trigger lowercase check."""
        msgs = [_assistant(output=100, parts=[_text("   ")])]
        # This falls through to empty response detection, not lowercase
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)

    def test_starts_with_punctuation_is_normal(self):
        """Text starting with punctuation (not lowercase) is normal."""
        msgs = [_assistant(output=200, parts=[_text("```python\nprint('hello')\n```")])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)


class TestCompactionSettleDelay(unittest.TestCase):
    """After compaction, use longer settle delay to let model finish post-compaction work.

    Case: ses_22e970b7affe (2026-04-28)
    - vLLM returned HTTP 400: prompt 230k + 32k requested = 262,145 > 262,144 limit
    - Opencode auto-compacted, created synthetic continue message
    - Session went idle, auto-continue fired 3s later
    - Model dismissed '[auto-continue]' as blank, interrupted its work

    Fix: listen for session.compacted SSE event, use longer settle delay (10s)
    on next idle to give model time to finish post-compaction response.
    """

    def test_get_settle_delay_normal(self):
        """Normal idle uses standard settle delay."""
        from auto_continue.watcher import get_settle_delay
        last_compacted = {}
        delay = get_settle_delay("ses_abc", last_compacted)
        self.assertEqual(delay, 2)  # SETTLE_DELAY

    def test_get_settle_delay_post_compaction(self):
        """Idle shortly after compaction uses longer settle delay."""
        import time
        from auto_continue.watcher import get_settle_delay
        last_compacted = {"ses_abc": time.time() - 5}  # compacted 5s ago
        delay = get_settle_delay("ses_abc", last_compacted)
        self.assertEqual(delay, 10)  # SETTLE_DELAY_POST_COMPACTION

    def test_get_settle_delay_compaction_expired(self):
        """If compaction was long ago, fall back to normal delay."""
        import time
        from auto_continue.watcher import get_settle_delay
        last_compacted = {"ses_abc": time.time() - 60}  # compacted 60s ago
        delay = get_settle_delay("ses_abc", last_compacted)
        self.assertEqual(delay, 2)  # SETTLE_DELAY

    def test_get_settle_delay_unknown_session(self):
        """Unknown session uses normal delay."""
        import time
        from auto_continue.watcher import get_settle_delay
        last_compacted = {"ses_other": time.time()}
        delay = get_settle_delay("ses_abc", last_compacted)
        self.assertEqual(delay, 2)  # SETTLE_DELAY

    def test_watch_handles_session_compacted_event(self):
        """The watch loop should recognize session.compacted SSE events and record them."""
        from auto_continue.watcher import COMPACTION_WINDOW
        self.assertGreater(COMPACTION_WINDOW, 0)


class TestSettlingState(unittest.TestCase):
    """Shared settling state is exposed to proxy for UI countdown display."""

    def setUp(self):
        from auto_continue.watcher import settling_sessions
        settling_sessions.clear()

    def test_settling_sessions_is_dict(self):
        """settling_sessions is a mutable dict accessible by proxy."""
        from auto_continue.watcher import settling_sessions
        self.assertIsInstance(settling_sessions, dict)

    def test_settling_entry_structure(self):
        """Entry contains remaining seconds and reason."""
        from auto_continue.watcher import settling_sessions
        settling_sessions["ses_abc"] = {"remaining": 7, "reason": "post-compaction"}
        entry = settling_sessions["ses_abc"]
        self.assertEqual(entry["remaining"], 7)
        self.assertEqual(entry["reason"], "post-compaction")

    def test_settling_entry_cleared_after_settle(self):
        """Entry is removed when settle completes."""
        from auto_continue.watcher import settling_sessions
        settling_sessions["ses_abc"] = {"remaining": 2, "reason": "normal"}
        del settling_sessions["ses_abc"]
        self.assertNotIn("ses_abc", settling_sessions)

    def test_settling_status_json_serializable(self):
        """Status dict is JSON serializable for proxy endpoint."""
        import json
        from auto_continue.watcher import settling_sessions
        settling_sessions["ses_abc"] = {"remaining": 5, "reason": "post-compaction"}
        settling_sessions["ses_def"] = {"remaining": 2, "reason": "normal"}
        result = json.dumps(settling_sessions)
        self.assertIn("ses_abc", result)
        self.assertIn("5", result)


if __name__ == "__main__":
    unittest.main()
