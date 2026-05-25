#!/usr/bin/env python3
"""Tests for conversational intent-to-continue detection.

New pattern: Agent sends a text-only response expressing intent to keep
working but doesn't actually execute tools or continue. E.g.:
  "I need to dig into a few more areas. Let me check the except Exception
   policy violation, the spec compliance, and the test warning."

This looks complete to the current logic (has text, no tools, finish=stop)
but the agent clearly intends to keep going.
"""

import sys
import unittest

sys.path.insert(0, "/home/weiner/repos/local-llm/proxy")
from auto_continue.watcher import _is_incomplete, _looks_like_intent_to_continue


def _assistant(output=100, finish="stop", parts=None, error=None):
    msg = {"info": {"role": "assistant", "tokens": {"output": output}, "finish": finish}, "parts": parts or []}
    if error:
        msg["info"]["error"] = error
    return msg


def _text(content):
    return {"type": "text", "text": content}


class TestIntentToContinuePhrases(unittest.TestCase):
    """Messages that express intent to continue should trigger auto-continue."""

    def test_let_me_check(self):
        """'Let me check...' is a clear intent to continue."""
        msgs = [_assistant(output=30, parts=[_text("I need to dig into a few more areas. Let me check the except Exception policy violation, the spec compliance, and the test warning.")])]
        should, reason = _is_incomplete(msgs)
        self.assertTrue(should, f"Expected continue, got: {reason}")

    def test_i_need_to(self):
        """'I need to...' expresses intent to continue."""
        msgs = [_assistant(output=25, parts=[_text("I need to verify this change doesn't break anything.")])]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)

    def test_i_should_check(self):
        """'I should check...' expresses intent to continue."""
        msgs = [_assistant(output=20, parts=[_text("That looks right. I should check the other files too.")])]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)

    def test_lets_me_look(self):
        """'Let me look...' expresses intent to continue."""
        msgs = [_assistant(output=15, parts=[_text("Let me look at the test results first.")])]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)

    def test_i_will_now(self):
        """'I will now...' expresses intent to do something next."""
        msgs = [_assistant(output=20, parts=[_text("Great. I will now update the configuration file.")])]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)

    def test_next_i_ll(self):
        """'Next, I'll...' expresses sequential intent."""
        msgs = [_assistant(output=25, parts=[_text("Done with that. Next, I'll review the error handling.")])]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)

    def test_let_me_run(self):
        """'Let me run...' expresses intent to execute something."""
        msgs = [_assistant(output=15, parts=[_text("Let me run the tests to make sure everything passes.")])]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)

    def test_i_gonna(self):
        """'I'm going to...' informal intent expression."""
        msgs = [_assistant(output=20, parts=[_text("Alright, I'm going to fix the remaining issues now.")])]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)

    def test_ending_with_colon(self):
        """Message ending with colon (listing items to check) is intent to continue."""
        msgs = [_assistant(output=20, parts=[_text("Areas to review:\n1. Policy compliance\n2. Error handling\n3. Edge cases")])]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)

    def test_next_steps(self):
        """'Next steps:' is an intent outline."""
        msgs = [_assistant(output=25, parts=[_text("Next steps:\n- Check the logs\n- Review the config\n- Run validation")])]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)

    def test_multiple_text_parts(self):
        """Intent phrase in one of multiple text parts should still trigger."""
        msgs = [_assistant(output=50, parts=[
            _text("The first issue seems resolved."),
            _text("Let me now check the second issue."),
        ])]
        should, _ = _is_incomplete(msgs)
        self.assertTrue(should)


class TestGenuinelyCompleteNoIntent(unittest.TestCase):
    """Messages that are complete answers should NOT trigger auto-continue."""

    def test_complete_answer(self):
        """A complete answer with no forward intent is done."""
        msgs = [_assistant(output=200, parts=[_text("The issue was a missing import statement on line 42. Adding `import os` resolved it. The test now passes.")])]
        should, reason = _is_incomplete(msgs)
        self.assertFalse(should, f"Should be complete, got: {reason}")

    def test_summary_without_action(self):
        """Summary statement without action intent is complete."""
        msgs = [_assistant(output=150, parts=[_text("Here's a summary of the changes made:\n1. Fixed the bug in parser.py\n2. Updated tests\n3. All green")])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)

    def test_here_is_followed_by_content(self):
        """'Here is...' followed by actual content is complete, not an intent."""
        msgs = [_assistant(output=200, parts=[_text("Here is the implementation you asked for:\n\ndef foo():\n    return bar")])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)

    def test_let_me_in_past_tense(self):
        """'Let me know' is not intent to continue, it's conversational close."""
        msgs = [_assistant(output=100, parts=[_text("I've completed all the changes. Let me know if you need anything else.")])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)

    def test_short_greeting(self):
        """Short greetings without substance are still complete if they don't signal work ahead."""
        msgs = [_assistant(output=10, parts=[_text("Hello! How can I help you today?")])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)

    def test_explains_what_was_done(self):
        """Explaining completed work is NOT intent to continue."""
        msgs = [_assistant(output=300, parts=[_text("I need to explain what happened: the build failed because of a dependency conflict. I resolved it by pinning the version to 2.1.0 and rerunning the build. It succeeded.")])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)

    def test_conditional_let_me(self):
        """'Let me know if...' is not intent to continue working."""
        msgs = [_assistant(output=100, parts=[_text("All changes are complete. Let me know if you need any further modifications.")])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)

    def test_let_me_see_past_tense(self):
        """'Let me see what happened' in retrospective context is not forward intent."""
        msgs = [_assistant(output=200, parts=[_text("Let me see what happened here. The error occurred because the variable was undefined. I traced the issue to line 15 and the fix was adding proper initialization.")])]
        should, _ = _is_incomplete(msgs)
        self.assertFalse(should)


class TestIntentHelper(unittest.TestCase):
    """Direct tests on the _looks_like_intent_to_continue helper."""

    def test_basic_phrases(self):
        for text in [
            "Let me check the file.",
            "I need to verify this.",
            "I should look at the logs.",
            "Let me run the tests.",
            "Next, I'll update the config.",
        ]:
            self.assertTrue(_looks_like_intent_to_continue(text), f"Should detect intent: {text}")

    def test_complete_responses(self):
        for text in [
            "Here is the complete answer.",
            "Let me know if you need help.",
            "I've finished the task.",
            "All done. Let me know if you need changes.",
        ]:
            self.assertFalse(_looks_like_intent_to_continue(text), f"Should NOT detect intent: {text}")


if __name__ == "__main__":
    unittest.main()
