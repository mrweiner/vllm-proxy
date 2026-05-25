#!/usr/bin/env python3
"""Tests for fix_assistant_whitespace_content (attention dilution fix)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_proxy.policies import fix_assistant_whitespace_content


def test_injects_after_tool_results():
    """Injects when conversation has tool results and model is about to respond."""
    req = {"messages": [
        {"role": "user", "content": "What does this file do?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "content": "file contents", "tool_call_id": "c1"},
        {"role": "tool", "content": "more contents", "tool_call_id": "c1"},
    ]}
    result = fix_assistant_whitespace_content(req)
    msgs = result["messages"]
    assert msgs[-1]["role"] == "user"
    assert "What does this file do?" in msgs[-1]["content"]


def test_injects_single_tool_result():
    """Injects even with single tool result."""
    req = {"messages": [
        {"role": "user", "content": "Read this file."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "content": "contents", "tool_call_id": "c1"},
    ]}
    result = fix_assistant_whitespace_content(req)
    assert result["messages"][-1]["role"] == "user"
    assert "Read this file" in result["messages"][-1]["content"]


def test_no_inject_without_tools():
    """Does not inject when no tool results in conversation."""
    req = {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]}
    result = fix_assistant_whitespace_content(req)
    assert len(result["messages"]) == 2, "Should not inject without tools"


def test_no_inject_on_user_message():
    """Does not inject when last message is a user message."""
    req = {"messages": [
        {"role": "user", "content": "What does this file do?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "content": "file contents", "tool_call_id": "c1"},
        {"role": "user", "content": "Can you also check the related files?"},
    ]}
    result = fix_assistant_whitespace_content(req)
    assert len(result["messages"]) == 4, "Should not inject when user is driving"


def test_no_inject_without_user():
    """Does not inject when no user message exists."""
    req = {"messages": [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "content": "contents", "tool_call_id": "c1"},
    ]}
    result = fix_assistant_whitespace_content(req)
    assert len(result["messages"]) == 2, "Should not inject without user message"


def test_truncates_long_user_message():
    """User reminder truncates long user messages."""
    long_msg = "x" * 300
    req = {"messages": [
        {"role": "user", "content": long_msg},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "content": "contents", "tool_call_id": "c1"},
    ]}
    result = fix_assistant_whitespace_content(req)
    reminder = result["messages"][-1]["content"]
    assert len(reminder) < 250, f"Reminder should be truncated, got {len(reminder)} chars"


def test_no_inject_after_30_messages():
    """Does not inject when conversation has 30+ messages (too late for dilution)."""
    messages = [{"role": "user", "content": "What does this file do?"}]
    # Build 28 more messages to reach 30 total
    for i in range(14):
        messages.append({"role": "assistant", "content": f"response {i}"})
        messages.append({"role": "user", "content": f"follow up {i}"})
    # Add tool results so the other conditions are met
    messages[-1] = {"role": "tool", "content": "file contents", "tool_call_id": "c1"}
    messages.append({"role": "assistant", "content": "I see the results."})
    # Total: 30 messages
    assert len(messages) == 30
    req = {"messages": messages}
    result = fix_assistant_whitespace_content(req)
    assert len(result["messages"]) == 30, f"Should not inject at 30 messages, got {len(result['messages'])}"


def test_injects_after_assistant_no_tools():
    """Injects when last message is assistant without tool_calls."""
    req = {"messages": [
        {"role": "user", "content": "What does this file do?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "content": "file contents", "tool_call_id": "c1"},
        {"role": "assistant", "content": "I see the file contents."},
    ]}
    result = fix_assistant_whitespace_content(req)
    assert result["messages"][-1]["role"] == "user"
    assert "What does this file do?" in result["messages"][-1]["content"]


if __name__ == "__main__":
    test_injects_after_tool_results()
    print("OK: test_injects_after_tool_results")
    test_injects_single_tool_result()
    print("OK: test_injects_single_tool_result")
    test_no_inject_without_tools()
    print("OK: test_no_inject_without_tools")
    test_no_inject_on_user_message()
    print("OK: test_no_inject_on_user_message")
    test_no_inject_without_user()
    print("OK: test_no_inject_without_user")
    test_truncates_long_user_message()
    print("OK: test_truncates_long_user_message")
    test_no_inject_after_30_messages()
    print("OK: test_no_inject_after_30_messages")
    test_injects_after_assistant_no_tools()
    print("OK: test_injects_after_assistant_no_tools")
    print("\nAll tests passed!")
