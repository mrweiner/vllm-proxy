#!/usr/bin/env python3
"""System prompt policy injection for chat completion requests.

Policies are loaded from config.toml [policies].system_prompts.
Falls back to hardcoded defaults if no config file is found.
"""

import logging

from . import config

log = logging.getLogger("vllm-proxy.policies")

_DEFAULT_POLICIES: list[str] = [
    """
<tdd-policy>
When writing new code or implementing features, use test-driven development:
write a failing test first, then implement minimal code to pass. One test at a time.

Skip TDD only for: configuration changes, trivial one-liners, documentation,
refactoring existing tested code, or when the user explicitly says so.
</tdd-policy>
""".strip(),

    """
<plan-mode-policy>
Before making decisions to write or edit files, consider whether you may be in
a planning or brainstorming mode based on conversational context. When
in doubt, ask for clarification on desired actions.
</plan-mode-policy>
""".strip(),
]

SYSTEM_POLICIES: list[str] = []


def _load_policies() -> None:
    global SYSTEM_POLICIES
    prompts = config.toml_list("policies", "system_prompts")
    if prompts:
        SYSTEM_POLICIES = [str(p).strip() for p in prompts if str(p).strip()]
    else:
        SYSTEM_POLICIES = _DEFAULT_POLICIES


_load_policies()


def build_policy_block() -> str | None:
    """Concatenate all active policies into a single string, or None if empty."""
    active = [p for p in SYSTEM_POLICIES if p.strip()]
    if not active:
        return None
    return "\n\n".join(active)


def _extract_anthropic_system_text(system: object) -> str:
    """Extract plain text from Anthropic system field (string or content block list)."""
    if isinstance(system, str):
        return system.strip()
    if isinstance(system, list):
        texts = [b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(texts).strip()
    return ""


def inject_policies(req_json: dict, fmt: str = "openai") -> dict:
    """
    Inject SYSTEM_POLICIES into the system prompt of a chat completions request.

    For OpenAI format: appends to existing system message or prepends one.
    For Anthropic format: appends to the top-level `system` field, preserving
    its type (string or list of content blocks).
    """
    policy_block = build_policy_block()
    if not policy_block:
        return req_json

    if fmt == "anthropic":
        system = req_json.get("system")
        if system is None:
            # Check if there's a system message in the messages array to extract
            messages = req_json.get("messages", [])
            for msg in messages:
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        req_json["system"] = content
                    elif isinstance(content, str):
                        req_json["system"] = [{"type": "text", "text": content}]
                    break

            # Re-read system after potential extraction
            system = req_json.get("system")

        if isinstance(system, str):
            req_json["system"] = system + "\n\n" + policy_block if system.strip() else policy_block
        elif isinstance(system, list):
            req_json["system"].append({"type": "text", "text": policy_block})
        else:
            req_json["system"] = policy_block
        log.debug("inject_policies: set Anthropic system field")
    else:
        messages = req_json.get("messages", [])
        if not messages:
            return req_json

        if messages[0].get("role") == "system":
            messages[0]["content"] += "\n\n" + policy_block
            log.debug("inject_policies: appended to existing system message")
        else:
            messages.insert(0, {"role": "system", "content": policy_block})
            log.debug("inject_policies: prepended new system message")

    return req_json


# ── Attention dilution fix: Inject synthetic user reminder after tool results ──
#
# Bug: Qwen3.6-27B via vLLM ignores the original user message after 2 rounds of
# tool calls, responding with "The user hasn't asked a specific question yet"
# despite the user message being clearly present in the request at index 1.
#
# Root cause: The model's attention mechanism cannot effectively attend back to
# the user message (position 1) after processing ~23K chars of tool results
# (positions 3, 4, 6, 7). The model loses track of the original question.
#
# Hypotheses tested (all in this function, evolved over iterations):
#   Test A: Rewrite `content: '\n\n'` → `content: null` for assistant+tool_calls
#     Result: Failed (~55% failure rate unchanged)
#     Conclusion: Not a ChatML template whitespace handling issue
#   Test C: Rewrite `content: '\n\n'` → `content: "Proceeding with tool calls."`
#     Result: Failed (~16% failure, possibly statistical outlier)
#     Conclusion: Not about assistant text content presence
#   Test D: Inject synthetic user reminder after consecutive tool results at end
#     Result: **Passed (30/30, 0% failure vs 55% baseline)**
#     Conclusion: Root cause is attention dilution; fix re-anchors attention
#
# Fix: After detecting the pattern [assistant(tool_calls), tool, tool] at the
# end of the message array, inject a synthetic user-role message containing a
# truncated reference to the original user question. This resets the model's
# attention to the most recent user input, avoiding the need to attend back
# to position 1 after processing 23K+ chars of intermediate tool content.
#
# Request structure before fix (8 messages, fails ~55%):
#   [0] system (40KB), [1] user (225 chars), [2] assistant+tools,
#   [3] tool, [4] tool, [5] assistant+tools, [6] tool, [7] tool
#
# Request structure after fix (9 messages, fails 0%):
#   [0] system, [1] user, [2] assistant+tools, [3] tool, [4] tool,
#   [5] assistant+tools, [6] tool, [7] tool,
#   [8] user: "[Context: continuing original request: ...]"
#
# See vllm-qwen3-tool-call-findings.md for full investigation details.

def fix_assistant_whitespace_content(req_json: dict, fmt: str = "openai") -> dict:
    """
    Fix attention dilution by injecting a user reminder before the model responds.

    Injects when:
    - There's a user message to reference (original question)
    - There are tool results in the conversation (model has processed tool output)
    - The last message is NOT a user message (model is about to respond)

    Does NOT inject when:
    - No user message exists (nothing to remind about)
    - No tool results in conversation (no dilution risk)
    - Last message is a user message (user is driving, model should respond to it)
    """
    messages = req_json.get("messages", [])
    if not messages:
        return req_json

    # Find the original user message to reference in the reminder
    user_message = None
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            user_message = msg.get("content")
            break
    if not user_message:
        return req_json

    # Only inject early in conversation (first 30 messages)
    if len(messages) >= 30:
        return req_json

    # Only inject if there are tool results (dilution risk) and last msg is not user
    has_tools = any(m.get("role") == "tool" for m in messages)
    if not has_tools:
        return req_json
    if messages[-1].get("role") == "user":
        return req_json

    # Inject reminder at the end
    reminder = f"\n\n<context-original-request-noop>{user_message[:150]}...</context-original-request-noop>"
    new_messages = list(messages)
    new_messages.append({
        "role": "user",
        "content": reminder,
    })
    req_json["messages"] = new_messages
    log.debug("fix_assistant_whitespace_content: inserted user reminder")
    return req_json
