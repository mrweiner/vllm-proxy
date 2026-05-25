#!/usr/bin/env python3
"""Format adapters for normalizing tool call SSE chunks across API formats."""

from dataclasses import dataclass


@dataclass
class ToolCallDelta:
    """Single tool call fragment from an SSE chunk."""
    idx: int
    name: str | None
    args_fragment: str


@dataclass
class StreamFinish:
    """Finish signal from the upstream stream."""
    finish_reason: str


# ── Format detection ──────────────────────────────────────────────────────────

def detect_format(path: str) -> str:
    """Return 'openai', 'anthropic', or 'unknown' based on request path."""
    if "/chat/completions" in path:
        return "openai"
    if "/messages" in path:
        return "anthropic"
    return "unknown"


def is_chat_endpoint(path: str) -> bool:
    """Check if this is any known chat/completions endpoint."""
    return "/chat/completions" in path or "/messages" in path


# ── OpenAI adapter ────────────────────────────────────────────────────────────

def openai_extract(chunk: dict) -> tuple[list[ToolCallDelta], StreamFinish | None]:
    """Extract tool call deltas and finish signal from OpenAI-format chunk."""
    choices = chunk.get("choices", [])
    if not choices:
        return [], None

    choice = choices[0]
    finish = choice.get("finish_reason")
    delta = choice.get("delta", {})
    tool_calls = delta.get("tool_calls", [])

    deltas: list[ToolCallDelta] = []
    for tc in tool_calls:
        idx = tc.get("index", 0)
        func = tc.get("function", {})
        name = func.get("name")
        args_frag = func.get("arguments", "")
        deltas.append(ToolCallDelta(idx=idx, name=name, args_fragment=args_frag))

    finish_sig = StreamFinish(finish) if finish is not None else None
    return deltas, finish_sig


def openai_repair_inplace(chunk: dict, idx: int, repaired_args: str) -> None:
    """Write repaired JSON back into an OpenAI-format chunk.

    Uses setdefault to get a mutable reference to the stored list, and appends
    a new tool call entry if none exists (finish chunk often has empty delta).
    """
    choices = chunk.get("choices", [{}])
    if not choices:
        return
    delta = choices[0].setdefault("delta", {})
    tcs = delta.setdefault("tool_calls", [])
    injected = False
    for tc in tcs:
        if tc.get("index", idx) == idx:
            tc.setdefault("function", {})["arguments"] = repaired_args
            injected = True
    if not injected:
        tcs.append({
            "index": idx,
            "function": {"arguments": repaired_args},
        })


# ── Anthropic adapter ─────────────────────────────────────────────────────────

def anthropic_extract(chunk: dict) -> tuple[list[ToolCallDelta], StreamFinish | None]:
    """Extract tool call deltas and finish signal from Anthropic-format chunk."""
    # content_block_start with tool_use
    if chunk.get("type") == "content_block_start":
        cb = chunk.get("content_block", {})
        if cb.get("type") == "tool_use":
            idx = chunk.get("index", 0)
            name = cb.get("name")
            inp = cb.get("input", {})
            args_frag = "" if not inp else _to_json_string(inp)
            return [ToolCallDelta(idx=idx, name=name, args_fragment=args_frag)], None

    # content_block_delta with input_json_delta
    if chunk.get("type") == "content_block_delta":
        delta = chunk.get("delta", {})
        if delta.get("type") == "input_json_delta":
            idx = chunk.get("index", 0)
            partial = delta.get("partial_json", "")
            return [ToolCallDelta(idx=idx, name=None, args_fragment=partial)], None

    # message_delta with stop_reason
    if chunk.get("type") == "message_delta":
        stop = chunk.get("stop_reason")
        if stop:
            return [], StreamFinish(stop)

    return [], None


def build_anthropic_tool_use_chunk(
    idx: int, tool_id: str, tool_name: str, repaired_input: str
) -> dict:
    """Build a content_block_start chunk with repaired input."""
    import json
    try:
        input_obj = json.loads(repaired_input)
    except (json.JSONDecodeError, ValueError):
        input_obj = {"_raw": repaired_input}

    return {
        "type": "content_block_start",
        "index": idx,
        "content_block": {
            "type": "tool_use",
            "id": tool_id,
            "name": tool_name,
            "input": input_obj,
        },
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_json_string(obj: object) -> str:
    """Serialize an object to compact JSON string."""
    import json
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, separators=(",", ":"))
