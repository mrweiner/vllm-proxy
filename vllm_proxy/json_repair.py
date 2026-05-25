#!/usr/bin/env python3
"""JSON repair utilities for handling truncated or malformed JSON in tool call arguments."""

import json
import logging

log = logging.getLogger("vllm-proxy.json_repair")


def repair_json(s: str) -> tuple[str, str | None]:
    """
    Close any unclosed { or [ in a JSON string.
    Returns (repaired_string, suffix_appended) or (original, None) if no repair needed.

    Also strips trailing {} artifacts and leading duplicate braces.
    """
    if not s or not s.strip():
        return s, None
    try:
        json.loads(s)
        return s, None
    except (json.JSONDecodeError, ValueError):
        pass

    working = s

    # Strip trailing {} artifacts (qwen3_coder parser)
    stripped = False
    while working.endswith('{}'):
        working = working[:-2].rstrip()
        stripped = True
    if stripped:
        try:
            json.loads(working)
            log.debug("repair_json: stripped trailing {} artifact")
            return working, None
        except (json.JSONDecodeError, ValueError):
            pass
        if not stripped:
            working = s

    # Strip leading duplicate braces + matching trailing braces
    if len(working) > 1 and working[0] == '{' and working[1] == '{':
        candidate = _strip_leading_braces(working)
        if candidate != working:
            working = candidate
            try:
                json.loads(candidate)
                log.debug("repair_json: stripped leading duplicate brace(s)")
                return candidate, None
            except (json.JSONDecodeError, ValueError):
                pass

    # Close any remaining unclosed brackets
    suffix = _close_brackets(working)
    if suffix is None:
        return s, None

    repaired = working + suffix
    try:
        json.loads(repaired)
        return repaired, suffix
    except (json.JSONDecodeError, ValueError):
        return s, None


def _close_brackets(s: str) -> str | None:
    """Count unclosed brackets and return the suffix needed to close them,
    or None if already balanced."""
    stack: list[str] = []
    in_string = False
    escape_next = False

    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append('}' if ch == '{' else ']')
        elif ch in ('}', ']') and stack:
            stack.pop()

    if not stack:
        return None

    return "".join(reversed(stack))


def _strip_leading_braces(s: str) -> str:
    """Strip excess leading braces and matching trailing braces.

    LLM sometimes emits {{"key": "val"}} or {{"key": "val"}} — this removes
    the leading duplicates and corresponding trailing ones.
    """
    head = 0
    while head < len(s) and s[head] == '{':
        head += 1

    tail = 0
    while tail < len(s) and s[-(1 + tail)] == '}':
        tail += 1

    # Try stripping head and tail together
    for i in range(1, head + 1):
        for j in range(0, tail + 1):
            candidate = s[i:len(s) - j] if j else s[i:]
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass

    return s


def strip_think_tags(s: str) -> tuple[str, bool]:
    import re
    cleaned = re.sub(r'\s*</?think[^>]*>\s*', '', s, flags=re.IGNORECASE)
    return cleaned, cleaned != s


def deduplicate_json(s: str) -> tuple[str, bool]:
    s = s.strip()
    if not s.startswith('{'):
        return s, False
    try:
        obj, end = json.JSONDecoder().raw_decode(s)
        remainder = s[end:].strip()
        if remainder:
            first = json.dumps(obj, separators=(',', ':'))
            return first, True
    except (json.JSONDecodeError, ValueError):
        pass
    return s, False
