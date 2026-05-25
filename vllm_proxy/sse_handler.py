#!/usr/bin/env python3
"""SSE stream handler with format-aware tool call repair."""

import json
import logging
from typing import Callable

from .format_adapters import (
    StreamFinish,
    ToolCallDelta,
    anthropic_extract,
    build_anthropic_tool_use_chunk,
    openai_extract,
    openai_repair_inplace,
)
from .json_repair import repair_json
from .tool_patches import apply_tool_patches

log = logging.getLogger("vllm-proxy.sse_handler")

WriteFn = Callable[[bytes], None]
State = dict  # args, names, tool_ids, buffered, openai_args_buffer


def _emit(data: dict, write: WriteFn) -> None:
    write(b"data: " + json.dumps(data, separators=(",", ":")).encode() + b"\n\n")


def create_stream_handler(fmt: str, write: WriteFn):
    """
    Create a streaming SSE handler that processes one line at a time.

    Returns (process_line, finalize) where:
      process_line(raw_bytes) — process one SSE line
      finalize() — flush any remaining buffered data

    This is designed for integration with server.py's socket read loop.
    """
    extract_fn = anthropic_extract if fmt == "anthropic" else openai_extract

    state: State = {
        "args": {},
        "names": {},
        "tool_ids": {},
        "buffered": {},
        "openai_args_buffer": {},
    }

    def process_line(raw: bytes) -> None:
        line = raw.rstrip(b"\r")
        if not line:
            return
        if line == b"data: [DONE]":
            write(b"data: [DONE]\n\n")
            return
        if not line.startswith(b"data: "):
            write(line + b"\n\n")
            return

        try:
            chunk = json.loads(line[6:])
        except (json.JSONDecodeError, ValueError):
            write(line + b"\n\n")
            return

        deltas, finish = extract_fn(chunk)

        for d in deltas:
            if d.name:
                state["names"][d.idx] = d.name
            if d.args_fragment:
                _accumulate_arg(state, d.idx, d.args_fragment)

        if fmt == "anthropic":
            _handle_anthropic_chunk(chunk, deltas, finish, write, state)
        else:
            if finish is not None and state["args"]:
                _handle_openai_finish(chunk, write, state)
            else:
                _handle_openai_non_finish(chunk, deltas, write, state)

    def finalize() -> None:
        if fmt == "anthropic" and state["buffered"]:
            _emit_repaired_tools(write, state)
            if state["buffered"]:
                _flush_buffered(write, state)
        if state["openai_args_buffer"]:
            # No finish chunk arrived — pass through buffered chunks as-is
            _flush_openai_args_buffer(write, state, strip_args=False)

    return process_line, finalize


def repair_all_tools(state: State, fmt: str) -> dict[int, tuple[str, str]]:
    """Run repair_json + apply_tool_patches for all accumulated tool args.

    Returns dict mapping idx to (final_args_string, patch_status).
    Patch status is one of: "ok", "patched", "invalid_json", "unrecoverable".
    """
    result = {}

    for idx, accumulated in state["args"].items():
        name = state["names"].get(idx, "?")

        repaired, suffix = repair_json(accumulated)
        if repaired != accumulated:
            log.warning("REPAIR tool[%d] '%s' — appended %r", idx, name, suffix)

        patched_args = repaired
        patched, status = apply_tool_patches(name, patched_args)

        if status == "patched":
            log.warning("PATCHED tool[%d] '%s' — overwrote args (%d chars)",
                        idx, name, len(patched))
            final_args = patched
        elif status == "invalid_json":
            log.error("INVALID_JSON tool[%d] '%s' — args not valid JSON: %.200s",
                      idx, name, repaired)
            final_args = repaired
        elif status == "unrecoverable":
            log.error("UNRECOVERABLE tool[%d] '%s' — required fields missing",
                      idx, name)
            final_args = repaired
        else:
            final_args = repaired

        result[idx] = (final_args, status)

    return result


def _accumulate_arg(state: State, idx: int, fragment: str) -> None:
    """Accumulate tool call arguments, detecting vLLM complete-resend duplication."""
    current = state["args"].get(idx, "")
    if current:
        combined = current + fragment
        is_combined_valid = _is_valid_json(combined)
        is_fragment_valid = _is_valid_json(fragment)
        if not is_combined_valid and is_fragment_valid:
            # vLLM re-sent complete JSON — replace instead of concatenate
            state["args"][idx] = fragment
            return
    state["args"][idx] = current + fragment


def _is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _handle_openai_non_finish(chunk, deltas, write, state):
    """For OpenAI non-finish chunks: buffer tool-args chunks, pass others through.

    Buffering args chunks prevents downstream from duplicating them when combined
    with the repaired args in the finish chunk. Chunks with only role/name pass
    through (the initial role+name chunk has empty arguments).
    """
    has_args = any(d.args_fragment for d in deltas)
    if has_args:
        for d in deltas:
            if d.args_fragment:
                state["openai_args_buffer"].setdefault(d.idx, []).append(chunk)
                return

    # Non-tool-call chunks (text, role, etc.) pass through immediately
    _emit(chunk, write)


def handle_sse_stream(raw: bytes, fmt: str, write: WriteFn) -> None:
    """
    Process raw SSE bytes, accumulate tool call args, repair at finish.

    For OpenAI: buffers tool-args chunks, injects repaired full args into
    finish chunk. Strips trailing {} artifacts and leading duplicate braces.
    For Anthropic: tool-use chunks are buffered and re-emitted with repaired
    input at finish time. Non-tool blocks (text, reasoning) pass through immediately.
    """
    process_line, finalize = create_stream_handler(fmt, write)
    lines = raw.split(b"\n")
    for raw_line in lines:
        process_line(raw_line)
    finalize()


def _handle_anthropic_chunk(
    chunk: dict,
    deltas: list[ToolCallDelta],
    finish: StreamFinish | None,
    write: WriteFn,
    state: State,
) -> None:
    """Route Anthropic chunks: buffer tool-use, pass text through, repair at finish."""
    chunk_type = chunk.get("type")

    if chunk_type == "content_block_start":
        cb = chunk.get("content_block", {})
        if cb.get("type") == "tool_use":
            idx = chunk.get("index", 0)
            state["tool_ids"][idx] = cb.get("id", "")
            state["buffered"][idx] = [chunk]
            return
        _emit(chunk, write)
        return

    if chunk_type == "content_block_delta":
        delta = chunk.get("delta", {})
        if delta.get("type") == "input_json_delta":
            idx = chunk.get("index", 0)
            state["buffered"].setdefault(idx, []).append(chunk)
            return
        _emit(chunk, write)
        return

    if chunk_type == "content_block_stop":
        idx = chunk.get("index", 0)
        if idx in state["buffered"]:
            state["buffered"][idx].append(chunk)
            return
        _emit(chunk, write)
        return

    if chunk_type == "message_delta":
        if state["buffered"]:
            _emit_repaired_tools(write, state)
        _emit(chunk, write)
        return

    _emit(chunk, write)


def _emit_repaired_tools(write: WriteFn, state: State) -> None:
    """Repair accumulated args and emit reconstructed tool-use blocks."""
    repaired_map = repair_all_tools(state, "anthropic")

    for idx in list(state["buffered"]):
        accumulated = state["args"].get(idx, "")
        if not accumulated:
            for c in state["buffered"][idx]:
                _emit(c, write)
            state["buffered"].pop(idx)
            continue

        name = state["names"].get(idx, "?")
        tool_id = state["tool_ids"].get(idx, "")
        final_args, _ = repaired_map[idx]

        reconstructed = build_anthropic_tool_use_chunk(
            idx=idx,
            tool_id=tool_id,
            tool_name=name,
            repaired_input=final_args,
        )
        _emit(reconstructed, write)
        _emit({"type": "content_block_stop", "index": idx}, write)
        state["buffered"].pop(idx)

    if state["args"]:
        details = {i: {"name": state["names"].get(i, "?"), "chars": len(v)}
                   for i, v in state["args"].items()}
        log.info("TOOL_FINISH fmt=anthropic tools=%d details=%s", len(state["args"]), details)


def _flush_buffered(write: WriteFn, state: State) -> None:
    """Emit any remaining buffered chunks without repair."""
    for chunks in state["buffered"].values():
        for c in chunks:
            _emit(c, write)
    state["buffered"].clear()


def _flush_openai_args_buffer(write: WriteFn, state: State, strip_args: bool = True) -> None:
    """Emit buffered OpenAI args chunks.

    When strip_args=True (called from finish handler), zero out partial args so
    downstream only sees the complete repaired args in the finish chunk.
    When strip_args=False (called from finalize), pass through original content
    since there was no finish chunk to carry the full args.
    """
    for idx, chunks in state["openai_args_buffer"].items():
        for c in chunks:
            if strip_args:
                for choice in c.get("choices", []):
                    for tc in choice.get("delta", {}).get("tool_calls", []):
                        if "arguments" in tc.get("function", {}):
                            tc["function"]["arguments"] = ""
            _emit(c, write)
    state["openai_args_buffer"].clear()


def _handle_openai_finish(chunk: dict, write: WriteFn, state: State) -> None:
    """Apply repair to OpenAI finish chunk and emit.

    First flushes any buffered args chunks (with partial args stripped),
    then injects the complete repaired args into the finish chunk so
    downstream sees the full tool arguments.

    NOTE: We do NOT rewrite finish_reason here. The model's original
    finish_reason (usually "stop" after tool calls) is passed through to
    opencode. Rewriting was removed because it caused the agentic loop
    to fire spurious extra generations after tool execution completed.

    References:
      https://github.com/anomalyco/opencode/issues/28986
      https://github.com/anomalyco/opencode/issues/28618
    """
    # Flush buffered chunks before finish, with partial args cleared
    _flush_openai_args_buffer(write, state)

    repaired_map = repair_all_tools(state, "openai")

    for idx, (final_args, patch_status) in repaired_map.items():
        # Always inject final args into the finish chunk so downstream consumer
        # sees completed tool arguments (the finish chunk typically has empty delta).
        openai_repair_inplace(chunk, idx, final_args)

    _emit(chunk, write)
