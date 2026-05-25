#!/usr/bin/env python3
"""Minimal watcher: monitor SSE, log session status transitions, auto-continue."""

import json
import logging
import threading
import time
import urllib.request
import urllib.error

from . import client

log = logging.getLogger("auto_continue.watcher")


# Minimum output tokens for a response to be considered non-empty
MIN_TOKENS = 5
# Minimum text length for a response to be considered "meaningful"
MIN_TEXT_LENGTH = 3
# Settle delay (seconds) after idle before evaluating
SETTLE_DELAY = 2
# Longer settle delay after compaction to let model finish post-compaction work
SETTLE_DELAY_POST_COMPACTION = 10
# How long (seconds) after session.compacted event to use the longer delay
COMPACTION_WINDOW = 30


# Shared state: populated during settle, read by proxy's /watcher/status endpoint
# Each entry: { "remaining": int, "reason": str }
settling_sessions: dict[str, dict] = {}


def _countdown_settle(session_id: str, seconds: int, reason: str) -> None:
    """Run a countdown thread that updates settling_sessions, then clears it."""
    settling_sessions[session_id] = {"remaining": seconds, "reason": reason}
    for i in range(seconds, 0, -1):
        time.sleep(1)
        settling_sessions[session_id]["remaining"] = i
    if session_id in settling_sessions:
        del settling_sessions[session_id]


def get_settle_delay(session_id: str, last_compacted: dict[str, float]) -> int:
    """Return settle delay for a session, using longer delay if recently compacted.

    Case: ses_22e970b7affe (2026-04-28) — vLLM HTTP 400 (context exceeded) triggered
    auto-compaction. Session went idle, auto-continue fired 3s later and interrupted
    the model's post-compaction work. Fix: use 10s settle delay after compaction.
    """
    compacted_at = last_compacted.get(session_id, 0)
    if compacted_at and (time.time() - compacted_at) < COMPACTION_WINDOW:
        return SETTLE_DELAY_POST_COMPACTION
    return SETTLE_DELAY


def should_continue(status, msgs):
    """Decide whether to auto-continue a session.

    Returns (bool, str) — should_continue and reason.
    """
    if status != "idle":
        return False, f"status is {status}, not idle"

    return _is_incomplete(msgs)


_INTENT_PATTERNS = [
    r"(?i)\blet\s+me\s+(now\s+)?\w+",
    r"(?i)\bi\s+need\s+to\b",
    r"(?i)\bi\s+want\s+to\b",
    r"(?i)\bi\s+should\s+\w+",
    r"(?i)\bi['']m\s+going\s+to\b",
    r"(?i)\bi\s+will\s+",
    r"(?i)\bnext,?\s+i['']\s*ll\b",
    r"(?i)\bnext\s+steps?",
    r"(?i)\bareas?\s+to\s+\w+",
    r"(?i)\btodo\b",
]


def _looks_like_intent_to_continue(text: str) -> bool:
    """Check if text expresses intent to do more work without actually doing it.

    Catches: 'Let me check...', 'I need to verify...', 'I should look...',
    'Next steps:...', 'Areas to review:...', etc.

    Avoids false positives on: 'Let me know if...', 'Here is the answer...',
    retrospective explains like 'Let me see what happened... the fix was...'.
    """
    import re

    combined = text.strip()

    # Quick reject: if the response is short (< 100 chars) and matches, likely intent.
    # For longer responses, be more cautious — the phrase might be buried in a
    # complete explanation.
    is_short = len(combined) < 100

    has_pattern = any(re.search(p, combined) for p in _INTENT_PATTERNS)
    if not has_pattern:
        return False

    if is_short:
        return True

    # For longer texts, require the intent phrase appears at the END of the
    # message (last sentence/paragraph), not buried mid-response.
    # Split on double-newline or period-space to find the last segment.
    segments = re.split(r'\n\s*\n|(?<=\.)\s+', combined)
    last_segment = segments[-1] if segments else combined

    # Also check if the text ends with a colon-followed list (no period at end)
    ends_with_colon_list = bool(re.search(r':\s*$', combined.rstrip()))

    if ends_with_colon_list:
        return True

    has_pattern_in_last = any(re.search(p, last_segment) for p in _INTENT_PATTERNS)

    # Reject if the overall text looks like it delivered content (has code,
    # multiple sentences, or summary-like language)
    delivered_content = bool(re.search(r"(?i)(here is|here['']s|i['']ve (fixed|completed|finished|resolved|done)|all (done|set|green)|let me know if)", combined))
    if delivered_content:
        return False

    return has_pattern_in_last


def _has_meaningful_text(parts):
    """Check if any text part has substantial content."""
    for p in parts:
        if p.get("type") == "text" and len(p.get("text", "").strip()) >= MIN_TEXT_LENGTH:
            return True
    return False


def _extract_text(parts):
    """Extract all text content from message parts."""
    texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
    return " ".join(t.strip() for t in texts if t.strip())


def _has_tools(parts):
    """Check if any part is a tool call."""
    return any(p.get("type") == "tool" for p in parts)


def _is_incomplete(msgs):
    """Check if the last assistant message is incomplete.

    Returns (bool, str) — should_continue and reason.
    """
    if not msgs:
        return False, "no messages"

    last = msgs[-1]
    info = last.get("info", {})
    parts = last.get("parts", [])
    finish = info.get("finish", "")

    if info.get("role") != "assistant":
        return False, "last message not assistant"

    output = info.get("tokens", {}).get("output", 0)
    if output == 0:
        return False, "compaction (0 output tokens)"

    if output < MIN_TOKENS:
        return True, f"low tokens ({output})"

    has_text = _has_meaningful_text(parts)
    has_tools = _has_tools(parts)

    if not has_text and not has_tools:
        return True, f"empty response (tokens={output})"

    if has_tools and finish == "stop":
        return True, "tools but finish=stop (truncated?)"

    if has_tools and not has_text and finish == "tool-calls" and info.get("error"):
        return True, "tools executed but no text response"

    if has_tools and finish == "tool-calls":
        return False, "has tools with tool-calls finish"

    if has_text and not has_tools:
        text = _extract_text(parts)
        # Detect reasoning leak: text starts lowercase means it's mid-sentence
        # continuation from reasoning that vLLM misrouted into content.
        stripped = text.lstrip()
        if stripped and stripped[0].islower():
            return True, f"reasoning leak (lowercase start): {stripped[:80]!r}"
        if _looks_like_intent_to_continue(text):
            return True, f"conversational intent to continue: {text[:80]!r}"
        return False, "has meaningful text, no trailing tools"

    if info.get("error"):
        return False, "last message has error"

    return False, "looks complete"


def _handle_compacted(props, last_compacted):
    """Handle session.compacted event."""
    sid = props.get("sessionID")
    if not sid:
        return
    last_compacted[sid] = time.time()
    log.info("[%s] compaction completed", sid[:20])


def _handle_status_transition(sid, prev, status_type, last_status):
    """Log and update status transition."""
    last_status[sid] = status_type
    if prev != status_type:
        log.info("[%s] %s... -> %s", time.strftime("%H:%M:%S"), sid[:20], status_type)


def _handle_idle(
    sid, last_compacted, base_url, directory
):
    """Handle idle status: settle, check, and potentially auto-continue."""
    settle = get_settle_delay(sid, last_compacted)
    reason = "post-compaction" if settle > SETTLE_DELAY else "normal"

    countdown = threading.Thread(
        target=_countdown_settle,
        args=(sid, settle, reason),
        daemon=True,
    )
    countdown.start()
    time.sleep(settle)

    if _session_is_busy_again(base_url, sid):
        return

    msgs = client.get_messages(base_url, sid)
    now = time.time()
    should, reason = should_continue(
        status="idle",
        msgs=msgs,
    )

    # Log what the model actually said for debugging
    last_msg = msgs[-1] if msgs else {}
    text = _extract_text(last_msg.get("parts", []))
    output = last_msg.get("info", {}).get("tokens", {}).get("output", 0)
    log.info("[%s..] last message: tokens=%d text=%r", sid[:20], output, text[:200] if text else "(none)")

    if not should:
        log.info("[%s..] no auto-continue — %s", sid[:20], reason)
        return

    log.info("[%s..] auto-continuing — %s", sid[:20], reason)
    auto_continue_msg = f"[your response was cut off briefly, please continue]"
    ok = client.send_message(base_url, sid, auto_continue_msg, directory=directory)
    if ok:
        log.info("[%s..] auto-continue sent", sid[:20])
    else:
        log.warning("[%s..] auto-continue FAILED", sid[:20])


def _session_is_busy_again(base_url, sid):
    """Check if session went busy again during settle period."""
    try:
        sess_info = client.get_session_info(base_url, sid)
        current_status = sess_info.get("status", {})
        if current_status.get("type") == "busy":
            log.info("[%s..] skipping auto-continue — session is busy again", sid[:20])
            return True
    except Exception:
        pass
    return False


def _process_event_line(line, last_status, last_compacted, base_url):
    """Process a single SSE data line."""
    if not line.startswith("data: "):
        return

    try:
        envelope = json.loads(line[6:])
    except json.JSONDecodeError:
        return

    event_data = envelope.get("payload", envelope)
    etype = event_data.get("type", "")
    props = event_data.get("properties", {})
    envelope_dir = envelope.get("directory", "")

    if etype == "session.compacted":
        _handle_compacted(props, last_compacted)
        return

    if etype != "session.status":
        return

    sid = props.get("sessionID")
    status_type = props.get("status", {}).get("type")
    if not sid or not status_type:
        return

    prev = last_status.get(sid)
    _handle_status_transition(sid, prev, status_type, last_status)

    if status_type != "idle":
        return

    directory = envelope_dir or None
    _handle_idle(sid, last_compacted, base_url, directory)


def watch(base_url="http://localhost:4096"):
    log.info("Watching %s/global/event ...", base_url)

    while True:
        last_status: dict[str, str] = {}
        last_compacted: dict[str, float] = {}

        try:
            req = urllib.request.Request(
                f"{base_url}/global/event",
                headers={"Accept": "text/event-stream"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                log.info("SSE connected, status=%s", resp.status)

                while True:
                    raw_line = resp.readline()
                    if not raw_line:
                        log.warning("SSE closed, reconnecting...")
                        break

                    line = raw_line.decode("utf-8").rstrip("\n").rstrip("\r")
                    _process_event_line(line, last_status, last_compacted, base_url)

        except urllib.error.URLError as e:
            if "Connection refused" not in str(e):
                log.error("[%s] error: %s, retrying in 5s...", time.strftime('%H:%M:%S'), e)
            time.sleep(5)
        except (ConnectionResetError, BrokenPipeError, ConnectionError, OSError) as e:
            # Socket-level errors (ConnectionResetError, BrokenPipeError, etc.) are NOT
            # urllib.error.URLError subclasses — they would silently kill the daemon thread.
            # Bug: ses_1a9c2876affeyPkDhdj1rWLapt — ConnectionResetError during readline()
            # was not caught, watcher died, auto-continue never fired.
            log.error("[%s] socket error: %s (%s), reconnecting in 5s...",
                      time.strftime('%H:%M:%S'), e, type(e).__name__)
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("Stopped.")
            import sys
            sys.exit(0)
        except Exception as e:
            # Catch-all for any unexpected error — never let daemon thread die silently
            log.error("[%s] unexpected error: %s (%s), retrying in 5s...",
                      time.strftime('%H:%M:%S'), e, type(e).__name__)
            time.sleep(5)
