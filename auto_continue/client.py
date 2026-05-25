#!/usr/bin/env python3
"""HTTP client helpers for communicating with opencode API."""

import json
import logging
import urllib.request
import urllib.error

log = logging.getLogger("auto_continue.client")


def _get(base_url, path, timeout=10):
    req = urllib.request.Request(
        f"{base_url}{path}",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _post(base_url, path, body=None, timeout=10, directory=None):
    payload = json.dumps(body or {}).encode()
    headers = {"Content-Type": "application/json"}
    if directory:
        headers["x-opencode-directory"] = directory
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get_session_info(base_url, session_id):
    try:
        return _get(base_url, f"/session/{session_id}")
    except Exception as e:
        log.warning("failed to fetch session info: %s", e)
        return {}


def get_messages(base_url, session_id):
    """Returns list of {info, parts} objects."""
    try:
        return _get(base_url, f"/session/{session_id}/message")
    except Exception as e:
        log.warning("failed to fetch messages: %s", e)
        return []


def send_message(base_url, session_id, text, directory=None):
    """Send a text message to a session."""
    try:
        _post(base_url, f"/session/{session_id}/message", {
            "parts": [{"type": "text", "text": text}],
        }, timeout=300, directory=directory)
        return True
    except Exception as e:
        log.warning("failed to send message: %s", e)
        return False
