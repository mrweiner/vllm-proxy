#!/usr/bin/env python3
"""
auto_continue — session status watcher for opencode

Connects to opencode's /global/event SSE endpoint and logs session status
transitions (busy/idle/retry). Currently passive — no auto-continue logic.

See auto_continue/INTENT.md for design constraints and re-addition guidance.

Usage:
    python -m auto_continue [--base-url http://localhost:4096] [-v]
"""

__all__ = [
    "client",
    "watcher",
]
