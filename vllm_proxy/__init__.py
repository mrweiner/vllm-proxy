#!/usr/bin/env python3
"""
vllm_proxy — tool-call JSON repair proxy for vLLM + qwen3_coder + MTP

Sits between opencode and vLLM. Forwards all traffic unchanged except it
repairs truncated tool call JSON (missing closing braces) caused by the
qwen3_coder streaming parser + speculative decoding bug.

Streaming is preserved: SSE chunks are forwarded to opencode as they arrive
from vLLM, not buffered until completion.

Also starts a minimal session-status watcher as a daemon thread that logs
busy/idle/retry transitions.

Designed for SSH-tunnel setups:
    opencode → localhost:LISTEN_PORT (this proxy) → localhost:VLLM_PORT → tunnel → RunPod

Usage:
    python -m vllm_proxy [--listen-port 4097] [--vllm-port 8000] [-v]
"""

__all__ = [
    "config",
    "json_repair",
    "policies",
    "tool_patches",
    "server",
]
