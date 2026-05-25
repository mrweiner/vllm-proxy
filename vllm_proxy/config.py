#!/usr/bin/env python3
"""Config for vllm-proxy + auto-continue.

Precedence (highest to lowest):
  1. CLI arguments
  2. Environment variables (VLLM_PROXY_*, OC_*)
  3. config.toml (found relative to running script or cwd)
  4. Hardcoded defaults
"""

import os
import tomllib

# ── Defaults ──────────────────────────────────────────────────────────────────

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 4097
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
OC_BASE = "http://localhost:4096"

# SSE read chunk size
SSE_READ_SIZE = 256

# Auto-continue settle delay (seconds) after session goes idle
IDLE_SETTLE_DELAY = 3

# Request tracking
_total_requests = 0

# ── Config loading ────────────────────────────────────────────────────────────

_toml_data = None


def _find_config_toml() -> str | None:
    """Find config.toml relative to running script, then cwd."""
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        candidate = os.path.join(current, "config.toml")
        if os.path.isfile(candidate):
            return candidate
        current = os.path.dirname(current)

    cwd_candidate = os.path.join(os.getcwd(), "config.toml")
    if os.path.isfile(cwd_candidate):
        return cwd_candidate

    return None


def load_toml() -> dict:
    """Load config.toml if available. Returns empty dict if not found."""
    global _toml_data
    if _toml_data is not None:
        return _toml_data

    path = _find_config_toml()
    if not path:
        _toml_data = {}
        return _toml_data

    try:
        with open(path, "rb") as f:
            _toml_data = tomllib.load(f)
    except Exception:
        _toml_data = {}

    return _toml_data


def toml_int(section: str, key: str, default: int) -> int:
    data = load_toml()
    try:
        return int(data.get(section, {}).get(key, default))
    except (ValueError, TypeError):
        return default


def toml_str(section: str, key: str, default: str) -> str:
    data = load_toml()
    val = data.get(section, {}).get(key, default)
    return str(val) if val is not None else default


def toml_bool(section: str, key: str, default: bool) -> bool:
    data = load_toml()
    val = data.get(section, {}).get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return default


def toml_list(section: str, key: str, default: list | None = None) -> list | None:
    data = load_toml()
    val = data.get(section, {}).get(key)
    if val is None:
        return default
    return val


def apply_env_and_toml():
    """Apply TOML + env var overrides to module-level globals."""
    global LISTEN_HOST, LISTEN_PORT, VLLM_HOST, VLLM_PORT, OC_BASE
    global IDLE_SETTLE_DELAY

    LISTEN_HOST = os.environ.get("VLLM_PROXY_LISTEN_HOST", toml_str("proxy", "listen_host", LISTEN_HOST))
    LISTEN_PORT = int(os.environ.get("VLLM_PROXY_LISTEN_PORT", toml_int("proxy", "listen_port", LISTEN_PORT)))
    VLLM_HOST = os.environ.get("VLLM_PROXY_VLLM_HOST", toml_str("proxy", "vllm_host", VLLM_HOST))
    VLLM_PORT = int(os.environ.get("VLLM_PROXY_VLLM_PORT", toml_int("proxy", "vllm_port", VLLM_PORT)))
    OC_BASE = os.environ.get("OC_BASE_URL", toml_str("watcher", "base_url", OC_BASE))

    IDLE_SETTLE_DELAY = toml_int("watcher", "settle_delay", IDLE_SETTLE_DELAY)


apply_env_and_toml()
