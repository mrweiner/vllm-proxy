#!/usr/bin/env python3
"""vLLM Proxy + Auto-Continue Watcher

Standalone launcher. Copy this directory to share.

Usage:
    python3 start.py [--listen-port 4097] [--vllm-port 8000] [-v] [--log-file PATH]

Config is loaded from config.toml in the same directory, overridden by
environment variables (VLLM_PROXY_*), then overridden by CLI flags.
"""

import argparse
import logging
import os
import signal
import sys
import threading

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from vllm_proxy import config
from vllm_proxy.server import ThreadedHTTPServer, ProxyHandler, WatcherStatusHandler
from vllm_proxy.format_adapters import detect_format, is_chat_endpoint
from vllm_proxy import policies


def main():
    parser = argparse.ArgumentParser(
        description="vLLM tool-call repair proxy + auto-continue watcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Config precedence: CLI > env vars (VLLM_PROXY_*) > config.toml > defaults\n"
            "\n"
            "Example:\n"
            "  python3 start.py --listen-port 4097 --vllm-port 8000 -v\n"
        ),
    )
    parser.add_argument("--listen-port", type=int, default=None,
                        help="Port to listen on (default: from config.toml or 4097)")
    parser.add_argument("--vllm-port", type=int, default=None,
                        help="Upstream vLLM port (default: from config.toml or 8000)")
    parser.add_argument("--vllm-host", default=None,
                        help="Upstream vLLM host (default: from config.toml or 127.0.0.1)")
    parser.add_argument("--log-file", default=None,
                        help="Log file path (stderr only if omitted)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    listen_port = args.listen_port or config.LISTEN_PORT
    vllm_host = args.vllm_host or config.VLLM_HOST
    vllm_port = args.vllm_port or config.VLLM_PORT
    listen_host = config.LISTEN_HOST
    oc_base = config.OC_BASE

    level = logging.DEBUG if args.verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stderr)]
    log_file_path = args.log_file or config.toml_str("logging", "log_file", "")
    if log_file_path:
        log_file_path = os.path.expanduser(log_file_path)
        handlers.append(logging.FileHandler(log_file_path, mode="a"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    log = logging.getLogger("start")

    if not os.path.isdir(os.path.join(_SCRIPT_DIR, "vllm_proxy")):
        log.error("vllm_proxy/ not found in %s", _SCRIPT_DIR)
        sys.exit(1)

    watcher_started = False
    try:
        import auto_continue.watcher as _watcher_mod
        threading.Thread(
            target=_watcher_mod.watch,
            args=(oc_base,),
            daemon=True,
        ).start()
        watcher_started = True
        log.info("Watcher started (in-process)")
    except ImportError:
        log.info("Watcher disabled (auto_continue/ not found)")
    except Exception as e:
        log.warning("Failed to start watcher: %s", e)

    server = ThreadedHTTPServer((listen_host, listen_port), ProxyHandler)

    def shutdown(signum, frame):
        signame = signal.Signals(signum).name
        log.info("Received %s, shutting down...", signame)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info("=" * 60)
    log.info("vLLM Proxy + Auto-Continue Watcher")
    log.info("=" * 60)
    log.info("Proxy:  %s:%d -> %s:%d", listen_host, listen_port, vllm_host, vllm_port)
    log.info("Watcher: %s (%s)", oc_base, "in-process" if watcher_started else "disabled")
    log.info("Config: %s", config._find_config_toml() or "(none, using defaults)")
    if log_file_path:
        log.info("Logs:   %s", log_file_path)
    log.info("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        log.info("stopped — lifetime: %d requests", config._total_requests)
        sys.exit(0)


if __name__ == "__main__":
    main()
