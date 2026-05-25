#!/usr/bin/env python3
"""Minimal vLLM proxy: forward SSE, repair tool call JSON, apply patches, log."""

import argparse
import datetime
import http.client
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from . import config
from .format_adapters import (
    detect_format,
    is_chat_endpoint,
)
from . import policies

log = logging.getLogger("vllm-proxy")
file_log = logging.getLogger("vllm-proxy.file")
file_log.propagate = False


# ── watcher status handler ───────────────────────────────────────────────────

class WatcherStatusHandler:
    """Handle GET /watcher/status and /watcher/sse for settling session countdown."""

    @staticmethod
    def do_GET(handler: "ProxyHandler") -> bool:
        """Handle watcher status endpoints. Returns True if handled, False to continue to proxy."""
        if handler.path == "/watcher/status":
            WatcherStatusHandler._handle_watcher_status(handler)
            return True
        if handler.path == "/watcher/sse":
            WatcherStatusHandler._handle_watcher_sse(handler)
            return True
        return False

    @staticmethod
    def _handle_watcher_status(handler: "ProxyHandler") -> None:
        """Return current settling sessions as JSON."""
        try:
            from auto_continue.watcher import settling_sessions
        except ImportError:
            settling_sessions = {}

        body = json.dumps(dict(settling_sessions)).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(body)

    @staticmethod
    def _handle_watcher_sse(handler: "ProxyHandler") -> None:
        """Stream settling session updates via SSE."""
        try:
            from auto_continue.watcher import settling_sessions
        except ImportError:
            settling_sessions = {}

        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Connection", "close")
        handler.end_headers()

        last_state = None
        try:
            while True:
                current = dict(settling_sessions)
                # Always send initial state on connect, then only on changes.
                # This resyncs the UI after EventSource reconnects from tab inactivity.
                is_initial = last_state is None
                if is_initial or current != last_state:
                    data = b"data: " + json.dumps(current, separators=(",", ":")).encode() + b"\n\n"
                    try:
                        handler.wfile.write(data)
                        handler.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    last_state = current
                time.sleep(1)
        except Exception:
            pass


def _extract_session_id(req_json: dict) -> str:
    for key in ("session_id", "sessionId", "conversation_id", "conversationId", "user"):
        if val := req_json.get(key):
            return str(val)[:32]
    msgs = req_json.get("messages", [])
    for msg in reversed(msgs):
        if msg.get("role") == "user" and msg.get("id"):
            return str(msg["id"])[:32]
    return "unknown"


# ── proxy handler ────────────────────────────────────────────────────────────

class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except ConnectionResetError:
            pass  # Client disconnected — normal during streaming

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def _forward_headers(self) -> dict:
        skip = {"host", "content-length", "transfer-encoding", "connection"}
        return {k: v for k, v in self.headers.items() if k.lower() not in skip}

    def _send_response_headers(self, status: int, headers: list[tuple[str, str]],
                                chunked: bool = False):
        self.send_response(status)
        skip = {"transfer-encoding", "content-length", "content-encoding", "connection"}
        for k, v in headers:
            if k.lower() in skip:
                continue
            self.send_header(k, v)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Connection", "close")
        self.end_headers()

    def _handle_sse(self, upstream: http.client.HTTPResponse,
                     status: int, headers: list[tuple[str, str]],
                     is_chat: bool, fmt: str, req_num: int, session_id: str):
        self._send_response_headers(status, headers, chunked=True)

        buf = b""

        def write_chunk(data: bytes):
            if not data:
                return
            try:
                self.wfile.write(f"{len(data):X}\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise

        # Use sse_handler for format-aware tool call repair
        from . import sse_handler
        if is_chat:
            process_line, finalize = sse_handler.create_stream_handler(fmt, write_chunk)
        else:
            def process_line(raw):
                write_chunk(raw + b"\n\n")
            def finalize():
                pass

        try:
            while True:
                chunk = upstream.read(config.SSE_READ_SIZE)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    process_line(line)
            if buf.strip():
                process_line(buf)
            finalize()
        except (BrokenPipeError, ConnectionResetError) as e:
            log.warning("CLIENT DISCONNECT req=#%d mid-stream (%s)", req_num, type(e).__name__)
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception:
                pass

    def _handle(self):
        # Check for watcher endpoints first
        if WatcherStatusHandler.do_GET(self):
            return

        body = self._read_body()
        fmt = detect_format(self.path)
        is_chat = is_chat_endpoint(self.path)
        req_num = 0
        session_id = self.headers.get("x-session-affinity", "") or ""

        if is_chat:
            config._total_requests += 1
            req_num = config._total_requests
            try:
                req_json = json.loads(body)
                tools = req_json.get("tools", [])
                if fmt == "anthropic":
                    tool_names = [t.get("name", "?") for t in tools]
                else:
                    tool_names = [t.get("function", {}).get("name", "?") for t in tools]
                streaming = req_json.get("stream", False)
                msgs = req_json.get("messages", [])
                if not session_id:
                    session_id = _extract_session_id(req_json)
                log.info("REQ #%d session=%s msgs=%d stream=%s",
                         req_num, session_id, len(msgs), streaming)
                file_log.debug("REQ #%d session=%s tools=[%s]",
                         req_num, session_id,
                         ", ".join(tool_names) if tool_names else "none")

                # Inject system prompt policies
                policies.inject_policies(req_json, fmt)

                # Fix attention dilution: inject user reminder after tool results
                policies.fix_assistant_whitespace_content(req_json, fmt)

                body = json.dumps(req_json).encode()
            except (json.JSONDecodeError, ValueError):
                log.info("REQ #%d (unparseable body)", req_num)

        conn = http.client.HTTPConnection(config.VLLM_HOST, config.VLLM_PORT, timeout=600)
        fwd_headers = self._forward_headers()
        fwd_headers["Host"] = f"{config.VLLM_HOST}:{config.VLLM_PORT}"
        if body:
            fwd_headers["Content-Length"] = str(len(body))

        try:
            conn.request(self.command, self.path,
                         body=body or None, headers=fwd_headers)
            upstream = conn.getresponse()
        except Exception as e:
            log.error("upstream error req=#%d: %s", req_num, e)
            self.send_error(502, f"Upstream error: {e}")
            return

        status = upstream.status
        resp_headers = upstream.getheaders()
        content_type = upstream.getheader("Content-Type", "")
        is_sse = "text/event-stream" in content_type

        if status != 200 and is_chat:
            log.warning("req=#%d vLLM returned HTTP %d", req_num, status)

        if is_sse:
            self._handle_sse(upstream, status, resp_headers, is_chat, fmt, req_num, session_id)
            return

        resp_body = upstream.read()
        self._send_response_headers(status, resp_headers)
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def do_GET(self):     self._handle()
    def do_POST(self):    self._handle()
    def do_PUT(self):     self._handle()
    def do_DELETE(self):  self._handle()
    def do_OPTIONS(self): self._handle()
    def do_HEAD(self):    self._handle()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="vLLM tool-call repair proxy")
    parser.add_argument("--listen-port", type=int, default=config.LISTEN_PORT)
    parser.add_argument("--vllm-port", type=int, default=config.VLLM_PORT)
    parser.add_argument("--vllm-host", default=config.VLLM_HOST)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    config.LISTEN_PORT = args.listen_port
    config.VLLM_HOST = args.vllm_host
    config.VLLM_PORT = args.vllm_port

    level = logging.DEBUG if args.verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="a"))
        # file_log goes to file only (detailed debug info, not streamed to terminal)
        file_h = logging.FileHandler(args.log_file, mode="a")
        file_h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        file_log.addHandler(file_h)
        file_log.setLevel(logging.DEBUG)
        file_log.propagate = False
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

    # Start session lifecycle watcher as daemon thread
    _watcher_started = False
    try:
        import auto_continue.watcher as _watcher_mod
        threading.Thread(
            target=_watcher_mod.watch,
            args=(config.OC_BASE,),
            daemon=True,
        ).start()
        _watcher_started = True
        log.info("Watcher started (in-process)")
    except Exception as e:
        log.warning("Failed to start watcher: %s", e)

    server = ThreadedHTTPServer((config.LISTEN_HOST, config.LISTEN_PORT), ProxyHandler)

    log.info("vLLM proxy listening :%d -> :%d", config.LISTEN_PORT, config.VLLM_PORT)
    log.info("intercepts: POST /v1/chat/completions, POST /v1/messages")
    log.info("repair: closes truncated JSON in tool call args")
    log.info("watcher: %s", "in-process" if _watcher_started else "disabled")
    if args.log_file:
        log.info("log file: %s", args.log_file)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("stopped — lifetime: %d requests", config._total_requests)
        sys.exit(0)
