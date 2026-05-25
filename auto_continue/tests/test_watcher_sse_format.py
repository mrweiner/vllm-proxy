#!/usr/bin/env python3
"""Test that watcher SSE endpoint sends proper SSE frames via actual handler."""

import sys
import io
import json
import unittest
import threading
import time

sys.path.insert(0, "/home/weiner/repos/local-llm/opencode")


class MockHandler:
    """Mock handler that captures what _handle_watcher_sse writes."""
    def __init__(self):
        self.wfile = io.BytesIO()
        self.headers_sent = []
        self.response_code = None

    def send_response(self, code):
        self.response_code = code

    def send_header(self, key, value):
        self.headers_sent.append((key.lower(), value))

    def end_headers(self):
        pass


class TestWatcherSSEActualOutput(unittest.TestCase):
    """Verify _handle_watcher_sse produces valid SSE output."""

    def setUp(self):
        from auto_continue.watcher import settling_sessions
        settling_sessions.clear()

    def tearDown(self):
        from auto_continue.watcher import settling_sessions
        settling_sessions.clear()

    def test_sse_output_has_no_chunked_hex_prefixes(self):
        """The SSE handler must write plain SSE frames, not chunked HTTP frames."""
        from auto_continue.watcher import settling_sessions
        from vllm_proxy.server import WatcherStatusHandler

        settling_sessions["ses_abc"] = {"remaining": 3, "reason": "normal"}

        handler = MockHandler()

        # Run the SSE handler in a thread, let it write one iteration, then stop
        stop_event = threading.Event()

        def run_sse():
            # Monkey-patch time.sleep to stop after first iteration
            original_sleep = time.sleep
            call_count = [0]
            def mock_sleep(secs):
                call_count[0] += 1
                if call_count[0] >= 2:
                    raise KeyboardInterrupt("stop")
                original_sleep(0.01)
            time.sleep = mock_sleep
            try:
                WatcherStatusHandler._handle_watcher_sse(handler)
            except (KeyboardInterrupt, BrokenPipeError, ConnectionResetError):
                pass
            finally:
                time.sleep = original_sleep

        t = threading.Thread(target=run_sse, daemon=True)
        t.start()
        t.join(timeout=5)

        output = handler.wfile.getvalue()

        # The output should contain SSE data lines, NOT chunked hex prefixes
        # Chunked encoding looks like: "1A\r\ndata: ...\n\n\r\n"
        # Plain SSE looks like: "data: ...\n\n"
        self.assertIn(b"data: ", output, "Output must contain SSE data lines")

        # Check that the output does NOT start with a hex chunk size prefix
        # If chunked, first bytes would be hex digits followed by \r\n
        if output:
            # Find the first \r\n - if there's hex digits before it, it's chunked
            first_crlf = output.find(b"\r\n")
            if first_crlf > 0:
                prefix = output[:first_crlf]
                # Valid SSE starts with "data:" - anything else before \r\n is chunked encoding
                self.assertTrue(
                    prefix.startswith(b"data:"),
                    f"Output appears to use chunked encoding. Prefix before first CRLF: {prefix!r}",
                )

    def test_sse_headers_include_cors_origin(self):
        """SSE endpoint must include Access-Control-Allow-Origin for cross-origin browser access."""
        from auto_continue.watcher import settling_sessions
        from vllm_proxy.server import WatcherStatusHandler

        settling_sessions["ses_abc"] = {"remaining": 2, "reason": "normal"}

        handler = MockHandler()

        def run_sse():
            original_sleep = time.sleep
            call_count = [0]
            def mock_sleep(secs):
                call_count[0] += 1
                if call_count[0] >= 1:
                    raise KeyboardInterrupt("stop")
                original_sleep(0.01)
            time.sleep = mock_sleep
            try:
                WatcherStatusHandler._handle_watcher_sse(handler)
            except (KeyboardInterrupt, BrokenPipeError, ConnectionResetError):
                pass
            finally:
                time.sleep = original_sleep

        t = threading.Thread(target=run_sse, daemon=True)
        t.start()
        t.join(timeout=3)

        header_dict = {k: v for k, v in handler.headers_sent}
        self.assertEqual(
            header_dict.get("access-control-allow-origin"),
            "*",
            "SSE endpoint must allow cross-origin requests from the openchamber UI",
        )

    def test_sse_headers_do_not_include_chunked_transfer_encoding(self):
        """The SSE handler should not set Transfer-Encoding: chunked."""
        from auto_continue.watcher import settling_sessions
        from vllm_proxy.server import WatcherStatusHandler

        settling_sessions["ses_abc"] = {"remaining": 2, "reason": "normal"}

        handler = MockHandler()

        def run_sse():
            original_sleep = time.sleep
            call_count = [0]
            def mock_sleep(secs):
                call_count[0] += 1
                if call_count[0] >= 1:
                    raise KeyboardInterrupt("stop")
                original_sleep(0.01)
            time.sleep = mock_sleep
            try:
                WatcherStatusHandler._handle_watcher_sse(handler)
            except (KeyboardInterrupt, BrokenPipeError, ConnectionResetError):
                pass
            finally:
                time.sleep = original_sleep

        t = threading.Thread(target=run_sse, daemon=True)
        t.start()
        t.join(timeout=3)

        header_names = [k for k, v in handler.headers_sent]
        self.assertNotIn(
            "transfer-encoding",
            header_names,
            "SSE endpoint should not use Transfer-Encoding: chunked (corrupts browser EventSource)",
        )


if __name__ == "__main__":
    unittest.main()
