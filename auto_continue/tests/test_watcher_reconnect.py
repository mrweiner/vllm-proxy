#!/usr/bin/env python3
"""Tests for SSE connection resilience -- watch() survives socket errors.

Bug: ses_1a9c2876affeyPkDhdj1rWLapt -- ConnectionResetError/BrokenPipeError
during readline() was not caught (only URLError was handled), killing the
daemon thread silently. Auto-continue never fired for any session.
"""

import sys
import unittest

sys.path.insert(0, "/home/weiner/repos/local-llm/proxy")

BUSY_EVENT = b'data: {"payload":{"type":"session.status","properties":{"sessionID":"x","status":{"type":"busy"}}},"directory":""}\n'
CLOSE = b""


class FakeResp:
    """Mock urllib response that yields specific readline results."""

    def __init__(self, lines_or_exc=None, exc=None):
        self.status = 200
        self.exc = exc
        self.lines = list(lines_or_exc) if isinstance(lines_or_exc, (list, tuple)) else []
        self.idx = 0

    def readline(self):
        if self.exc:
            raise self.exc
        line = self.lines[self.idx] if self.idx < len(self.lines) else b""
        self.idx += 1
        return line

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestWatchReconnectsOnSocketError(unittest.TestCase):
    """watch() catches socket errors and reconnects."""

    def _run_watch(self, resp_sequence):
        """Run watch() with mocked urlopen and time.sleep.

        resp_sequence: list of FakeResp or Exception for sequential urlopen calls.
        After exhausted, urlopen raises SystemExit (BaseException, not caught
        by watcher's except Exception handler) to cleanly stop the loop.

        Each error->reconnect cycle triggers exactly one time.sleep. Returns
        (call_count, sleep_count).
        """
        import auto_continue.watcher as w
        import time as _time
        import urllib.request as _urllib

        idx = [0]
        call_count = [0]
        sleep_count = [0]
        original_sleep = _time.sleep

        def fast_sleep(n):
            sleep_count[0] += 1
            original_sleep(0.01)

        def fake_urlopen(req, timeout=300):
            call_count[0] += 1
            if idx[0] >= len(resp_sequence):
                raise SystemExit("test done")
            entry = resp_sequence[idx[0]]
            idx[0] += 1
            if isinstance(entry, Exception):
                raise entry
            return entry

        orig_urlopen = _urllib.urlopen
        orig_sleep = _time.sleep

        try:
            _urllib.urlopen = fake_urlopen
            _time.sleep = fast_sleep
            w.watch()
        except SystemExit:
            pass
        finally:
            _urllib.urlopen = orig_urlopen
            _time.sleep = orig_sleep

        return call_count[0], sleep_count[0]

    def test_reconnects_after_connection_reset(self):
        """ConnectionResetError in readline -> caught -> sleep -> reconnect."""
        resp1 = FakeResp(exc=ConnectionResetError("reset"))
        resp2 = FakeResp(lines_or_exc=[BUSY_EVENT, CLOSE])
        count, sleeps = self._run_watch([resp1, resp2])
        self.assertGreaterEqual(count, 2,
            "Should have reconnected after ConnectionResetError")
        self.assertGreaterEqual(sleeps, 1,
            "Should have slept through error recovery + reconnect cycles")

    def test_reconnects_after_broken_pipe(self):
        """BrokenPipeError in readline -> caught -> sleep -> reconnect."""
        resp1 = FakeResp(exc=BrokenPipeError("broken"))
        resp2 = FakeResp(lines_or_exc=[BUSY_EVENT, CLOSE])
        count, sleeps = self._run_watch([resp1, resp2])
        self.assertGreaterEqual(count, 2,
            "Should have reconnected after BrokenPipeError")
        self.assertGreaterEqual(sleeps, 1,
            "Should have slept through error recovery + reconnect cycles")

    def test_reconnects_after_urlopen_error(self):
        """URLError in urlopen -> caught -> sleep -> reconnect."""
        import urllib.error
        resp1 = urllib.error.URLError("connection refused")
        resp2 = FakeResp(lines_or_exc=[BUSY_EVENT, CLOSE])
        count, sleeps = self._run_watch([resp1, resp2])
        self.assertGreaterEqual(count, 2,
            "Should have reconnected after URLError")
        self.assertGreaterEqual(sleeps, 1,
            "Should have slept through error recovery + reconnect cycles")

    def test_reconnects_after_unknown_exception(self):
        """Unknown exception in urlopen -> catch-all -> sleep -> reconnect."""
        resp1 = RuntimeError("unexpected crash")
        resp2 = FakeResp(lines_or_exc=[BUSY_EVENT, CLOSE])
        count, sleeps = self._run_watch([resp1, resp2])
        self.assertGreaterEqual(count, 2,
            "Should have reconnected after unknown exception")
        self.assertGreaterEqual(sleeps, 1,
            "Should have slept through error recovery + reconnect cycles")


class TestExceptionClasses(unittest.TestCase):
    """Verify the exception hierarchy that motivated the fix."""

    def test_connection_reset_not_urllib_error(self):
        import urllib.error
        self.assertFalse(issubclass(ConnectionResetError, urllib.error.URLError))

    def test_broken_pipe_not_urllib_error(self):
        import urllib.error
        self.assertFalse(issubclass(BrokenPipeError, urllib.error.URLError))

    def test_connection_reset_is_oserror(self):
        self.assertTrue(issubclass(ConnectionResetError, OSError))

    def test_broken_pipe_is_oserror(self):
        self.assertTrue(issubclass(BrokenPipeError, OSError))


if __name__ == "__main__":
    unittest.main()
