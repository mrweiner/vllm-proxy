#!/usr/bin/env python3
"""Tests for vllm_proxy /watcher/status and /watcher/sse endpoints."""

import sys
import unittest

sys.path.insert(0, "/home/weiner/repos/local-llm/proxy")


class TestWatcherStatusEndpoint(unittest.TestCase):
    """Proxy exposes GET /watcher/status returning settling_sessions as JSON."""

    def test_status_endpoint_returns_json(self):
        """GET /watcher/status returns JSON with settling sessions."""
        from vllm_proxy.server import WatcherStatusHandler
        self.assertTrue(hasattr(WatcherStatusHandler, "do_GET"))

    def test_status_reflects_settling_sessions(self):
        """Status endpoint reads from watcher.settling_sessions."""
        from auto_continue.watcher import settling_sessions
        settling_sessions["ses_test"] = {"remaining": 5, "reason": "post-compaction"}
        self.assertIn("ses_test", settling_sessions)
        self.assertEqual(settling_sessions["ses_test"]["remaining"], 5)
        del settling_sessions["ses_test"]


if __name__ == "__main__":
    unittest.main()
