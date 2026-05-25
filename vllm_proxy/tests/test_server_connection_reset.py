#!/usr/bin/env python3
"""Test that ProxyHandler.handle() swallows ConnectionResetError silently."""

import os
import sys
import unittest
from http.server import BaseHTTPRequestHandler
from unittest.mock import MagicMock, patch

# Save original modules so we can restore them after testing
_original_modules = {}
_mocked_modules = [
    "vllm_proxy.config",
    "vllm_proxy.format_adapters",
    "vllm_proxy.policies",
    "vllm_proxy.sse_handler",
]


def _apply_mocks():
    """Mock upstream deps and load server module in isolation."""
    global _original_modules
    # Save originals
    for mod in _mocked_modules:
        _original_modules[mod] = sys.modules.get(mod)
    _original_modules["vllm_proxy"] = sys.modules.get("vllm_proxy")

    # Replace with mocks
    for mod in _mocked_modules:
        sys.modules[mod] = MagicMock()
    sys.modules["vllm_proxy"] = MagicMock()

    # Load the actual module from the parent directory (not tests/)
    import importlib.util
    _server_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),  # tests/
        "..",  # vllm_proxy/
        "server.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vllm_proxy.server",
        _server_path,
        submodule_search_locations=[],
    )
    server_mod = importlib.util.module_from_spec(spec)
    sys.modules["vllm_proxy.server"] = server_mod
    spec.loader.exec_module(server_mod)
    return server_mod


def _restore_modules():
    """Restore original modules after testing."""
    # Remove all mocked modules so Python re-imports the real ones
    sys.modules.pop("vllm_proxy.server", None)
    for mod in _mocked_modules:
        sys.modules.pop(mod, None)

    # Restore or remove vllm_proxy so real package can be re-imported
    orig = _original_modules.get("vllm_proxy")
    if orig is not None:
        sys.modules["vllm_proxy"] = orig
    else:
        sys.modules.pop("vllm_proxy", None)


class TestProxyHandlerConnectionReset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server_mod = _apply_mocks()

    @classmethod
    def tearDownClass(cls):
        _restore_modules()

    def test_handle_swallows_connection_reset(self):
        """ConnectionResetError during handle() should not propagate."""
        handler = self.server_mod.ProxyHandler.__new__(self.server_mod.ProxyHandler)
        handler.server = MagicMock()
        handler.client_address = ("127.0.0.1", 9999)
        handler.request = MagicMock()

        with patch.object(BaseHTTPRequestHandler, "handle", side_effect=ConnectionResetError("Connection reset by peer")):
            handler.handle()  # should not raise

    def test_handle_propagates_other_errors(self):
        """Non-ConnectionResetError should still propagate."""
        handler = self.server_mod.ProxyHandler.__new__(self.server_mod.ProxyHandler)
        handler.server = MagicMock()
        handler.client_address = ("127.0.0.1", 9999)
        handler.request = MagicMock()

        with patch.object(BaseHTTPRequestHandler, "handle", side_effect=RuntimeError("other error")):
            with self.assertRaises(RuntimeError):
                handler.handle()


if __name__ == "__main__":
    unittest.main()
