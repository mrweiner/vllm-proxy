#!/usr/bin/env python3
"""Test runner that ensures correct sys.path for package imports.

Usage:
    python3 run_tests.py              # all tests
    python3 run_tests.py sse_handler  # specific test module
"""
import os
import sys
import unittest

# Ensure the proxy/ directory is on sys.path so vllm_proxy and auto_continue
# import correctly regardless of how this script is invoked.
_PROXY_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROXY_DIR not in sys.path:
    sys.path.insert(0, _PROXY_DIR)

def main():
    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=2)

    if len(sys.argv) > 1:
        # Run specific test module
        for name in sys.argv[1:]:
            # Try as dotted module path first, then as filename
            if "." in name or name.endswith(".py"):
                suite = loader.loadTestsFromName(name.replace(".py", ""))
            else:
                suite = loader.loadTestsFromName(f"vllm_proxy.tests.test_{name}")
            runner.run(suite)
    else:
        # Discover all tests in both packages
        suite = loader.discover(
            start_dir=os.path.join(_PROXY_DIR, "vllm_proxy", "tests"),
            pattern="test_*.py",
        )
        suite.addTests(loader.discover(
            start_dir=os.path.join(_PROXY_DIR, "auto_continue", "tests"),
            pattern="test_*.py",
        ))
        runner.run(suite)


if __name__ == "__main__":
    main()
