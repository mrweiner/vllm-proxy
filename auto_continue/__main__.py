#!/usr/bin/env python3
"""Allow running as python3 -m auto_continue"""
import argparse
import logging
import sys

from .watcher import watch


def main():
    parser = argparse.ArgumentParser(description="Session status watcher")
    parser.add_argument("--base-url", default="http://localhost:4096")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

    watch(base_url=args.base_url)


if __name__ == "__main__":
    main()
