#!/usr/bin/env python3
"""Tests for tool schema patches."""

import json
import unittest
from vllm_proxy.tool_patches import apply_tool_patches, _patch_read, _patch_bash, _patch_write, _patch_edit


class TestPatchRead(unittest.TestCase):
    def test_returns_unrecoverable_when_filepath_missing(self):
        patched, recoverable = _patch_read({}, "read")
        self.assertFalse(patched)
        self.assertFalse(recoverable)

    def test_returns_unrecoverable_when_filepath_empty(self):
        patched, recoverable = _patch_read({"filePath": ""}, "read")
        self.assertFalse(patched)
        self.assertFalse(recoverable)

    def test_returns_recoverable_when_filepath_present(self):
        patched, recoverable = _patch_read({"filePath": "/some/path"}, "read")
        self.assertFalse(patched)
        self.assertTrue(recoverable)


class TestPatchBash(unittest.TestCase):
    def test_returns_unrecoverable_when_command_missing(self):
        patched, recoverable = _patch_bash({}, "bash")
        self.assertFalse(patched)
        self.assertFalse(recoverable)

    def test_patches_missing_description(self):
        args = {"command": "ls"}
        patched, recoverable = _patch_bash(args, "bash")
        self.assertTrue(patched)
        self.assertTrue(recoverable)
        self.assertEqual(args["description"], "[auto]")

    def test_returns_recoverable_with_command(self):
        args = {"command": "ls", "description": "list files"}
        patched, recoverable = _patch_bash(args, "bash")
        self.assertFalse(patched)
        self.assertTrue(recoverable)


class TestPatchWrite(unittest.TestCase):
    def test_returns_unrecoverable_when_filepath_missing(self):
        patched, recoverable = _patch_write({}, "write")
        self.assertFalse(patched)
        self.assertFalse(recoverable)

    def test_returns_unrecoverable_when_content_missing(self):
        patched, recoverable = _patch_write({"filePath": "/tmp/x"}, "write")
        self.assertFalse(patched)
        self.assertFalse(recoverable)

    def test_returns_recoverable_when_complete(self):
        patched, recoverable = _patch_write({"filePath": "/tmp/x", "content": "hello world enough"}, "write")
        self.assertFalse(patched)
        self.assertTrue(recoverable)


class TestPatchEdit(unittest.TestCase):
    def test_returns_unrecoverable_when_filepath_missing(self):
        patched, recoverable = _patch_edit({}, "edit")
        self.assertFalse(patched)
        self.assertFalse(recoverable)

    def test_returns_unrecoverable_when_oldstring_missing(self):
        patched, recoverable = _patch_edit({"filePath": "/tmp/x", "newString": "foo"}, "edit")
        self.assertFalse(patched)
        self.assertFalse(recoverable)

    def test_returns_recoverable_when_complete(self):
        patched, recoverable = _patch_edit(
            {"filePath": "/tmp/x", "oldString": "a", "newString": "b"}, "edit"
        )
        self.assertFalse(patched)
        self.assertTrue(recoverable)


class TestApplyToolPatches(unittest.TestCase):
    def test_read_missing_filepath_returns_unrecoverable(self):
        args_str = json.dumps({})
        result_str, status = apply_tool_patches("read", args_str)
        self.assertEqual(status, "unrecoverable")

    def test_read_empty_filepath_returns_unrecoverable(self):
        args_str = json.dumps({"filePath": ""})
        result_str, status = apply_tool_patches("read", args_str)
        self.assertEqual(status, "unrecoverable")

    def test_read_valid_filepath_returns_ok(self):
        args_str = json.dumps({"filePath": "/tmp/test.txt"})
        result_str, status = apply_tool_patches("read", args_str)
        self.assertEqual(status, "ok")
        self.assertEqual(result_str, args_str)

    def test_bash_missing_description_returns_patched(self):
        args_str = json.dumps({"command": "ls"})
        result_str, status = apply_tool_patches("bash", args_str)
        self.assertEqual(status, "patched")
        result = json.loads(result_str)
        self.assertEqual(result["description"], "[auto]")

    def test_invalid_json_returns_invalid_json(self):
        result_str, status = apply_tool_patches("read", "{broken")
        self.assertEqual(status, "invalid_json")

    def test_unknown_tool_returns_ok(self):
        args_str = json.dumps({"foo": "bar"})
        result_str, status = apply_tool_patches("unknown_tool", args_str)
        self.assertEqual(status, "ok")


if __name__ == "__main__":
    unittest.main()
