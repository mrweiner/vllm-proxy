#!/usr/bin/env python3
"""Tool schema patches for fixing malformed tool call arguments."""

import json
import logging

log = logging.getLogger("vllm-proxy.tool_patches")


def _patch_bash(args: dict, tool: str) -> tuple[bool, bool]:
    patched = False
    if "command" not in args or not args.get("command"):
        return False, False
    if "description" not in args or not args.get("description"):
        args["description"] = "[auto]"
        patched = True
    return patched, True


def _patch_read(args: dict, tool: str) -> tuple[bool, bool]:
    if "filePath" not in args or not args.get("filePath"):
        return False, False
    return False, True


def _patch_write(args: dict, tool: str) -> tuple[bool, bool]:
    if "filePath" not in args or not args.get("filePath"):
        return False, False
    if "content" not in args or args.get("content") is None:
        return False, False
    content = args.get("content", "")
    if isinstance(content, str) and len(content) < 20:
        log.warning(
            "write tool has very short content (%d chars) for path %s — may be truncated",
            len(content), args.get("filePath", "?"),
        )
    return False, True


def _patch_edit(args: dict, tool: str) -> tuple[bool, bool]:
    if "filePath" not in args or not args.get("filePath"):
        return False, False
    for field in ("oldString", "newString"):
        if field not in args or not args.get(field):
            return False, False
    return False, True


def _patch_multiedit(args: dict, tool: str) -> tuple[bool, bool]:
    if "filePath" not in args or not args.get("filePath"):
        return False, False
    edits = args.get("edits")
    if not edits or not isinstance(edits, list):
        return False, False
    return False, True


def _patch_glob(args: dict, tool: str) -> tuple[bool, bool]:
    if "pattern" not in args or not args.get("pattern"):
        return False, False
    return False, True


def _patch_todowrite(args: dict, tool: str) -> tuple[bool, bool]:
    if "todos" not in args:
        return False, False
    return False, True


TOOL_PATCHES: dict[str, callable] = {
    "bash": _patch_bash,
    "shell": _patch_bash,
    "read": _patch_read,
    "write": _patch_write,
    "edit": _patch_edit,
    "multiedit": _patch_multiedit,
    "glob": _patch_glob,
    "todowrite": _patch_todowrite,
}



def apply_tool_patches(tool_name: str, args_str: str) -> tuple[str, str]:
    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, ValueError):
        if tool_name in TOOL_PATCHES:
            return args_str, "invalid_json"
        return args_str, "ok"

    patch_fn = TOOL_PATCHES.get(tool_name)
    if patch_fn is None:
        return args_str, "ok"

    patched, recoverable = patch_fn(args, tool_name)

    if not recoverable:
        return args_str, "unrecoverable"

    if patched:
        return json.dumps(args).strip(), "patched"

    return args_str, "ok"
