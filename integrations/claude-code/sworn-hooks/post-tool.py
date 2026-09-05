#!/usr/bin/env python3
"""PostToolUse entry for the sworn hooks (see README.md in this directory).

Loads ``styxx/harness/claude_code.py`` by file path under a private module name, so that
``styxx/__init__.py`` — and the eager imports it carries — never runs on this path, which fires
once per tool call. Stages one event file; exits zero on every input, because a hook that
blocks the agent is a gate without a measured precision.
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent.parent.parent / "styxx" / "harness" / "claude_code.py"


def load():
    spec = importlib.util.spec_from_file_location("_styxx_harness_claude_code", str(MODULE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    try:
        load().main(["post-tool"] + sys.argv[1:])
    except BaseException as exc:                                        # never block the agent
        try:
            sys.stderr.write("sworn-hooks post-tool: %s: %s\n" % (type(exc).__name__, exc))
        except Exception:
            pass
    sys.exit(0)
