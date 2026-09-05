#!/usr/bin/env python3
"""Stop entry for the sworn hooks (see README.md in this directory).

Folds the session's staged events into ``<dir>/<session_id>.manifest.json`` at L1. This is the
one place the hook pair imports ``styxx`` — once per turn, not once per tool call. Prints
nothing on stdout (Stop's stdout is parsed for a decision); the manifest path goes to stderr.
Exits zero on every input.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    try:
        from styxx.harness import claude_code
        claude_code.main(["stop"] + sys.argv[1:])
    except BaseException as exc:                                        # never block stopping
        try:
            sys.stderr.write("sworn-hooks stop: %s: %s\n" % (type(exc).__name__, exc))
        except Exception:
            pass
    sys.exit(0)
