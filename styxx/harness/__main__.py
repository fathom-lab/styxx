# -*- coding: utf-8 -*-
"""``python -m styxx.harness {junit,github,claude-code} ...`` — dispatch to one adapter's main.

Each adapter's own exit contract holds: junit and github exit zero when they wrote a manifest
and two on a usage error; claude-code exits zero on every input. An unknown word here is a
usage error.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence

USAGE = ("usage: python -m styxx.harness {junit,github,claude-code} ...\n"
         "  junit REPORT --rung {L1,L2} [--turn ID] [--authored FILE ...] --out M.json\n"
         "  github --event EVENT.json --event-name {pull_request,pull_request_target,push} "
         "[--diff DIFF] [--diff-complete] --rung {L1,L2} [--after-turn-on-base] "
         "[--base-pinned-workflow] [--turn ID] --out M.json\n"
         "  claude-code {post-tool,stop} [--dir DIR]   (payload on stdin)\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        sys.stderr.write(USAGE)
        return 2
    if argv[0] in ("-h", "--help"):
        sys.stdout.write(USAGE)
        return 0
    name, rest = argv[0], argv[1:]
    if name == "junit":
        from styxx.harness import junit as mod
    elif name == "github":
        from styxx.harness import github as mod
    elif name == "claude-code":
        from styxx.harness import claude_code as mod
    else:
        sys.stderr.write("unknown adapter %r\n%s" % (name, USAGE))
        return 2
    return mod.main(rest)


if __name__ == "__main__":
    sys.exit(main())
