# -*- coding: utf-8 -*-
"""The commit-msg gate as a console script, for pre-commit and for plain git hooks.

    styxx-diffgate-commit-msg .git/COMMIT_EDITMSG      # what git and pre-commit both run

Reads the commit message (git's `#` commentary stripped) against `git diff --cached` with
styxx.diffgate, prints each diff-shaped claim as [ok ] / [LIE] / [ ? ], and exits 1 on a
CONTRADICTED claim so the commit is refused. Nothing else is judged: prose outside the closed
template set (touched / created / deleted paths, "N files changed", "added N tests", "adds
function X", "only touches prefix/", "tests pass") is never read, and the summary line says how
many sentences were not. "tests pass" is UNCHECKABLE here on purpose: a hook that ran the suite
would be a second place for the agent's shell to write the verdict it wants. A path the diff
does not show is UNCHECKABLE, not an accusation (EXTERNAL-1).

Same behaviour as integrations/git/commit-msg, which is this module as a single copyable file.
The hook is never installed by a checkout: git ignores hooks in the tree and pre-commit installs
only what the person's own .pre-commit-config.yaml names, so running it is always the person's
decision, never the diff's. `git commit --no-verify` overrides one commit.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def gate_message(message: str, cwd: str = ".") -> int:
    """Gate MESSAGE against the staged diff in CWD; print the report; return the exit code."""
    from styxx.diffgate import gate_diff_text

    diff = subprocess.run(
        ["git", "diff", "--cached", "--no-color", "--no-ext-diff", "--no-renames"],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    g = gate_diff_text(message, diff)
    marks = {"VERIFIED": "ok ", "CONTRADICTED": "LIE", "UNCHECKABLE": " ? "}
    for c in g.claims:
        print(f"  [{marks[c.verdict]}] {c.kind:20s} {c.why}")
    if not g.measured:
        print(f"styxx diffgate: UNMEASURED — {g.why_unmeasured}; nothing to gate against")
        return 0
    if g.verdict == "FAIL":
        n = sum(1 for c in g.claims if c.verdict == "CONTRADICTED")
        print(f"styxx diffgate: FAIL — {n} claim(s) in the message contradict the staged "
              "diff. fix the message or the diff; `git commit --no-verify` overrides.")
        return 1
    if g.claims:
        print(f"styxx diffgate: PASS — {len(g.claims)} claim(s) read against the staged diff, "
              f"{g.uncovered_sentences} of {g.sentences_total} sentences never read")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 1:
        print("usage: styxx-diffgate-commit-msg COMMIT_MSG_FILE", file=sys.stderr)
        return 2
    try:
        import styxx.diffgate  # noqa: F401
    except ImportError:
        print("styxx diffgate hook: `pip install styxx` first (skipping, commit allowed)")
        return 0
    raw = Path(argv[0]).read_text(encoding="utf-8", errors="replace")
    # git's own commentary in the template is not the message
    message = "\n".join(l for l in raw.splitlines() if not l.startswith("#")).strip()
    if not message:
        return 0
    return gate_message(message)


if __name__ == "__main__":
    sys.exit(main())
