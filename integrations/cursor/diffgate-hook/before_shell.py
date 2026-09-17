#!/usr/bin/env python3
"""Cursor beforeShellExecution hook: the agent cannot run a commit or PR command whose message lies about its diff.

Wire it in ~/.cursor/hooks.json (every project) or <project>/.cursor/hooks.json (that project):

    {"version": 1, "hooks": {"beforeShellExecution": [
        {"command": "python3 /ABS/PATH/integrations/cursor/diffgate-hook/before_shell.py"}]}}

Cursor hands every shell command the agent is about to run to this script as JSON on stdin
({"command": ..., "cwd": ...}). When the command is a `git commit -m …` or a `gh pr create/edit
--body …`, the message is read against the diff it describes with styxx.diffgate (staged diff for
a commit; merge-base..HEAD against the PR's base for a pull request), and a CONTRADICTED claim
answers {"permission": "deny"} with the verdict in agent_message, which Cursor hands back to the
agent as the reason. The agent then fixes the message or the diff; it does not get to choose the
verdict. Anything else, and any command this script cannot parse, answers {"permission": "allow"}
untouched — a hook that guessed would be an instrument that accuses.

The command parser is the Claude Code hook's (integrations/claude-code/diffgate-hook/pretool.py),
verbatim; tests/test_cursor_hook.py checks the two stay identical. Never runs the agent's tests,
never executes anything from the command. Read-only git.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys


def _git(cwd: str, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def _heredoc(command: str) -> str | None:
    """Claude Code writes commit messages as  -m "$(cat <<'EOF' … EOF )"  — read that body."""
    m = re.search(r"<<-?\s*['\"]?(\w+)['\"]?\n(.*?)\n\1\b", command, re.S)
    return m.group(2) if m else None


def _flag_values(tokens: list[str], names: tuple[str, ...]) -> list[str]:
    out = []
    for i, t in enumerate(tokens):
        for n in names:
            if t == n and i + 1 < len(tokens):
                out.append(tokens[i + 1])
            elif t.startswith(n + "=") and n.startswith("--"):
                out.append(t[len(n) + 1:])
    return out


def _parse(command: str, cwd: str):
    """-> (kind, message, diff) or None when the command is not one this hook reads."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    heredoc = _heredoc(command)
    # git commit
    if "git" in tokens and "commit" in tokens and tokens.index("commit") > tokens.index("git"):
        if "--amend" in tokens and not any(t in ("-m", "--message") or t.startswith("--message=") for t in tokens):
            return None
        msgs = _flag_values(tokens, ("-m", "--message"))
        message = heredoc if heredoc else "\n\n".join(msgs)
        if not message.strip():
            return None
        staged = _git(cwd, "diff", "--cached", "--no-color", "--no-ext-diff", "--no-renames")
        if "-a" in tokens or "--all" in tokens or any(t.startswith("-a") and t[1:].isalpha() and "a" in t for t in tokens if t.startswith("-") and not t.startswith("--")):
            unstaged = _git(cwd, "diff", "--no-color", "--no-ext-diff", "--no-renames")
            staged = (staged or "") + (unstaged or "")
        return ("commit", message, staged if staged is not None else "")
    # gh pr create / edit
    if "gh" in tokens and "pr" in tokens and any(t in ("create", "edit") for t in tokens):
        body = heredoc
        if body is None:
            bodies = _flag_values(tokens, ("--body", "-b"))
            files = _flag_values(tokens, ("--body-file", "-F"))
            if files:
                try:
                    body = open(os.path.join(cwd, files[0]), encoding="utf-8").read()
                except OSError:
                    body = None
            elif bodies:
                body = bodies[0]
        if not body or not body.strip():
            return None
        bases = _flag_values(tokens, ("--base", "-B"))
        base = bases[0] if bases else None
        if base is None:
            head_ref = _git(cwd, "symbolic-ref", "refs/remotes/origin/HEAD")
            base = head_ref.strip().rsplit("/", 1)[-1] if head_ref else "main"
        mb = _git(cwd, "merge-base", f"origin/{base}", "HEAD") or _git(cwd, "merge-base", base, "HEAD")
        if not mb:
            return None
        diff = _git(cwd, "diff", "--no-color", "--no-ext-diff", "--no-renames", f"{mb.strip()}..HEAD")
        return ("pr", body, diff if diff is not None else "")
    return None


def _answer(permission: str, user_message: str | None = None, agent_message: str | None = None) -> int:
    out = {"permission": permission}
    if user_message:
        out["user_message"] = user_message
    if agent_message:
        out["agent_message"] = agent_message
    sys.stdout.write(json.dumps(out) + "\n")
    return 0


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return _answer("allow")
    command = payload.get("command") or ""
    cwd = payload.get("cwd") or os.getcwd()
    parsed = _parse(command, cwd)
    if parsed is None:
        return _answer("allow")
    kind, message, diff = parsed
    try:
        from styxx.diffgate import gate_diff_text
    except ImportError:
        return _answer("allow", user_message="styxx diffgate hook: `pip install styxx` to gate commit messages and PR bodies (allowed through)")
    g = gate_diff_text(message, diff)
    if not g.measured:
        return _answer("allow")      # nothing to read against; not this hook's call to block
    marks = {"VERIFIED": "ok ", "CONTRADICTED": "LIE", "UNCHECKABLE": " ? "}
    lines = [f"  [{marks[c.verdict]}] {c.kind:20s} {c.why}" for c in g.claims]
    what = "commit message vs the staged diff" if kind == "commit" else "PR body vs the diff against its base"
    if g.verdict == "FAIL":
        n = sum(1 for c in g.claims if c.verdict == "CONTRADICTED")
        head = (f"styxx diffgate: BLOCKED — the {what} contradicts the diff in {n} claim(s). "
                "Fix the message or the diff; the verdict is not yours to choose.")
        return _answer("deny", user_message=head, agent_message=head + "\n" + "\n".join(lines))
    return _answer("allow")


if __name__ == "__main__":
    sys.exit(main())
