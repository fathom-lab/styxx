#!/usr/bin/env python3
"""Claude Code PreToolUse hook: the agent cannot post a message that lies about its diff.

Wire it in ~/.claude/settings.json (or a checkout's .claude/settings.local.json):

    {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [
        {"type": "command", "command": "python3 /ABS/PATH/integrations/claude-code/diffgate-hook/pretool.py"}]}]}}

Every Bash tool call the agent is about to make is handed here as JSON on stdin. When the
command is a `git commit -m …` or a `gh pr create/edit --body …`, the message is read against
the diff it describes with styxx.diffgate (staged diff for a commit; merge-base..HEAD against
the PR's base for a pull request), and a CONTRADICTED claim blocks the call — exit 2, with the
verdict on stderr, which Claude Code feeds back to the agent as the reason. The agent then
fixes the message or the diff; it does not get to choose the verdict. Anything else, and any
command this script cannot parse, is allowed through untouched — a hook that guessed would be
an instrument that accuses.

Codex CLI and Gemini CLI speak the same hook protocol (a PreToolUse / BeforeTool event, the same
stdin object, exit 2 with the reason on stderr); Gemini names its shell tool run_shell_command, so
that name is accepted beside Bash. See integrations/codex/ and integrations/gemini-cli/.

Never runs the agent's tests, never executes anything from the command. Read-only git.
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


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    if payload.get("tool_name") not in ("Bash", "run_shell_command"):   # Claude Code / Codex; Gemini CLI
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""
    cwd = payload.get("cwd") or os.getcwd()
    parsed = _parse(command, cwd)
    if parsed is None:
        return 0
    kind, message, diff = parsed
    try:
        from styxx.diffgate import gate_diff_text
    except ImportError:
        print("styxx diffgate hook: `pip install styxx` to gate commit messages and PR bodies (allowed through)", file=sys.stderr)
        return 0
    g = gate_diff_text(message, diff)
    marks = {"VERIFIED": "ok ", "CONTRADICTED": "LIE", "UNCHECKABLE": " ? "}
    lines = [f"  [{marks[c.verdict]}] {c.kind:20s} {c.why}" for c in g.claims]
    what = "commit message vs the staged diff" if kind == "commit" else "PR body vs the diff against its base"
    if not g.measured:
        return 0            # nothing to read against; not this hook's call to block
    if g.verdict == "FAIL":
        n = sum(1 for c in g.claims if c.verdict == "CONTRADICTED")
        sys.stderr.write("styxx diffgate: BLOCKED — the " + what + " contradicts the diff in "
                         f"{n} claim(s). Fix the message or the diff; the verdict is not yours to choose.\n"
                         + "\n".join(lines) + "\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
