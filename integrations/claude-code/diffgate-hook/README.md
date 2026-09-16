# diffgate hook for Claude Code — the agent cannot post a message that lies about its diff

`pretool.py` is a `PreToolUse` hook. Every Bash call the agent is about to make passes
through it; when the call is a `git commit -m …` or a `gh pr create / edit --body …`, the
message is read against the diff it describes with `styxx.diffgate` — the staged diff for a
commit, `merge-base..HEAD` against the PR's base for a pull request — and a CONTRADICTED
claim blocks the call. Claude Code hands the verdict back to the agent as the reason, so the
agent fixes the message or the diff. It does not get to choose the verdict.

```
pip install styxx
```

`~/.claude/settings.json` (every session on your machine) or a checkout's
`.claude/settings.local.json` (that checkout only):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          { "type": "command",
            "command": "python3 /ABSOLUTE/PATH/TO/styxx/integrations/claude-code/diffgate-hook/pretool.py" }
        ]
      }
    ]
  }
}
```

What the agent sees when it tries:

```
$ git commit -m "Refactored src/retry.py. Added 3 tests. Only touches files under src/."
styxx diffgate: BLOCKED — the commit message vs the staged diff contradicts the diff in 2 claim(s). Fix the message or the diff; the verdict is not yours to choose.
  [ok ] file_touched         diff status 'M' for 'src/retry.py'
  [LIE] tests_added          diff adds 1 test functions, claim says 3
  [LIE] only_touches         paths outside 'src': ['tests/test_retry.py']
```

The heredoc form Claude Code writes by default (`-m "$(cat <<'EOF' … EOF)"`) is read;
`--body-file` is read; `--amend` without a new message, commands the parser cannot make
sense of, a repository with nothing to diff against, and a missing `styxx` install are all
allowed through with at most a note — a hook that guessed would be an instrument that
accuses. What it reads is the closed template set of the CLI and the Action; "tests pass"
stays UNCHECKABLE (the hook never runs the suite: that would hand the agent's shell a second
place to write the verdict it wants); a path the diff does not show is UNCHECKABLE, not an
accusation (EXTERNAL-1).

Read-only git, no network, nothing executed from the command. Tested by driving the script
the way Claude Code drives it — JSON on stdin — against a temporary repository: lying `-m`
and heredoc messages blocked with the lies named, honest ones allowed, non-git commands
untouched, `gh pr create` with a lying `--body` and `--body-file` blocked against the real
merge-base, honest bodies allowed.
