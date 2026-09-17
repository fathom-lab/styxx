# diffgate hook for Cursor — the agent cannot run a commit or PR command whose message lies about its diff

`before_shell.py` is a Cursor `beforeShellExecution` hook. Cursor hands it every shell command
the agent is about to run; when the command is a `git commit -m …` or a `gh pr create / edit
--body …`, the message is read against the diff it describes with `styxx.diffgate` — the staged
diff for a commit, `merge-base..HEAD` against the PR's base for a pull request — and a
CONTRADICTED claim answers `{"permission": "deny"}` with the verdict in `agent_message`, which
Cursor hands back to the agent as the reason. The agent fixes the message or the diff. It does not
get to choose the verdict.

```
pip install styxx
```

`~/.cursor/hooks.json` (every project) or a project's `.cursor/hooks.json` (that project only):

```json
{
  "version": 1,
  "hooks": {
    "beforeShellExecution": [
      { "command": "python3 /ABSOLUTE/PATH/TO/styxx/integrations/cursor/diffgate-hook/before_shell.py" }
    ]
  }
}
```

What Cursor receives when the agent tries (the hook's stdout, for the command
`git commit -m "Refactored src/retry.py. Added 3 tests. Only touches files under src/."` with one
modified file and one new test staged; pretty-printed here, one line in reality):

```json
{
  "permission": "deny",
  "user_message": "styxx diffgate: BLOCKED — the commit message vs the staged diff contradicts the diff in 2 claim(s). Fix the message or the diff; the verdict is not yours to choose.",
  "agent_message": "styxx diffgate: BLOCKED — the commit message vs the staged diff contradicts the diff in 2 claim(s). Fix the message or the diff; the verdict is not yours to choose.\n  [ok ] file_touched         diff status 'M' for 'src/retry.py'\n  [LIE] tests_added          diff adds 1 test functions, claim says 3\n  [LIE] only_touches         paths outside 'src': ['tests/test_retry.py']"
}
```

Everything else answers `{"permission": "allow"}` untouched: any command the parser cannot make
sense of, `--amend` without a new message, a repository with nothing to diff against, an
unmeasurable diff, unparseable stdin, and a machine without `styxx` (that one carries a note in
`user_message`). A hook that guessed would be an instrument that accuses. The heredoc form
(`-m "$(cat <<'EOF' … EOF)"`) is read; `--body-file` is read. What it reads is the closed
template set of the CLI and the Action; "tests pass" stays UNCHECKABLE (the hook never runs the
suite: that would hand the agent's shell a second place to write the verdict it wants); a path
the diff does not show is UNCHECKABLE, not an accusation (EXTERNAL-1). One known false-VERIFIED
class is open as #101 (a changed `def` counts as added) and applies here as everywhere.

The command parser is the Claude Code hook's (`integrations/claude-code/diffgate-hook/pretool.py`)
verbatim; `tests/test_cursor_hook.py` checks the two files' `_git`, `_heredoc`, `_flag_values`
and `_parse` stay identical. Read-only git, no network, nothing executed from the command. Tested
by driving the script the way Cursor drives it — the documented JSON on stdin — against a
temporary repository: a lying `-m` and a lying heredoc denied with the lies named, an honest
message allowed, non-git commands and claim-free messages allowed, unparseable stdin allowed.

Rung: weak, and said so. This hook shares a shell with the agent that wrote the message. It is
the habit; the enforcement is the same gate in CI where the agent's shell cannot reach.
