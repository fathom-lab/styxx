# diffgate hook for Codex CLI — the same file as the Claude Code hook

Codex CLI's hooks speak the same protocol as Claude Code's: a `PreToolUse` event with a
`matcher` of `"Bash"`, the tool call as JSON on stdin with `tool_name`, `tool_input.command` and
`cwd`, and exit code 2 with the reason on stderr to block the call. So the Claude Code hook,
`integrations/claude-code/diffgate-hook/pretool.py`, is the Codex hook too — one file, no copy.
When the agent is about to run a `git commit -m …` (the heredoc form included) or a
`gh pr create` / `gh pr edit --body …`, the message is read against the diff it describes with
`styxx.diffgate`; a CONTRADICTED claim blocks the command, and Codex hands the verdict back to
the agent as the reason. Anything the parser cannot read is allowed through untouched.

```
pip install styxx
```

`~/.codex/hooks.json` (every project) or a project's `.codex/hooks.json` (that project only):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          { "type": "command",
            "command": "python3 /ABSOLUTE/PATH/TO/styxx/integrations/claude-code/diffgate-hook/pretool.py",
            "statusMessage": "styxx diffgate: the message vs the diff" }
        ]
      }
    ]
  }
}
```

The same block goes in `~/.codex/config.toml` as `[hooks]` tables if you keep hooks there. If
hooks are switched off in your config (`[features] hooks = false`), nothing here runs.

What the agent sees is the Claude Code hook's stderr, verbatim:

```
styxx diffgate: BLOCKED — the commit message vs the staged diff contradicts the diff in 2 claim(s). Fix the message or the diff; the verdict is not yours to choose.
  [ok ] file_touched         diff status 'M' for 'src/retry.py'
  [LIE] tests_added          diff adds 1 test functions, claim says 3
  [LIE] only_touches         paths outside 'src': ['tests/test_retry.py']
```

`tests/test_codex_hook.py` drives the file with Codex's stdin object — its extra fields
(`turn_id`, `tool_use_id`, `permission_mode`, `transcript_path`, `model`) alongside the shared
ones — and checks the lying message exits 2 with the lies on stderr, the honest one exits 0, and
an `apply_patch` call is left alone. The field names and the exit-code rule come from
developers.openai.com/codex/hooks as of 2026-09-16; the protocol is what was tested here, not
the Codex app itself. Everything the Claude Code hook's README says about scope applies: the
closed template set, "tests pass" UNCHECKABLE, a path the diff does not show UNCHECKABLE and
never accused (EXTERNAL-1), the open false-VERIFIED class #101, and the weak rung of any hook
that shares the agent's shell. The enforcement is the Action.
