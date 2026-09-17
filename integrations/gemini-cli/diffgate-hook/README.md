# diffgate hook for Gemini CLI — the same file as the Claude Code hook

Gemini CLI's hooks speak the same protocol as Claude Code's and Codex's: a `BeforeTool` event
matched on the tool name, the tool call on stdin as JSON with `tool_name`, `tool_input.command`
and `cwd`, and exit code 2 with the reason on stderr to block the call. Gemini names its shell
tool `run_shell_command`, and the Claude Code hook accepts that name beside `Bash`, so
`integrations/claude-code/diffgate-hook/pretool.py` is the Gemini hook too — one file, no copy.
When the agent is about to run a `git commit -m …` (the heredoc form included) or a
`gh pr create` / `gh pr edit --body …`, the message is read against the diff it describes with
`styxx.diffgate`; a CONTRADICTED claim blocks the command, and Gemini hands the verdict back to
the agent as the rejection reason. Anything the parser cannot read is allowed through untouched.

```
pip install styxx
```

`~/.gemini/settings.json` (every project) or a project's `.gemini/settings.json` (that project
only):

```json
{
  "hooks": {
    "BeforeTool": [
      {
        "matcher": "run_shell_command",
        "hooks": [
          { "type": "command",
            "name": "styxx-diffgate",
            "description": "the commit message or PR body vs the diff",
            "command": "python3 /ABSOLUTE/PATH/TO/styxx/integrations/claude-code/diffgate-hook/pretool.py" }
        ]
      }
    ]
  }
}
```

What the agent sees is the Claude Code hook's stderr, verbatim:

```
styxx diffgate: BLOCKED — the commit message vs the staged diff contradicts the diff in 2 claim(s). Fix the message or the diff; the verdict is not yours to choose.
  [ok ] file_touched         diff status 'M' for 'src/retry.py'
  [LIE] tests_added          diff adds 1 test functions, claim says 3
  [LIE] only_touches         paths outside 'src': ['tests/test_retry.py']
```

`tests/test_gemini_hook.py` drives the file with Gemini's stdin object (`hook_event_name`
`BeforeTool`, `timestamp`, `mcp_context`, `tool_name` `run_shell_command`) and checks the lying
message exits 2 with the lies on stderr, the honest one exits 0, and a `write_file` call is left
alone. The field names and the exit-code rule come from geminicli.com/docs/hooks/reference as
of 2026-09-16; the protocol is what was tested here, not the Gemini app itself. Everything the
Claude Code hook's README says about scope applies: the closed template set, "tests pass"
UNCHECKABLE, a path the diff does not show UNCHECKABLE and never accused (EXTERNAL-1), the open
false-VERIFIED class #101, and the weak rung of any hook that shares the agent's shell. The
enforcement is the Action.
