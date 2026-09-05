# sworn hooks for Claude Code — the weak rung

This is the weak rung. These two hooks are adapters: they turn what Claude Code's `PostToolUse`
and `Stop` events hand them into a `sworn/manifest/0.2` file and print `L1` on it, because a hook
that shares the agent's filesystem and shell can be fed what the agent's shell wrote a moment
earlier. They are not a recorder — in-toto Witness records what a step did and signs it; these
sign nothing, fetch nothing and verify no signature. They are blind, permanently, to files
written by shell commands (`cat > f`, a heredoc, `python -c`, `git apply`): only `Write` and
`Edit` enter `authored_sha256`, and a later `Read` of a shell-written file mints a receipt the
verifier will accept. The manifest lives outside your working tree because a manifest the agent
can `Read` into a receipt of itself is L0 wearing an L1 label — not because the agent's shell
cannot reach it; it can, and that is what L1 means.

Design: `papers/sworn/DESIGN_harness_adapters_2026_09_02.md`. Code: `styxx/harness/claude_code.py`
(not `styxx/hooks.py`, which is an unrelated OpenAI telemetry patch).

## What the pair does

- `post-tool.py` runs after every matched tool call. It reads the hook payload from stdin and
  stages one small JSON file per event under `<dir>/<session_id>/events/`, atomically, without
  importing `styxx` (the module is loaded by file path, so the package's eager imports never run
  on this path). Bash stdout and stderr become `tool_stdout` / `tool_stderr` receipts, a `Read`
  becomes `file_read`, a `WebFetch` becomes `http_fetch` (never complete — the tool returns a
  rendering, not the page); `Write` and `Edit` mint nothing and record what the agent wrote
  (`tool_input.content` / `new_string`, the reconstructed post-edit text, and the file on disk
  right after the tool) into `authored_sha256`.
- `stop.py` runs when the turn ends. It folds the staged events, in order of capture time, into
  `<dir>/<session_id>.manifest.json` at rung `L1`, records the last assistant message into
  `authored_sha256` as well, and prints the path on stderr. It is idempotent: the same staged set
  yields the same bytes. `claude --resume` reuses the session id, so the manifest accretes across
  resumed sessions; every receipt's note carries the `prompt_id` so a reader can cut by turn.

## Enabling it — in your own settings, never in a repository's

Put the block below in `~/.claude/settings.json` (every session on your machine) or in a
checkout's `.claude/settings.local.json` (that checkout only; add the file to
`.git/info/exclude`). Do not commit it as `.claude/settings.json`: hooks merge across settings
levels, so a committed block runs for every contributor and every session in that checkout,
subagents included, beside whatever `Stop` hook they already have. Replace `<repo>` with the
absolute path of a checkout of this repository; on Windows use forward slashes or doubled
backslashes inside the JSON string.

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash|Read|Write|Edit|WebFetch",
        "hooks": [
          {
            "type": "command",
            "command": "python \"<repo>/integrations/claude-code/sworn-hooks/post-tool.py\"",
            "timeout": 30
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python \"<repo>/integrations/claude-code/sworn-hooks/stop.py\"",
            "timeout": 60
          }
        ]
      }
    ]
  }
}
```

The manifest directory is `STYXX_SWORN_MANIFEST_DIR` when set, else `%LOCALAPPDATA%\styxx\sworn-manifests`
on Windows and `~/.local/share/styxx/sworn-manifests` elsewhere; `--dir DIR` after the script name
overrides both. Whatever you choose, it must resolve outside the session's `cwd` and outside
`CLAUDE_PROJECT_DIR` (the directory the session started in, which does not follow a worktree the
way `cwd` does), or the hook writes nothing and says so on stderr. The check is by resolved path
components, so a sibling directory whose name merely extends the workspace's is outside; a
`\\?\` path or a drive-less spelling such as `/c/Users/...` is refused rather than guessed at.

Both scripts exit zero on every input, including stdin they cannot read. A hook that blocks the
agent — exit code two — is a gate, and no precision has been measured for one.

## Swearing a document against the manifest

After the turn, the manifest is at `<dir>/<session_id>.manifest.json`. Author the report with
inline spans that name its receipts — `<sworn r="r3" k="quote">the test run printed `12 passed`</sworn>` —
then:

```
python -m styxx.sworn canon REPORT.md --manifest <dir>/<session_id>.manifest.json --commit <40-hex>
python -m styxx.sworn verify REPORT.sworn.json --repo . --out REPORT.sworn-receipt.json
```

Every span that rests on this manifest prints `L1` beside its verdict, and the receipt's `rungs`
block counts them. A span over bytes that are in `authored_sha256` is MALFORMED
`receipt_author_minted`: the agent swearing to itself. `python -m styxx.harness claude-code
post-tool|stop [--dir DIR]` runs the same code through the package (it pays the package import
each time, so the thin scripts above are the ones to configure as hooks).

## What this does NOT attest — read before relying on it

- That the agent did not write a file. Shell-written files, `MultiEdit`, `NotebookEdit` and every
  MCP filesystem tool are invisible to the hook; a `Read` of such a file yields a receipt the
  verifier accepts. The manifest's `harness` string says so on every manifest.
- That a receipt's bytes are the bytes on disk. A tool response is what the tool returned; it
  can drop a trailing newline, keep a CRLF the input lacked, or match an edit after whitespace
  normalisation. The on-disk read after `Write` and `Edit` narrows this; it does not close it.
- That anything ran after the turn, or on a machine the agent could not reach. That is L2, and
  this adapter offers no way to declare it.
- That a hook fired for every tool call. A hook can be disabled, time out, or be skipped by a
  tool the matcher does not name; a missing receipt is a gap, not a verdict.
- That the payload shapes are stable. The hook documentation and observed transcripts disagree
  on `tool_response`; both shapes are read, and a shape neither matches is serialised whole with
  a note saying so.

## Handle the manifest like the transcript

The manifest holds every matched tool response verbatim, base64-encoded: a `Read` of a secrets
file or a `Bash` that printed a token lands in it at rest, outside the working tree. A receipt
over `RECEIPT_BYTE_CAP` (see the module) is kept as its sha256 alone. Treat the manifest
directory with the care you give `~/.claude/projects/`.
