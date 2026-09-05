# CANNED — the Claude Code hook pair's manifest, verified through the command line

Fathom Lab · 2026-09-05 · **A canned run, not a result.** No Claude Code session was run. The
four payloads under `papers/sworn/canned_harness_claude_code_*.json` were fed on stdin to
`integrations/claude-code/sworn-hooks/post-tool.py` three times and to `stop.py` once, with
`--dir` naming a directory outside the payload's `cwd`; the manifest the Stop hook folded is
embedded in this document's sidecar. The turn they describe is a Bash test run, a Write of
`summary.txt` and a Read of the same file. One span below is MALFORMED on purpose: it is the
agent swearing to a file it wrote, and the verifier refuses it by set membership over
`authored_sha256`, end to end through the command line. Nothing here is a measurement of
anything.

<sworn r="r1" k="quote">The Bash receipt holds the test run's `3 passed in 0.42s`.</sworn>
<sworn r="r2" k="numeric">The file the agent wrote and then read says 3 tests passed.</sworn>

The second span resolves MALFORMED `receipt_author_minted` and this document's verdict is
therefore SWORN-FAILED, as intended: the receipt beside this document records the refusal, and
the RESULT that ships the adapters cites that leaf. The Write minted no receipt. The Bash
receipt is HELD at L1, and it would be HELD just the same had a shell command written the file
the Read later returned: that is the blindness the manifest's harness string prints.
