# CANNED — the GitHub adapter's manifest, verified through the command line

Fathom Lab · 2026-09-05 · **A canned run, not a result.** The manifest embedded in this
document's sidecar was minted by `python -m styxx.harness github --event papers/sworn/canned_harness_github_event.json --event-name pull_request --diff papers/sworn/canned_harness_github.diff --diff-complete --rung L1 --turn canned_harness_github`
over an event payload and a diff written by hand: a pull request from a fork. The same event at
L2 with `--after-turn-on-base` and without `--base-pinned-workflow` was refused at the command
line with no manifest written, which is the fork caveat doing its work; the refusal is pinned by
`tests/test_harness_github.py` and leaves no receipt here. Nothing here is a measurement of
anything.

<sworn r="r5#/pull_request/number" k="numeric">The payload names pull request 7.</sworn>
<sworn r="r5#/pull_request/head/repo/full_name" k="quote">Its head repository is `someone-else/styxx-fork`.</sworn>
<sworn r="r2" k="quote">The base sha recorded is `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`.</sworn>
<sworn r="r1" k="absent">The diff, complete as the caller asserted, carries no `TODO`.</sworn>

The rung is L1 because the caller declared nothing about when or where anything ran; the
manifest's harness string prints `fork: true` and the fork caveat beside it, as it does on every
manifest this adapter mints.
