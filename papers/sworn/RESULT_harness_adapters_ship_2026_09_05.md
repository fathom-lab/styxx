# RESULT — harness adapters v0.1 ship: manifest minters at L1 and L2, adapters and never a recorder

Fathom Lab · 2026-09-05 · Design: `DESIGN_harness_adapters_2026_09_02.md`, re-sworn and frozen in
its own commit before any code. Module: `styxx/harness/` (`junit.py`, `github.py`,
`claude_code.py`) and the hook pair under `integrations/claude-code/sworn-hooks/`. Tests:
`tests/test_harness_junit.py`, `tests/test_harness_github.py`, `tests/test_harness_claude_code.py`,
`tests/test_harness_purity_boundary.py`, `tests/test_harness_cli.py`. Harness: `styxx.harness.junit`
itself, turn `turn_2026_09_05_harness_adapters`, rung L1. **This document is itself sworn**: every
count in it is bound to a receipt in a manifest the JUnit adapter minted over pytest's own report
of the run, or to a leaf of a file committed at the commit the sidecar names. Nothing here is a
measurement of the format; that remains `DESIGN_sworn_measurement_v2_2026_09_02.md`, waiting on a
signature.

## What was built

An adapter turns bytes a harness already holds into a `sworn/manifest/0.2` file and signs,
fetches, observes and verifies nothing. <sworn r="path:styxx/harness/__init__.py" k="quote">The package docstring leads with `adapters, never a recorder`, which is the plan's label for this leg.</sworn>
The JUnit adapter reads a test report through `styxx.evidence` and copies the reader's counts into
receipts, withholding them when the reader parsed no source; the GitHub adapter records an event
payload, its shas and a diff the caller fetched, prints the fork caveat on every manifest and
refuses L2 without the caller's declarations; the Claude Code pair stages one file per PostToolUse
event without importing `styxx` and folds them at Stop into a manifest at L1.
<sworn r="path:styxx/harness/claude_code.py" k="quote">That adapter's docstring opens by saying it is `blind, permanently, to files written by shell commands`, and the same sentence is printed into every manifest it mints.</sworn>
<sworn r="path:pyproject.toml" k="quote">The wheel ships the package: the explicit packages list carries `"styxx.harness",` and nothing else in that file moved.</sworn>
<sworn r="path:styxx/sworn.py" k="hash">The verifier the adapters target did not move under this leg and hashes, at the commit this document names, to 110f3461bb6772248c987f9707ac7ef3a0d5df4d37b4d26124b7aa82551c04c9.</sworn>

## The run this document swears to

The adapter tests were run with `--junitxml` and the report was handed to the JUnit adapter:
`python -m styxx.harness junit papers/sworn/turn_2026_09_05_harness_adapters.junit.xml --rung L1 --turn turn_2026_09_05_harness_adapters`.
The adapter is dogfooded by the document that announces it.

<sworn r="r1" k="numeric">The reader resolved 139 passed testcases from the report.</sworn>
<sworn r="r2" k="numeric">It resolved 0 failures.</sworn>
<sworn r="r4#/totals/errors" k="numeric">It kept 0 harness errors apart from those failures.</sworn>
<sworn r="r4#/outcome" k="quote">The reader's outcome over the report reads `PASSED`.</sworn>
<sworn r="r4#/sources/0/producer_guess" k="quote">The reader's producer guess for the report is `pytest`.</sworn>
<sworn r="path:papers/sworn/turn_2026_09_05_harness_adapters.junit.xml" k="hash">The report the manifest was minted over is committed and hashes to 2160794ea61c302aeb8fa898eaf56ff632e45152fdf3fd1e384c36437eaa0042, the same digest the manifest records for the report receipt.</sworn>

## The rung, and why

L1, declared and printed. The manifest was minted on the author's machine by the author's own
shell right after the same shell ran the tests; the adapter read a file that shell could have
written, and nothing about the run was after the turn or on a machine the agent could not
reach. That is the rung R6 calls weak, and the honest one for a run like this. A reader who wants
L2 for these counts runs the same command on a runner the author cannot write to and declares it
so; the adapter offers the flag and checks nothing.

## The canned runs, verified through the command line

One manifest per adapter was minted over inputs written by hand and committed under
`papers/sworn/canned_harness_*`, and a short document was sworn against each with
`python -m styxx.sworn verify ... --manifest ...`.
<sworn r="path:papers/sworn/CANNED_harness_junit_2026_09_05.sworn-receipt.json#/rungs/L1" k="numeric">The JUnit document's receipt counts 5 spans standing on the rung its manifest declared, and none standing anywhere else.</sworn>
<sworn r="path:papers/sworn/CANNED_harness_github_2026_09_05.sworn-receipt.json#/counts/HELD" k="numeric">The GitHub document's receipt counts 4 spans HELD over a pull request from a fork.</sworn>
<sworn r="path:papers/sworn/CANNED_harness_claude_code_2026_09_05.sworn-receipt.json#/spans/1/reason" k="quote">The Claude Code document swears, on purpose, to a file the agent wrote and then read, and its receipt records the refusal as `receipt_author_minted`.</sworn>
<sworn r="path:papers/sworn/CANNED_harness_claude_code_2026_09_05.sworn-receipt.json#/document_verdict" k="quote">That document's verdict is therefore `SWORN-FAILED`, by design, and its own text says so.</sworn>

## What this does not say

That any rung has been verified: R6 prints, nothing checks, and L3 is reserved. That an L2
manifest is trustworthy beyond the caller's declaration and the runner it describes. That the
Claude Code adapter records what the agent wrote: it records what the Write and Edit tools
reported and what was on disk after them, and it is blind to the shell, which every manifest it
mints says. That the hook payload shapes are stable: the adapter tolerates two shapes because
neither is a contract. That the counts above are true of any run other than the one on the
author's machine that the committed report describes. That a canned run is a result. That the
format has been measured. That anyone outside this lab has minted a manifest with these adapters.
No adversarial pass has been run against this module; the standing rule says it is not announced
until one has, and the battery is owed below.

## Owed

An adversarial pass against `styxx/harness/` as committed, with a dated ERRATA appended to the
design and a battery pinned by tests. The Action that would consume the GitHub adapter (leg 3,
item 4), whose README owes the fork caveat. A run of the adapter tests on a runner the author
cannot write to, so that a manifest over them can honestly print L2. The measurement of the
format, unchanged.

---

*A harness is where the world gets in. These adapters let it in through a named door, write down
which door, and print the rung on the way out; this document was sworn against a manifest one of
them minted, at the rung it could honestly print, and no higher.*
