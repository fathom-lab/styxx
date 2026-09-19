# AUDIT — the papers, read against each other: one stale headline, and a guard that had stopped firing

Fathom Lab · 2026-09-18 · **Not a measurement and not preregistered.** A sweep of the committed
papers for the defect class that surfaced by accident on 2026-09-18, plus the reading that
decided each candidate. Scanner: `audit_cross_paper_numbers.py`, committed here and re-runnable.
Instrument unchanged at sha256 `9b620e00…` and not run.

## Why this exists

Writing a module header, an agent quoted the instrument's `only_touches` coverage as 5.4%, then
"corrected" it to DECIDE-1's *17 of 299*, then discovered that its own correction was the
regression: DECIDE-1 measured against instrument `4ba947a8…` on 2026-09-17, and SCOPE-1
re-measured against `9b620e00…` a day later and got *16 of 299*. Both papers are right about
their own instrument. Neither mentions the other.

That was luck. This document is what looking on purpose found.

## Method, and what it deliberately does not do

The scanner takes every `N of M` in `papers/**.md`, pairs occurrences that share `M`, differ in
`N`, sit in different files and share at least three content words nearby, and **prints them**.
It labels nothing. A script that scanned the record and announced contradictions would repeat
the error this lab has now recorded three times in two days — BENCH-1 with the prefix extractor,
BENCH-2 with the oracle's acceptance rate, DECIDE-1 with the instrument's own admission rate,
each one machinery's output mistaken for ground truth. So the script proposes and a reader
decides, and the reading is below.

`AUDIT_*.md` files are skipped, this one included: an audit quotes the numbers it is discussing,
so leaving it in makes the scan find its own commentary and makes the count below stop
reproducing. That is the only exclusion and it is named in the script.

**20 candidate pairs. One real supersession.** A 5% hit rate is what an honest proposer looks
like. Re-run it:

```
python papers/closed-model-frontier/audit_cross_paper_numbers.py
```

## The one that is real

| | |
|---|---|
| `RESULT_decide1_decidable_fraction_2026_09_17.md` | instrument returns a verdict on **17 of 299** `only_touches` claims — abstains on 94% |
| `RESULT_scope1_ABANDONED_2026_09_18.md` | instrument returns a verdict on **16 of 299** — **5.4%** [3.3%, 8.5%] |

Not a disagreement: the instrument moved between them (`4ba947a8…` → `eba8f5fc…` → `9b620e00…`,
three versions in two days), and each paper pins the sha it used in its own header. The record is
traceable. What it is missing is a pointer: **DECIDE-1's title makes a present-tense claim — "The
instrument abstains on 94% of them" — and nothing in that document tells a reader arriving today
that its instrument-derived figure has been superseded.** Every other number in DECIDE-1 is hand
adjudication (71% decidable, 52% for `only_touches`, the per-kind table) and is untouched by any
instrument change.

This repository already has the convention for exactly this. `FINDING_mount_fpr_live_2026_06_12.md`
carries a `> **RESOLVED (2026-06-13, see ...)**` banner under its own title, pointing at the
finding that superseded its catch rate, and the successor points back. Nothing new had to be
invented; the banner simply was not written. It is now appended to DECIDE-1, in that form.

## The nineteen that are not

Read in full, not dismissed by pattern:

- **`4 of 36` vs `~2 of 36` wrong under path-diverse derivation.** The finding says 4 items
  remained wrong and 2 of those were stably wrong across all five methods; the synthesis quotes
  the 2. A subset, not a conflict. (The synthesis writes `~2` for a count that is exactly 2 and
  names both items; that is loose, not wrong.)
- **`8 of 20` vs `17 of 20` caves caught.** Superseded — and **handled correctly**. The 0.40
  finding carries a RESOLVED banner naming the regime-matched re-run, and the 0.85 finding names
  the paper it resolves. This pair is the worked example of what DECIDE-1 was missing.
- **`119 of 269` vs `117 of 269` battery catch.** v0.4 cycle 25 and v0.5 cycle 38, against a
  preregistered floor of 116. Different versions, both stated with theirs.
- **`≤ 13 of 269` vs `≤ 26 of 269` false-verify.** Gate thresholds in different preregistrations,
  one a deliberate strict halving of the other. Thresholds, not results.
- **`27 of 30` vs `30 of 30` decoys.** A gate bar and the outcome that cleared it.
- **`1 of 40` vs `16 of 40` pairs improved.** Different N-comparisons in a scaling series.
- The remainder are table-row indices, corpus sizes reused across unrelated quantities, and line
  truncations that put two unrelated numbers in one window.

## The more serious finding — and the correction it needed

> **Correction, appended 2026-09-19.** The first version of this section said the class had not
> been repaired. That was wrong, and wrong in this document's own way: measured against `main`
> and described as the state of the repository. `tests/test_port_is_current.py` — introduced in
> **#127**, commit `325667bd`, *"tests: the guard that would have caught the port falling a cycle
> behind"* — already pins the instrument, already holds `web/gate/README.md` to the same hash,
> already asserts every corpus `check_pairs.js` names exists and is whitelisted in `.gitignore`,
> and already runs both implementations over the pinned pairs. It was sitting unmerged in the same
> stack as this note. The sweep found no new defect here; it found a branch its author had not
> read. What survives is narrower, and is stated below.

`web/gate/differential/py_side.py` pins the instrument's sha256 and refuses to run against
anything else, so "0 disagreements between the Python and the browser port" always means
"against the file the port claims to be". Good design. But **`web/gate/` appears in no CI
workflow**, so the guard only fires when a person runs it by hand.

The consequence is already written down, in a comment above the pin:

> ... which is why this script has been REFUSING TO RUN on main ever since -- the check that
> guards the two-implementation claim was itself disabled, which is how the port fell a whole
> cycle behind without anything failing.

The port fell a cycle behind, the public bookmarklet served a stale reading, and every check
stayed green. That instance was found and repaired, and so was the class — in #127, a branch
above this one.

`benchmarks/silent_pass/CORPUS.md` catalogues this shape as SP-1, an absent measurement
surfacing as a passing check, and `tests/test_ledger.py` makes the same argument about the
ledger's regeneration guarantee — in its own words, "an absent measurement surfacing as a
passing check is the defect class this repository exists to document ... It had been sitting in
our own suite." It was sitting in the browser door too.

What #127 did not close is one line at the end of its own file:

```python
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; ...")
```

**A skip is green.** On any runner without node, the single test that holds the browser port to
the pinned pairs reports success while checking nothing — failure mode (1) of that file's own
docstring, reintroduced two screens below where it is named. `tests/test_ledger.py` had already
settled how this repository answers that: repair the precondition, and **fail in CI instead of
skipping**, because "a developer with a shallow clone is not the person hiding a defect."

Both halves are now applied. `test.yml` installs node for the test job, so the precondition is
repaired rather than assumed; and the node branch raises in CI instead of skipping, so removing
that step sets off an alarm rather than going quiet. Checked in both directions: with `CI` set and
node off `PATH` it fails with the message naming the workflow step, and without `CI` it still
skips for a contributor who simply has no node.

## What this audit does not cover

Percentages stated without a denominator. Numbers written only as prose. Quantities that agree
across papers but were measured against different instruments — the case a reader should worry
about most, and the one no scan can settle, because two different numbers can both be right and
only the pinned sha in each header says which. Every RESULT here carries that sha. Whether a
reader thinks to check it is not something a test can guarantee.
