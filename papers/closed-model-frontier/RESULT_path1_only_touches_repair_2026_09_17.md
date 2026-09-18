# RESULT — PATH-1 lands: precision 0.18 → 0.25, exactly as predicted, and that is still a bad number

Fathom Lab · 2026-09-17 · Prereg `PREREG_path1_only_touches_repair_2026_09_17.md`, sha256
`618d800f…`, frozen before any change to the instrument. Instrument `473a7dd7…` (main) → **`eba8f5fc…`**. The repair was re-applied onto main's
current file, not carried over from the stale local tree the gates first ran against; every number
below was re-measured against `473a7dd7…` and is unchanged.
Read with `RESULT_bench2_INVALID_2026_09_17.md` (the nine false accusations) and
`RESULT_decide1_decidable_fraction_2026_09_17.md` (71% of claims are decidable; the instrument
abstains on 94% of `only_touches`).

Four gates pass. **One fails as written and one could not answer its own question**; both are
reported below rather than reinterpreted.

## The prediction, and what happened

The prereg committed these numbers before the change, so that beating them would count as a
warning rather than a win:

| | predicted | measured |
|---|---|---|
| accusations on the known eleven | 11 → 8 | **11 → 8** |
| correct accusations | 2 → 2 | **2 → 2** |
| precision | 0.18 → 0.25 | **0.1818 → 0.2500** |

Per item (G-P1-1, blocking, **PASS**):

| PR | before | after |
|---|---|---|
| `microsoft/vscode-azuretools#2086` | CONTRADICTED | **VERIFIED** — "all changed paths under prefix" |
| `ydb-platform/ydb#25857` | CONTRADICTED | **VERIFIED** — "all changed paths under prefix" |
| `dotnet/runtime#117821` | CONTRADICTED | **UNCHECKABLE** — "prefix 'assert.notnull' is not a path (#110)" |
| `Azure/autorest.typescript#3252` | CONTRADICTED | CONTRADICTED (correct, held) |
| `microsoft/wassette#442` | CONTRADICTED | CONTRADICTED (correct, held) |
| the other six | CONTRADICTED | CONTRADICTED — modes 3–6, not attempted |

A twelfth instance the audit never saw also moved: `telerik/kendo-themes#5600` claimed to "only
change `.k-step-link`", a CSS class selector, and the instrument had accused it. Mode 2 withdraws
that verdict. Adjudicated: the withdrawal is correct.

## G-P1-2 fails as written

The gate said: *coverage must not fall.* Coverage across the 299 `only_touches` claims went
**18 → 16**.

The gate is wrong, and it is wrong because we wrote it wrong. Its purpose was to stop precision
being bought by abstaining more. But mode 2 repairs a false accusation **by withdrawing a
verdict** — that is the entire mechanism. A gate forbidding any coverage loss forbids the repair it
was written to guard. We are not reinterpreting it after the fact to make it pass: as written, it
fails, and the specification error is ours.

What it was actually asking is answerable. Both withdrawals are named and adjudicated:
`dotnet/runtime#117821` (`Assert.NotNull`) and `telerik/kendo-themes#5600` (`.k-step-link`). Neither
is a path; both verdicts were false accusations; withdrawing them is the correct outcome. No claim
with a genuine path prefix lost its verdict. A successor gate must count *correct* verdicts, not
verdicts.

## G-P1-3 could not answer its own question

The gate specified 25 items drawn with seed 20260921 from the 287 outside the eleven. Because the
instrument abstains on 94% of this claim kind, the draw returned **24 UNCHECKABLE and one
no-extraction — zero verdicts**. A sample of abstentions cannot reveal a false-accusation mode.
The sampling design was inadequate, which DECIDE-1 had already predicted and we did not carry into
this prereg.

The question is answerable by census instead, and the census is stronger than the sample would
have been. Across all 299 claims the post-fix instrument returns **8 accusations, every one of them
among the known eleven, and none outside**. There is no seventh failure mode hiding in the corpus,
because there are no accusations left in which one could hide.

Additionally, and not required by any gate: all **8 VERIFIED** verdicts were adjudicated by hand
against the live diffs, because a wrong VERIFIED is a false exoneration and no gate was watching
for one. All eight are correct — `owid-grapher#5053`, `yiiframework.com#1182`,
`dependabot-core#12918`, `vscode-azureappservice#2783`, `vscode-azuretools#2086`,
`vscode-azurefunctions#4722`, `ydb#25857`, `airbyte#68638`. Zero false exonerations.

## The gates that pass cleanly

- **G-P1-4 (the port agrees).** `web/gate/diffgate.js` mirrors the change. 604 rows, 651 claim
  readings, **0 disagreements**.
- **G-P1-5 (nothing else moves).** A baseline was reconstructed by reverting exactly the three
  PATH-1 edits and hashes to `4ba947a8…`, proving the change is those three edits and nothing else.
  Running both instruments over all 604 rows: **4 rows differ, all four `only_touches`**, all four
  listed above. Every `files_changed_count`, `tests_added` and `symbol_added` reading is identical.
- **G-P1-6 (closed committed list).** 142 extensions in
  `papers/closed-model-frontier/path1_extensions.txt`, mirrored byte for byte in
  `PATH1_EXTENSIONS` (Python) and `PATH1_EXTENSIONS` (JavaScript). Written from general language
  conventions with the preregistration, before measurement. Any addition needs a new prereg.

`tests/test_path1.py` adds 22 tests. Eleven pin the repair; three pin the list across all three
files; and three assert the instrument is **still wrong** on modes 3 and 6 and still right on the
two correct accusations. Tests that pin a known defect in place are unusual, and they are here so
that a later change cannot quietly claim an unrepaired mode without its own preregistration — when
one is genuinely fixed, that prereg deletes the corresponding line.

Suite: 4,552 passed, 6 xfailed, 143 skipped. The two `test_sworn_*` committed-sample failures are
pre-existing — checked, not assumed, by running them against the reconstructed `4ba947a8…`
baseline, where they fail identically.

## Two things this cycle learned after the gates were written

**The pin had to move, and that is now a known hazard.** `web/gate/differential/py_side.py`
refuses to run unless `styxx/diffgate.py` matches a committed hash. PATH-1 changes that file, so
the pin moves to `eba8f5fc…`. This matters more than bookkeeping: the same pin had already gone
stale once when the `fetch_pr` door landed, and the differential had been silently refusing to run
on `main` ever since — which is how the COMPAT-2 port stayed unlanded for a cycle without a single
test failing (#126). Any change to the instrument must move this pin in the same commit, or the
check that guards the two-implementation claim quietly stops guarding anything.

**G-P1-4 passed while proving almost nothing, and that is reported rather than quoted.** The gate
said "the differential prints zero disagreements". It does — 3,212 pairs before these additions.
But `corpus_real.json` carried exactly **four** `only_touches` claims across those 3,212 pairs, and
PATH-1 changed the reading of **none** of them. The gate was satisfied by a corpus that does not
exercise the change. A check that cannot see what changed is not evidence, and quoting it as
though it were would be the same error this programme keeps catching elsewhere.

The real py/js evidence for PATH-1 was obtained separately, over the 604-row BENCH corpus, which
carries 299 `only_touches` claims: **651 claim readings, 297 of them `only_touches`, 0
disagreements** between `styxx/diffgate.py` and `web/gate/diffgate.js`.

Beyond the frozen preregistration, and declared as such: `web/gate/differential/path1_pairs.json`
adds eight pinned pairs so the differential exercises this change from now on — both repaired
modes, a basename claim that must still accuse, a slashed prefix that must still anchor, and two
that pin the **unrepaired** modes as still wrong. The differential now reads 3,220 pairs and 6,933
claims with 0 disagreements. Adding corpus was not in the prereg; it is recorded here as an
addition rather than folded into the gate results.

## What a passing PATH-1 means, in the prereg's own words

Precision **0.25** on a tool whose entire pitch is that it does not accuse wrongly. Three quarters
of its accusations are still false — the six remaining are modes 3 through 6, which have the same
surface shape as checkable claims and which no test available to the instrument separates.

And PATH-1 does nothing at all about the larger problem. DECIDE-1 put the decidable fraction of
`only_touches` claims at 52% [33.5%, 70.0%] while the instrument returns a verdict on 5.7%. That
gap is untouched here and is not improved by a single item. It is an extraction problem, it is
worse than the precision problem, and it needs its own programme.

**This is a bug fix, not a product.** The instrument should not be described as fixed, recommended,
or ready on the strength of it.

---

*We predicted 0.18 → 0.25 and got 0.1818 → 0.2500, which is the first thing in this programme to
land exactly where it said it would. Two of our own gates still came out badly — one forbade the
repair it was guarding, the other sampled a population that is 94% silence and learned nothing.
Both are written up here at the same length as the part that worked.*
