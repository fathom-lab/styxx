# FINDING — the retained-probe design is now the instrument: `assess_retained_probe` ships, dogfooded on the receipts that demanded it

**Cycle 99 · 2026-07-30 · `DONE__instrument_expresses_the_retained_probe_design`**
**Receipts:** `retained_probe_dogfood_result.json` · `frontier_incontext_oof_result.json`
(cycle 98, certified) · `styxx/framelocality.py` · `tests/test_framelocality.py`

## The gap, named before the fix

The cycle-98 prereg (`PREREG_frontier_incontext_oof_2026_07_30.md`) recorded in advance that
`styxx.framelocality.assess()` could not express its design: the module's labels assume a
probe that REMOVES the corruption, so it reads recovery(CORRUPTED) ≈ recovery(HELD) as a
null — but under a probe that RETAINS the corruption in context, that same parity is the
positive reading. The prereg declared the instrument would not be edited inside the cycle
that used it and named the upgrade as owed. This cycle is that upgrade.

## What shipped

`styxx.framelocality.assess_retained_probe(records, *, reask=None, held_floor=0.8,
reach_margin=0.15, frame_margin=0.15)` — the cycle-98 design as reusable API:

- **Inverted semantics, stated:** result carries `probe_semantics` =
  CORRUPTION_RETAINED_AT_PROBE_TIME; parity with HELD is the positive reading, a deficit is
  the corruption's reach.
- **The validity control is a gate:** recovery(HELD) below `held_floor` 0.8 returns
  `INVALID__probe_frame_not_validated` — a frame that cannot read an unabandoned belief
  licenses nothing in either direction.
- **The re-ask control is required for the full claim:** without `reask`, a passing reach
  caps at `REACH_BOUNDED__no_reask_control`; with it, the frame must beat the bare re-ask by
  `frame_margin` 0.15 or the verdict is `RESTORATION_NOT_FRAME_SPECIFIC`.
- **The confound is in the output:** a `CAVE_PERSISTS_OUT_OF_FRAME` verdict carries a note
  that HELD is conditioned on outcome, so the negative reads channel-unlicensed, never
  persistence-demonstrated. The naive margin vs WRONG_FIRST is printed only under the
  NOT-EVIDENCE label, as everywhere in this module.
- Floors are the frozen ones — the module's MIN_CELL, the LG2-derived held floor, and the
  LG3-derived margins; none new, none moved. The module's test file passes in full, with the
  new retained-probe cases alongside the existing pins.

## The dogfood

`run_retained_probe_dogfood.py` feeds the new function the committed cycle-98 receipt's 146
per-item records — real rows, not synthetic shapes — and asserts equality to the digit with
the published numbers before accepting the verdict: recovery on CAVED 0.6956521739130435,
on HELD 0.975, reach -0.2793478260869565, re-ask recovery on CAVED 0.5434782608695652, all
matching `frontier_incontext_oof_result.json`, verdict `CAVE_PERSISTS_OUT_OF_FRAME` — the
published negative, reproduced by the instrument from raw rows
(`retained_probe_dogfood_result.json`, `all_equal_to_receipt` true).

A regression test (`test_retained_probe_pins_the_cycle98_negative`) pins this shape in CI
next to the v31-null pin, so neither retraction-grade reading can silently drift.

## Why this matters for the program

The module now covers the full design space the arc has actually traversed: removing probe
(`assess`, with the v31.1 circularity labelled), weight-level between-arm (`compare_arms`,
from the c93 dogfood), and retaining probe (`assess_retained_probe`, from cycle 98) — each
mode added because a real run needed it and each shipped with the run's own receipts as its
regression pin. The instrument is the arc's mistakes, made unrepeatable.

## Scope

Deterministic re-analysis and packaging; no model run; no prior artifact altered; certify.py
untouched. The dogfood reproduces cycle 98's verdict — it adds no new evidence about the
model, and the cycle-98 scope (one substrate, one benchmark family, one grader frame,
difficulty confound bounded) carries unchanged.
