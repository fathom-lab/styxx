# PREREG -- the BURIED-JUDGE family (cycle 61, autopilot, 2026-07-24)

**Frozen before any scored run exists. Committed separately, ahead of results.
Bars below are binding; a missed bar is CLOSED_NEGATIVE, never SURVIVED.**

## Motivation (the named next step)

Cycle 60 closed the four-family PRICE/REFUSE partition of `styxx.anchors.audit_panel`:
attr + numeric **PRICE** (an informative judge exists under honest anchors), chain + temporal
**REFUSE** (`VOID_PANEL__uninformative` -- no judge clears the gate). Its logged next step:

> "A genuinely-informative-but-HARD family (real judge buried under noise) would sharpen the
> price/refuse boundary."

Every prior family lands cleanly on one side. This family is built to sit **on the boundary**:
a single genuinely-informative judge whose *true* separation is swept from deaf through the
noise-margin gate to strong, so the run measures exactly where PRICE turns into REFUSE and whether
the transition is honest.

## Construction (SEALED DGP + SHIPPED instrument -- nothing invented)

- Generating process: `anchored_stage_a.simulate_panel` / `make_anchors` (the sealed Stage-A DGP).
- Instrument under test: `styxx.anchors.audit_panel` (the shipped public surface, v3 selective),
  called with `n_boot=300`, `null_sims=200` (its shipped defaults for per-dataset tau + misfit).
- Panel: J = 4. **Judge 0 informative**: alpha0 = 0.30, beta0 = 0.30 + sep. **Judges 1-3 deaf**:
  alpha = 0.45, beta = 0.50 (true separation 0.05, below the 0.15 gate always).
- Anchors drawn from the **same** per-judge (alpha, beta) as the organic panel -> exchangeable,
  honest (the load-bearing assumption held, not attacked, so the boundary is what is measured).
- PI = 0.35; N = 6000 organic; K = 400 per anchor stratum (the Stage-A design point).
- Separation grid (frozen): **{0.00, 0.16, 0.22, 0.25, 0.28, 0.34, 0.40}**. 0.22/0.25/0.28
  bracket the effective noise-margin gate (~0.255 at K=400), the split region where the keep
  decision selects on the anchor draw.
- R = **60** replicates per cell; disjoint seeds `SEED_BASE(610000) + cell_index*100000 + i`,
  audit seed `+ 5_000_000`.

**Identifiability note (fixed by construction, not by luck):** with one informative judge kept,
pi is exactly identified in expectation -- t = 0.30 + 0.35*sep, A = 0.30, B = 0.30 + sep, so
(t - A)/(B - A) = 0.35 for every sep. The estimate is never wrong in expectation; the entire
question is whether the gate + anchor sampling keep the *interval* honest as separation is buried.

**Why the kill can fire (this is not a victory-lap family).** The noise-margin gate keeps judge 0
iff its ANCHOR-measured separation beta_hat - alpha_hat >= 0.15 + 3*sqrt(...) ~ 0.255 at K=400.
Near that gate the keep decision **selects on the anchor draw**: a replicate survives only when its
draw shows a large beta_hat - alpha_hat (equivalently a low alpha_hat and/or high beta_hat), which
biases the point estimate pi = (t - A)/(B - A). If that selection bias exceeds the bootstrap CI
width, the priced replicates are systematically MISCOVERED -- an over-pricing danger. The gate
below can therefore return CLOSED_NEGATIVE on real behaviour.

## Frozen bars

- **PD_MOVE** (validity precondition, two-sided sanity -- NOT the discovery): deaf cell
  (sep = 0.00) `VOID_PANEL` rate >= 0.90 **AND** strong cell (sep = 0.40) `ESTIMATED` rate >= 0.90.
  If either fails, the instrument is not reading separation and the run is
  `INVALID__instrument_not_reading_separation` (reported as a block, not a result).

- **PD_DANGER** (THE KILL -- can fire CLOSED_NEGATIVE): for every cell the ladder **prices**
  (`ESTIMATED` rate >= 0.50), pi-CI coverage of PI = 0.35 among the ESTIMATED replicates must be
  **>= 0.80** -- the instrument's own worst measured regime (styxx.anchors docstring: boundary
  coverage 0.81). If ANY priced cell covers < 0.80 ->
  `CLOSED_NEGATIVE__ladder_over_prices_buried_judge`. Reported verbatim, not rescued.

- **PD_HONEST** (refusal is clean): no `VOID` replicate (either class) may emit a non-null pi.
  A refusal that leaks a number is not a refusal. Any leak ->
  `CLOSED_NEGATIVE__refusal_leaks_a_number`.

- **PD_BOUNDARY** (the deliverable, REPORTED not gated): the smallest sep with `ESTIMATED` rate
  >= 0.50 AND coverage >= 0.90 -- the "buried but recoverable" threshold that sharpens the
  price/refuse boundary. No bar; it is a measured characteristic.

## Verdict mapping (frozen)

- PD_MOVE fails -> `INVALID__instrument_not_reading_separation` (blocked cycle, honest).
- PD_MOVE holds AND any priced cell coverage < 0.80 ->
  `CLOSED_NEGATIVE__ladder_over_prices_buried_judge`.
- PD_MOVE holds, no over-pricing, but a VOID leaks a pi ->
  `CLOSED_NEGATIVE__refusal_leaks_a_number`.
- PD_MOVE holds, every priced cell covered >= 0.80, refusals clean ->
  `SURVIVED__prices_where_covered_refuses_below_gate`. The claim earned is exactly this scope: on
  a single-informative-judge panel with honest exchangeable anchors, the ladder prices where its
  interval stays covered and refuses (VOID) below the gate -- the price/refuse boundary is
  well-placed, characterized by PD_BOUNDARY.

## Scope / what this does NOT claim

Simulation on the sealed DGP with honest (exchangeable) anchors; a single informative judge.
It says nothing about correlated multi-judge panels, non-exchangeable anchors (priced elsewhere
and out of scope here), or real LLM judges (Stage B). It sharpens the price/refuse boundary of the
existing instrument; it does not extend the field-level claim.

## Harness / receipts

`run_buried_judge.py` (frozen with this prereg); scored output `buried_judge_result.json`;
smoke writes only `buried_judge_SMOKE_INVALID.json` (never read as a result). CPU only,
deterministic, ASCII.
