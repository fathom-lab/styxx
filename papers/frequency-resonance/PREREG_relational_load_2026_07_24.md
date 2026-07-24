# PREREG — does oscillation's advantage scale with RELATIONAL LOAD?

**Frozen:** 2026-07-24, before this evaluated run. Tests the sharpest prediction of the dissociation
(`RESULT_recall_horizon_2026_07_24`: oscillation is required to COMPARE across distance, not to REMEMBER).

## The prediction being risked

If oscillation is a *relating* mechanism rather than a memory one, then a decay channel's failure should
grow with the number of RELATIONS a task requires while staying flat in the number of FACTS it must merely
hold. This is a dose-response, and it can fail three distinct ways (all reportable):
- decay fails equally at every relational load (a fixed deficit, not relational) → the dissociation's
  "relating" reading is wrong or incomplete;
- decay also degrades with pure storage load (facts without relations) → the effect is memory capacity
  after all, contradicting the recall result;
- decay succeeds at all loads → no effect to explain.

## Design — two orthogonal load axes, one knob

Same CLRU phase-clamp (FREE θ-learnable vs CLAMPED θ≡0, matched, RNG-matched), T=256, distances held
constant (facts scattered over the first half, probe at the end), success-probability over 5 CLAMPED
seeds (accuracy ≥ 0.80; decay trainability is bimodal), 2 FREE seeds, 2000 steps.

**Axis R — relational load (the treatment).** R ∈ {1, 2, 3, 4}. R independent premise bits are placed at
fixed distant positions; the probe carries R claim bits. Label = 1 iff ALL R claims match their premises
(an R-fold conjunction of comparisons). Every additional relation adds one product a decay channel must
form; storage grows only linearly.

**Axis S — storage load (the control).** S ∈ {1, 2, 3, 4}. S premise bits are placed identically, but the
label is the PARITY-FREE readout of a single designated premise (the probe names which one via a one-hot
selector): the model must hold S facts and report one, with NO comparison at all. Storage grows exactly as
in axis R; relations do not.

Distances, sequence length, number of stored facts, and parameter count are matched across the two axes at
equal load index; the only difference is whether the label requires comparisons or a selection.

## Post-smoke calibration (honest note; gates unchanged)

The pipeline smoke placed the premises at gaps 186-246 — far beyond decay's measured half-horizon (~gap
32) — so CLAMPED scored 0.00 at load 1 and the run would have ABSTAINed under the headroom clause with
nothing learned. Before any evaluated run, the premise slots were moved INSIDE decay's competent range
(gaps 4-11 from the probe) so that load 1 has headroom and a dose-response is measurable at all. This
tests the relational axis where decay is otherwise competent — the only regime where the prediction is
falsifiable. Distances remain identical across both axes and all loads, so the R-vs-S contrast is
unaffected. Gates, axes, and controls below are unchanged.

## Frozen gates

Let `pR(k)` = CLAMPED solve rate at relational load k, `pS(k)` = CLAMPED solve rate at storage load k.

- **ABSTAIN** iff FREE solve rate < 1.0 at any cell (oscillation not solving → no contrast) OR
  `pR(1) < 0.60` (decay fails the single-relation case outright at this distance — no headroom to show a
  dose-response; the horizon result already covers that regime).
- **CONFIRM (relational dose-response)** iff FREE range-free AND `pR` is non-increasing in k (within seed
  noise, tolerance 0.20) AND `pR(4) ≤ pR(1) − 0.40` AND the storage control is spared:
  `pS(4) ≥ pS(1) − 0.20`.
- **NULL (not relational)** iff `pR(4) ≥ pR(1) − 0.10` (no relational dose-response) OR the storage
  control degrades as much as the relational axis (`pS(1) − pS(4) ≥ pR(1) − pR(4)`), which would mean the
  deficit is storage, not relating. Ship the negative; it would demote the dissociation's reading.
- **PARTIAL** otherwise — reported verbatim.

## Confounds controlled (frozen)

- **Storage matched:** axis S holds the same number of facts at the same positions as axis R; only the
  required operation differs. This is the load-bearing control — without it, "more relations" is confounded
  with "more things to remember."
- **Distance matched:** premise positions are identical across both axes and all loads, so nothing here
  varies distance (that axis was already characterized by the horizon result).
- **Single knob, RNG-matched, matched-compute** (FREE/CLAMPED differ only in θ).
- **Bimodality respected:** success rate over seeds, never a mean of a bimodal accuracy.
- **Chance stated:** both axes are binary/selection tasks with chance ≤ 0.5; the solve threshold 0.80 sits
  well above chance for every cell.

## Scope (non-negotiable)

A controlled state-space-model characterization of what kind of computational load a decay channel fails
under. NOT a real-LLM claim (no language model is run; transformers have no θ to clamp). It sharpens the
mechanism and its prediction for recurrent/SSM architectures.

## Red-team asserts

1. `lin_scan == seq_scan` < 1e-4. 2. FREE/CLAMPED share `B_re`/`nu` at init (only θ differs).
3. Premise positions identical across axes/loads; labels ≈ balanced; the storage task's answer is
determined by the selector (not guessable from the stored set alone). 4. Both tasks are solvable in
principle by the FREE model (else the contrast is rigged).
