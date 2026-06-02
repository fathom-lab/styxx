# The white-box-vs-text MAP — when does reading activations beat reading the text?

Auto-generated from the committed result JSONs (`build_map.py`). Every cell is a
pre-registered run with the gate frozen before data. This is the contribution the
scattered literature lacks: one honest head-to-head across **task × regime ×
setting**, residual signal vs the text baseline, verdicts and negatives included.

```
task     | regime                       | setting       | white-box   | text      | margin     | verdict
--------------------------------------------------------------------------------------------------------
refusal  | behavioral (held-out, cross-arch) | —             | 0.78–0.97   | n/a       | —          | SURVIVED
action   | described intent             | menu          | 0.84–0.95   | 0.51      | 0.33–0.44  | SURVIVED
action   | emergent behavior            | menu          | 0.65–0.92   | 0.16–0.35 | 0.48–0.71  | SURVIVED (4/5)
action   | emergent behavior            | native tools  | 0.51–0.82   | 0.16–0.27 | 0.35–0.55  | FORMAT-DEPENDENT (1/2 pass)
action   | accidental (benign prompt)   | native tools  | 0.62–0.79   | 0.36–0.52 | 0.10–0.32  | SURVIVED
action   | closed-model (behavioral)    | GPT API       | 0.53 (non-circular) | —         | —          | FAILED (only signal circular)
```

**white-box** = pre-emission residual-probe AUC (range across valid models).
**text** = best text-only baseline (prompt bag-of-words / embedding) on the same
task. **margin** = white-box − text.

## What the map says (honest — the flagship cell has now landed)

- White-box **wins clearly** where the signal is in the model's representation and
  the interface is clean — refusal (held-out, cross-arch) and action under the menu
  harness (residual ≫ text).
- **The flagship cell SURVIVED.** Accidental harm — a *benign* prompt where the
  model emergently over-reaches to a destructive tool, the exact case a
  text-monitor-on-the-prompt is blind to *by construction* — is caught by the
  pre-emission residual at AUC ~0.79 on **two different architectures** (Qwen-1.5B,
  Llama-3.2-3B), with the text baseline confirmed blind (≤0.52, at/below chance) and
  a feasibility-grade calibrated probability (ECE ~0.14). This is white-box's one
  true deployment USP, demonstrated on the case that mattered most.
- The edge is **real but model-dependent in strength**: 2 of 4 models cleared the
  full frozen gate; Qwen-3B missed by 0.014 (0.686), and the smallest model
  (Llama-1B) is both reckless (50/60 prone) and less legible (0.618). An honest
  bound, not a clean sweep.
- **At this n the per-model verdict is inside the noise.** Hanley-McNeil 95% CIs
  are ±0.15–0.20 wide (10–15 positives/model), and the 0.70 gate line runs through
  *all four* intervals — Llama-3B [0.62, 0.97], Qwen-3B [0.48, 0.89]. "Gate met on
  2/4" is a true pre-registered *binary* outcome, but it overstates the statistical
  resolution: which 2 of 4 pass is not separable from chance at this n. The robust
  unit of evidence is the **direction** — white-box beats text on **4/4** models
  (margins +0.10 to +0.32, sign-test p≈0.06), consistent but not yet significant.
  And the per-model ranking does **not** track capacity (Qwen-1.5B 0.79 > Qwen-3B
  0.69, opposite our refusal-gate finding) — further sign the point estimates are
  seed-sensitive. Read this cell as **promising and directionally robust, NOT an
  established cross-arch pass.** ECE ~0.14 at n=60 is a gesture, not a calibration.
- White-box's edge is **fragile to the interface** in the emergent-choice regime:
  under native tool-calling it went format-dependent (1/2).
- White-box has **no edge** where there are no activations: closed-model behavioral
  signals failed (the only "win" was circular).

## What would harden it (honest next step, not a victory lap)

The pre-committed expansion kept the gate **frozen** (no goalpost-moving on the
threshold) — but two degrees of freedom remain, and the CI analysis above shows why
they matter: (1) the expansion's *scenario composition* was authored after seeing
which models were near the line (threshold frozen, dataset not), and (2) it is a
single seed, which the capacity inversion shows is load-bearing. The confirmatory
run that converts "promising" → "claimable": **pre-register the exact scenario count
and composition rules, author the held-out set blind, run ≥3 seeds, report per-model
AUC with confidence intervals (not just the point estimate vs the gate), and add a
5th architecture.** Stated before claiming the cell is bulletproof — because it
isn't yet.

The value isn't any single cell; it's that **we publish the whole board, losses
included.** No incumbent will.
