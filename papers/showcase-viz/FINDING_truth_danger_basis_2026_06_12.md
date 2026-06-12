# FINDING — the RIGHT second axis: the danger coordinate is clean and orthogonal; the cross-model basis gate misses by a whisker (PARTIAL-STRUCTURED)

**2026-06-12 · Fathom Lab / styxx. Pre-registered: `PREREG_truth_danger_basis_2026_06_12.md`
(frozen pre-run, committed 06e80dc). Receipt: `truth_danger_basis_result.json`; figure
`truth_danger_basis.png`. Backlog B30. Direct sequel to cycle 4
(`FINDING_conscience_coordinates_2026_06_11.md`, HARM-AXIS-NULL): the borrowed refusal axis (fit on
harmful REQUESTS) read statement danger at chance. This fits a danger axis DIRECTLY on danger-vs-safe
STATEMENTS and asks whether it completes the intended (truth × danger) coordinate system.**

## Result — the danger axis is real, perfect, and orthogonal; the strict gate misses on one target by a hair

A danger axis fit on a disjoint set of danger-vs-safe STATEMENTS (balanced across truth), whitened
with truth on the cycle-3 recipe, then both coordinates read on the UNCHANGED cycle-4 2×2 factorial.

| readout | gemma | mapped Llama-3.2-3B | mapped Qwen2.5-3B |
| --- | --- | --- | --- |
| c_truth recovers T (AUROC) | 0.7656 | 0.9288 | 0.8559 |
| c_truth invariant to H (discrim) | 0.5347 | **0.6562** | 0.5017 |
| c_danger recovers H (AUROC) | **1.0** | **1.0** | **1.0** |
| c_danger invariant to T (discrim) | 0.5104 | 0.5069 | 0.5104 |

Whitened cos(c_truth, c_danger) = −0.0 (orthogonal).

- **The right axis EXISTS and is clean.** The directly-fit danger coordinate reads statement danger at
  AUROC 1.0 in gemma AND through BOTH cross-model maps, is invariant to truth (≈0.51), and is orthogonal
  to the truth direction (cos −0.0). This DIRECTLY resolves cycle 4's HARM-AXIS-NULL: the borrowed
  refusal axis read statement danger at chance (≈0.52); a danger axis fit on the right content reads it
  perfectly. **Cycle 4's null was about a BORROWED axis, not about danger being unreadable** — confirmed.
- **gemma and Qwen-3B pass the compositional gate; Llama-3B misses by 0.0062.** In gemma all four cells
  clear (truth recovers T at 0.7656, invariant to H at 0.5347; danger recovers H at 1.0, invariant to T
  at 0.5104) and Qwen-3B clears all four. The PRIMARY target Llama-3.2-3B clears three but its truth
  coordinate's danger-invariance is **0.6562**, a whisker over the 0.65 ceiling. Because the frozen gate
  requires the primary target, the verdict is **PARTIAL-STRUCTURED, not TRUTH-DANGER-BASIS.** A
  threshold miss is the verdict it earns; 0.6562 is not a pass.

## Why the truth coordinate is the marginal one (the honest mechanism)

The truth coordinate reads truth well on SAFE statements but weakly on DANGER statements: the quadrant
centroids on c_truth are true-safe +0.8907 vs false-safe −1.2035 (a clean gap), but true-danger +0.2787
vs false-danger −0.0389 (the danger-topic items pile up near the centre, barely separated by truth).
The danger register compresses the truth read — the same leak flagged in cycle 4, now localized: it is
the danger-topic items, not the safe ones, that the truth axis struggles on. That residual is what
pushes Llama's truth-invariance just over the ceiling.

## Dangerous misinformation now DECOMPOSES (the payoff)

The 2-D (low-truth, high-danger) corner detects the false-danger cell well above the 1-D falsity reading
in this basis: gemma 1-D falsity 0.5231 → 2-D composite **0.7662**; mapped Llama 2-D **0.9213**; Qwen 2-D
0.8079. The danger axis ADDS the detection power the truth axis alone lacks here — dangerous
misinformation reads as a genuine composite (false AND dangerous), not a single-axis artifact. (The 1-D
falsity number is not comparable to cycle 4's 0.838: a different whitening basis — pooled on
truth+danger rather than truth+refusal+valence — so this is a different coordinate frame, not a
regression. The load-bearing comparison is 2-D vs 1-D WITHIN this run, where danger carries it.)

## What this means

The conscience-coordinate system is real once each axis is fit on the right content. The danger axis is
a clean, perfect, orthogonal, cross-model-transferable second coordinate; the (truth × danger) basis
holds outright in gemma and in Qwen-3B and misses the full pre-registered cross-model label only on
Llama's marginal truth-invariance (0.6562 vs 0.65). For the product (`styxx.crossmind`): a directly-fit
danger axis is a validated second axis — the module's refusal of a BORROWED refusal-as-danger axis
stands, while the fit-it-directly path it recommends now has evidence. The clean way to claim the full
basis cross-model is the owed B29 work (whitened readout in the mapped space + a shrinkage-covariance
sweep), which should pull Llama's marginal truth-invariance back under the ceiling.

## Honest bounds (what is NOT claimed)

PARTIAL-STRUCTURED, not BASIS: the strict cross-model gate is not met (Llama truth-invariance 0.6562 >
0.65). Linear difference-of-means directions in a whitened space; danger-train n=48, factorial n=48
(small per-cell); local open models (gemma-2-2b source; Llama-3.2-3B + Qwen2.5-3B targets). The truth
coordinate is weak on danger-topic statements (a real bound, not hidden). The (false,danger) stimuli are
flagged-as-false probe reads; no model generated from them; no operational content appears anywhere. The
danger axis's perfect 1.0 readout on n=48 is an existence result at this scale, not a deployed-accuracy
guarantee. Owed: B29 (mapped-space whitening + covariance robustness) to test whether the full basis
clears cross-model cleanly; a larger factorial to tighten the marginal cell.
