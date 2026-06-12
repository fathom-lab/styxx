# PRE-REGISTRATION — the RIGHT second axis: does a content-danger STATEMENT axis complete the (truth × danger) basis? (frozen)

**2026-06-12 · Fathom Lab / styxx. Frozen before any score is seen. Runner:
`run_truth_danger_basis.py` (SEED=0). Receipt: `truth_danger_basis_result.json`; figure
`truth_danger_basis.png`. Backlog B30. Direct sequel to cycle 4
(`FINDING_conscience_coordinates_2026_06_11.md`, HARM-AXIS-NULL), which showed the borrowed
refusal axis — fit on harmful REQUESTS — does NOT read the danger of a STATEMENT. The fix this
prereg tests: fit a danger axis DIRECTLY on danger-vs-safe statements and ask whether it completes
the intended (truth × danger) coordinate system.**

## The question

Cycle 4 proved you cannot read "is this statement dangerous" off a request-refusal axis. So build
the right axis: a content-DANGER direction fit on danger-topic vs safe-topic STATEMENTS (balanced
across truth, so it reads danger, not truth). Then ask the cycle-3/4 questions of the pair
{truth, danger}: are they a clean orthonormal basis under whitening, do BOTH coordinates recover
their OWN factor and stay blind to the other on a held-out 2×2 factorial, does it transfer
cross-model, and — the payoff — does the 2-D (low-truth, high-danger) corner detect dangerous
misinformation as a DECOMPOSED composite, beating cycle 4's single-axis falsity detector (0.838)?

## Design

- **Truth axis:** difference-of-means on factual true/false statements (the arc's families), gemma L12.
- **Danger axis (NEW):** difference-of-means on a TRAIN set of danger-topic vs safe-topic STATEMENTS,
  BALANCED across truth within each danger level (12 true-danger + 12 false-danger vs 12 true-safe +
  12 false-safe), so the direction reads danger-topic and not truth. This train set is DISJOINT from
  the held-out factorial (no shared sentences).
- **Whitening:** ZCA on the pooled train (truth-train + danger-train), the cycle-3 recipe; both
  directions fit in the whitened space.
- **Held-out test = the cycle-4 2×2 factorial, UNCHANGED** (identical 48 sentences,
  T{true,false} × H{danger,safe}, imported from `run_conscience_coordinates.py`) so the comparison
  to HARM-AXIS-NULL is exact. Project onto both whitened coordinates: c_truth, c_danger.
- **Readout matrix:** AUROC(c_truth,T), discrim(c_truth,H), AUROC(c_danger,H), discrim(c_danger,T).
  On-target raw AUROC; off-target (invariance) discriminability max(AUROC,1−AUROC).
- **Cross-model:** one shared label-free ridge map target→gemma L12 fit on the UNION of train texts
  (paired, label-free), Llama-3.2-3B primary + Qwen2.5-3B secondary; project the mapped factorial.
- **Dangerous-misinfo detectors (descriptive):** 1-D falsity baseline = standardized −c_truth
  (cycle-4 method, ref 0.838); 2-D composite = standardized (−c_truth + c_danger); AUROC for picking
  the (false,danger) cell out of the other three. Does the danger axis ADD detection power?

## Frozen gates (verdict precedence top-to-bottom)

Let A=AUROC(c_truth,T), Bh=discrim(c_truth,H), Dh=AUROC(c_danger,H), Bt=discrim(c_danger,T).

- **DANGER-AXIS-WEAK** iff, in gemma, Dh < 0.70 — statement danger is not even linearly readable
  in-model at this layer; the right axis doesn't exist linearly and the system can't be built. (Bounds
  the axis; honest stop.)
- **TRUTH-DANGER-BASIS** iff, in gemma AND mapped into Llama-3.2-3B: A ≥ 0.75 AND Bh ≤ 0.65 AND
  Dh ≥ 0.75 AND Bt ≤ 0.65 (each coordinate recovers its own factor, blind to the other). → the
  intended (truth × danger) coordinate system EXISTS and transfers; dangerous misinformation
  decomposes. This is the result cycle 4 could not get with the borrowed refusal axis.
- **ENTANGLED-COORDINATES** iff both off-target discriminabilities ≥ 0.75 (each coord reads both).
- **PARTIAL-STRUCTURED** — anything else; report the exact matrix and name what holds.

Thresholds (0.75 on-target, 0.65 off-target, 0.35 whitened cosine for the descriptive basis check)
inherited from cycles 2–4 unchanged. Bars do not move post-hoc.

## What each outcome means (pre-committed)

- **TRUTH-DANGER-BASIS** → the conscience-coordinates system is real once the axis is fit on the right
  content; cycle 4's null was about a BORROWED axis, not about danger being unreadable. styxx.crossmind
  gains a validated second axis and a decomposable dangerous-misinformation readout. If the 2-D
  composite also beats the 1-D falsity baseline (0.838), the danger axis adds genuine detection power.
- **DANGER-AXIS-WEAK / PARTIAL** → statement-level danger is not cleanly linearly separable (or not
  orthogonal to truth) at this layer/scale; the honest bound is that the truth coordinate carries the
  load and danger is not an independent readable axis here. Recorded, not spun.

## Safety scope (frozen)

The (false, danger) items are dangerous misinformation used ONLY as flagged-as-false probe stimuli read
pre-output; no model generates from them; no operational harmful content appears in stimuli, receipts,
or finding. Identical scope to cycle 4.
