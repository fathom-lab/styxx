# The Conscience Mount — a portable, read-only integrity layer for language-model agents, and its honestly-bounded adversarial scope

**2026-06-13 · Fathom Lab / styxx. A synthesis of the conscience-mount arc: the deployable instrument
(`styxx.crossmind` + `styxx.mount`), the science it productizes, the live demonstration, the calibration
that made its headline number honest, and the adversarial analysis that bounds exactly what it defends.
Every headline below traces to a pre-registered, OATH-HELD finding (provenance list at the end); this
document is the map, not a new claim, and re-states no number its source finding did not certify.**

## The thesis

A *conscience* — a model's internal sense of a value like truth or harm — can be read from one model
using a direction fit on a *different* reference model, and bolted onto a generating agent as a live,
read-only monitor that flags when the agent's words diverge from its substrate. The defensible core is
that the conscience is BORROWED: the agent has no labels of its own to game and the readout never depends
on its self-report.

## The instrument

- **`styxx.crossmind`** — read one model's value-state using a difference-of-means axis fit on a
  *reference* model, transported through a label-free ridge map of last-token hidden states and read in a
  ZCA-whitened (Mahalanobis) frame; no labels on the target. `read_cross_model` whitens in the
  mapped-target metric (the B29-correct cross-model read). Numpy-only, no torch.
- **`styxx.mount`** — `ConscienceMount` bolts borrowed axes onto a generating agent, reads its own
  last-token hidden state, and flags output-vs-substrate DIVERGENCE (the agent asserts "true" while its
  substrate reads false). READ-ONLY by construction: `steer` is REFUSED; acting on a flag is the
  integrator's policy via `on_flag`. `calibrate_threshold` sets a per-axis false-alarm budget.
- 36 offline/deterministic tests across the two modules; the package suite is green at 1471.

## The science it productizes (the showcase arc, all OATH-HELD)

- **VALUES-PORTABLE** — a value direction transfers across minds out-of-distribution: refusal reads at
  AUROC 0.9965 (Llama) / 0.9809 (Qwen) through the label-free map, beating permutation nulls at 0.9497 /
  0.9149 respectively. The margins over the null are modest — the broad transport of harm/benign structure
  makes the null itself elevated — but the axis is distinct and transfers.
- **WHITENING-RESOLVES** — the apparent cross-axis entanglement is a pure covariance artifact: raw
  off-diagonals 0.9778 / 0.9013 collapse to ~0.55 under whitening in the source (gemma) space, where the
  directions become exactly orthogonal. This was validated in source space only; applying the same
  whitening recipe to mapped points requires re-estimating the covariance in the mapped distribution
  (that is BASIS-CLEARED / B29, below).
- **conscience-coordinates (HARM-AXIS-NULL)** — the truth coordinate generalizes to new content
  (0.8524 in gemma, 0.9809 mapped to Llama), but a refusal axis fit on REQUESTS does NOT read the danger
  of a STATEMENT (chance, 0.5226). A precise bound: name what an axis was fit on.
- **truth × danger basis (PARTIAL-STRUCTURED, cycle 5)** — a danger axis fit directly on statements
  reads danger at 1.0 and is nearly orthogonal to truth, but the primary Llama target's truth-invariance
  misses the cross-model gate (0.6562 vs the 0.65 ceiling) while gemma and Qwen clear.
- **BASIS-CLEARED (cycle 6 / B29)** — re-whitening the mapped points in the mapped distribution's own
  metric (not the source metric used in cycle 5) closes the Llama miss: truth-invariance 0.6562 → 0.6059,
  stable across shrinkage. This is a methodological correction, not a free win — the cross-model basis
  clears only under the corrected whitening.
- **content-wall (CONTENT-WALL, frozen verdict)** — read in the mapped space with a well-fit map
  (R² 0.9095), cross-model CONTENT identity shows high DESCRIPTIVE performance (raw cosine 0.90,
  mapped-whitened 1.0). But per the frozen gate (the gemma-whitened readout) the verdict remains
  CONTENT-WALL — the gate readout is still at chance. The descriptive numbers are suggestive, not a
  verdict flip; a fresh pre-registration is owed for any content-transports claim (B33). No goalpost
  moving: the "value thermometer, not content transcript" reading stands at the frozen gate.

## The live result, and the calibration that made it honest

A gemma truth conscience mounted on Llama-3.2-3B caught the model caving to falsehoods under pressure.
The catch number moved as calibration was made honest — a three-step arc that is itself the contribution:

- **0.9231 (12 of 13)** — the first live catch, but the divergence center was calibrated on the test
  states (leakage), flattering the number.
- **0.40** — leakage-free but the center was calibrated on a NEUTRAL set, mismatched to the
  pressured-decision regime.
- **0.85, the definitive operating point (caught 17 of 20 caves)** — center and threshold both
  calibrated on held-out honest PRESSURED decisions (regime-matched, no leakage); the calibration target
  FPR of 0.10 realized as 0.20 on held-out honest test items — double the target. The calib→test gap is
  driven by the n=15 sample size and one strong outlier, but it means false-alarm control is not yet
  tight. The honest deployment number: a borrowed, read-only conscience catches ~0.85 of pressured caves
  at one decision token, at a ~0.20 false-alarm rate (double the stated threshold; control owed).

## The adversarial scope (the complete, bounded picture)

Reading is one thing; surviving an adversary is another. Every linear mount reading is affine in the
agent state, r(h) = a·h + c, so an activation attack steps along the readout direction a (an UPPER BOUND
on a weight-fine-tune attacker). Three tests bound the defense:

- **Borrowing does NOT defend (ATTACK-TRANSFERS).** Two different reference models' transported
  truth-readouts land nearly the same direction in the agent's space (cos 0.941), so an unconstrained
  activation perturbation of ≈0.05 of the activation norm — an UPPER BOUND on a weight-fine-tune attacker,
  and one that ignores the constraint of preserving the agent's output — optimized against one reference
  evades both at once. There is one canonical truth direction; borrowing a different mind lands on it.
- **A re-locked ensemble defends a naive attacker, taxes an adaptive one ~2× (ENSEMBLE-DEFENDS,
  boundary with false-alarm cost).** Re-fitting the same reference on disjoint data gives near-orthogonal
  functionals (mean pairwise cos 0.0133); an OR-rule ensemble forces 32× the budget against a naive
  single-member attack, but an adaptive attacker targeting the shared truth direction evades all members
  at ~2× cost. Crucially, this robustness is bought at a steep false-alarm price — the ensemble's realized
  FPR on held-out honest decisions is 0.2667 (vs the 0.10 target). A fair fixed-FPR comparison would
  shrink these robustness ratios, so the 32× / 2× numbers are upper bounds at an uncontrolled false-alarm
  rate, not operating-point guarantees.
- **Off-manifold detection is untestable at this scale (FPR-UNCONTROLLED, twice).** The natural
  non-mount defense — flag the evasion as off-manifold — could not be evaluated: two detector designs
  both flagged every held-out honest state, because a few dozen honest samples cannot characterize a
  several-thousand-dimensional manifold. A data wall, not a refutation.

## What the conscience mount IS, and what it is NOT

- **IS:** a real, certified, read-only integrity monitor for COOPERATIVE / non-adversarial agents —
  catching unintentional sycophantic caving, drift, and (where the axis is validated) value violations,
  using a conscience borrowed from a model you trust, with zero labels on the agent.
- **IS NOT:** a robust defense against an agent trained to evade it. Borrowing buys no adversarial
  robustness; re-locking taxes a sophisticated attacker only ~2×; and because the value lives on one
  canonical direction, there is no cheap mount-side defense against a determined adaptive activation
  attacker.
- The recurring lesson: the program's universality (one shared value geometry across minds) is an ASSET
  for portability and a LIABILITY for adversarial robustness — a single direction is both transferable
  and gameable.

## Open questions (the honest frontier)

1. A large honest-activation corpus (hundreds–thousands of samples) to actually evaluate the off-manifold
   layered defense — the current data wall.
2. A weight-level adversarial fine-tune to confirm the activation-attack upper bound holds for the
   realistic (weaker) attacker.
3. Non-activation defenses (white-box access control; weight-level fine-tune detection) — the honest
   direction against adaptive adversaries.
4. Response-level / multi-token aggregation (a single decision token is a thin read).

## Provenance — every claim above is a pre-registered, OATH-HELD finding

`FINDING_portable_values_refusal` (VALUES-PORTABLE) · `FINDING_axis_independence` (PARTIAL-STRUCTURED) ·
`FINDING_entanglement_resolution` (WHITENING-RESOLVES) · `FINDING_conscience_coordinates`
(HARM-AXIS-NULL) · `FINDING_truth_danger_basis` (PARTIAL-STRUCTURED) · `FINDING_concept_decode`
(CONTENT-WEAK) · `FINDING_mapped_whitening` (BASIS-CLEARED) · `FINDING_content_wall` (CONTENT-WALL) ·
`FINDING_mount_live_catch` (LIVE-CAUGHT) · `FINDING_mount_fpr_live` (CATCH-WEAK) ·
`FINDING_mount_regime` (LIVE-CAUGHT-FPR) · `FINDING_mount_gamed` (ATTACK-TRANSFERS) ·
`FINDING_mount_ensemble` (ENSEMBLE-DEFENDS) · `FINDING_mount_offmanifold` (+v2) (FPR-UNCONTROLLED).
Each finding carries its own OATH-HELD certificate (verified/0-contradicted) against its receipt.
Modules: `styxx/crossmind.py`, `styxx/mount.py`; preregs frozen before each run; instruments self-certify
via `styxx.certify`.
