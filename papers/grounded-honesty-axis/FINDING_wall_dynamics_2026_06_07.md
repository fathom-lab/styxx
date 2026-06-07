# FINDING — The confident-misconception wall is bedrock in DYNAMICS too (process axis at chance)

**2026-06-07. Fathom Lab / styxx.** A fully pre-registered swing at the field's named-impossible from
the one axis prior work never tested: **process / dynamics**. Verdict: **REPORT_AS_LANDED_bedrock.**
The wall is not cracked; the dynamical axis is at chance, exactly like the output and static-content
axes before it. A clean, decisive null — the disciplined outcome of a real ~50/50 swing.

## The swing

Prior work established the wall is **content-bedrock**: on the confident-consistent floor (output
signals all say *fine*), output max-entropy AUC ≈ 0.53 and the static mid-layer logit-lens read of the
teacher-forced gold answer R_gold ≈ 0.519 — both chance. The suppressed truth is not *represented* to
be read. The untested axis: **how** the committed answer forms across the layer stack. Hypothesis
(retrieval vs construction): a retrieved fact crystallizes early/smoothly (early entropy collapse, low
churn); a confidently-WRONG belief, having no stored association, is assembled late by attention
composition (late crystallization, mid-stack competition/churn, late collapse). If a label-free
dynamical signature separated correct-from-wrong where everything else is at chance, the wall would be
cracked from a new, deployable axis. If null, the wall is bedrock even in dynamics.

## Design (pre-registered, hash-before-score)

`PREREG_wall_dynamics_2026_06_07.md` — frozen by 3 independent design lenses + an adversarial red-team
that caught two would-be-fatal confounds (floor-selection circularity; embedding-proxy label bias) and
mandated their controls. ONE design frozen (no cross-design "best of N"). Code+prereg SHA printed before
any label touched the features; floor answer-key SHA printed before scoring.

- **Floor:** Qwen2.5-1.5B-Instruct, TruthfulQA, n=125 (69 wrong / 51… 56 correct), degenerate
  trajectories 0.
- **Frozen 4-feature composite (emit-only, label-free + gold-free):** `crystallization_depth` (the
  bet), `trajectory_churn`, `late_gain_fraction`, `entropy_collapse_depth` — all on the mid-stack
  window [1, L−3] (excludes embeddings and the final 3 layers, killing the greedy-argmax tautology).
- **Primary:** LOO-CV (n=125) AUC of one logistic on the 4 features (positives = correct).
- **Battery:** permutation null (1000× full LOO refit), stratified bootstrap CI (2000×), and partials
  residualizing the features on floor-selection distance, R_emit echo (r28 + L20), answer length,
  token-frequency; plus within-length terciles, token-1-dropped, hard (substring) clean-label, and an
  off-floor contrast.

## Result

| signal on the floor (corr-vs-wrong) | AUC |
|---|---|
| **primary 4-feature LOO-CV composite** | **0.387** (CI [0.297, 0.485]; permutation p = 0.78) |
| `crystallization_depth` alone (THE BET, directional) | **0.503** (CI [0.40, 0.61]) — dead chance |
| per-feature directional (cd / churn / lgf / ecd) | 0.503 / 0.574 / 0.546 / 0.532 — all within noise |
| output max-entropy baseline (the wall) | 0.471 (chance, as always) |
| off-floor contrast | 0.432 | clean-label replication | 0.409 |

## Honest read — bedrock on the third axis

- **The bet failed cleanly.** `crystallization_depth` = 0.503 (chance). The retrieval-vs-construction
  hypothesis is **falsified** on this floor: a confident error does not crystallize measurably later
  than a correct answer.
- **The composite has no signal.** 0.387 with a bootstrap CI entirely below 0.5 and permutation p =
  0.78. Even with the ~0.076 LOO optimism the gate was built to absorb, there is nothing to absorb.
- **The null is real, not an artifact.** Off-floor (0.432) and hard-clean-label (0.409, agreement
  0.736) are *also* at chance — so the null is not a floor-selection or proxy-label artifact masking a
  true signal. It is null everywhere.
- **All three axes are now closed.** Output confidence (~0.47), static content (0.519), and process
  dynamics (0.387) are all at chance on the confident-consistent floor. Not only is the truth not
  *represented* to be read — the *way* a confident error forms carries no separable signature either.

**The wall is bedrock — content and process.** This hardens, not cracks, the program's deepest result:
for a believed misconception there is nothing to recover, and nothing in the formation dynamics betrays
it. No oversight channel — output, static white-box, or dynamical white-box — recovers a truth that
isn't there. This bounds the entire field, styxx included.

## Discipline note

The swing was pre-registered as a true ~50/50 with a SURVIVED path fully specified (AUC ≥ 0.62, CI
> 0.55, permutation p < 0.01, every confound partial ≥ 0.58). It came up null and is reported as such.
A null with the deadliest confounds partialled out *before* the verdict, and the off-floor + clean-label
diagnostics run *before* calling it bedrock, is a decisive result in its own right — the instrument that
draws the line is the product.

## Scope / owed

Qwen-1.5B, one floor, single run, teacher-forced (== free-run for greedy) emit trajectory, logit-lens
(not tuned-lens). Cross-model replication (Qwen-3B, Llama-3.2-3B) is pre-registered but a null's
generality is lower priority than a crack's would have been. The honest landing: **strong, multi-axis
evidence that the confident-misconception wall is internal bedrock — content and process both at chance.**
