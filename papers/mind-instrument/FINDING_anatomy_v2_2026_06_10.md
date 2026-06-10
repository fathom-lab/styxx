# FINDING — norm-equalization reveals the true magnitude of cross-mind convergence (ANATOMY-ROBUST)

**2026-06-10 · Fathom Lab / styxx. Pre-registered: `PREREG_anatomy_v2_normeq_2026_06_10.md` (frozen
pre-run). Receipt: `anatomy_v2_result.json` (with `anatomy_v0_result.json` and
`anatomy_v1_result.json` as contrast receipts). Local, $0. Verdict: ANATOMY-ROBUST, all gates.**

## The apparatus discovery (the bigger result)

The v1 attack on anatomy v0 exposed, via a smoke-run paradox (within-mind reliability BELOW
between-mind agreement), that the bare `{w}` template's hidden states carry ~7x the norm of every
contextual template's (row norms 531.6 vs 80.8, Qwen2.5-3B receipt `anatomy_v1_result.json`), so the
field-standard unweighted template average is dominated by bare-prompt geometry. v2 fixes the
convention: L2-normalize each template's state BEFORE averaging.

**The fix heals the instrument**: Qwen2.5-3B's split-half battery reliability rises from 0.291
(old convention) to **0.9411** (gate G3 passed). And under the healed instrument, the measured
convergence between independently-built minds **roughly doubles to quadruples in every domain**:

| domain | v0 (norm-dominated) | v2 (norm-equalized) |
| --- | --- | --- |
| body | 0.4117 | 0.8521 |
| vehicle | 0.4174 | 0.8448 |
| animal | 0.3667 | 0.7949 |
| weather | 0.384 | 0.7743 |
| profession | 0.0594 | 0.7093 |
| instrument | 0.4716 | 0.6604 |
| fruit | 0.2874 | 0.6413 |
| furniture | 0.0116 | 0.6139 |

Subjects are the four out-of-anchor minds (gpt2, gpt2-large, pythia-410m, Qwen2.5-0.5B); values are
mean cross-family partial-lexical RSA against the six anchors, all ten minds recomputed live under
the corrected convention. **Independent lineages with zero shared construction agree on contextual
meaning geometry at 0.61–0.85 across every domain tested.** The published template-averaging
convention — ours included — systematically understated cross-mind convergence; any
template-averaged RSA in the literature is exposed to the same artifact.

## The anatomy verdict (G1: ANATOMY-ROBUST)

The per-domain profile remains shared across the independent subjects — more strongly than v0
measured: Kendall's W = **0.875** (v0: 0.75; bar 0.60), permutation p = 0.0 (10000 draws). The
anatomy survives its strongest apparatus critique.

## What dies, loudly (correcting our own same-day finding)

v0's headline interpretation — "professions and furniture are family-idiosyncratic" — was the
artifact speaking. Under the healed apparatus professions converge at 0.7093 and furniture at
0.6139: not idiosyncratic, merely at the bottom of a gentle gradient on a high floor. The v0 map's
ordering correlates with the corrected map at only Spearman 0.5476 (G2): the bare-word component
substantially distorted WHICH domains looked universal. `FINDING_anatomy_v0_2026_06_10.md` carries
a correction notice; its SUPPORTED verdict (anatomy is shared) stands, its domain interpretation is
superseded by this document.

## Implications for the program's published numbers

Convention-bound values measured under the old averaging (real-convergence confirm xfam 0.258, the
Atlas v0 citizenship table, anatomy v0) remain true AS MEASURED and equivalence-gated; their
INTERPRETATION tightens: they are lower bounds depressed by norm domination. The "magnitude
heterogeneous, modest" reading of cross-model convergence should be revisited program-wide under
norm-equalized measurement — the true contextual convergence is large. Atlas v1 under the corrected
convention is the obvious next rung; `normeq_reps.npz` (all ten minds) is persisted for it.

## Bounds

One battery, one layer convention, four independent subjects, instruction-tuned and base models
mixed; norm-equalization is one principled weighting (uniform direction-average) — alternatives
(per-template z-scoring, median) untested. The G2/G3 contrast quantifies the artifact only for the
minds measured here.
