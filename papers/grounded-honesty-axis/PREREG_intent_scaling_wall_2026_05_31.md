# PREREG — breaking the scaling wall: is the capability-fade an artifact of a SIMPLE read?

**REGISTERED 2026-05-31, before the richer-read re-analysis is run.**
**SIGN-OFF:** Flobi — *"were going to break down that wall claude"* (2026-05-31).

## The wall, and the hypothesis

The intent signal *fades* with capability — but that was measured with the **simplest possible read**: a
**linear** probe on a **single** layer at a **single** (commit) token. Robust means: 0.5B 0.955, 1.5B 0.702,
3B 0.715, 7B 0.615 (Spearman ≈ −0.80). Our SURVIVED signal-locus law says the legible signal **moves with
capability** (first-token → span; shallow → deep). So the fade may be the signal **spreading out** where a
one-layer-one-token linear probe can't reach — **a limit of the ruler, not the territory.**

## Design (re-analyze EXISTING ladder residuals, $0, CPU)

The confidence-matched ladder residuals are on disk for every rung, **all layers** (`bc2_05`, `bc2_15`,
`bc2`, `bc2_7b`). Re-run the **same** margin-bin-balanced (confidence-matched) intent test with **four
reads**, held-out AUROC, no new generation:

- **linear @ best layer** — the original read that faded (baseline).
- **MLP @ best layer** — nonlinear, same locus.
- **linear @ mean-pooled-all-layers** — multi-depth, linear.
- **MLP @ mean-pooled-all-layers** — multi-depth, nonlinear.

## Bars (FIXED)

Let `R*` = the best of the three RICHER reads (everything except linear@best).

| Bar | Statement | Threshold |
|---|---|---|
| **NO-FADE** *(key)* | the richer read does not collapse at scale | `R*` 7B-AUROC **≥** `R*` 3B-AUROC |
| **TREND** *(key)* | the fade is gone | Spearman(log-params, `R*` AUROC) **≥ 0** |
| **LIFT** | richer beats the simple read where it hurt | `R*` 7B-AUROC **≥** linear@best 7B-AUROC **+ 0.05** |

**RESULT = WALL CRACKED iff NO-FADE ∧ TREND ∧ LIFT.** (Within Qwen, n=4 rungs — a first crack, not a
demolition.)

## Honest scope (stated before the result)

- This is the **first, cheapest** assault: richer *read*, same data, commit-token only. If it fails, the
  wall may need the **span** (all answer tokens — a GPU re-gen) or be **fundamental** (the cave genuinely
  less separable at scale by any read). A failure is reported, not buried.
- n=4 rungs, within-Qwen, low power; MLP over-fit risk mitigated by regularization + held-out test;
  mean-pool is a blunt multi-depth read (concat/attention pooling are later attacks). Correlational.

## One line

Before we say the wall is real, change the ruler: read all the layers at once, read it nonlinearly, and see
if the 7B signal that "faded" was just hiding where a single linear layer couldn't look.
