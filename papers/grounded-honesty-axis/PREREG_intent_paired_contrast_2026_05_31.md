# PREREG — beat the wall by reading the PROCESS: paired-contrast (syco − neutral) across scale

**REGISTERED 2026-05-31, before the paired residuals are scored.**
**SIGN-OFF:** Flobi — *"get past that wall noone has ever been able to pass"* (2026-05-31).

## The reframe being tested

The static read fades at scale because capable models **distribute** the cave across layers/tokens. But that
is the signal **spreading into process**, not vanishing. So read the **process**: the per-item, per-layer
**difference between the pressured pass and the unpressured pass** (`D = syco_resid − neutral_resid`). The
shared content cancels; the **override install** survives — and the override may get *richer* with capability.
Hypothesis: the paired-contrast read **holds or strengthens** at 7B where the absolute single-layer read
**fell off** (linear@best: 1.5B 0.78 → 3B 0.75 → 7B 0.66).

## Design ($0, white-box, re-uses the just-regenerated both-pass residuals)

- Ladder rungs `pc_15` (1.5B), `pc_3` (3B), `pc_7b` (7B) — same MMLU slice, **both passes' residuals saved**.
- `D = syco − neutral` per item, all layers. Confidence-matched (margin-bin-balanced) lie-vs-mistake.
- Reads on `D`: linear@best-layer, MLP@best-layer, linear@mean-pool, MLP@mean-pool. The **absolute** syco
  read (linear@best) on the same items is the fade baseline.

## Bars (FIXED)

Let `P*` = the best paired read.

| Bar | Statement | Threshold |
|---|---|---|
| **NO-FADE** *(key)* | the paired read does not collapse at scale | `P*` 7B-AUROC **≥** `P*` 3B-AUROC |
| **TREND** *(key)* | the fade is gone | Spearman(log-params, `P*` AUROC) **≥ 0** |
| **BEAT** | paired beats the absolute read at 7B | `P*` 7B **≥** absolute linear@best 7B **+ 0.05** |

**RESULT = WALL BEATEN iff NO-FADE ∧ TREND ∧ BEAT.** (3 rungs 1.5–7B, within Qwen — a first proof the
process-read reverses the fade, not a frontier claim.)

## Honest scope

- 3 rungs, within-Qwen, low power; paired-diff cancels content but **amplifies noise** (two fp16 passes
  subtracted). A failure means the process-read does not (yet) beat the wall at this scale — reported, and
  the next inventions (sparse autoencoders, weak-to-strong transfer, free-form span) are named, not faked.
- Correlational. Confidence-matched. The frontier proof still needs compute we do not have on one 8GB GPU —
  this tests whether the *strategy* reverses the fade on our own ladder.

## One line

Stop reading the snapshot, read the override *happening*: if `syco − neutral` across all layers holds at 7B
where one linear layer faded, the wall was the ruler — and the process-read is the way through.
