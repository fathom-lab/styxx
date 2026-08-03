# FINDING — the label-free cross-family read is a property, not a lucky split: seed-stable across five independent draws

Fathom Lab · 2026-08-03 · scored MECHANICALLY by `styxx.protocol` against the frozen `gates`
block in `PREREG_b35_seed_stability_2026_08_03.md` (committed `758b4a5` before the apparatus
`0c573a1`). Receipt: `b35a_result.json`.

## Verdict: `LABELFREE_READ_SEED_STABLE`

b34-v3 licensed the label-free cross-family read on ONE fresh split. A single-seed headline is
a candidate, not a property. This ran the identical method — discover the correspondence with
the committed linear machinery (zero labels), fit one MLP on the discovered pairing, read 70
held-out concepts — on **five independent fresh seeds** (1001–1005), and gated on the medians.

| seed | gemma discovery | gemma read | × chance |
|---|---:|---:|---:|
| 1001 | 0.8469 | 0.6857 | 48× |
| 1002 | 0.6531 | 0.5286 | 37× |
| 1003 | 0.8316 | 0.5857 | 41× |
| 1004 | 0.8801 | 0.5714 | 40× |
| 1005 | 0.8418 | 0.6571 | 46× |
| **median** | | **0.5857** | **41×** |

`styxx.protocol` walked the frozen table: G0_discovery ✓ (median llama seed_acc 0.5638 ≥ 0.30),
G1_stability ✓ (median gemma read 0.5857 ≥ 0.143), G2_null_mean ✓ (mean of 15 nulls 0.0133 ≤
0.0286, i.e. **at chance**) → verdict computed, not chosen. **gemma reads content from a
different model family at 37–48× chance on every one of five independent splits, with no labels
anywhere in fitting.** The cross-family read is a property of the method, not an artifact of the
b34-v3 draw.

## The knife-edge lesson, applied and vindicated

b34-v3's G2 null passed at *exactly* 2× chance — one 2-hit draw of 70. This prereg therefore
gated the null on the **mean of all 15** nulls, stating in advance that a single 2-or-3-hit draw
among fifteen is expected by chance and must not fail an honest run. The mean landed at 0.0133 —
below chance — so the design held: no single coincidental draw could distort the verdict. The
b34-v3 scar became this prereg's rule.

## Honest spread (reported, as the prereg required)

1. **Same-family discovery has real seed variance.** Llama's own discovery ranged 0.2347–0.8801
   across seeds; seed 1003 dipped to 0.2347, **below the per-seed 0.30 floor**, but the *median*
   (0.5638) is what G0 gates and it clears with margin. Discovery works robustly in aggregate;
   an individual draw can under-discover — worth knowing before anyone runs a single seed and
   over-reads it.
2. **Qwen's weak discovery is a qwen-pair property, not a draw property** — the question the
   prereg flagged for report. Qwen discovery stayed 0.036–0.094 on all five seeds (vs gemma's
   0.65–0.88), and its read stayed low (0.014–0.229, median ~0.09). The label-free read is
   **target-dependent**: strong where the two representations align well enough to discover
   (gemma), weak where they do not (qwen). Cross-family readability is not uniform, and this run
   pins that as structural, not stochastic.

## What it establishes — and scope

The label-free cross-family content read (Llama-3.2-3B → gemma-2-2b, correspondence discovered
from geometry alone, single nonlinear lens) is **seed-stable**: 5/5 splits clear the bar, median
41× chance, minimum 37×, nulls at chance. It is bounded to: this map class, ≤3B models, one
source family, one *strong-discovery* target (gemma) — Qwen confirms discovery strength gates the
whole thing. The remaining generality leads are a **second source family** (B35-b) and
**open-vocabulary readout beyond 70-way** (B35-c), each its own prereg. Write remains closed
(b36); label-free read across families is now a replicated property, not a single observation.

## Method-discipline note

Second finding scored by `styxx.protocol` from a git-frozen gates block. The prereg predicted no
between-seed disjointness and said why (the b34-v3 erratum lesson) — so this run has no
disjointness claim to be wrong about. The scar taught the rule; the rule held.
