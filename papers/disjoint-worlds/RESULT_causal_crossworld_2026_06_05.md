# RESULT — causal meaning transfers across disjoint worlds through an unsupervised map (faithfulness-gated, NOT distinctiveness-gated)

**Date:** 2026-06-05 · **Verdict: CAUSAL TRANSFER CONFIRMED — universal forms carry transferable causal
structure, not just correlational geometry.** With one pre-registered sub-prediction corrected (the effect
does **not** need distinctive structure) and one genuinely new finding (causal transfer **decouples from**
exact instance recovery). Frozen: `PREREG_causal_crossworld_2026_06_05.md` (P2 amended pre-confirmatory,
P1 untouched). Two disjoint worlds, zero shared data, unsupervised GW map, 3 seeds.

## Numbers — cross-world attribute transfer (cosine)
| condition | faith | recovery | same-attr (diag) | separable (off) | diag advantage | null | true-map ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|
| distinctive σ=0.02 | 0.97 | 0.31 | **0.73** | −0.01 | **0.74** | −0.09 | 1.01 |
| distinctive σ=0.05 | 0.97 | 0.29 | **0.76** | −0.02 | 0.78 | −0.12 | 1.01 |
| isotropic σ=0.02 | 0.99 | 0.14 | **0.50** | −0.03 | 0.53 | +0.05 | 1.01 |
| isotropic σ=0.12 | 0.98 | 0.11 | 0.35 | −0.02 | 0.38 | +0.01 | 1.01 |
| distinctive σ=0.7 (low faith) | 0.85 | 0.04 | 0.23 | −0.03 | 0.26 | +0.01 | 0.98 |

## What it shows (read straight)
**P1 — CONFIRMED, the decisive result.** An attribute-manipulation direction from world A, pushed through a
map learned with **zero paired concepts**, lands on the **same** attribute in world B (cosine 0.73) and
stays **orthogonal** to separable attributes (cosine ≈ 0). It beats the shuffled-map null (≈0 or negative)
by ~0.8 and approaches the true-map ceiling (1.0). The cross-world causal-inner-product structure
(Park–Veitch) is preserved by an unsupervised geometric alignment. **The shared geometry of meaning is not
merely correlational — it carries transferable causal structure across zero-shared-data worlds.** The
strongest Half-A result to date.

**P3 — CORRECTED (a pre-registered sub-prediction that did not hold).** I predicted the transfer would
*need* distinctive structure (isotropic collapses). It does **not**: isotropic transfers at 0.50 (same
attr) with clean orthogonality (−0.03), faithfulness-gated just like distinctive. **The transfer needs
FAITHFULNESS, not distinctiveness** — it collapses as faithfulness drops (0.73 → 0.23 as faith 0.97 →
0.85), but holds for *any* shared structure at high fidelity. The auto-gate's "needs distinctive structure"
reading is wrong for transfer; it held only for the *recovery* channel. Reported as a failed sub-prediction.
The metric-artifact concern is independently ruled out by the null (random maps fail) and the true-map
ceiling (1.0), not by isotropic collapse.

**The new finding — causal transfer DECOUPLES from exact recovery.** Isotropic transfers cleanly (0.50) at
only **0.14** instance-recovery; distinctive transfers at 0.73 with 0.31 recovery. **You can recover *what
the structure means* (attribute/causal directions transfer) even when you cannot recover *which concept is
which* (instance identity).** The unsupervised alignment pins the geometry's *rotation* — enough for causal
directions to ride across — far more easily than it pins instance correspondence (a sharp, ~0.6-ceiling
threshold per the 06-03 scaling data). Meaning transfers at the structural level below the identity-recovery
threshold. This is the part the pre-registration did not anticipate.

## Pre-registration accounting (full transparency)
- **Original gate** (recovery ≥ 0.80): would read INCONCLUSIVE — recovery (0.31) never reaches 0.80, because
  that bar sat above the instrument's known ~0.6 ceiling (the mis-calibration the amendment fixed).
- **Amended gate** (recovery ≥ 0.30, P1 untouched): P1 ✓ (0.74, beats null by 0.83), P2 ✓ (0.31, thin),
  P3 ✓ *via the recovery channel only*. Auto-reading "CAUSAL UNIVERSALITY" — **corrected here** to "causal
  transfer confirmed, faithfulness-gated not distinctiveness-gated," because P3's *intent* (distinctiveness
  needed) is falsified for transfer. The decisive P1 claim stands on its own (null + ceiling controls).

## Honest bounds
Linear synthetic construction: each world's geometry is an orthogonal embedding of the *same* latent z, so
"directions transfer through the correct rotation" is partly built in. The empirical, non-trivial content
is: (1) the **unsupervised** map (GW + Procrustes on recovered correspondence) approximates the true
rotation well enough to transfer (0.73 vs ceiling 1.0) **at only 0.31 instance-recovery**; (2) the
**faithfulness threshold**; (3) the **transfer⊥recovery decoupling**; (4) clean **separable-orthogonality**
everywhere. Real nonlinear embeddings (where geometry is a *warp*, not a rotation, of shared structure)
remain the open frontier — and the place this could break.

## Arc & instrument meaning
This is the causal arm Half A named and never ran: the geometry of meaning is shared (RSA) **and** carries
transferable causal structure across zero-shared-data worlds — universal forms in the strongest in-silico
sense, bounded to the attribute level and the linear regime. For styxx the instrument: a steering/attribute
direction transferring through an **unsupervised cross-model map** is exactly the cross-model representation
control (and attack surface) the integrity layer must reason about — concept manipulation can ride from one
model into another with zero paired data, below the identity-recovery threshold. Logged as a capability
foundation, not yet a feature (per the geometry-probe productization lesson).
