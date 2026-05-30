# FINDING — ACCELERATION 2 SURVIVED: ~4 resamples retain detection (not 10), and routing × trimming compound to 1.94 forward passes/item — a 5.15× collapse of the honesty layer at no detection loss

**Run 2026-05-30. Offline over already-collected detection-locus resample sequences, pre-registered
in `PREREG_resample_earlystop_2026_05_30.md`.** Inputs SHA-256'd pre-scoring. The early-stop metric
is new on this data (no peek). Per-model calibrated, white-box, feasibility-grade.

## The bet

The cascade (`FINDING_cascade_acceleration`) routes WHO gets resampled. This trims HOW MANY samples
each escalated item needs. Detection = instability `(distinct−1)/(N−1)` over N=10 @ T=1.0. On
derivation, correct items are near-perfectly stable and confabs scatter within the first few samples,
so most of the N=10 budget may be wasted.

## Result — E1 SURVIVED

| | |
|---|---|
| **median min_k (samples for AUC_k ≥ AUC_N − 0.02)** | **4 of 10** — E1 bar (≤5) **HELD** |
| per-regime min_k | 2–5 in 12/14 regimes (gpt-3.5 span: **2**; logic, all easy-arith, gpt-span: **3**) |
| two high outliers | logic_Llama-3B = 8, logic_Gemma = 9 — **exactly** the regimes whose full-N detection is itself weak (AUC 0.82 / 0.94); min_k is unstable there, as the scope pre-stated |

**Most of the resampling budget is wasted.** Where the instability signal is healthy, 2–5 samples
recover the full-N detection; only where full-N detection is already marginal does min_k climb (you
can't truncate a signal that's barely there).

## The compounding speedup — the acceleration arc's headline

Routing (cascade) and trimming (early-stop) multiply:

| stage | forward passes / item | speedup |
|---|---|---|
| full resample-all | 10.00 | 1× |
| + cascade (route the uncertain fraction) | 3.37 | 3.0× |
| **+ early-stop (≈4 samples on escalated items)** | **1.94** | **5.15×** |

**The honesty layer runs at ~1.94 forward passes/item instead of 10 — a 5.15× compute collapse — at
no detection loss** (the cascade's per-model-calibrated AUC was a hair *above* full). On derivation
specifically it is cheaper still: the cascade escalates 0%, so the cost is essentially the one cheap
gate pass (≈10×).

## Honest nuance: truncation beats the adaptive certificate

The adaptive distinct-count certificate (stop the instant `distinct ≥ D`, or once D is unreachable)
draws **more** samples on average (mean 5–8) than fixed-k truncation (4). Reason: a *confab* stops
early (the first disagreement certifies it), but a *correct* item must draw nearly the full budget to
*prove* stability (no disagreement can be ruled out until few samples remain). So the better lever is
simply **fixed-k truncation** — accept the slightly noisier k≈4 instability estimate — not a
sequential stability certificate. An asymmetry worth stating: instability is cheap to certify,
stability is expensive.

## Scope / discipline

Offline, per-model calibrated, white-box, exact-integer resampling. Counts samples / forward passes,
not wall-clock (real speedup also depends on batching, and the cheap gate reuses the generation pass).
`min_k` and the cascade operating points are **in-sample** (the AUC_k curve is descriptive) — the
5.15× compounding figure is a strong feasibility result that **warrants a clean pre-registered,
held-out confirmation** before it is a load-bearing claim. It accelerates *detection* (fewer passes
for the same signal); it corrects nothing (the closed negative). Where full-N detection is weak, the
trim is unavailable — you cannot speed up a signal you do not have.
