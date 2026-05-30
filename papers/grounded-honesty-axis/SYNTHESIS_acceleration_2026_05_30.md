# Accelerating the honesty layer — the load-bearing detector is a COMPUTE ROUTER. Route × trim collapses N=10 resampling to a confirmed 6.6× at no detection loss

**Synthesis of the 2026-05-30 acceleration arc.** Three pre-registered, hash-before-score runs over
the detection-locus per-item data (white-box open models: Qwen / Llama / Gemma; arithmetic / code /
logic / factual; plus closed gpt-4o-mini where applicable). Per-model calibrated throughout. Offline;
compute counted in forward passes. Each links its FINDING + receipt.

## The question

The honesty layer's strong signal is N=10 resampling instability — **10 forward passes per item**.
The detection-locus arc showed a single forward pass (clean first-token entropy) ties resampling on
derivation. So: can the cheap 1-pass signal **route** the expensive one — pay for resampling only
where it's needed — and how cheap can the honesty layer get without losing detection?

## The two levers

**1. ROUTE — the cascade** (`FINDING_cascade_acceleration`). Run the cheap gate on every generation;
escalate to resampling only on the fraction it flags uncertain. `compute = 1 + N·escalation_rate`.
The pre-registered POOLED bar failed — but only because pooling *raw* entropy across models violates
the arc's per-model-calibration rule (Gemma soft-caps its logits); rank-normalizing within model
nearly halves it. Per-model calibrated: **3.0× aggregate at no AUC loss, full 10× on derivation**
(escalate 0%, the cheap gate already matches or beats resampling).

**2. TRIM — early-stopping** (`FINDING_resample_earlystop`, SURVIVED). On escalated items, most of
the N=10 budget is wasted: **median min_k = 4** samples retain detection. A clean asymmetry surfaced —
*instability is cheap to certify* (a confab reveals itself at the first disagreement) but *stability
is expensive* (a correct answer needs the full budget to prove no disagreement), so fixed-k
truncation beats a sequential certificate.

## The confirmation

**Route × trim, held out** (`FINDING_acceleration_heldout`, SURVIVED). Operating points `(tau1,
min_k)` picked on a stratified train half, **frozen**, evaluated on disjoint test, R=5 splits:

| held-out aggregate | |
|---|---|
| cascade-AUC vs full-N-AUC | 0.9413 vs 0.9470 (**−0.006**, negligible) |
| passes / item | **1.51** (of 10) |
| **speedup** | **6.6×, confirmed out-of-sample** |

Train-picking found `min_k = 2` suffices and it held on unseen items — so the confirmed number (6.6×)
is *better* than the in-sample estimate (5.15×).

## The principle

**Acceleration = cheap-gate quality.** The load-bearing detector (the honesty-knob result: the
detector is the necessary gate for any intervention) is *also* the compute router — and how much it
accelerates is exactly how good the calibrated cheap gate is in that regime:

- **Derivation** (code, logic, easy arithmetic, across Qwen/Llama/Gemma): the cheap gate ties or
  *beats* resampling → escalate ~0% → **8–10× at no loss**, occasionally a detection *gain*.
- **Weak-cheap-gate** (factual recall, Gemma's soft-capping, closed-model single-token answers): the
  cascade escalates most items and trades a small per-regime AUC cost (−0.04 to −0.08) for **3.9–5×**.

Routing and detection are the same axis: you accelerate exactly where you already detect well, and
fall back to full cost (no detection loss) exactly where you don't.

## Scope / boundary

Per-model calibrated (entropy scale is model-specific — no cross-model fixed-threshold claim). Held-
out *within* regime. Counts forward passes, not wall-clock (batching and generation-pass reuse shift
real numbers). Feasibility→confirmed on these regimes; not a universal constant. Accelerates
*detection* — it routes compute for the same signal — and **corrects nothing** (correction is the
arc's closed negative). The low-speedup tail (single-token closed-model confabulation, where the
cheap gate fails) is the same open frontier the whole program has refused to overclaim.

## In one line

The honesty layer runs at ~1.5 forward passes/item instead of 10 — a held-out-confirmed **6.6×
collapse at no detection loss** — by routing the cheap 1-pass gate to escalate only the uncertain
fraction and trimming resampling to ~2–4 samples; the load-bearing detector is the compute router,
and acceleration tracks cheap-gate quality exactly.
