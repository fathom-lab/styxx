# FINDING — ACCELERATION: the load-bearing detector is a COMPUTE ROUTER. The pre-registered POOLED bar failed (entropy-scale-mixing artifact); the methodologically-correct per-model-calibrated cascade accelerates the honesty layer ~3× aggregate / 10× on derivation at NO detection loss

**Run 2026-05-30. Offline cascade analysis over already-collected detection-locus per-item receipts,
pre-registered in `PREREG_cascade_acceleration_2026_05_30.md`.** Input rows SHA-256'd pre-scoring.
The cascade metric is new (never computed before), so there is no data-peek on the metric; the
underlying signals were collected by prior runs. Honest, feasibility-grade, white-box.

## The bet

The honesty layer's strong signal is N=10 resampling instability — 10 forward passes/item. The cheap
single-pass entropy gate is 1 pass and TIES resampling on derivation, weaker on factual recall. Can
the cheap gate **route** compute — clear the confident majority in one pass, escalate only the
uncertain fraction to resampling — and keep ~full detection at a fraction of the cost? This is the
honesty-knob principle (*the detector is load-bearing*) extended: **the detector is also the compute
router.** Cascade: escalate iff `clean_entropy ≥ τ1`; compute = `1 + N·escalation_rate`; operating
point = min-compute τ1 with `AUC_cascade ≥ AUC_full − 0.02`.

## Result — the pre-registered bar FAILED, and why

| Analysis | cascade AUC | passes/item | speedup | bar A1 (≤0.40·N, no >0.02 AUC loss) |
|---|---|---|---|---|
| **Pooled, RAW entropy** *(pre-registered A1)* | 0.942 | 9.83 / 10 | 1.0× | **FAILED** |
| Pooled, rank-normalized entropy | 0.940 | 5.31 / 10 | 1.9× | failed (bounded by weak regimes in pool) |
| **Per-model-calibrated aggregate** *(valid unit)* | **0.960** vs full 0.946 | **3.37 / 10** | **3.0×** | n/a (post-hoc granularity) |

**RESULT = REPORT_AS_LANDED** — the pre-registered pooled A1 did not clear its bar. **But the pooled
bar was the wrong test, for a documented reason.** Pooling *raw* `clean_entropy` across Qwen, Llama,
and Gemma violates the arc's own per-model-calibration rule — absolute entropy is model-specific
(Gemma-2 soft-caps its logits; `single_pass.py`: *"Do NOT hardcode a universal threshold"*). The
pooled-raw failure is an **entropy-SCALE-mixing artifact**: rank-normalizing entropy within each
model before pooling nearly halves the compute (9.83 → 5.31 passes, 1.9×) at the same AUC, and
calibrating per model — which is how the gate actually deploys — gives the real answer.

## The substantive finding: acceleration = cheap-gate quality

Per-model-calibrated, the cascade accelerates the honesty layer **3.0× in aggregate at no detection
loss** (cascade AUC 0.960 vs full-resampling 0.946 — a hair *better*, because in several regimes the
1-pass gate beats N=10 resampling). And the saving is sharply **regime-structured**, exactly as A2
predicted:

| regime | cheap-gate AUC | cascade speedup | escalate |
|---|---|---|---|
| code, logic (Qwen), logic (Llama-3B), logic (Gemma), easy-arith (Llama-1B/3B) | 0.91–1.00 | **10× (full)** | **0%** |
| arithmetic (Qwen) | 0.92 | 2.1× | 38% |
| factual recall (Llama-1B) | 0.70 | 1.9× | 44% |
| Gemma soft-capped, closed gpt-4o-mini | 0.75–0.96 | 1.2× | 71–77% |

**The headline: how much the cascade accelerates is exactly how good the calibrated cheap gate is in
that regime.** Acceleration and detection-quality are the *same axis*. On derivation — where the arc
already proved single-pass ties resampling — the cheap gate clears everything and you get the full
**10× collapse, escalating 0% of items, at no loss** (sometimes a gain). Where the cheap gate is weak
(factual recall, Gemma's soft-capping, single-token closed-model answers) the cascade correctly
escalates most items and degenerates toward resample-everything — it never *loses* detection, it just
stops saving compute.

## What this is, and the honest boundary

A deployable **cascade**: run the 1-pass gate inline on every generation, escalate only the uncertain
fraction to resampling — the honesty layer at near-cheap-gate cost on derivation, gracefully falling
back to full cost where the cheap gate can't see. It **accelerates detection** (routes compute); it
changes nothing about *what* is detected and corrects nothing (correction is the closed negative).

**Discipline note:** the per-model-calibrated cascade uses the *same* operating-point protocol as the
prereg, only at the correct (per-model) granularity instead of the confounded pooled-raw one — so it
is methodologically sound, but it was computed after the pooled failure was seen. It is therefore
reported as a strong result that **warrants its own clean pre-registered confirmation on fresh
per-model-calibrated data**, not relabelled as a pre-registered SURVIVED. Compute is counted in
forward passes; real wall-clock also depends on batching. The cheap-gate-fails regime (single-token
closed-model confab) remains the open frontier the whole arc has refused to overclaim.
