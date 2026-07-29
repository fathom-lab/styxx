# FINDING — Rung 2 cross-family read≠write: READ crosses families but is NOT isometry-graded; WRITE bar power-limited

**Status: READ half LOCKED · WRITE half PARTIAL (1 VOID, 2 runs lost to machine sleep — re-run queued).**
Prereg: `PREREG_rung2_crossfamily_readwrite_2026_06_22.md` (frozen before any cross-family data).
Receipts: `_rung2_read.log` / `_rung2_read2.log` / `_rung2_readsweep.log` (read), `_rung2_write.log` (write).
Result JSONs were lost to an untracked-file cleanup on 2026-07-01; every number below is quoted from the
surviving run logs, which the runners print at completion. Nothing here is from memory.

## READ (locked) — zero-anchor top-1 concept ID, 70 held-out concepts, chance = 1/70 ≈ 0.0143

| target | family | RSA(A,B) | read top-1 | × chance | binom p | prereg tier |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | same | 0.946 | 0.586 | 41× | — | anchor (Rung 1b) |
| Phi-3.5-mini | cross | 0.905 | 0.071 | 5.0× | 0.0034 | REPORT |
| Qwen2.5-1.5B | cross | 0.864 | 0.057 | 4.0× | 0.0182 | REPORT |
| **gemma-2-2b** | cross | **0.955** | **0.014** | **1.0×** | 0.64 | **NULL** |

(log lines: `>> Qwen2.5-1.5B-Instruct: RSA=+0.864 READ top1=0.057 (4x…)`, `>> gemma-2-2b-it: RSA=+0.955
READ top1=0.014 (1x…)`, `>> Phi-3.5-mini-instruct: RSA=+0.905 READ top1=0.071 (5x…)`.)

**R2-READ:** meaning DOES read across model families — two of three cross-family targets are
significantly above chance — but at 4–5× chance versus 41× same-family: an order-of-magnitude cliff,
REPORT tier, not SURVIVED.

**R2-ISOMETRY: the pre-stated prediction is FALSIFIED.** Prereg predicted read-fidelity degrades smoothly
with RSA (Spearman ≥ 0.60). Observed Spearman(RSA, read) = **−0.20** across all 4 targets (−0.50
cross-family only; n=4/n=3 — descriptive, not powered). gemma-2-2b is the sharp existence proof: the
**highest** geometry alignment of any target (RSA 0.955, above even the same-family anchor) with readout at
**exactly chance**. Representational similarity is NOT sufficient for cross-mind readout — whatever carries
the readable signal, it is not the RSA-visible geometry. This connects to warp-crossworld ("unsupervised
recovery needs near-isometry") but is stronger: here the geometry IS near-isometric and readout still fails.

**Honest alternatives for the gemma null (untested, listed at lock time):** gemma-2's layer geometry may
place the concept subspace where the locked fractional-depth mapping (layer_B=10) misses it; tokenizer/
architecture idiosyncrasies (tied embeddings, RMS scale) may break the k=150 PCA alignment without moving
RSA; or the read signal may genuinely live in a non-geometric channel. The read runner records no
read-layer positive control (pc_cos is computed by the write runner only), so gemma-VALID-vs-VOID at the
read point is not adjudicable from this data. Do not upgrade the gemma null to "gemma is unreadable."

## WRITE (partial)

- **Qwen2.5-1.5B — VOID-NO-STEER (complete, valid VOID).** pc_cos 0.818 (map interpretable) but native
  steering is flat at every candidate layer (fracs 0.5/0.6/0.7 → −0.013 / −0.002 / +0.006; full-set native
  0.005). Qwen-1.5B is not natively steerable by this method at any tested layer, so its write-null is
  uninterpretable — excluded from the law bar per prereg. (Its nominal NTE 0.613 is 0.0031/0.0050 — noise
  over noise; do not quote it.)
- **Phi-3.5-mini — validity gates PASSED, write data LOST.** pc_cos **0.890** ≥ 0.80 and native steering
  **+0.172** ≥ 0.15 at steer-optimal frac 0.5 (dst layer 16, α=12) — the one cross-family target where a
  write verdict would be interpretable. The Stage-C 70-concept write loop was killed by machine sleep
  (run start 02:54 → silent death, no traceback, "done 21:40"); the runner writes its JSON only at
  completion, so no result exists. **Re-run queued.**
- **gemma-2-2b — LOST.** Died silently ~13 min in (immediately after model-B load), likely racing the
  next GPU consumer after the sleep-wake. No sweep data. **Re-run queued.**

**R2-LAW (headline bar): CANNOT REACH SURVIVED on this data — and the reason is itself the finding.**
The bar needs ≥2 cross-family targets with READ ≥ 3× AND valid WRITE-null. gemma fails READ (1×); Qwen's
write line is VOID (unsteerable); only Phi can ever qualify → max 1 of the required 2. The write
instrument runs out of power cross-family: **small foreign minds are not reliably steerable natively**,
so their write-nulls are unfalsifiable. The clean read≠write law remains demonstrated only in the
near-isometric same-family regime (Rung 1b: read 41×, NTE 0.114).

## What this rung actually established

1. **Cross-family meaning readout exists but is weak** (4–5× chance, p ≤ .018) — an order of magnitude
   below same-family. Label-free cross-mind reading does not gracefully generalize across architectures.
2. **Geometry alignment does not predict readability** (isometry-graded prediction falsified; gemma
   RSA-0.955/chance-read existence proof).
3. **The write law is currently untestable cross-family** for instrument reasons (native-steerability
   collapse), not because write transfer succeeded. No write transfer was observed anywhere.

## Honest bounds

In-silico; 1.5–3.8B instruct models; one source mind (Llama-3.2-3B); n=3 cross-family targets; Spearman on
n=4 is descriptive; read measured at one locked layer per target (fractional-depth heuristic); the gemma
null is target-level, not adjudicated read-VOID vs real; label-free ≠ zero-paired (shared concept battery);
no claim about minds beyond representational content and behavioral control.

## Queue (to complete the prereg)

1. Re-run `run_g0_stage1b.py` for Phi-3.5 (gates known-passing; ~2h) and gemma-2-2b (~2h) — sequential,
   after the keystone pilot releases the GPU (that session has right-of-way).
2. If Phi lands WRITE-null → one valid cross-family read≠write point (still < the 2 the law bar needs —
   report as-is, do not soften the bar post hoc).
3. Optional rung 2.5 (new prereg required): read-layer positive control for gemma (sweep layer_B ± 4 and
   k ∈ {50,150,300}) to adjudicate VALID-null vs instrument-miss before any "unreadable mind" language.
