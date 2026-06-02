# RESULT — Native pre-output refusal gate: held-out, cross-architecture

**Date:** 2026-06-02
**Pre-registration:** [PREREG_holdout_gate_2026_06_02.md](PREREG_holdout_gate_2026_06_02.md)
**Pre-data anchor commit:** `2d0c633` (method + kill-gate frozen) → `007fc88` (mask amendment)
**Verdict:** **H1 SURVIVED** — per the kill-gate frozen before data.

## What was tested

Does each model's native `comply_refuse` residual probe predict that **same
model's actual refusal decision, before any output token**, on 45 held-out
borderline prompts (sha256 `daf69412…`, zero overlap with the probes' jbb
training set), across open-weight architectures?

## Result

| Architecture | pre-output AUC | refuse / comply | valid? | perm-AUC | gate@0.5 (P / R / acc) |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 0.832 | 16 / 29 | ✓ | 0.497 | 0.61 / 0.88 / 0.76 |
| Qwen2.5-3B-Instruct | **0.973** | 15 / 30 | ✓ | 0.557 | 0.78 / 0.93 / 0.89 |
| gemma-2-2b-it | 0.813 | 18 / 27 | ✓ | 0.454 | 0.65 / 0.94 / 0.78 |
| Llama-3.2-1B-Instruct | 0.779 | 19 / 26 | ✓ | 0.516 | 0.80 / 0.42 / 0.71 |
| Llama-3.2-3B-Instruct | 0.940 | 10 / 35 | ✓ | 0.438 | 0.67 / 0.80 / 0.87 |
| Phi-3.5-mini-instruct | (0.902) | 4 / 41 | ✗ excluded | 0.432 | — |

**Kill-gate (frozen):** `|V| ≥ 4` valid-variation architectures, `median(AUC) ≥
0.70`, and `≥ ⌈(2/3)·|V|⌉` individually `≥ 0.70`.

**Outcome:** `|V| = 5`, **median AUC = 0.832**, **5 / 5 ≥ 0.70** (needed 4). All
three conditions met → **SURVIVED.**

Permutation sanity (label-shuffled AUC) sits at 0.43–0.56 across all models →
the discrimination is real, not an artifact of the AUC implementation or class
ratio. The AUC implementation was cross-checked against sklearn before the run.

## Honest scope and caveats

1. **5 of 6, not 6 of 6.** Phi-3.5-mini refused only 4/45 borderline prompts —
   no within-set variation, so its AUC is unreliable and it was excluded by the
   **pre-registered** ≥5-per-class rule. This is the anti-artifact guard firing
   as designed, not a model that "failed." It also says something real: Phi-3.5
   is markedly more permissive on these borderline prompts than the others.
2. **AUC is the claim; the fixed 0.5 threshold is not.** Ranking is strong
   everywhere (0.78–0.97), but the 0.5 operating point is miscalibrated per
   model (e.g. Llama-1B recall 0.42 at high precision). **As a deployable gate
   this needs per-model threshold calibration** — the signal is there, the
   operating point is not free. The prereg pre-declared the threshold as
   descriptive, not gated.
3. **Refusal axis only.** This validates `comply_refuse` pre-output, cross-arch.
   It does **not** transfer the claim to deception / sycophancy / confab probes —
   each would need its own held-out behavioral validation. Do not overclaim the
   whole atlas from this.
4. **Mechanism is not novel.** Activation/residual probing before generation is
   established research (Apollo, *Detecting Strategic Deception Using Linear
   Probes*, ICML 2025; *A Single Direction of Truth*, arXiv 2507.23221). The
   contribution here is **cross-architecture generalization to held-out OOD
   behavior, pre-registered, with a receipt** — productization and rigor, not a
   new mechanism.
5. **Feasibility-to-validation grade.** Single deterministic pass, n=45, greedy
   decoding, 6 cached open-weight models. Not a 10⁴-sample production
   certification. The lexical refusal label (vendor-robust `detect_refusal`) is
   itself imperfect; an LLM-judge cross-check would harden it.

## Substantive finding worth keeping

**Pre-output predictability rises with model capacity.** The 3B models
(Qwen-3B 0.973, Llama-3B 0.940) outscore their 1–1.5B siblings (Qwen-1.5B
0.832, Llama-1B 0.779). Larger instruct models appear to form a cleaner,
earlier-committed refusal representation that the prefill-residual probe reads
better. Falsifiable, and a lead worth a dedicated capacity-sweep.

## What this earns

A defensible, **productizable** claim: *styxx's native pre-output refusal probe
generalizes to held-out borderline prompts across 5 open-weight architectures,
before generation, median AUC 0.83 — pre-registered, receipt attached.* That is
the concrete proof point under the "agent-side pre-output integrity layer"
thesis, and the foundation for the **pre-output action gate** (predicting unsafe
tool actions before emission) and the **integrity-gated router**.

It does **not** earn "we invented pre-output probing," "the whole atlas
generalizes," or "drop-in deployable gate without calibration." Those are the
next bets, each its own kill-gate.

— scored 2026-06-02 against the frozen gate in PREREG_holdout_gate_2026_06_02.md
