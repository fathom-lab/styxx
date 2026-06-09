# RESULT — a consistency detector is structurally immune to the representation attack that defeats a probe

> **TEMPERED 2026-06-04 (proper-research correction).** A stronger attack disciplined this result's
> framing. A **learned minimal per-item perturbation** does **not** collapse the probe here (AUROC
> ~unchanged at a large cost) — on clean MMLU the honest/deceptive separation is wide, so the probe
> has a **big margin and is robust to cheap per-item attacks**. Only a *maximal* attack (full steering
> to the honest centroid) collapses it, which is unrealistic. So **this experiment does NOT show the
> probe is cheaply fragile** — that half genuinely rests on the literature's *distribution-level*
> RL-obfuscation, which I have not reproduced. And grounded_honesty's invariance, while real, is
> **near-tautological** (it reads no residuals, so no residual attack can touch it). **Net: the
> thesis is plausible and structurally argued, but NOT proven by my experiments.** The decisive tests
> — real RL-obfuscation, or a *direct* attack on grounded_honesty (adversarial-question framing) — are
> not yet run. The "MECHANISM SUPPORTED" language below over-reached; read it through this correction.

**Date:** 2026-06-04 · **Verdict (as first written, now tempered above): MECHANISM SUPPORTED.** Frozen:
`PREREG_obfuscation_robustness_2026_06_04.md`. Qwen2.5-3B-Instruct, 160 in-knowledge MMLU items
(80 held-out test), mid-layer (L18) static activation probe vs styxx `grounded_honesty`
(stateless sampling-divergence).

## Numbers

| attack | static probe AUROC | grounded_honesty AUROC |
|---|---:|---:|
| clean (HONEST vs DECEPTIVE) | 1.000 | 0.937 |
| behavioral obfuscation ("look honest") | 1.000 | 0.949 |
| steering α=0.25 / 0.5 / 0.75 | 1.000 / 1.000 / 1.000 | 0.937 (uses no residuals) |
| **steering α=1.0 (full shift to honest centroid)** | **0.525** | **0.937** |

Stateless belief accuracy on obfuscated items: **1.00** (no deep-deception escape).

## What is demonstrated (the real finding)
**`grounded_honesty` is invariant to the representation attack** — 0.937 at every level, because it
scores the stated claim against *stateless resamples of the model's belief*, never touching the
generation-time residual the attack perturbs. The same steering that drives the static probe to
chance (0.525) is, by construction, **invisible** to it. This is the first **head-to-head**
demonstration that a consistency detector survives the exact manipulation that collapses a signature
probe — grounding the strategic claim that styxx's consistency-based primitives are the
obfuscation-robust ones.

## Honest scope (what is NOT demonstrated)
- **My linear steering is a crude attack.** The probe held at 1.000 until α=1.0 (full shift to the
  honest centroid). So this experiment does **not** by itself show probes are *easily* fragile — the
  probe-collapse half of the thesis rests on the literature's *learned* attacks (RL-obfuscation
  collapses probes to chance, arXiv 2505.27958), which are far more efficient than a centroid shift.
  What is clean here is the **invariance of grounded_honesty**, which holds against *any* residual
  attack regardless of its strength.
- **Behavioral prompt-attack too weak.** "Answer as if it were simply true" did not move the probe
  (drop 0.00) — a separate finding: surface 'look honest' prompting does not shift the deception
  representation. The representation must be attacked directly (steering / RL / GCG).
- One model, MCQ factual deception, in-knowledge items. `grounded_honesty` is single-model
  self-consistency (a confidently-wrong belief → wrong verdict; hence in-knowledge).
- **Different threat model, not universal immunity.** grounded_honesty has its OWN attack surface —
  in-session context injection (divergence.py SECURITY MODEL). The exact claim: it is immune to the
  **representation-obfuscation** surface that collapses activation probes — complementary, not total.

## Why it matters
Static probes are a fixed target an adversary routes around; a consistency check is not a target at
all — to beat it the model must make the lie its own stable, resampled belief (stateless accuracy
here stayed 1.0, so it didn't). For the RSI future — where AI builds AI and a monitor must survive
an adversary that can optimize against it — the only durable signal is the one with a *structural*
reason it can't be cheaply gamed. This is direct evidence that styxx already has one.

## Next (to harden, in order)
1. **Real learned attack:** RL-obfuscation / GCG against the probe (collapses it efficiently),
   confirm grounded_honesty's invariance holds — replaces the crude steering stand-in.
2. **Multi-model / cross-vendor** (the council arm) for the residual confabulations a single model
   can't catch.
3. **Deep-deception stress:** force a coherent consistent false belief and test whether the
   cost-escalation argument (consistency converts strategic lies into expensive stable beliefs)
   actually bites.
