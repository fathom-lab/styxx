# Pre-Registration · The Council — reference-free truth via inter-model agreement

**Committed BEFORE data.** The boldest swing of the arc. We validated *intra-model*
inconsistency (a model invents a different fact each sample → confabulation). The Council
moves the axis to *inter-model*: **independent models should converge on what's real and
diverge on what's fake** — because a real fact is a shared attractor (they all learned
it) while a fabrication has no attractor (each model invents its own). If so, **agreement
across models is a truth signal that needs no reference** — exactly what the synthesis
marked *closed* for text-only methods (reference-less deception ≈ chance).

## Hypothesis

For a question, poll K independent models; cluster their committed answers by meaning.
**High agreement on a substantive (non-abstention) answer ⇒ likely real; divergence (or
convergence only on "doesn't exist") ⇒ likely fake.**

## Design (run once)

- **Council (OpenAI-only, cross-vendor key-blocked):** `gpt-4o-mini`, `gpt-4o`,
  `gpt-3.5-turbo`, `gpt-4.1-mini` (skip any that errors; ≥3 required). Each model's
  "vote" = modal of N=3 samples (denoise), prompt = "Answer in one short sentence with a
  specific answer." (forces a committal vote).
- **Questions:** 6 real-common + 8 real-obscure + 8 fake (reliable labels).
- **Agreement score** per question = (size of largest cluster of *non-abstention* votes)
  / (number of voting models). Clustering by cosine@0.90 (+ LLM-judge cross-check).
  Abstention votes are excluded from the "substantive answer" cluster (a council that
  agrees only by all saying "no such thing" is detecting *fakeness*, not agreeing on a
  fact).

## Kill-gate (PASS iff C1 ∧ C2)

| ID | Bar |
|----|-----|
| **C1 (truth signal exists)** | AUC(agreement → is_real) ≥ **0.75** over all REAL vs FAKE. |
| **C2 (the holy grail — decisive)** | on **REAL-OBSCURE vs FAKE** alone, AUC ≥ **0.70** — agreement distinguishes obscure-but-real from nonexistent *without a reference*. This is the case single-model abstention can't do (a model abstains on both). |

**PASS** → inter-model agreement is a genuine reference-free validity signal → a new
grounding substrate (the council *is* the grounding) → build it + a paper that takes a
real swing at the closed reference-less-truth line. **FAIL shapes (all informative):**
(a) **correlated confabulation** — models share a fake (same training corpora) → false
agreement on fake → C1/C2 fail → an important negative about wisdom-of-crowds for truth;
(b) obscure-real known by too few models → false divergence on real → agreement tracks
*popularity*, not truth; (c) works on real-common but not obscure (C1 pass, C2 fail) →
agreement = "is this widely known," a weaker but real signal.

## Honest prior

Real-common → near-certain convergence; fake → likely divergence (intra-model
inconsistency already shows each model invents freely, so across models even more). The
whole bet rides on **C2**: obscure-real. If the obscure fact is genuinely in multiple
models' training, they converge (truth signal works); if it's known to one and
confabulated by others, agreement collapses and the Council only measures *fame*, not
*truth*. Correlated confabulation is the deepest risk — if it appears, report it loudly;
it's the reason this problem is hard. Do not reinterpret a fame-detector as a
truth-detector.
