# Finding · The Dark Matter — a candle in the cave (PASS, but marginal, partial, tiny-n)

**2026-05-25.** Prereg `preregistration_darkmatter_2026_05_25.md`. The hardest swing of
the arc: reference-free detection of **consensus hallucination** (shared misconceptions)
via **perturbation-fragility**. **Verdict: PASS on the pre-registered bars — and
deliberately not oversold.** A genuine first signal *and* a confirmed floor, in one
result.

## Result (cross-vendor: gpt-4o-mini + Qwen2.5-3B + gemma-2-2b-it)

| metric | value | bar | verdict |
|---|---|---|---|
| consensus-misconception items (filtered) | **10** | — | tiny n |
| consensus-truth controls | 14 | — | — |
| flip-rate, misconceptions (neutral challenge) | **0.40** | — | catches 40% |
| flip-rate, truths | **0.00** | — | perfectly robust |
| **D1** AUC(fragility → misconception) | **0.70** | ≥0.70 | **PASS (exactly at bar)** |
| **D2** truth robust (<0.25) | 0.00 | <0.25 | PASS |
| **D2** misconception flips that self-corrected | **0.75** | ≥0.50 | PASS (3 of 4) |
| **PASS = D1 ∧ D2** | | | **TRUE** |

## What is real (and genuinely a first)

- **Truths do not move.** 0/14 flipped under a neutral "reconsider carefully" challenge.
  Real knowledge is robust to reflection. (Believable even at this n — clean signal.)
- **Fragile misconceptions self-correct.** When a shared misconception flips, it flips
  *toward the truth* (3/4). So fragility is **correction, not sycophantic churn** — a
  **zero-false-positive, reference-free** signal that catches *some* consensus
  hallucination. No one has shown that before. A crack of light in the dark matter.

## What is NOT solved (and I will not let it read as solved)

- **Low recall: 40%.** It caught 4 of 10 shared misconceptions and **missed 6.** The
  misses are the ones that *held* under reflection — the **stubborn** misconceptions,
  which are precisely the **most dangerous** (a model that won't reconsider a false belief
  even when prompted). This is exactly the failure mode the prereg's honest prior named:
  *"fragility catches the weak ones and misses the dangerous ones — a partial result at
  best."* It did.
- **Tiny n.** 10 misconception items; AUC sits *exactly* on the 0.70 bar; the
  self-correction rate is 3-of-4. A couple of items either way moves every number. This is
  a **suggestive first signal, not a validated result** — it demands replication at scale
  before any stronger claim.
- **Consensus misconception is itself rarer than feared (good news):** only 10 of 45
  TruthfulQA candidates produced a high-agreement *shared* wrong answer — mostly the
  vendors scattered, disagreed, or got it right. The dark matter is real but not
  ubiquitous.

## The honest verdict: a first AND a floor

Perturbation-fragility **does** reference-free-detect the *fragile* consensus
hallucinations (novel, zero false alarms on truth) — **and it confirms the dark core
stays dark**: stubborn shared misconceptions resist reflection and remain undetectable by
this method. We lit a candle in the cave and proved most of the cave is still black. Both
are real contributions: a method that catches the soft consensus errors, and a *mapped
boundary* on the hard ones.

## Scope & next

n=10/14, single run, neutral-challenge only, 2–3B local models in the council. The pattern
(truth-robust / fragile-misconception-corrects / stubborn-misconception-resists) is
qualitatively believable; the quantitative bars are tiny-n and must not be cited as
validated. Real next step: scale (n in the hundreds), stronger perturbations (adversarial
vs neutral — though adversarial reintroduces the sycophancy confound), bigger/cross-vendor
councils, and a calibrated recall/precision curve. Until then: **a pre-registered first
signal that the dark matter is *partly* visible — and a proof that its dangerous core is
not, by this method.**
