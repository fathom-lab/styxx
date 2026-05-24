# H1d (cross-instrument, hallucination) — FAIL

One-shot on the holdout hashed before scoring (sha `529073743c1bde97dbaf6db34a298864abd93fe8265328a76bff886d4bded035`).

- instrument = styxx hallucination check(use_nli=True); n=450; gold via gpt-4o judge (val 0.90).
- pooled ρ(validity_lp, −error) = **-0.1838**, p = 1.0000.
- within hallucinated (n=217): ρ=-0.0269; within correct (n=233): ρ=+0.4755.
- bar: pooled ≥ 0.40, p<0.01, min within-class ≥ 0.20.

**VERDICT: FAIL** — INSTRUMENT-SPECIFIC — logprob-validity was a refusal property; it does not transfer to the hallucination instrument. The map gains a sharp boundary.
