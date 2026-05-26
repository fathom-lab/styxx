# Finding · TriviaQA validation — semantic_entropy works (B1), does NOT beat logprob (B2), shipped overclaim corrected

**2026-05-25.** Prereg `preregistration_triviaqa_2026_05_25.md`. The credible test of the
7.7.0 `semantic_entropy` primitive on a public benchmark. **Verdict: PASS = FALSE** — B1
passes, B2 fails — and B2's failure corrects an overclaim already shipped to PyPI.

## Result (TriviaQA rc.nocontext, 150-question hashed holdout, gpt-4o-mini N=5)

| signal | AUC(→ incorrect) | note |
|---|---|---|
| **semantic_entropy (judge clustering)** | **0.785** | B1 PASS (≥0.75); matches the literature (~0.75–0.79) |
| semantic_entropy (cosine default) | 0.727 | the shipped cheap backend — weaker, as documented |
| **neg mean-logprob** | **0.817** | the single-response baseline — **beats semantic_entropy** |

- Base error rate 12.7% (19/150 wrong — TriviaQA rc.nocontext is easy for gpt-4o-mini).
- Clean separation for semantic_entropy: mean **0.555** (incorrect) vs **0.057** (correct).
- **B1 (judge AUC ≥ 0.75): PASS** (0.785).
- **B2 (judge beats logprob by ≥ 0.05): FAIL** (−0.032).
- **PASS = B1 ∧ B2 = FALSE.**

## What's earned (B1)

`semantic_entropy` **generalized off the homemade fictional-entity set to a standard
benchmark**: AUC 0.785, in the published band, clean separation. As a *detector of model
error from sampling alone*, it is validated — no longer feasibility-grade on this point.

## What's corrected (B2) — a shipped overclaim

7.7.0's docstring and CHANGELOG say semantic_entropy "catches what single-response
confidence (logprob) **provably misses**." TriviaQA refutes that as stated: **logprob
(0.817) beat semantic_entropy (0.785)** at separating correct from incorrect. The
"provably misses" line over-generalized a *narrow* grounded-arc result (logprob's
*within-hallucinated reliability ranking* is ρ≈0) into an *across-item* claim it doesn't
support. On a real benchmark, where logprobs are available, logprob is the better — and
far cheaper (1 call vs N) — signal. I shipped that overclaim to PyPI; it is corrected in
7.7.1.

## The honest, stronger repositioning

`semantic_entropy` is a **validated, sampling-based hallucination signal whose niche is
settings WITHOUT logprob access** — e.g. the Anthropic Messages API (no `logprobs=True`),
or any provider/gateway that doesn't expose token logprobs. There it delivers
field-matching AUC (~0.79) using only resampling. It is **not** a replacement for, and
does not beat, logprob where logprobs exist. That is a real, defensible niche — and a
more honest claim than the one 7.7.0 shipped.

Secondary: the cosine backend (0.727) again trailed the judge backend (0.785),
reconfirming the clustering finding — use a judge (`same_fn`) for the real signal; cosine
is the lossy convenience default.

## Honest scope

n=150, one model (gpt-4o-mini), one benchmark, single run; logprob baseline is mean
per-sample mean-token-logprob. A multi-model / multi-benchmark run is the next step before
any stronger claim. TriviaQA (knowledge-boundary) is favorable to the mechanism;
adversarial-misconception sets (TruthfulQA) would stress the consistent-confabulation
floor where neither divergence nor (likely) logprob helps.

## Net

The benchmark did its job: it **validated the detector (B1)** and **caught a shipped
overclaim (B2)** within hours of release. Seventh self-correction of the arc — the only
one that reached into a published wheel. 7.7.1 ships the corrected scope: validated on
TriviaQA, logprob-free niche, no logprob-beating claim.
