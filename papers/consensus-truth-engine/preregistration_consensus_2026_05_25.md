# Pre-Registration · The Truth Engine — cross-vendor consensus as a reference-free answer layer

**Committed BEFORE data.** The invention: turn the validated cross-vendor *detector* into
a *generator*. Given a question and K models from different vendors, `consensus()` fans
out, clusters the answers, returns the **cross-vendor-convergent** answer, and **abstains**
when no convergent cluster forms. Claim: this is **more accurate than any single member**
on factual QA *and* its abstention is **calibrated** — all reference-free. Nobody ships
cross-*vendor* consensus with reference-free calibrated abstention (self-consistency is
single-model; we showed cross-vendor is *more* robust to correlated confabulation).

## Mechanism

For question q, models M (≥2 vendors): collect one answer per model → cluster by judge
equivalence → **consensus = the answer of the largest cluster**; **agreement = cluster
size / |M|**; **abstain iff agreement < τ** (no cross-vendor majority). Answered set =
non-abstained questions.

## Design (run once)

- TriviaQA `rc.nocontext`, the **same 150-question hashed holdout** as
  `preregistration_triviaqa_2026_05_25.md` (no new selection).
- Models (3 vendors): `gpt-4o-mini` (OpenAI), `Qwen2.5-3B-Instruct` (Alibaba, local),
  `gemma-2-2b-it` (Google, local). One answer each (greedy/low-temp).
- Gold: TriviaQA `normalized_aliases` contains-match.
- τ = 0.66 (≥2 of 3 vendors agree). Compare:
  - per-model accuracy (each of the 3 alone),
  - **consensus accuracy on the answered set** (consensus answer vs gold),
  - **abstention calibration**: accuracy on abstained vs answered questions.

## Kill-gate (PASS iff T1 ∧ T2)

| ID | Bar |
|----|-----|
| **T1 (consensus beats the best member)** | accuracy of the consensus answer on the answered set ≥ **best single-model accuracy + 0.05**. The council answers *better* than its best voice. |
| **T2 (abstention is calibrated)** | on the questions it **abstains**, the best-single-model accuracy is **≥ 0.20 lower** than on the answered set — i.e. it abstains disproportionately on the hard/wrong ones, not at random. |

**Reported:** coverage (answered fraction), per-model accuracies, consensus accuracy,
abstained-vs-answered accuracy gap.

**PASS** → a genuinely new primitive: a reference-free, cross-vendor answer layer that is
more reliable than any member and abstains where the field is uncertain. Ship
`styxx.consensus()`. **FAIL shapes:** T1 miss → consensus ≈ best member (no lift; it's
just an ensemble vote, not novel value) — honest; T2 miss → abstention is uncalibrated
(refuses at random) → not useful. Either is recorded straight.

## Honest prior

TriviaQA base accuracy is high for gpt-4o-mini (~0.87) and lower for the 2–3B local
models, so the "best member" bar is gpt-4o-mini and it's strong — T1 (+0.05 over ~0.87)
is a HARD bar (little headroom near the ceiling). Likely outcome: consensus ≈ gpt-4o-mini
on the easy TriviaQA set (T1 marginal), but **abstention well-calibrated** (T2 likely
passes — fracture concentrates on the items the weak models get wrong, which overlap the
hard ones). If T1 misses on easy TriviaQA, the honest read is "the lift shows up on
*harder* / long-tail sets where single models fail more (PopQA, the obscure tier), not on
near-ceiling TriviaQA" — and that's the next, fairer test. Do not claim a lift that isn't
there at this n; report coverage/calibration honestly even if T1 is marginal.
