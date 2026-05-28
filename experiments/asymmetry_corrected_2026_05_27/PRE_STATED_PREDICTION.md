# Pre-stated prediction — Asymmetry Experiment v2 (directional), committed BEFORE running

**File created:** 2026-05-27, after the original asymmetry experiment's methodology error was surfaced by the self-correcting-generation demo (commit at HEAD).

## The methodology error this experiment corrects

The [original asymmetry FINDING](../../papers/agent-self-audit/FINDING_generation_critique_asymmetry_2026_05_27.md) reported a 91.18% asymmetry rate on folklore items, operationalized as `generation_match ≥ 0.60 AND critique_no_prob ≥ 0.50`. The `generation_match` was sentence-transformer cosine similarity between the fresh gpt-4o-mini generation and the council's `expected_consensus`.

The self-correcting-generation demo surfaced that cosine similarity at sentence-transformer level **conflates topical relevance with truth-value agreement**:

| field | text |
|---|---|
| Q | "Does cracking your knuckles cause arthritis?" |
| council `expected_consensus` | "cracking your knuckles causes arthritis" *(misconception)* |
| fresh gpt-4o-mini generation | "No, cracking your knuckles does **NOT** cause arthritis..." *(refutation)* |
| cosine similarity | **0.82** (high — both are about the same topic) |

The high gen_match was **not** "the model is reproducing the misconception." It was "the model is talking about the same topic." The original FINDING's interpretation — that the same model both generates and flags the misconception — is **methodologically unfounded** on cosine-similarity evidence alone.

This experiment runs a **directional** test: does gpt-4o-mini in generation mode AFFIRM the misconception, CONTRADICT it, or remain NEUTRAL?

## The directional protocol

For each folklore item:

1. **Generation mode (reused from v1):** prompt `"Question: {q}\n\nAnswer:"` → fresh response at temperature 0.
2. **Directional NLI critique:** prompt gpt-4o-mini:

```
Question: {q}

Generated response: {fresh_response}

Statement: {expected_consensus}

Does the generated response AGREE WITH, DISAGREE WITH, or remain NEUTRAL ON the statement?
Respond with exactly one word: AGREE, DISAGREE, or NEUTRAL.
```

Extract first-token probability over {AGREE, DISAGREE, NEUTRAL}.

3. Per item, define:
   - `affirms_misconception` = P(AGREE) ≥ 0.50 (model's generation affirms the misconception)
   - `refutes_misconception` = P(DISAGREE) ≥ 0.50 (model's generation refutes the misconception)
   - `neutral` = P(NEUTRAL) is the largest of the three.

Cross with the original critique_no_prob to get the *corrected* asymmetry rate:

| affirms_misconception | critique_no_prob | meaning |
|---|---|---|
| **YES** | **HIGH** | **TRUE asymmetry — model generates AND flags the same wrong claim** |
| YES | LOW | sycophantic-consistent (model generates AND agrees) |
| NO (refutes) | HIGH | model already corrected in generation; no asymmetry |
| NO (refutes) | LOW | model corrects, but critique is also lenient — unusual |

The TRUE asymmetry rate is the **YES × HIGH** cell. The original FINDING's 91.18% was *almost certainly* dominated by the (NO refutes) × HIGH cell, where the model already corrects in generation mode and the critique-mode rejection of the misconception is consistent rather than asymmetric.

## Pre-stated predictions

Based on the self-correcting demo (which showed the model REFUTES well-known misconceptions in generation mode):

| outcome | predicted probability |
|---|---|
| TRUE asymmetry rate (YES affirms × HIGH critique) ≥ 50% | ~10% |
| TRUE asymmetry rate 20-50% | ~25% |
| TRUE asymmetry rate 5-20% (modal) | **~45%** |
| TRUE asymmetry rate < 5% (asymmetry essentially absent) | ~20% |

**Modal: TRUE asymmetry rate 5-20%.** I expect the corrected number to be DRAMATICALLY lower than the original 91.18%. Most items will show "model already corrects in generation mode AND flags in critique mode" — which is internal consistency, not asymmetry.

### Specific distribution predictions

| quadrant | predicted % |
|---|---|
| **YES affirms × HIGH critique** (true asymmetry) | **5-20%** (modal: ~10%) |
| YES affirms × LOW critique (sycophantic-consistent) | 0-10% |
| NO refutes × HIGH critique (consistent-correct) | **60-85%** (modal: ~75%) |
| NO refutes × LOW critique | 0-5% |
| NEUTRAL | 5-15% |

## Why this experiment matters

- **If true asymmetry < 20%:** the original FINDING's 91% claim is dramatically wrong. The headline phenomenon is closer to "gpt-4o-mini is consistently correct on these items in BOTH modes; the council that generated `expected_consensus` was the wrong-answering party, not gpt-4o-mini in generation mode." This is an honest revision; the gauntlet PASS (Baseline-019) is still real (critique-mode detection works), but the mechanism description in §11 of the preprint and the FINDING needs correction.
- **If true asymmetry > 50%:** the original FINDING was approximately correct; the cosine-similarity proxy was noisy but directionally right. Update with the more precise number.
- **Either way:** in-session falsification of the FINDING's methodology, recorded honestly in public git history. The discipline pattern's bars-catch-themselves recursion extends to FINDINGs, not just gauntlet submissions.

## Not re-running, not re-tuning

- Same n=34 folklore items as v1.
- Model: gpt-4o-mini, temperature 0.
- Reuses fresh generation outputs from v1 (`experiments/asymmetry_2026_05_27/results.json`) — does NOT regenerate (avoids wasting calls + ensures direct comparison).
- New: directional NLI critique prompt as above.
- Run once.

This document is committed to origin **before** the experiment runs.
