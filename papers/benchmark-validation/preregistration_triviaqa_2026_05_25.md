# Pre-Registration · Validate semantic_entropy on TriviaQA (the credible test)

**Committed BEFORE any model output.** 7.7.0 shipped `styxx.semantic_entropy` on
feasibility-grade evidence (16 homemade fictional entities). This is the move that makes
it real or honestly kills it: a **public benchmark, real n, head-to-head with the
literature and with the closed single-response signal.** Run once.

## Setup

- **Benchmark:** `trivia_qa`, config `rc.nocontext`, split `validation` (real factual QA
  with gold `answer.normalized_aliases`).
- **Hashed holdout (selection fixed before data):** take the **150** questions with the
  smallest `sha256(question_id)` hex. Deterministic, content-blind, fixed by this prereg.
- **Model under test:** `gpt-4o-mini`, **N=5** samples/question, temp 1.0, `logprobs=True`.
- **Gold:** the modal answer is **correct** iff its normalized text contains any
  `normalized_alias` (TriviaQA-standard contains-match); else **incorrect**.
- **Signals per question:**
  - `se_judge` = `styxx.semantic_entropy(samples, same_fn=<gpt-4o-mini "same answer?" judge>)`
    — the SHIPPED primitive with the recommended (cleanest) clustering.
  - `se_cosine` = `styxx.semantic_entropy(samples, method="cosine")` — the shipped cheap
    default (we expect it weaker; report the gap honestly).
  - `neg_logprob` = − mean per-sample mean-token-logprob — the single-response confidence
    baseline the grounded-arc CLOSED on hallucination (within-hallucinated ρ≈0).

## Kill-gate (PASS iff B1 ∧ B2)

| ID | Bar |
|----|-----|
| **B1 (matches the field)** | AUC(`se_judge` → incorrect) ≥ **0.75** on the 150-question holdout (Farquhar et al. report ~0.75–0.79 AUROC on tuned QA; we must land in that band on a standard benchmark). |
| **B2 (beats the closed single-response signal)** | AUC(`se_judge`) ≥ AUC(`neg_logprob`) **+ 0.05** — across-sample divergence must beat mean-logprob, the signal that dies on hallucination. |

**Reported regardless:** `se_cosine` AUC (the shipped default's real-benchmark number +
its gap to the judge), and the base rate (fraction the model gets wrong).

**PASS** → `semantic_entropy` is no longer feasibility-grade: a validated, sampling-based
hallucination detector that matches the literature and beats logprob on a public
benchmark. Upgrade the docstring scope from "feasibility" to "validated on TriviaQA
(n=150)", and *that* is the defensible game-changer claim. **FAIL shapes:** B1 miss →
honest "underperforms the field on TriviaQA, X" (the homemade-entity AUC didn't
generalize); B2 miss → divergence ties/loses to logprob here (the across-sample edge is
distribution-specific). Either is recorded, not spun.

## Honest prior

TriviaQA is a good fit for the mechanism (knowledge-boundary confabulation → divergent
when wrong), so I expect `se_judge` ~0.75–0.85 and a clear win over logprob. Real risks:
(1) entity-alias answers ("David Seville" vs "Ross Bagdasarian" for the same person) can
make the judge/cosine over-split correct answers, depressing AUC; (2) consistent
misconception errors (the model confidently+consistently wrong) are the irreducible
Tier-3 floor and will be misses for any divergence signal — TriviaQA has fewer of these
than TruthfulQA, which is why TriviaQA (not TruthfulQA) is the fair first test. n=150,
single model, single run — a real validation, not yet multi-model/multi-seed.
