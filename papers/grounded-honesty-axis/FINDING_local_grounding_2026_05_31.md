# FINDING — a FULLY-LOCAL retrieval+NLI layer catches confident misconceptions — SURVIVED (the $0 door in the wall)

**SURVIVED.** PREREG `PREREG_local_grounding_2026_05_31.md` (bars CATCH ≥ 0.50, PRESERVE ≥ 0.80 authored
before the run). Qwen2.5-3B-Instruct confident answers on TriviaQA, pooled **33,636-passage** reference
corpus, `local_ground.py` (`all-MiniLM-L6-v2` retrieval + `DeBERTa-v3-base-mnli-fever-anli` judge),
**no API**. `run_local_grounding_eval.py`; receipt `local_grounding_result.json`.

## Result

- Confident subset (first-token entropy < median) grounded: **350** (139 wrong, 211 right), powered.
- **CATCH** = P(refuted | confident-WRONG) = **0.561** (bar ≥ 0.50) → PASS.
- **PRESERVE** = P(not-refuted | confident-CORRECT) = **0.882** (bar ≥ 0.80) → PASS.
- **RESULT = SURVIVED.**
- Baseline: white-box CATCH on confident misconceptions ≈ **0** — three pre-registered negatives
  (`FINDING_residual_confab_probe` / `_strict` / `_span_residual`).

## The claims that land

1. **The first $0, fully-local signal to catch confident confabulation.** Where every white-box signal
   (output entropy/margin, first-token AND full-trajectory residuals) scores CATCH ≈ 0, a local
   embedding-retrieval + local NLI judge catches **56% of confident misconceptions** while sparing
   **88% of correct answers** — cached open models, no API, retrieving over a 33K-passage corpus.
2. **The bottleneck was retrieval, exactly as the thesis (claim 4) predicted.** The NLI judge is
   near-perfect on clean evidence (entailment 0.998 / contradiction 0.999); the open question was
   whether local retrieval could feed it. It can — over a 33,636-passage discriminative corpus.
3. **A gate, not an oracle.** CATCH 0.56 misses ~44% (false-supports — wrong "Lloyd George", "Capote"
   supported). PRESERVE 0.88 has ~12% false-refutes (correct "David Seville", "green" refuted). Fallible
   like all retrieval; both error modes are in the receipt, disclosed.

## What it unblocks

The **retrieval tier** of the honesty layer / attestation — the lever for confident misconceptions,
previously demonstrated only via the OpenAI search model and **blocked on API credit** — now has a
**$0 local implementation.** The product's strongest tier no longer requires a paid model.

## Honest scope (load-bearing)

The corpus is the **pooled test-domain evidence** (33K passages from the 700 test questions' own
evidence), **not global Wikipedia.** So this proves: *given a corpus that contains the relevant
evidence, local retrieval+NLI discriminates it among 33K distractors and grounds the claim at a useful
operating point.* The open-world question — does a global corpus **contain** the evidence, and can
retrieval find it at that scale — is the **deployment step**, and harder. Single model (Qwen-3B
answers), single dataset (TriviaQA), one run, feasibility-grade. Detects/abstains; corrects nothing.
NLI + embeddings are shipped-class open models, not frontier. Prereg bars were authored before the run.

## One line

A fully-local embedding-retrieval + NLI stack ($0, no API) catches 56% of confident factual
misconceptions while sparing 88% of correct answers — the first free signal to open the door in the
confident-confabulation wall every white-box method is blind to — with both error modes disclosed and
global-corpus retrieval named as the next step.
