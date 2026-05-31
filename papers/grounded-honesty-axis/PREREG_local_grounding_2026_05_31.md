# PREREG — a FULLY-LOCAL retrieval+NLI layer catches confident factual misconceptions (the $0 retrieval lever)

**REGISTERED 2026-05-31, before the eval is run.** Test set hashed pre-run.
**SIGN-OFF:** Flobi — *"push the tech farther than anyone thought possible / defy all odds"* (2026-05-31).

## Why this is the real frontier (not a re-tread)

White-box detection of **confident** confabulation is **closed**: three pre-registered negatives
(first-token VOID, strict VOID, trajectory REPORT_AS_LANDED). The output is the white-box ceiling; the
representation adds nothing. The arc's standing conclusion: the only lever for confident misconceptions
is **external evidence (retrieval)** — and every prior demonstration of it used an **API**
(`FINDING_retrieval_grounding`, gpt-4o-mini-search). The thesis (claim 4) names the next build:
*a free, local evidence-retrieval layer good enough to feed the NLI judge that already works.* Feasibility
just confirmed the judge is near-perfect on clean evidence (entailment 0.998 / contradiction 0.999), so
the **only open question is whether local retrieval can feed it.** That is genuinely unbuilt and high-ceiling.

## Design (no API; all local, all $0)

- **Test set:** Qwen2.5-3B-Instruct's **confident** answers on TriviaQA (bottom-entropy subset) —
  **confident-wrong** (the misconceptions) + **confident-correct** controls. The regime where every
  white-box signal scores **CATCH ≈ 0**. Hashed pre-run.
- **Corpus:** a local reference corpus (Wikipedia intro passages), embedded with `all-MiniLM-L6-v2`.
- **Pipeline (`local_ground.py`):** claim = "`<question> <model answer>`" → retrieve top-k passages by
  cosine → local NLI judge (`DeBERTa-v3-base-mnli-fever-anli`) per passage → **refuted** (strong
  contradiction) / **supported** (entailment) / **unclear**. No external calls.

## Metric & bars (FIXED)

- **CATCH** = P(refuted | confident-WRONG) — confident misconceptions caught.
- **PRESERVE** = P(not refuted | confident-CORRECT) — correct answers spared.
- Baseline: white-box CATCH on this subset ≈ **0** (the wall).

| Bar | Statement | Threshold |
|---|---|---|
| **CATCH** *(key)* | local grounding catches what white-box can't | P(refuted \| confident-wrong) **≥ 0.50** |
| **PRESERVE** *(key)* | it doesn't over-refute the truth | P(not refuted \| confident-correct) **≥ 0.80** |

**RESULT = SURVIVED iff CATCH ∧ PRESERVE.** Powered: ≥ 20 confident-wrong + 20 confident-right.
A SURVIVED is the **first $0 local signal to catch confident confabulation** — the wall's free door.

## Honest scope

Local retrieval may simply be too weak (corpus coverage, retriever quality) to feed the judge — that is
the real risk, and a FAILURE means *retrieval isn't good enough yet*, not that grounding is impossible
(the judge is validated). Retrieval is fallible (can mislead). Single model, single dataset,
feasibility-grade, one corpus snapshot. Detects/abstains; corrects nothing. NLI + embeddings are the
shipped-class open models, not frontier.

## One line

Build the retrieval lever **local and free** — does an embedding+NLI stack with no API catch the
confident misconceptions every white-box signal is blind to, the one thing the field assumes you need a
paid model for.
