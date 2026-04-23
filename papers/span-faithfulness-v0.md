# Span-Faithfulness v0: Remediation Plan for DROP and FinanceBench

**Research proposal — 2026-04-23. For styxx v4.1.**

**Status update (2026-04-23, same day): the cheap-heuristic version
of both proposed signals is NULL on HaluBench-DROP (AUC 0.500 and
0.520, near chance). Full writeup in §8. Section 2–6 retained for the
approach framing; §7–8 carry the null result and the pivot.**

## What this document is

v4.0.0 of `styxx.guardrail` achieves AUC 0.998 on HaluEval-QA and AUC
0.807 on HaluBench-RAGTruth but falls below chance on HaluBench-DROP
(AUC 0.424) and HaluBench-FinanceBench (AUC 0.492). These two
failure modes are published in the v4.0.0 CALIBRATION_NOTES with
their mechanisms explicitly characterized. This document proposes
the v4.1 remediation and provides a minimal reproducible probe so
the approach can be evaluated before a full integration.

## Why the current signal stack fails

### DROP — extractive-span arithmetic errors

Looking at 20 DROP FAIL rows sampled at seed 31:

- **13/20** are numeric "how many" questions answered with the wrong
  number ("How many yards was the longest field goal?" → answered
  42 when the reference says 38).
- **4/20** are selection questions answered with the wrong choice
  ("Which team won: Patriots or Packers?" → answered "Packers" when
  the reference documents the Patriots winning).
- **3/20** are multi-part questions with incomplete or
  over-aggregated answers.

Every signal in the v4.0.0 stack fails on these by construction:

- `content_novelty`, `*_novelty` — the wrong number (e.g., "42") is
  frequently a token *present in the passage* but attached to a
  different entity, so novelty is near-zero.
- `nli_contradict` — "the longest field goal was 42 yards" is NOT
  contradicted by a passage that contains "a 38-yard field goal" in
  one sentence and "a 42-yard field goal" elsewhere. Whole-response
  NLI treats the claim as consistent with the passage.
- `knowledge_grounding` — the response tokens overlap heavily with
  the reference. Coverage is high on both truth and hallucination.

### FinanceBench — arithmetic on verbatim source numbers

Same mechanism, even more pronounced. The hallucinated answer
"operating cash flow ratio 0.25" contains a single token "0.25" that
*may or may not* appear in the source, but the question and
relevant words ("operating cash flow ratio") are directly in the
source. No signal in the current stack detects that the
*arithmetic* producing 0.25 is wrong.

## Proposed v4.1 signals

### Signal 10: `role_mismatch`

The cheap pass. Extract the semantic-role slot demanded by the
question's wh-word + first argument, extract the semantic role of
the response, emit a mismatch signal if they differ. Picks up the
minority of DROP failures that are type-level mismatches (e.g.,
"Which team won?" answered with a player name).

Heuristic mapping:

    how many | how much    → NUMBER
    who                    → PERSON
    which team/group/band  → ORG
    when | what year/date  → DATE | TIME
    where                  → LOC
    how long | how far     → NUMBER + unit

Implementation: regex on the first 6 tokens of the question plus a
lightweight POS/NER pass (`spacy.load("en_core_web_sm")` or a small
CRF) on the response. Sub-millisecond per call after model warmup.

Expected DROP lift: +5 to +10 AUC points. Does not address
FinanceBench.

### Signal 11: `answer_context_adjacency`

The expensive pass, and the one that actually addresses arithmetic
errors. For a (Q, A, Ref) triple:

1. Extract the answer's anchor token — the primary numeric or
   entity token.
2. If the anchor is not present in the reference, skip to the NLI
   signal (already covered).
3. For each occurrence of the anchor in the reference, compute a
   distance metric to the nearest occurrence of the question's key
   content words (noun phrases, verbs, modifiers excluding the
   wh-word).
4. Return `min(distances) / max_distance`, bounded to [0, 1]. A
   correctly-contextualized answer will be adjacent to the question's
   key phrase in the reference; a hallucinated-but-in-passage answer
   will be far away.

Worked example:

    Q:   How many yards was the longest field goal?
    Key: longest field goal, yards
    A:   42
    Ref: "...a 38-yard field goal broke the record as longest ...
          later in the game, a 42-yard punt was dropped..."

    anchor = "42"
    occurrences of "42" in ref: [position 78]
    occurrences of key phrase "longest field goal" in ref: [position 12]
    adjacency distance: 66 tokens → high mismatch signal

Cost: O(|ref|) string search per signal call. ~0.5 ms on 5KB ref.

Expected DROP lift: +15 to +25 AUC points (this is the dominant
DROP failure mechanism).

### Signal 12: `symbolic_number_check`

The expensive and specialized pass for FinanceBench. Detect when
the question implies an arithmetic operation (ratio, sum,
difference, percentage) and try to recompute it from numbers
visible in the reference.

1. Detect arithmetic keywords in Q: "ratio", "difference", "sum",
   "percent", "more than", "times".
2. Extract ALL numeric tokens from the reference with their
   immediate left-context noun phrase.
3. Try to match the question's two operand descriptions
   ("operating cash flow" / "current liabilities") to
   reference-grounded numbers.
4. Compute the expected result via the arithmetic keyword.
5. If the answer number differs from the computed result by more
   than 1% → emit high signal.

Cost: 5–30 ms per call (dominated by NP chunking). Gates on
arithmetic-keyword presence so most non-finance traffic pays zero.

Expected FinanceBench lift: +10 to +20 AUC points, modest gain on
DROP.

## Minimal viable v4.1 path

Priority order, each independently shippable:

1. **role_mismatch** (half-day engineering, spacy dep)
2. **answer_context_adjacency** (1-day engineering, no deps)
3. **symbolic_number_check** (2-day engineering, arithmetic parser)

We can ship 1 + 2 as v4.1.0rc1 within a day. Signal 3 is the bigger
research lift and gates on whether FinanceBench shows meaningful
movement from 1 + 2 alone.

## Success criteria

A v4.1.0 ship is warranted when:

- HaluBench-DROP AUC ≥ 0.60 (from 0.424), or
- HaluBench-FinanceBench AUC ≥ 0.60 (from 0.492)

Either threshold turns a "below chance" cell in the v4.0.0 table
into "real signal" — the claim becomes **6/8 above chance, 6/8
above AUC 0.60** rather than 5/8 and two declared failures. That
is a headline-worthy delta.

If neither signal moves the number, we publish the null result with
the same honesty as the v4.0.0 failure modes. Disconfirmation is
reporting too.

## Reproducer stub

`styxx/guardrail/role_mismatch.py` (to be shipped in v4.1):

```python
def role_mismatch_signal(question: str, response: str) -> float:
    """Return [0, 1] mismatch between question's expected role and
    response's primary entity type. 0 = match, 1 = mismatch."""
    expected = _question_role(question)
    actual = _response_role(response)
    if not expected or not actual:
        return 0.0
    return 0.0 if actual == expected else 1.0
```

`styxx/guardrail/answer_context.py` (to be shipped in v4.1):

```python
def answer_context_adjacency(question: str, response: str,
                               reference: str) -> float:
    """Return [0, 1] distance between the response's anchor token
    and the question's key phrase within the reference. 1 = farthest,
    0 = adjacent or anchor absent."""
    anchor = _primary_token(response)
    if not anchor or anchor not in reference:
        return 0.0
    q_phrase = _question_key_phrase(question)
    if q_phrase not in reference:
        return 0.0
    # Character-level distance between nearest co-occurrence
    ...
```

Probe script to evaluate on HaluBench-DROP before full integration:

```bash
python benchmarks/hallucination_test/probe_dropfix.py --n 100
# Reports: AUC of role_mismatch alone, answer_context alone,
# and (v4 signals) + new signals as a combined 10- or 11-signal LR.
```

## Scope exclusions

- **Cross-vendor arithmetic LLMs** — we are not proposing to call an
  external model to verify arithmetic. All signals run locally with
  no API cost.
- **Span-level NLI** — decomposing the response into atomic claims
  and running NLI per claim IS a separate and complementary
  improvement (targeting HaluEval-Dialog's 0.68 ceiling) that
  belongs to its own v4.2 track, not v4.1.
- **Multi-hop reasoning errors** — questions that require chained
  inference (HaluBench-RAGTruth edge cases) are already at AUC 0.81.
  Further gains there are v5+ territory.

## Who this is for

Users running `@trust` on production RAG over reference documents
with numeric answers. Finance ops, reading-comp tutoring systems,
factual-QA assistants over structured corpora. These are exactly
the users v4.0.0's failure modes told they should NOT rely on the
detector. v4.1 aims to bring them inside the envelope.

---

## 7. Probe result — the cheap heuristics don't work

Before committing engineering time to the full v4.1 signals, we ran a
quick probe (`benchmarks/hallucination_test/probe_dropfix.py`, 2026-04-23)
implementing both proposed signals at the most naive possible level:

- `role_mismatch`: regex match the question's wh-word + first argument
  to an expected semantic-role class, regex classify the response's
  first token, emit `1.0` on mismatch.
- `answer_context_adjacency`: token-level search for the response's
  first numeric/capitalized anchor in the reference, compute the
  shortest distance to any non-stopword question keyword, normalize
  to [0, 1] with 0.0 = adjacent.

Tested on n=200/class DROP rows at seed 31. Results:

    role_mismatch                  AUC 0.500  (chance)
    answer_context_adjacency       AUC 0.520  (barely above chance)
    naive combined (0.6·role + 0.4·ctx)  AUC 0.515

The role_mismatch signal fires on exactly 8/200 PASS rows and 8/200
FAIL rows — it is too conservative to discriminate. The adjacency
signal picks up a 2-point lift which is within noise.

**Neither signal clears the 0.55 threshold we set for "worth
integrating."** The sophisticated versions described in §3 will
presumably do better than the regex version, but the magnitude of
the null on the cheap version suggests the v4.1 budget needs to grow.

## 8. What the null result tells us

Two updates to the v4.1 plan:

### 8a. The DROP failure mode is harder than a regex fix

The real DROP errors are not wh-word/response-type mismatches (the
sampled 20 FAIL rows were overwhelmingly "how many X" questions
answered with a number — the wh-word and the response TYPE both
match). They are *semantic grounding* failures: the wrong number
inside a numeric answer to a numeric question, or the wrong team
inside a team-selection answer. Catching these needs one of:

- A trained span-faithfulness model (non-trivial)
- Actual NER on both question and reference with typed matching
- A small verification LM queried at inference time (defeats
  local-first / zero-cost design)

None of these are a half-day of engineering. The v4.1 budget is
closer to a full week if we want a real lift.

### 8b. Even v4.0.0's *below-chance* result contains signal

AUC 0.424 on DROP is not random — random would be 0.500. Below
chance means the v4.0.0 signal stack is making systematically
WRONG predictions on DROP. Inverting the sign on DROP would give
AUC 0.576 — better than any of the v4.1 stubs produced here. This
is not a real fix (you can't invert a production classifier based on
a hunch about which domain you're in), but it IS a hint that the
existing signals *do* carry information about DROP correctness, just
with the wrong polarity. Likely mechanism: novelty-based signals
interpret "wrong number that doesn't appear in the passage" as MORE
hallucinated, but the RIGHT answer to a DROP question is often also
a novel number (the result of an extraction + minor inference).

Follow-up probe for v4.1.1: train a DROP-specific LR head using the
existing 9 signals but with the correct sign, measure lift.

### 8c. Pivot — v4.1 plan as of 2026-04-23 end of day

- **DROP remediation: de-prioritized from v4.1.** The regex approach
  confirmed dead. Investment path forward is a trained span-
  faithfulness head or NER-typed matching, which is a v4.2 research
  track (1–2 weeks).
- **FinanceBench remediation: unchanged.** The symbolic_number_check
  signal (§3) is a distinct mechanism not tested here, and remains
  the plausible v4.1 direction.
- **Honest communication: updated.** The v4.0.0 launch and cognometry
  manifesto already list DROP + FinanceBench as published failure
  modes. The correct update is "we tried the obvious fix for DROP,
  it did not work, here is the full probe output." Report in
  progress.

## 9. Reproducing the null

```bash
pip install styxx==4.0.0[nli]
python benchmarks/hallucination_test/probe_dropfix.py --n 200 --seed 31
# Expected output:
#   role_mismatch                  AUC 0.500
#   answer_context_adjacency       AUC 0.520
#   naive combined                 AUC 0.515
```

The probe uses the corrected Mann-Whitney U AUC implementation (the
original `_auc` in `cross_dataset_calibrate.py` had a tie-breaking bug
that inflated the probe's first-pass result by ~0.4 AUC points; fixed
in probe_dropfix.py's `_auc` and documented in the commit that lands
this paper).
