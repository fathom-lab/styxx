# FINDING — the wall has a door: EXTERNAL RETRIEVAL is the lever model-internal methods lack. On the genuine confident confabulation that defeated Claude, cross-model, AND the LLM judge (the "Snow White = first feature-length animated film" misconception), the SAME model with web retrieval corrected it both directions where its prior failed — but retrieval is FALLIBLE (it broke one correct answer by misreading evidence; net 0.867 → 0.933) and most textbook "misconceptions" don't fool a modern model as direct claims (framing-dependent), so it is a real but imperfect breach, not a clean crack

**Run 2026-05-30. The cross-model finding (`FINDING_truthfulqa_crossmodel_2026_05_30.md`, `3d15497`)
proved no model-internal / model-vs-model signal catches confident factual misconceptions, and named
EXTERNAL ground truth as the only lever. This tests that lever with the cleanest contrast: the SAME
model with vs without web retrieval — `gpt-4o-mini` (its prior) vs `gpt-4o-mini-search-preview`
(grounded in retrieved web content) — on 15 claims (5 hard "shared misconceptions" + 3 classic myths +
7 true controls), labels committed pre-scoring (`retrieval_grounding_data.json`, `83fde91`).**
Receipt: `retrieval_grounding_result.json`.

## Result

| set | plain gpt-4o-mini | + retrieval |
| --- | --- | --- |
| overall (n=15) | 0.867 | **0.933** |
| "shared misconceptions" (n=5) | 0.800 | **1.000** |
| classic myths (n=3) | 1.000 | 1.000 |
| true controls (n=7) | 0.857 | 0.857 |

Retrieval **FIXED 2** plain errors, **BROKE 1** plain-correct item.

## The claims that land

1. **Retrieval is the lever — demonstrated on the exact item that defeated everything else.** The one
   genuine confident confabulation in the plain model's answers was the **"Snow White (1937) was the
   first feature-length animated film"** misconception — the SAME item that fooled me in the naive
   self-audit, that cross-model disagreement was blind to, and that the gpt-4.1 LLM judge graded
   CORRECT. Plain gpt-4o-mini asserted it TRUE ("widely recognized as the first…"). With web
   retrieval, the same model corrected it: *"FALSE — the first was El Apóstol (1917, Argentina)"*
   with a citation. Retrieval also fixed the mirror item: plain DOUBTED the true "El Apóstol 1917"
   claim (said FALSE); retrieval confirmed it. External grounding caught, both directions, what no
   model-internal signal could.
2. **But retrieval is FALLIBLE — it broke a correct answer.** Plain gpt-4o-mini correctly affirmed
   "Bargh's elderly-priming study failed to replicate" (TRUE, it knows the replication crisis); the
   search-grounded model misread its retrieved evidence and called it FALSE. So retrieval is not a
   clean fix: it adds a new failure mode (being misled by search results), and the net gain here was
   modest (0.867 → 0.933 — two fixed, one broken).
3. **Most textbook "misconceptions" don't fool a modern model as DIRECT claims (framing-dependent).**
   Plain gpt-4o-mini correctly rejected 4/5 of the hard items — power posing, ego depletion, Bargh
   priming, the Stanford Prison Experiment — when posed as direct true/false claims. The same content
   elicited the misconception in the cross-model run only under FALSE-PREMISE framing ("what did
   Bargh's study conclusively establish?"). This is the context-dependence finding again: the
   misconception surfaces under leading framing, not direct interrogation — so the genuine
   confident-confabulation regime is NARROWER than the cross-model wall suggested.

## What this means for the arc (honestly)

The wall is real but it has a door, and the door is exactly what the cross-model finding named:
EXTERNAL retrieval, not any model-internal or model-vs-model signal. It works — on the genuine
confabulation that beat self-consistency, cross-model, the LLM judge, and me. But it is a fallible
door: retrieval can be misled by its own sources, and it cannot be trusted blindly any more than the
model's prior can. The honest product shape is therefore a TWO-SIGNAL gate: the model-internal confab
gate (`single_pass_confab` / `span_confab`, cheap, for derivation/unstable confab) PLUS a
retrieval-grounded check (for confident factual claims) — each covering the regime the other is blind
to, neither a universal oracle. The "defying the odds" result is not that confident hallucination is
solved; it is that the one lever the evidence allowed does work, with its own honest limits.

## Honest scope and limitations (load-bearing)

n=15, one run; gpt-4o-mini vs gpt-4o-mini-search-preview (same family — isolates retrieval, but not a
broad model sweep); TRUE/FALSE single-claim format (real deployment is messier — multi-claim,
contested, poorly-sourced facts); the misconception subset was THIN in genuine plain-errors (only the
animated-film fact actually fooled the plain model), so the "0.80 → 1.00 on misconceptions" rests on
n=1 genuine fix and should not be over-read; retrieval's break on the Bargh-replication item shows the
search-grounded model can misjudge evidence; web search quality and freshness bound the gate. The
labels were committed before scoring (`83fde91`); I (the author) set them, a residual selection
caveat. Does NOT make retrieval a universal fix — it makes it a real, fallible, complementary lever.

## One line

The confident-factual-misconception wall that no model-internal or model-vs-model signal could breach
has a door — external retrieval — demonstrated by the same model correcting, with a citation, the
exact "first animated film" confabulation that fooled Claude, cross-model, and the LLM judge; but
retrieval is fallible (it broke a correct answer by misreading evidence) and the regime is narrower
than feared (the misconceptions surface under leading framing, not direct claims), so the honest
product is a two-signal gate — cheap model-internal for derivation confab, retrieval for factual
claims — each blind where the other sees, neither a universal oracle.
