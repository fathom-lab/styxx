# PREREG -- SOURCE INDEPENDENCE: is abstention a property of MODELS, or of ITEMS?

**Cycle 67 (operator-directed "get ambitious"). Frozen before any scored run exists. Committed
ahead of results. Bars are binding; a missed bar is CLOSED_NEGATIVE, never SURVIVED.**

## The question cycles 65 and 66 could not answer

`FINDING_tiered_channel_2026_07_24.md` (cycle 65) and `FINDING_scale_channel_2026_07_24.md`
(cycle 66) both escalated the loop's abstention slice to a second model and both hit the same wall:
co-abstention of **0.8478** (different family, same scale) and **0.8043** (same family, 2x scale),
with agreement of 0.9837 and 0.9919 where both channels spoke. Cycle 66's pass margin was four
tenths of an item, so the practical reading of both is: model-side escalation does not reach the
core.

**Neither cycle could say WHY.** Two very different worlds produce that data:

- **World A -- it is about MODELS.** Every LLM shares a training distribution, so they are ignorant
  of the same things. A source of knowledge that is not an LLM would decline on *different* items.
- **World B -- it is about ITEMS.** The declined items are intrinsically hard or underdetermined,
  and *every* knowledge source declines on them, LLM or not.

These have opposite consequences. In World A the route forward is external knowledge and the
program has a real next move. In World B the abstention slice is a floor, no channel reaches it,
and the honest product is the refusal itself. This cycle separates them.

## The design: two tier-2 channels of different KIND, one slice, paired

Over the same items and the same tier-1 abstention slice, under an **identical
adjudicate-or-abstain contract**:

| channel | kind | independence from tier-1 |
|---------|------|--------------------------|
| tier-2a | **model** -- Llama-3.2-3B | different family, same scale (the cycle-65 design) |
| tier-2b | **retrieval** -- dense top-5 over a 20,233-passage haystack | different *kind of knowledge source* |

The retrieval channel adjudicates exactly as the model channels do -- it never supplies an answer,
it only reports which of the two existing candidates actually **appears** in the retrieved
passages, and abstains when neither or both do. No reader LLM is involved, so its verdict cannot
be contaminated by model priors.

Because both tier-2 channels are evaluated on the *same* items with the *same* contract, their
co-abstention rates with tier-1 are **paired and directly comparable** -- which is what makes
World A and World B distinguishable.

## The corpus is data-derived, not authored (the cycle-63 trap, avoided again)

An authored fact table would BE the answer key. Everything here comes from SQuAD v2:

- **Items**: questions and gold answers written by SQuAD annotators over Wikipedia; filtered to
  short answers (1-3 words) so the frozen `mentions()` scorer stays reliable.
- **Distractor Y**: a REAL short answer span drawn from a **different passage**, selected as the
  most embedding-similar candidate to the true answer -- type-plausible, and from a different
  passage precisely so both candidates cannot co-occur in the gold paragraph (which would make the
  retrieval channel abstain by construction).
- **Haystack**: all 20,233 unique SQuAD contexts. The gold passage is present but must be **found**
  among ~20k distractors; the smoke measured gold-in-top-5 at roughly 0.70, so retrieval is
  genuinely imperfect rather than a lookup.

**Disclosed domain change:** this pool is SQuAD, not the factual pool of cycles 62-66, so results
are NOT directly comparable to those cycles and the whole pipeline (agent, tier-1) is re-run here.
That change is forced: testing retrieval requires a domain where an independent corpus exists, and
authoring one for the old domain is exactly the forbidden move.

## Frozen bars

- **FV1 (validity):** >= 25 items in each condition AND >= 25 items in the tier-1 abstention slice.
- **FG1:** retrieval-tiered final coverage >= tier-1 coverage **+ 0.05** (inherited from cycle 65).
- **FG2 (the kill):** retrieval-tiered answered accuracy >= tier-1 answered accuracy **- 0.05**
  (inherited). Coverage may not be bought with the errors the refusal was catching.
- **FG3 (paired):** on items retrieval rescues, its accuracy must exceed the fallback's accuracy on
  **those same items** by **>= 0.15** (inherited).
- **FG4 (THE NOVEL CLAIM):** retrieval's co-abstention with tier-1 must be **at least 0.15 BELOW**
  the model channel's co-abstention, measured on the same slice in the same run. This is the World
  A / World B discriminator and it is the reason the cycle exists.

## Both outcomes pre-committed, so neither can be spun

- **FG4 passes -> World A.** Ignorance is shared *because the channels are all language models*.
  Knowledge-source diversity is a real and previously unmeasured axis of independence, orthogonal
  to the architectural diversity that failed in cycles 65 and 66, and external knowledge is the
  route forward.
- **FG4 fails -> World B.** Abstention is a property of the ITEMS. Hard items are hard for
  retrieval too, the ~80% core is a floor no channel reaches, and the selective predictor's ceiling
  is intrinsic rather than an engineering gap. That would make **the refusal itself the product**,
  and would close the entire escalation direction -- a more useful result than another marginal
  coverage gain.

## Reported, NOT gated

Gold-in-top-5 rate overall and on the slice (retrieval quality, so a null cannot be blamed on a
broken index without evidence); model-tier-2 final coverage and accuracy; per-condition
breakdowns; stubborn at matched coverage.

## Scope

0.5B agent, Qwen2.5-3B tier-1, Llama-3.2-3B tier-2a, dense retrieval tier-2b, 200 SQuAD items,
two-turn pressure. Motivating-grade; a fresh-pool confirmation would be owed before any pass is
claimed as more. No frontier model, no capability claim, no training claim.

## Receipts

`build_squad_pool.py`, `run_source_independence.py`, `squad_pool.json` (all frozen with this
prereg). `squad_corpus.json` and `squad_corpus_emb.npy` are large derived artifacts, regenerable
deterministically from the builder, and are therefore not committed. Scored output
`source_independence_result.json`; `--smoke` writes only `*_SMOKE_INVALID*`.
