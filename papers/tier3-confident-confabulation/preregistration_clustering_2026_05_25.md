# Pre-Registration · Tier-3 clustering redesign — does a meaning-judge solve the step that form can't?

**Committed BEFORE data.** The arc converged on one open problem: semantic entropy's
*clustering step*. Cosine is form-only (threshold-brittle: 0.70 merges template-sharing
lies, 0.95 works only because correct answers were verbatim). `nli-deberta-base` is a
weak meaning-judge (false-positived on 5/8 correct explanations). The untested candidate
is the **strongest available equivalence judge: the model judging its own answers**
("do these two convey the same core answer?"). This is the form→meaning move, applied
inside the instrument.

## Fix to the prior VOID

The focused probe VOIDed because the model answers near-verbatim. Here we **force**
surface variation with a uniform nudge — system prompt: *"Answer in one short sentence,
specific and direct; phrase it in your own words (vary the wording, don't reuse a fixed
template)."* — applied to ALL classes, so correct answers actually paraphrase.

## Design (run once)

N=6, temp 1.0, gpt-4o-mini. Classes:
- **C1 fact** (6) + **C2 explanation** (8): correct; with the nudge, expected surface-
  varied. Together = the *correct* (negative) set.
- **C3 fictional-confab** (8): nonexistent person/year/place → invents specifics.

Clustering methods compared on the IDENTICAL samples:
- **cosine** at {0.70, 0.80, 0.90, 0.95} (form);
- **nli-deberta** bidirectional entailment (weak meaning, the one that failed);
- **LLM-judge**: gpt-4o-mini pairwise "same core answer? YES/NO", greedy-clustered
  (strong meaning — the candidate fix).

Per method: semantic entropy per item; AUC(entropy → confab) over correct (C1∪C2 valid)
vs C3-confab; and the false-positive load on the surface-varied-correct subset.

## Validity gate (VOID if unmet)

The nudge must actually create paraphrase: **≥ 4** correct items (C1∪C2) with mean
within-item pairwise cosine **< 0.85** AND judged correct vs a curated reference. Else
VOID (model resists paraphrasing — itself a finding; redesign, don't reinterpret).

## Kill-gate (PASS iff G2 ∧ G3, given G1 validity)

| ID | Bar |
|----|-----|
| **G2 (judge solves it)** | LLM-judge AUC ≥ **0.80** on correct-vs-confab, AND its mean entropy on the surface-varied-correct subset ≤ **0.40 ×** its mean on C3-confab (no paraphrase false-positive). |
| **G3 (cheap methods don't)** | the best of {cosine-sweep, nli-deberta} either scores AUC < **0.75** OR false-positives on paraphrase (varied-correct mean entropy > **0.60 ×** its C3 mean). I.e., the meaning-judge is *necessary*. |

**PASS** → the clustering step is solvable with a meaning-judge and cheap form/weak-NLI
methods are insufficient under paraphrase → validates an LLM-judge-clustered
`semantic_entropy` design + confirms the form↔meaning boundary *inside* the instrument.
**FAIL shapes:** (a) cheap methods also fine → no judge needed (cheaper primitive, good);
(b) even the LLM-judge false-positives on paraphrase → the divergence signal itself
conflates paraphrase with confabulation (a real Tier-3 limit, not a clustering bug).

## Honest prior

I expect the LLM-judge to win (it is the strongest equivalence signal and should ignore
wording). Real risks: (1) the nudge may degrade correctness (varied wording drifts in
meaning) → muddies C2; (2) the judge may be inconsistent (lenient/strict) → noisy
clusters; (3) confabulations that the model commits to *consistently* (low divergence by
any method) remain undetectable — an irreducible floor. Record whichever happens.
