# Within-category control — is the LLM↔brain match just coarse category structure?

**2026-06-30 · fathom-lab / styxx · reproduce: `within_category_control.py` → `within_category_result.json`**

The standard objection to any RSA-to-brain result: the stimulus set splits into coarse semantic
categories (the Mitchell-2008 60 nouns = 12 categories × 5: animal, body, building, building-part,
clothing, furniture, insect, kitchen, man-made, tool, vegetable, vehicle), and *any* representation
trivially recovers "animal vs tool" blocks. So the headline match could be cheap taxonomy, not meaning.
We tested it two ways. All RSA is partial-lexical (word-length + token-length removed), vs the same brain.

## Positive control — categories are present but weak
Across-category / within-category mean distance ratio: **brain 1.15, gpt2-large 1.20** (>1 = categories
separate). The brain RDM is *not* category-dominated to begin with — the blocks are mild.

## Test 1 — partial out category (well-powered: 1770 / 1378 pairs)
Add a same/different-category indicator to the lexical controls and re-measure. **The match survives, and
at the human-behavior level:**

| representation | full RSA | + category-controlled (95% CI) |
|---|---|---|
| gpt2-large (text-only LLM) @60 | 0.264 | **0.206  [0.116, 0.310]** |
| gpt2-large @53 shared | 0.222 | 0.151 |
| **VICE / human behavior** @53 | 0.247 | **0.186  [0.088, 0.291]** |
| mpnet (embedder) @53 | 0.161 | 0.084 |
| MiniLM (embedder) @53 | 0.156 | 0.084 |

The LLM's category-controlled match (0.206 [0.116, 0.310]) excludes zero and is statistically
indistinguishable from human behavior's (0.186 [0.088, 0.291]). **The match is not reducible to the 12
category blocks** — it captures cross-category meaning structure beyond taxonomy, as well as 1.5M human
judgments do. The objection, as posed, is defeated.

## Test 2 — within-category only (underpowered; honest limit)
Restrict RSA to same-category pairs (does the model recover structure *finer* than category — e.g. which
animals the brain holds closest?). With only 5 nouns/category there are just **102–120 within-category
pairs**, and the test is far too noisy to resolve:

| representation | within-category RSA (95% CI) |
|---|---|
| gpt2-large @60 | +0.032  [−0.344, +0.421] |
| VICE / human @53 | +0.171  [−0.279, +0.590] |
| mpnet / MiniLM | −0.041 / −0.013 |

Every estimate's CI spans zero — **including human behavior's.** This is a power limit of the Mitchell
dataset (5 items/category), not evidence against the LLM: neither the model nor 1.5M human judgments can
resolve sub-category structure here. The finest-grain test needs a denser fMRI set (e.g. THINGS-fMRI).

## Bottom line
The headline — *a free text-only LLM shares the brain's geometry of concrete meaning, as well as human
behavior does* — **survives the strongest standard objection** (coarse category structure) at the level
it can be posed and tested. Resolving meaning *within* categories is a future experiment, not a present
claim. We report both, and stop where the data stops.
