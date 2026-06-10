# FINDING — the interlingua is species-bound (SPECIES-BOUND)

**2026-06-10 · Fathom Lab / styxx. Pre-registered: `PREREG_gavagai_x_2026_06_10.md` (frozen
pre-run). Receipt: `gavagai_x_result.json` (v0 reference receipt: `gavagai_v0_result.json`).
Local, $0. Gate X1 failed exactly as gates are meant to fail: loudly, with the boundary mapped.**

## Question

GAVAGAI v0 recovered concept identity between causal LLMs; telepathy v0 transmitted novel content
over that channel. Is the identity-carrying geometry a property of LEARNED MEANING in general, or
of a SPECIES of mind — models sharing a training-objective class? Test: the same label-free
translator between 10 causal LLMs (norm-equalized) and 2 contrastively-trained sentence embedders
(MiniLM-L6, mpnet-base; per-template convention mirrored), 20 cross-species pairs.

## Result

- **X1 FAILED — SPECIES-BOUND.** Mean cross-species identification accuracy **0.0292** (chance
  0.0104; bar 0.1042; identity-decoupled null 95th percentile 0.0312). At the assignment level,
  the cross-species channel is statistically indistinguishable from a broken one.
- **The internal control is decisive (X3):** the within-other-species pair MiniLM↔mpnet translates
  at **0.1875** (category 0.4688) — at the LLM↔LLM cross-family level of 0.1661 (X2). The matcher
  works inside BOTH species; what fails is the bridge between them.
- **Kind leaks, identity doesn't:** cross-species category accuracy 0.1927 vs chance 0.125 —
  coarse taxonomic structure crosses the objective boundary; concept-level correspondence does not.
- Pipeline control (self-pair): 1.0.

## Reading

The convergent, identity-carrying geometry demonstrated tonight is a **within-species
phenomenon** at this scale: minds trained on the same objective class share assignable structure;
minds trained on different objectives share correlated structure (the old confirm receipt's MiniLM
anchor partial RSA 0.36) WITHOUT assignable identity — second-order similarity is necessary but
not sufficient for translation. This bounds GAVAGAI v0 and telepathy v0 precisely: the channel
they demonstrate runs within the species of next-token minds (and, by X3, a parallel channel runs
within the species of contrastive minds). The boundary itself is new, quantified knowledge: nobody
had measured where structure-only translation dies.

## Rungs

Whether the species barrier is fundamental or an alignment-method limit is open: the oracle
question (does a LABELED LLM↔embedder mapping carry novel content the way telepathy v0's oracle
did within-species?) separates code-loss from correspondence-loss; richer matchers (seeded GW,
soft assignment) and intermediate species (instruction-tuned embedders, encoder-decoders) map the
boundary's thickness. All cycle-sized.
