# FINDING — structure loses to search space at this matcher (STRUCTURE-LOSES)

**2026-06-10 · Fathom Lab / styxx. Pre-registered: `PREREG_gavagai_scale_2026_06_10.md` (frozen
pre-run). Receipt: `gavagai_scale_result.json`. Local, $0, no new model runs.**

## Question

Is Quine's residual indeterminacy a principle or a quantity that shrinks as the interpreter sees
more of the foreign mind's world? Nested worlds of 48 ⊂ 96 ⊂ 192 concepts (seeded subsets of the
telepathy reps), the frozen label-free matcher, 40 cross-family LLM pairs.

## Result: the frozen gate answers RAW assignment — and it degrades

- Mean cross-family accuracy by world size: **0.1693 (N=48) → 0.1573 (N=96) → 0.1073 (N=192)**.
  Only 1 of 40 pairs improved from 96 to 192; two-sided sign test p = 0.0. Verdict per the frozen
  S1 gate: **STRUCTURE-LOSES** — at this matcher, vocabulary growth outpaces constraint growth.
- **The chance-normalized story runs the other way (S2, descriptive):** accuracy-over-chance climbs
  **8.13× → 15.1× → 20.6×** as the world grows (chance 0.0208 / 0.0104 / 0.0052). The search space
  doubled; raw accuracy fell by roughly a third — much slower than chance halved. Per guess, the
  geometry carries MORE identifying information at larger N; the assignment solver just cannot
  cash it in.

## The convergent lesson of the night

Three independently pre-registered gates now point at the same wall: telepathy v0's oracle gap
(0.8322 achievable with the true alignment vs 0.1441 achieved without), GAVAGAI-X's
correlation-without-assignability at the species boundary, and this scaling verdict. **The
relational code is rich and near-lossless; the unsupervised MATCHER is the bottleneck of
structure-only translation.** "Indeterminacy of translation" at this scale is not a property of
meaning — it is a property of the interpreter's algorithm. That is a falsifiable, attackable
engineering target, not a philosophical wall.

## Rungs

Better matchers under the same frozen evaluation: seeded Gromov-Wasserstein with entropic
regularization; soft-assignment EM over the correlation refinement; category-blocked two-stage
matching (the anatomy receipt says the blocks are shared); multi-restart consensus. Each is
cycle-sized against the persisted reps, with this receipt as the baseline to beat at N=192.

## Bounds

One matcher family tested; nested subsets share composition but inherit the battery's concrete-noun
bias; 10 minds. The S1 verdict is matcher-relative by construction — that is its content, not a
caveat.
