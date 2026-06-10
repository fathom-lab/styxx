# PREREG — GAVAGAI-SCALE: does indeterminacy shrink as the shared world grows?

**Frozen 2026-06-10, before any scored run. Fathom Lab / styxx.**

GAVAGAI v0 recovered concept identity across minds at 16x chance on a 96-concept world. The deep
question behind Quine's indeterminacy: is the residual ambiguity a PRINCIPLE, or a QUANTITY that
shrinks as the interpreter sees more of the foreign mind's world? Identification gets mechanically
HARDER with more candidates (chance falls as 1/N) while the relational structure gets RICHER (more
constraints per concept). Which force wins is the whole question: if raw accuracy holds or climbs
as N doubles, structure beats search-space and translation scales toward open vocabularies.

## Apparatus (frozen)

- Reps: `telepathy_reps.npz` (10 minds × 192 words: the 96-concept anchor battery + the 96-word
  fresh battery, norm-equalized convention; receipt of telepathy v0). NO new model runs.
- Worlds: nested seeded subsets of the 192 words — N=48 ⊂ N=96 ⊂ N=192 (numpy default_rng(0)
  permutation of the 192; first 48 / first 96 / all). Same subsets for every pair.
- Translator: the frozen GAVAGAI matcher (`run_gavagai_v0.translate`), labels hidden per pair by
  seeded permutation (seed 0); RDMs via `styxx.mind.distmat`.
- Population: the 33 cross-family unordered LLM pairs (A→B in word order), as v0.

## Pre-registered gates (frozen)

- **S1 (the scaling direction):** mean cross-family raw accuracy at N=192 EXCEEDS that at N=96
  (paired over the 33 pairs) AND a two-sided sign test over the 33 paired differences gives
  p < 0.05. PASS → **INDETERMINACY-SHRINKS** (structure beats search space; chance halved, accuracy
  rose anyway). If the mean falls with sign-test significance → **STRUCTURE-LOSES** (assignment
  degrades with vocabulary at this matcher; scaling needs better alignment, not more context).
  Neither significant → **SATURATED**.
- **S2 (descriptive):** accuracy and accuracy/chance ratios at 48/96/192; per-step deltas
  (48→96, 96→192); category accuracy alongside.
- **VOID-PIPELINE:** self-pair at N=192 with hidden labels must decode 1.0.

## Honest prior

Genuinely split. The matcher's refinement constraint count grows linearly with N (favoring
SHRINKS); the assignment space grows factorially (favoring LOSES). v0's smoke history warns
against trusting intuitions here. Any of the three verdicts is a finding: SHRINKS is a scaling
law; LOSES locates the matcher as the bottleneck (mirroring telepathy v0's oracle gap); SATURATED
calibrates the sweet spot.
