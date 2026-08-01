# FINDING — read ≠ write survives its strongest attack: the machinery that opened reading does not open control

Fathom Lab · 2026-08-01 · scored under `PREREG_b36_write_door_2026_08_01.md` (frozen
`5d6e2e6`; apparatus `d774ce8`, both committed before any scored generation). Receipt:
`b36_result.json`. Prior-run comparators quoted from `../synthesis_minds_addendum.json`.

## Verdict: `READ_NEQ_WRITE_SURVIVES_CAPACITY`

The program's flagship dissociation — *what a mind means crosses; the means to move it does
not* — was measured with a **linear** map. Then b31v2 proved the linear class was leaving
readable content on the table (a target that read at exact chance read at 55× chance through
a paired MLP). So we turned that same machinery on our own law, at matched-maximal
supervision, at the layer where the target demonstrably steers.

| arm | mean steering gain | gate | |
|---|---:|---|---|
| **native** (the target's own directions — the ceiling) | **0.173** | PC ≥ 0.15 | **PASS** |
| **M1** paired MLP transfer (the attack) | **0.0504** | G1 ≥ 0.15 | **FAIL** |
| L1 paired linear transfer (comparator) | 0.0353 | — | — |
| random unit direction (the null) | 0.0217 | — | — |
| M1 − random | 0.0286 | G2 magnitude ≥ 0.10 | **FAIL** |
| M1 beats random (sign rate) | 0.6571 | G2 sign ≥ 0.70 | **FAIL** |
| NTE = M1 / native | 0.291 | G3 ≥ 0.40 | **FAIL** |

70 held-out concepts, dose locked on native directions at α=16.0, finite-difference
direction stability 0.999 (the transfer method was not broken — the extracted directions are
essentially exact). **The positive control fired and every transfer gate failed.**

## What capacity bought — and where it stopped

Against the committed linear-map run at the same operating point (transfer 0.0245, NTE 0.114):
heavier machinery with maximal supervision roughly **doubled** the transferred control
(0.0245 → 0.0504) and lifted the fraction of native control recovered from 0.114 to 0.291.
Real, ordered, and reproducible: M1 > L1 > random, in that order, exactly as capacity should
behave. And it lands **three times below its own floor.**

The comparison that makes this the sharpest statement of the dissociation yet: on **this same
model pair**, the same class of upgrade took *reading* from 0.3429 to 0.8000 — over a floor
it cleared with room to spare — and took *writing* from 0.0245 to 0.0504, under a floor it
misses by 3×. Capacity multiplies both channels by about the same factor. Only one of them
was ever close enough for that to matter.

**Reading was capacity-limited. Control is not.**

## The honest nuance we did not want

The sign test moved the wrong way: under the linear map the transferred direction beat the
random null on 0.71 of concepts; under the MLP it beats it on 0.6571. Cross-run and
descriptive (different runs, not a controlled contrast) — but it says the extra capacity
bought *magnitude*, not directional reliability. The faint correctly-aimed shadow the
writelayer-decouple run named did not sharpen. It got slightly blurrier while getting
slightly stronger.

## Scope and threats to validity

One model pair (Llama-3.2-3B → Llama-3.2-1B, same family, near-isometric), one seed, one
layer pair (src 11 → dst 11, the committed steer-optimal point), one steering metric
(the committed `steer_gain` protocol: fixed carriers, greedy continuation, MiniLM concept-similarity gain over clean), 70 held-out concepts,
maximal supervision by design (392 true pairs — a null cannot be blamed on correspondence
discovery). This does NOT establish that no map class can ever write across minds; it
establishes that **the map class which opened reading does not open writing at this
operating point**, which is precisely the attack the b31v2 result licensed and the one that
mattered.

Also disclosed: this prereg predates `styxx.protocol` and carries its gates in prose, not in
a machine-readable ```gates block — so this FINDING's seal exercises the OATH layer only.
Every prereg from here carries the block, and this one is the reason.

## What it means

Every deployable this program ships inherits read ≠ write as a design rail — the conscience
is READ-ONLY, `styxx.witness` has no `steer` method at all, not as a policy choice but
because the physics refused. That rail was built on a linear-map measurement. It now stands
on the strongest attack we know how to mount, with the positive control firing and four
gates imported unchanged from the run it was trying to overturn.

*What a mind means crosses. The means to move it does not.* We tried to break it with the
best weapon we own, in public, under bars frozen before the data. It held.
