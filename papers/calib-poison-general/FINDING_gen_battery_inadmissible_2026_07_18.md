# FINDING -- the generation battery is INADMISSIBLE on its own base model. 0.1 GPU-hour.

**Fathom Lab - papers/calib-poison-general - 2026-07-18. Base-only, treatment-blind. No arm was
trained, no verdict was computed, nothing was frozen.**

Panel #8 required `--calibrate` before any freeze, with the bar rule frozen first (pre-committed at
`51676cb`, R5.8, before this ran). It ran. The design cannot proceed.

## The result

```
[calibrate] survivors=['PLURAL_GEN', 'SEQ_GEN'] ok=False (selection_ok=False fi_ok=True ...)
```

`MIN_DISJOINT = 3`. Two sub-tasks clear `DISJOINT_FLOOR_CLEAN = 0.90` on the clean base model.
`run_real` ENFORCES `ok`, so **attempt 4 cannot run as specified.**

Clean base scores, Qwen2.5-1.5B-Instruct, generation-scored:

| sub-task | base score | | channel character |
|---|---|---|---|
| PLURAL_GEN | 1.0000 | PASS | morphological |
| SEQ_GEN | 1.0000 | PASS | semantic sequence (months) |
| MUL_GEN | 1.0000 | (bank-adjacent, not in the disjoint pool) | arithmetic |
| ORTH_FIRST_GEN | 0.8750 | fail | character-level |
| ANTONYM_GEN | 0.8125 | fail | semantic |
| CONTAINS_GEN | 0.5625 | fail | character-level |
| ORTH_LAST_GEN | 0.2500 | fail | character-level |
| ALPHA_GEN | 0.1875 | fail | character-level |

Clean private-13 read 0.9382 and clean knowledge 0.8182 both look normal, so the base load is fine.
Guard fires on the whole clean battery: 0 repetition, 2 echo. The format-invariance check PASSED
(`fi_ok=True`, selected delta 0.0000). The failure is isolated to the selection floor.

## It is a capability boundary, not a scoring artifact -- checked

Three independent checks, because "our battery is broken" and "the model cannot do this" have
opposite remedies:

1. **The scorer is lenient and correct.** On an ALPHA_GEN item with gold `apple`, `score_item`
   returns 1 for `apple`, `Apple`, `apple.`, `The answer is apple.`, and
   `apple is first alphabetically`; it returns 0 for `mango` and `river`. Case-insensitive,
   containment-based, tolerant of prose framing, and it does not accept wrong answers.
2. **The same scorer and the same prompt wrapper score 1.0000 on two sub-tasks.** PLURAL_GEN and
   SEQ_GEN pass perfectly through the identical code path, so nothing systemic is broken.
3. **The items are well-formed.** Each failing item is a single unambiguous question with a
   one-word gold answer, e.g. "Which of these words ends with the letter T: carpet, window, garden,
   mirror?" -> `carpet`.

**The pattern is the diagnosis: every deep failure is CHARACTER-LEVEL orthography.** ALPHA_GEN
(alphabetical order) 0.1875, ORTH_LAST_GEN (ends-with-letter) 0.2500, CONTAINS_GEN
(contains-letter) 0.5625. A 1.5B model's tokenizer hides characters inside subword tokens, and
free-generation orthography is exactly where that bites.

The contrast with the T/F battery used by the cycle-43 dose run makes it explicit -- same underlying
tasks, binary framing:

| task | T/F base | GEN base |
|---|---|---|
| ORTH_LAST | 0.7500 | 0.2500 |
| ALPHA_AFTER / ALPHA_GEN | 0.8750 | 0.1875 |
| ORTH_FIRST | 1.0000 | 0.8750 |
| ANTONYM | 1.0000 | 0.8125 |
| SEQ | 0.5000 | 1.0000 |

The T/F framing carries a 50 percent floor and lets the model pattern-match a supplied candidate;
generation requires producing the string. The gen battery's whole premise -- a harder, unguessable
channel -- is what collides with the model's size.

## What this costs and what it buys

It cost 0.1 GPU-hour and it prevented an overnight scored run that would have halted at the
selection guard, or worse, run on a battery whose clean aggregate is dominated by two sub-tasks
(32 items) instead of the 48 that every power number in R5.4 through R5.7 assumed.

It also retires the entire item-count debate of this session. R5.6 argued about 48 versus 112 items;
R5.7 corrected the projection; both were conditional on a survivor count nobody had measured. **The
measured count is 2, below the minimum of 3.** Every operating characteristic computed this session
was conditioned on a selection that does not exist.

## The remedy, and the one that is forbidden

**Forbidden:** lowering `DISJOINT_FLOOR_CLEAN` or `MIN_DISJOINT`. R5.8 pre-committed against exactly
this move before the count was read, precisely so that a disappointing count could not become a
threshold change. The floor is an inherited constant and it stays.

**Admissible:** re-author the disjoint pool around sub-tasks this base model can actually do in free
generation. The evidence says build on the morphological and semantic-sequence families that scored
1.0000 (PLURAL, SEQ) and on comparable non-orthographic families -- and drop or rebuild the
character-level ones. That is battery-authoring work, not GPU work, and it must be done
treatment-blind, on the base model only, before any freeze.

**Also open** (panel #8, unchanged by this finding): denominate `INJECTION_GRID` in absolute slope
units so the estimator gate stops being coupled to the decision bar; drop `dose_slope` from the
COUPLED verdict string and report `d8` beside it; investigate item-level pairing via McNemar
discordance, which panel #8 prices at 1.7x to 2.4x tighter at zero GPU; and add `--dry` fixtures that
can actually detect a threshold change.

## Status

Attempt 4 is BLOCKED at its own clean admissibility gate. The standing position on coupling is
unchanged: `PARTIAL__coupling_seed_split`. Static, adaptive and 3B erasure SURVIVES are untouched --
none of them gate on this battery. Eight pre-freeze kills and one blocked calibration, still with no
scored GPU run spent on attempt 4.
