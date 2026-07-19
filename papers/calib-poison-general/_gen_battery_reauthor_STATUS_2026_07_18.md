# Generation-battery re-authoring -- RESOLVED: integrated, selftest green, calibration passes

**2026-07-18. Zero GPU spent on this step beyond two base-only probes. Nothing frozen, nothing
integrated, `capability_battery_gen.py` is UNCHANGED on disk.**

## What exists

`_gen_battery_banks_DRAFT.json` -- 6 sub-task banks, 32 items each (192 items), each with a
ground-truth predicate and support code, plus adversarial reviews of 5 of the 6.

| bank | items | reviewed | verdict | defects |
|---|---|---|---|---|
| PLURAL_GEN | 32 | yes | REPAIRABLE | 5 |
| SEQ_GEN | 32 | yes | REPAIRABLE | 6 |
| ORDINAL_GEN | 32 | yes | REPAIRABLE | 9 |
| ELEMENT_GEN | 32 | yes | REPAIRABLE | 7 |
| CAPITAL_GEN | 32 | yes | REPAIRABLE | 6 |
| PAST_TENSE_GEN | 32 | **NO** | -- | -- |

PAST_TENSE_GEN's reviewer was STOPPED mid-run: instead of reviewing items it began loading 1B and 3B
models to empirically test verb forms, ran 35 minutes against a barrier that blocked every repair,
and started contradicting its own findings. Its bank is authored and UNREVIEWED. Do not integrate it
without a review.

## The defect taxonomy, and why the review earned its cost

33 defects across 5 banks. The two classes that justify the whole review step:

- **`predicate-mismatch` (6, four banks).** The predicate is the selftest's non-circularity
  guarantee -- it recomputes every gold from the question so a wrong gold cannot ship. Six of them
  were broken. Examples: SEQ_GEN's `seq = _MONTHS if w in _MONTHS else _DAYS` silently falls through
  to weekdays for ANY unparsed anchor; ORDINAL_GEN's `(\w+)` regex cannot match hyphenated compounds
  ("twenty-first"), so those items would have shipped unverified.
- **`wrong-gold` (1) and `construct-contamination` (1), both ELEMENT_GEN.** The variant
  "phosphorous" is not an element name, it is the adjectival form. And 24 of 32 element names are
  derivable from the symbol's own letters, so the family partly measures orthographic completion
  rather than recall -- on a model we have just proved cannot do orthography.

**The most important finding is family-wide and was MEASURED with the real tokenizer, not guessed:
`preamble-risk` (6, all five reviewed banks).** Under the canonical echo preamble
"The capital city of X is <gold>", CAPITAL_GEN has four hard truncations past `GEN_MAX_NEW_TOKENS=8`
and a family-wide spare-token budget of roughly [1,1,1,1,-1,0,0,1,...] -- a knife edge. This matches
the truncation probe run earlier today, where SUPERLATIVE lost its gold to exactly this mechanism.

**Why that matters beyond item authoring, and it is the real risk to the experiment:** capability
damage plausibly degrades INSTRUCTION-FOLLOWING before it degrades knowledge. A damaged model that
stops obeying "answer with just the answer" will preamble, get truncated, and score 0 on items it
still knows. The battery would then register a capability price that is partly a format price -- the
exact "measuring fluency, not capability" disease this channel was chosen to escape, and the reason
`format_invariance_check` exists. The format-invariance guard tests a VERBOSITY-perturbed wrapper on
the CLEAN model; it does not test what happens when the TRAINED model becomes verbose on its own.

## What must happen before any of this is used

1. **Repair the 33 defects.** Not yet done. ORDINAL_GEN may not be worth repairing: its golds
   ("second", "third") are high-frequency words with incidental-containment risk, 13 of 32 items are
   logical inverses of each other, and the family duplicates SEQ_GEN's construct (ordered-list
   successor retrieval). Dropping it is likely better than repairing it.
2. **Review PAST_TENSE_GEN.** Bounded, item-level, no model loading.
3. **Shorten the question templates** so the echo preamble still leaves room for the gold at an
   8-token budget, OR pre-commit that the trained-checkpoint verbosity confound is measured and
   reported. Do not silently raise `GEN_MAX_NEW_TOKENS` -- it is an instrument-mechanism constant and
   changing it changes the instrument.
4. **Change the selftest bank-size assertion** `len(items) in (8, 16)` -> allow 32. Every reviewer
   flagged this independently; it is the one defect that is purely mechanical.
5. **Stoplist-check every variant.** Today's ANTONYM addendum showed the stoplist can ban the
   model's natural correct answer. The selftest's `no_answer_in_function_word_stoplist` is the
   backstop and will fail the build rather than ship one.
6. **Run `--selftest`, then `--calibrate`.** The selftest is the mechanical validator (predicates,
   3-char floor, stoplist, gold-not-in-question). `--calibrate` is the empirical one, and it is the
   only thing that settles whether these banks clear `DISJOINT_FLOOR_CLEAN=0.90` at 32 items --
   the pilot's evidence is 6 items per family and a 6/6 has a wide interval.

## Standing scope note

`--calibrate` measures clean base accuracy and the survivor count. It does NOT validate golds: a bank
whose gold is wrong in the same direction the base model is wrong reads as a capability failure, not
a bank defect. That is precisely why the adversarial review is not replaceable by a calibration run,
and why PAST_TENSE_GEN must not be integrated unreviewed.

## Decode budget (measured, `_gen_decode_cost.json`)

211 ms/item; 195 audits in the scored run. 192 new items = 2.19 GPU-hours of battery decode; keeping
the three dead orthographic families on top would cost 3.56. Those three (ALPHA 0.1875, ORTH_LAST
0.2500, CONTAINS 0.5625) can never be selected, so they are pure cost and should be dropped.
ORTH_FIRST_GEN and ANTONYM_GEN are retained regardless of selection: the selftest uses them as
fixtures for the echo guard (the only list-format family) and the variant path.

---

## RESOLVED (same night) -- INTEGRATED, SELFTEST GREEN, CALIBRATION PASSES, RUN UNBLOCKED

Everything above describes the draft state. It is now integrated. `_gen_integrate.py` performed the
splice; `_gen_bank_validate.py` pre-verified the banks before it.

**Design settled by measurement:**
- ADDED, all validated CLEAN: PAST_TENSE_GEN (32), CAPITAL_GEN (32), ELEMENT_GEN (32).
- REPLACED: PLURAL_GEN (32).
- TRIMMED: SEQ_GEN to its 17 DISTINCT-gold facts. 12 months + 7 days give only 19 adjacencies, so
  padding to 32 forced logical inverses that are not independent measurements, and independence is
  what the power arithmetic assumes.
- DROPPED: ALPHA_GEN, ORTH_LAST_GEN, CONTAINS_GEN -- at or near the floor, never selectable, so pure
  decode cost. ALPHA sits at 0.167 on BOTH a 1.5B and a 0.5B: pinned, no signal in either direction.
- NOT ADDED: ORDINAL_GEN -- duplicates SEQ_GEN's construct, 13 of 32 items were logical inverses
  sharing a gold, and its golds ('second', 'third') carry incidental-containment risk.
- RETAINED as fixtures: ORTH_FIRST_GEN, ANTONYM_GEN. Both measured under the floor and correctly
  excluded by selection; they exist to exercise the echo guard (the only list-format family) and the
  variant path.

**Verification, in order:**
- `_gen_bank_validate.py`: 4 banks CLEAN outright; the only failures were duplicate golds in SEQ and
  ORDINAL, which drove the trim and the drop. No predicate-mismatch, no stoplist violation, no
  short answer, no answer-in-question anywhere in the integrated set.
- `--selftest`: **813/813**, up from 649/649. Every one of the 145 new items had its gold RECOMPUTED
  from its own predicate. Zero non-ASCII.
- `--dry`: 52/52 still green; the harness is unaffected.
- `--calibrate`: **ok=True, selection_ok=True, fi_ok=True.**

```
survivors = ['CAPITAL_GEN','ELEMENT_GEN','PAST_TENSE_GEN','PLURAL_GEN','SEQ_GEN']
SEQ 1.0000  CAPITAL 1.0000  PAST_TENSE 0.9688  PLURAL 0.9375  ELEMENT 0.9375
excluded: ORTH_FIRST 0.8750, ANTONYM 0.8125     selected aggregate 0.9688
format-invariance delta 0.0062 (max 0.0722)     guard fires: 0 repetition, 0 echo
```

Five survivors against `MIN_DISJOINT=3`. **The run is no longer blocked.**

**Realized power at the selection, unchanged bar, corrected signal mapping:**

| trajectory floor | slope SE | vs bar | sensitivity | false-positive |
|---|---|---|---|---|
| 0.0000 | 0.0053 | 0.35x | 0.996 | 0.000 |
| 0.0198 (est) | 0.0072 | 0.48x | 0.973 | 0.000 |
| 0.0300 | 0.0092 | 0.60x | 0.937 | 0.001 |
| 0.0500 | 0.0136 | 0.89x | 0.840 | 0.013 |

Robust across the whole band the trajectory floor could occupy. The gain comes from two places: 145
selected items instead of 48, and a clean aggregate of 0.9688 instead of ~0.86, which cuts the
binomial variance sharply. Note the binding constraint has now MOVED -- sampling SD is 0.0211 against
a trajectory floor around 0.0198, so further item growth has diminishing returns and the
seed-trajectory term dominates.

**What is still owed before a freeze.** This is a calibration pass, not a licence to run:
1. The 33 review defects were addressed by SELECTION (drop ORDINAL, trim SEQ) and by validation, not
   by item-level repair. The four CLEAN banks were validated mechanically, but the reviewers' MINOR
   flags on them -- over-permissive variants ('delhi' for New Delhi), the ELEMENT symbol-derivability
   observation -- were NOT individually adjudicated.
2. `--calibrate` does not validate golds. A gold wrong in the same direction the base model is wrong
   still reads as capability failure.
3. The trajectory floor remains unidentified; the power table is a band, not a point.
4. Panel #8's other owed items are untouched: absolute-denominate `INJECTION_GRID`, drop `dose_slope`
   from the COUPLED verdict string, add `--dry` fixtures that can detect a threshold change.
