# ADDENDUM — three limits on the disjoint-worlds reads: what the battery and the held-out set do and do not license

Fathom Lab · 2026-09-01 · a bounding document, not a retraction.

**These three limits bound the arc's published reads. They do not overturn them.** The gemma
cross-family paired read of 0.7857 (`b31v2_result.json`) and the label-free read of 0.5714
(`b34v3_result.json`, 0.5263 on the 57 genuinely-unseen concepts per
`b34v3_fresh_split_addendum.json`) were computed by the code that is committed, against gates
frozen before the runs, with nulls at or near chance. Nothing below contradicts any of those
numbers, and nothing below is evidence that the reads are artifacts. What the three limits do is
narrow the population those numbers describe and name one claim whose label is wider than its
method. Each was verified against the committed code for this document rather than taken from a
summary; where our verification disagreed with the account that prompted it, the verification is
what is written here.

Nothing in the 2026-08 arc is edited by this addendum. `run_g0clear.py`, every `*_result.json`,
every certificate and every seal stay exactly as committed. The successor work lives in new files:
`build_concept_pool.py`, `concept_pool.json`, `PREREG_b52_pooled_battery_2026_09_01.md`.

---

## Limit 1 — the selection that produced the battery is not recorded, so the extraction term is UNCHECKABLE

### What is in the code

`run_g0clear.py` line 31 opens a triple-quoted string literal `_BANK`, which runs to line 65 and
holds **465 whitespace-separated words** typed directly into the source across category blocks
(animals, fruit, food, tools, vehicles, furniture, clothing, nature, materials, buildings,
instruments, emotions, roles, body parts, colours, abstractions). Lines 66-67 are the entire
filtering chain:

```
_seen = set()
CONCEPTS = [c for c in _BANK if not (c in _seen or _seen.add(c))]
```

An order-preserving dedup. Verified by re-running the literal through `.split()` and a counter: it
removes exactly three words — **chicken, orange, mushroom** — each of which the author wrote into
two different category blocks. 465 becomes 462.

There is nothing else. Verified by reading the whole 188-line file: **no tokenization filter, no
check against any target model's vocabulary, no frequency threshold, no part-of-speech filter, no
representation-quality gate.** `CONCEPTS` is used unchanged from line 67 onward, and the only
other thing that ever touches it is `CONCEPTS = CONCEPTS[:40]` under `--smoke`.

The docstring at line 5 and the receipt field `parent_baseline.N` in
`g0clear_result_llama3b.json` both record the parent bank as **110**. The parent bank at
`run_thought_transfer.py` lines 31-40 contains **121** words, with no duplicates. Verified by
counting. All 121 appear in the 462. **341 words are net new, and how they were chosen is recorded
nowhere.**

### Why this matters, stated precisely

A read scored over a battery has two terms a reader needs: how well the reader identified the
items, and what share of candidate items reached the battery in the first place. Call the second
the **extraction term**. For these reads it is not merely unmeasured but **UNCHECKABLE**: no pool
was recorded, no rejection log exists, there is no source corpus and no seed to replay, so no
later work can reconstruct what was considered and refused. Provenance of the *file* is complete —
it is in git, it is byte-stable, every receipt names it. Provenance of the *selection* is absent.

This is not evidence that the battery is biased. It is the statement that the question cannot be
asked of it. UNCHECKABLE is the verdict, and the absence of a rejection log is not itself a
contradiction of anything.

### One measurable consequence, since a filter that was absent can be priced after the fact

The extraction *chain* is unrecoverable, but the effect of one filter it lacked can be measured
now. Using the tokenizers of the four models this arc compares, **35 of the 462 words are
multi-token in Llama-3.2 (3B and 1B) and in Qwen2.5**; 2 of 462 are multi-token in gemma-2. **6 of
the 70 held-out concepts** are multi-token in at least one target: apricot, gorilla, tambourine,
peacock, cactus, melon. (Reproduced by `build_concept_pool.py`, which records the full list in
`concept_pool.json` under `comparison_to_2026_08_battery`.)

The consequence should not be overstated, and the smaller version is the true one.
`CONCEPT_TEMPLATES` never puts the concept at the end of a sentence — every template continues
past the `{c}` slot — and `extract_multi` reads the **last** token of the whole sentence. So a
multi-token concept does not mean the read was taken on a word-piece. What it means is that
`template.format(c=concept)` and the neutral contrast `template.format(c="object")` are then
different lengths, so the two hidden states being subtracted sit at different absolute positions;
and that a string which is one token in gemma and two in Llama is not the same stimulus in the two
models being compared. Both are confounds of unmeasured size. Neither is shown here to have moved
any published number, and no claim is made that it did.

### What would have made it checkable

A pool recorded as an artifact before the selection, and a filter chain that counts what it
removes at every stage. That is `build_concept_pool.py`, committed alongside this addendum. It
draws 547,373 candidate types by a stated rule from a version-pinned corpus, applies eight named
filters each recording its rule, its removal count and a sha256 of what it left, and samples the
battery from the survivors under a recorded seed. Its extraction term is
**462 / 547,373 = 0.00084403**, with 462 / 5,199 = 0.088863 of the eligible set. The same quantity
for the 2026-08 battery is UNCHECKABLE, and `concept_pool.json` records it with that word rather
than with a number.

---

## Limit 2 — the held-out set has been scored repeatedly, and the arc's hyperparameters were selected on a sibling split

### What is in the code

`split_concepts(seed=0)` in `run_g0clear.py` partitions the 462 concepts into 323 anchors, 69
`SEL_dirs` and 70 `FIN_dirs`. Verified by reproducing the split: 323 / 69 / 70.

Nine committed scripts import that function and score its `fin` set. Verified by reading each
one's use of the variable, not by grepping the name:

| script | committed receipt | what it scores on FIN |
|---|---|---|
| `run_g0clear.py` | `g0clear_result_llama3b.json` | the locked `pc_cos` positive control |
| `run_b31v2.py` | `b31v2_result.json` | paired-MLP top-1, the 0.7857 cell |
| `run_b34.py` | `b34_result.json` | label-free top-1 |
| `run_b34v2.py` | `b34v2_result.json` | label-free top-1 |
| `run_b36.py` | `b36_result.json` | steering gain per concept |
| `run_g0_stage1.py` | `thought_transfer_g0clear_result_llama3b.json` | zero-anchor read top-1 and steer gains |
| `run_g0_stage1b.py` | `writelayer_decouple_result_llama3b.json` | native and transferred steer gains |
| `run_read_verify.py` | `read_verify_result.json` | read top-1 across a sweep |
| `run_rung2_read.py` | `rung2_read_result_Qwen2_5_1_5B_Instruct.json` | read top-1 |

That is **nine, not seven**; `run_g0_stage1.py` and `run_g0_stage1b.py` also score it and were not
in the account that prompted this check. Two further scripts, `run_g0_stage3_truthaxis.py` and
`run_rung3_steerlayer.py`, take only `tr` from the split and are not on the list.

### The map is never fitted on FIN. Two hyperparameters are selected on a slice of it.

The cross-model map is fitted on anchors in every one of the nine. **In the two headline read
experiments — `run_b31v2.py` and `run_b34v3.py` — `fin` appears only in scoring**, and that is
worth saying plainly because the loose word for this problem is "leakage" and this is not leakage
in the training sense.

Three write-side scripts do more than score it, and the account that prompted this check did not
mention it:

- `run_g0_stage1.py` line 85: `alpha = P.lock_dose(..., fin[:5], ...)` — the injection dose is
  chosen on the first five held-out concepts.
- `run_b36.py` lines 99-100: `dose_pool = list(fin)[:8]`, then `lock_dose` over it.
- `run_g0_stage1b.py` lines 26, 42, 68: `N_NATIVE_SEL = 12` with the comment *"held-out concepts
  used to pick the steer-optimal layer"*; `native_sel = fin[:12]`; `f_star` is the write fraction
  maximising mean gain over `native_sel`.

In all three, `lock_dose` and the layer sweep score the **native** target direction `vecsB[c]` —
not a transferred one — so neither the map nor the transfer quantity is tuned on held-out data.
The bound is narrower than "leakage" and it is real: a hyperparameter was selected using items
that were subsequently scored, in the write-side experiments. **The two headline reads are clean
on this point.**

### The layer and k the arc runs on were selected on SEL_dirs

`g0clear_result_llama3b.json` records `locked: {layer: 11, k: 150}`, chosen as the argmax of
`pc_cos` over `SEL_dirs`. **Seventeen committed scripts in this directory hard-code that exact
receipt filename and read `k` (and in several cases the layer) out of it** — counted by grep and
confirmed by reading each call site: `run_b31v2`, `run_b34v2`, `run_b34v3`, `run_b35a`,
`run_b35b`, `run_b35c`, `run_b35c_nullreplicate`, `run_b37`, `run_b38`, `run_b39`, `run_b41`,
`run_b42`, `run_b44`, `run_b46`, `run_read_verify`, `run_rung2_read`, `diag_b35c_collapse`. Three
more (`run_g0_stage1`, `run_g0_stage1b`, `run_rung3_steerlayer`) read a `g0clear_result_<tag>.json`
chosen by command-line tag, which may resolve to one of the axis receipts instead.

`run_b34v3.py` draws a fresh split at seed 343 with `n_fin = 70` and `n_tr = 462 - 70 = 392`, so
its training set is every concept not in its held-out set. Reproducing both splits: **60 of the 69
`SEL_dirs` concepts fall inside b34v3's 392-concept training set, and 9 fall inside its 70-concept
held-out set.** The claim that SEL_dirs "sits inside" b34v3's training set is therefore true of 60
of 69 and false of 9, and the precise version is the one that belongs in the record. The
hyperparameters b34v3 runs under were selected on concepts that are, for the most part, its own
training items — and for 9 items, its own test items.

Two smaller observations, recorded because they are checkable and neither is a defect: the locked
pair sits at two corners of the swept grid (`layer 11` is the lowest of `[11, 14, 17, 20]`, `k 150`
the highest of `[60, 90, 120, 150]`), so the selection surface's optimum may lie outside what was
swept; and `grid_max_sel` equals `pc_cos_sel_at_locked` at 0.9131, as it must by construction.

### The bound, stated without inflation

A set that is scored across nine committed experiments is not a fresh held-out set for the ninth.
Each individual experiment's gate was frozen before its own run, and the arc's per-experiment
discipline is not in question. What accumulates across the arc is a different quantity: the
program's own choices — which method to keep, which successor to spawn, which map class to
abandon — were made by people who had seen this set's numbers many times. That is a bound on the
*arc*, not a defect in any single run, and no correction to any single number follows from it.
It is also not measurable retrospectively, so it is stated and left stated.

The arc already carries the precedent for handling this properly. `run_b34v3.py` was written
specifically to draw a fresh split, its frozen prereg asserted the new held-out set was disjoint
in membership from `split_concepts(0)`, verification falsified that at 13 of 70, and
`b34v3_fresh_split_addendum.json` persists the true overlap together with the decisive read
recomputed on the 57 genuinely-unseen concepts (gemma 0.5263, llama 0.6667). We reproduced the
split independently and obtained 13 of 70, matching the committed receipt exactly. The prereg was
left frozen and wrong, and the correction was published rather than buried. That is the right
handling, and it is the reason the successor prereg
(`PREREG_b52_pooled_battery_2026_09_01.md`) measures freshness with a gate instead of asserting it.

---

## Limit 3 — the "open-vocabulary" readout scores against a closed list of 462

### What is in the code

`run_b35c.py` builds its candidate set at line 46:

```
vocab = fin + tr                             # the full open candidate set
```

`fin` is 70 held-out concepts, `tr` is the remaining 392 — the same hand-typed 462 imported as
`from run_g0clear import CONCEPTS as C`. `read462` then takes, for each held-out query, the
`argmin` of the distance from the mapped source point to `vocabB`, the 462 target-space points.
Chance is recorded as `1/462 = 0.00216`.

**So "open vocabulary" here names a 462-way closed-set identification, not open recall.** The
candidate set is larger than the 70-way set it replaced, and it now includes the anchors the map
was fitted toward, which is genuinely the adversarial direction. It is still a closed list, and it
is still the same hand-authored list whose selection provenance Limit 1 records as UNCHECKABLE. An
open-vocabulary readout in the ordinary sense would score against a vocabulary the battery did not
define — a tokenizer vocabulary, a frequency list, an unrestricted lexicon — and no committed
script in this directory does that.

### Being fair about what was claimed

The code says what it does. `run_b35c.py`'s own docstring states that each query is scored
"against ALL 462 concept points in the target space", and
`PREREG_b35c_open_vocab_2026_08_03.md` states it too: *"the query is scored against all 462
concepts — the 392 anchors the MLP trained toward plus the 70 held-outs"*, with the new chance
written out as 1/462. Nothing was concealed at any point. The overclaim is in the **name**: the
prereg's framing sentence is "Real reading has no shortlist", and what the design then does is
replace a 70-item shortlist with a 462-item shortlist drawn from the same literal.

`b35c_result.json` returned `INVALID__null_artifact` — `G2_null` failed at `max_null462 = 0.01429`
against a bar of 0.00648, exactly the single-coincidental-hit failure the prereg predicted in
advance and said it would not respond to by moving the bar. **No open-vocabulary claim is
published from this run**, and `FINDING_b35bc_generality_invalids_2026_08_03.md` carries the
invalids. The limit recorded here is therefore about vocabulary in this arc generally, not about a
standing published number: wherever the phrase "open vocabulary" appears in this program's prose,
it should be read as "462-way closed set over the 2026-08 battery" until a run scores against a
vocabulary the battery did not author.

---

## What is now measurable that was not

Three artifacts are committed with this addendum.

- **`build_concept_pool.py`** derives a candidate pool from a version-pinned corpus by a stated
  rule, then applies eight named filters, each recording what it removed and why and what that
  cost. Every stage carries a sha256 of its survivors. The battery is a seeded sample of the
  eligible set, not a choice.
- **`concept_pool.json`** is that record: pool 547,373 → eligible 5,199 → battery 462, survival
  ratio 0.00084403, byte-reproducible across processes (verified by re-running under different
  `PYTHONHASHSEED` values and comparing bytes).
- **`PREREG_b52_pooled_battery_2026_09_01.md`** preregisters the read on that battery with the
  extraction term reported beside it, a fresh split seed, a measured rather than asserted
  freshness gate, six null cells, and a two-sided gate requiring the pooled read to land within
  0.15 of the published 0.7857 — a bar derived from the binomial standard error at n = 70 rather
  than chosen. It commits in advance to publishing a failure as a failure, and it names the
  outcome in which the hand-authored battery turns out to have been load-bearing as the most
  informative thing the design can produce.

None of this recovers an extraction term for the existing reads. That number does not exist and
cannot be made to exist, and the honest form of the sentence is that these reads are real,
preregistered, null-controlled, and unpriced on one axis that their successors will be priced on.
