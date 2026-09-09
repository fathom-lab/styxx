# Calibrating the sampled challenge, and the miss list it produces

Fathom Lab · 2026-09-09 · **A measurement about a detector, on one published artifact.** It does
not show the challenge mechanism works, because the challenge mechanism has still never been run
by a second party. It shows what a k-item challenge would and would not catch if one were, and
names the forgeries it would not reach. Not sworn.

Everything below reads `papers/v8/first_verdict_2026_09_09/` — the verdict on
`google/gemma-2-2b-it`, 64-prompt battery, greedy, 16 new tokens, five bf16 floor runs, one fp16
comparison, a seven-entry log. No model was run. Receipts beside this file: `artifact.py`
(loaders and the two detectors), `forgeries.py` (the mutation classes), `calibrate.py` (the sweep,
writes `results.json`), `verdict_flip.py` (the worked attack), the captured output of both as
`transcript_calibrate.txt` and `transcript_verdict_flip.txt`, and
`tests/test_sampled_challenge_calibration.py` (25 guards, all passing). Every number in this
document is in one of those files.

## Why this exists

`THE_BOUNDARY_2026_09_09.md` concludes that a check on bytes an issuer wrote can only ask whether
that party contradicted itself, and that the challenge — a second party running the same battery
and putting their own bytes in the log — is the only part of the system that introduces a byte the
issuer did not write. It also says a challenge costing a full battery re-run is a challenge nobody
performs. So the question is how much of a re-run is enough, and this lab's own standing rule is
that an agreement number without its detection power is not a number.

A sampled challenge is a detector. This is its calibration.

## The two detectors, and the third

| | what it costs | what it compares |
|---|---|---|
| **A · k-item output-digest challenge** | k forward passes by a second party | its own `output_sha256` and `token_ids_sha256` against the certificate's, on k of 64 items drawn uniformly |
| **B · internal-consistency battery** | nothing | one certificate against itself — `class_two_empty_2026_09_09/first_claim_battery.py`, imported and called here, not reimplemented |
| **C · k-item full-record challenge** | k forward passes | the same draw, plus `seq_logprob` and the recorded `topk` distributions |

C is an extension proposed by this calibration, not part of the published system. It is here
because A turns out to be blind to an entire half of the forgery space, and because the artifact
contains the evidence needed to say what C would cost in false accusations.

Detection for A and C is a set intersection, so its probability is exact:
`1 − C(64−b, k) / C(64, k)` where b is the number of items whose compared fields moved.
`calibrate.py` re-derives three of those numbers from 20,000 draws each and prints both
(0.125000 / 0.122000, 0.584869 / 0.580950, 0.421933 / 0.412250) so the formula is checked rather
than trusted.

## False alarm first, because it decides the protocol

The five bf16 floor runs are five honest executions of the same battery under assignments the
plan fixed. Each of the ten pairs stands in for an honest challenger's re-run. Measured:

| challenger's batch size | pairs | items disagreeing on the output digest | items disagreeing on the full record |
|---|---|---|---|
| same as the reference | 3 | **0, 0, 0** | **0, 0, 0** |
| different | 7 | 2, 1, 3, 2, 2, 1, 1 | **64, 64, 64, 64, 64, 64, 64** |

Two consequences, and the second is the sharper one.

**A challenge must be run at the reference run's batch size.** At the same batch size this machine
reproduced itself exactly: 192 item comparisons, zero disagreements on either comparison, so the
false-alarm rate is 0 at every k on both A and C. Off-batch, A false-alarms with probability
0.5849 at k = 16 against an honest issuer — which is the same number as its detection power
against a three-item fabrication, so off-batch the detector cannot distinguish the two cases at
all. And C off-batch false-alarms with probability 1.000 at k = 1: every one of 64 items disagrees
on `seq_logprob` and `topk` across every cross-batch pair. **The fine channels are unusable as an
exact comparison unless the batch size is fixed**, which is a protocol requirement the challenge
mechanism does not currently state.

This also means the batch-size field is load-bearing for the challenge in a way it is not for
anything else, and a challenger who cannot reproduce the reference batch size cannot run detector
C at all.

## The population

15 mutation classes, 72 (class, f) cells, 200 forgeries per cell, 14,400 forgeries, 0 vacuous
draws after the generator was fixed to avoid them. Target: the bf16 canonical fingerprint —
64 items, **37 with a recorded distribution for every emitted token, 27 whose generation runs past
the last recorded distribution, 193 emitted tokens carrying no distribution at all**. (Across all
five published fingerprints the corresponding figures are 185 / 135 / 965, reported by the
first-claim battery.) The honest certificate passes every predicate of detector B, which is the
control.

Writing the mutations found two format holes that were not in the brief:

**H1 — nothing ties `output_text` to `token_ids`.** The battery digests each separately (P3, P4)
and no predicate detokenizes. So the rendered answer a reader sees can be replaced wholesale while
every number stays honest — and the `exact` channel does not notice either, because
`styxx/v8/distances.py` compares `token_ids`, not text. The `output_sha256` field is therefore
covered by *no* verdict channel and by *no* internal predicate beyond its own self-consistency.
Detector A is the only thing in the system that looks at it.

**H2 — `seq_logprob` is pinned by equality only where every emitted token has a recorded
distribution.** Past that, P8b gives one inequality: the score may not *exceed* the recorded prefix
sum. On the 27 partial items of this certificate the honest score already sits below that ceiling
by between 0.000463 and 5.524638 nats, mean 1.236859, and the whole interval below the prefix is
writable. This is the gap the brief named; the measurement below is how large it is.

## Detection

`internal` is the fraction of forgeries detector B catches. The k columns are detector A's exact
detection probability. `resid` is the fraction of forgeries that **neither** reaches at any k.
Full table in `results.json`; the abridged version:

| class | f | internal | resid | k=1 | k=4 | k=16 | k=64 |
|---|---|---|---|---|---|---|---|
| fabricate_careful | 4 | 0.000 | 0.000 | 0.062 | 0.233 | 0.694 | 1.000 |
| fabricate_careful | 16 | 0.000 | 0.000 | 0.250 | 0.694 | 0.995 | 1.000 |
| fabricate_careless | 1 | **1.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| copy_output | 1 | 0.000 | 0.005 | 0.016 | 0.062 | 0.249 | 0.995 |
| swap_pairs | 1 | 0.000 | 0.015 | 0.031 | 0.120 | 0.434 | 0.985 |
| text_only_rewrite | 4 | 0.000 | 0.000 | 0.062 | 0.233 | 0.694 | 1.000 |
| text_swap | 4 | 0.000 | 0.000 | 0.125 | 0.421 | 0.914 | 1.000 |
| seqlp_tied | 1 | **1.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| seqlp_partial_under | 1 | 0.000 | **1.000** | 0.000 | 0.000 | 0.000 | **0.000** |
| seqlp_partial_to_prefix | 27 | 0.000 | **1.000** | 0.000 | 0.000 | 0.000 | **0.000** |
| seqlp_partial_over | 1 | **1.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| topk_reshape_naive | 1 | 0.755 | 0.245 | 0.000 | 0.000 | 0.000 | 0.000 |
| topk_reshape_coordinated | 64 | 0.000 | **1.000** | 0.000 | 0.000 | 0.000 | **0.000** |
| topk_partial_free | 1 | 0.470 | 0.530 | 0.000 | 0.000 | 0.000 | 0.000 |
| truncate_careful | 4 | 0.000 | 0.000 | 0.062 | 0.233 | 0.694 | 1.000 |
| truncate_careless | 1 | **1.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

The shape of it: **every class splits cleanly into one column or the other, and no class is in
both.** A mutation either moves a token — in which case detector B is blind to it and detector A
sees it with probability k/64 per moved item — or it does not, in which case detector A is blind
at every k and the question is only whether the forger kept the arithmetic straight. The two
detectors partition the space rather than overlapping, which means the challenge budget question
has a clean answer: spend nothing on the classes below, and everything on the classes above them.

**Do not spend a re-run on these.** Caught by detector B on every forgery at every f, free:
`fabricate_careless`, `seqlp_tied`, `seqlp_partial_over`, `truncate_careless`. The predicates that
fire are P8 and P8b (the arithmetic ties), and P1/P3/P4 (the stale digests).

**Spend it on these.** Invisible to detector B on every forgery, and reached only by a second
party's bytes: `fabricate_careful`, `copy_output`, `swap_pairs`, `text_only_rewrite`, `text_swap`,
`truncate_careful` — all at 1.000 by k = 64, and at 0.694 by k = 16 for a four-item fabrication.
A 16-item challenge is a quarter of a battery and reaches two thirds of a four-item fabrication;
a 32-item challenge reaches 0.943.

## The miss list

**26 of 72 cells contain forgeries that neither detector reaches. 18 of them miss on every one of
the 200 forgeries generated.** Named in full:

| class | f | resid | why nothing reaches it |
|---|---|---|---|
| `seqlp_partial_under` | 1, 2, 4, 8, 16, 27 | **1.000** | H2. The score is pushed down on partial items. P8b bounds it above only; the tokens do not move, so no digest changes. |
| `seqlp_partial_to_prefix` | 1, 2, 4, 8, 16, 27 | **1.000** | H2 at its limit: the score set to the highest value P8b permits, claiming the unrecorded tail was free. |
| `topk_reshape_coordinated` | 1, 2, 4, 8, 16, 64 | **1.000** | The recorded distributions are rewritten and the score is moved to match, so P8 and P8b both hold. At f = 64 the certificate's entire confidence record is fabricated and detector A's probability is still 0.000 at k = 64. |
| `topk_partial_free` | 1, 2, 4, 8 | 0.530, 0.270, 0.060, 0.005 | Distributions rewritten on partial items only. Where the rewrite raises the prefix sum, P8b gets looser and nothing objects; where it lowers it below the untouched score, P8b fires. The split is the item's own confidence profile. |
| `topk_reshape_naive` | 1, 2 | 0.245, 0.050 | Same, without moving the score: caught on the 37 tied items by P8 and on some partial ones by P8b, missed on the rest. |
| `copy_output` | 1 | 0.005 | A copy between two items that honestly produced the *same* answer. 26 of the 4,032 ordered item pairs share both digests (0.0064), and in all 13 such unordered pairs the two items' `seq_logprob` and `topk` still differ — so the copy is a real forgery that no digest can see. |
| `swap_pairs` | 1 | 0.015 | The same coincidence, for a transposition. |

The three 1.000 rows are the finding. They are not partial misses at small k; they are misses at
**k = 64**, which is re-running the entire battery. A second party who reproduces all 64 prompts
and compares output digests learns nothing about any of them.

The last two rows are worth separating from the rest, because they are the only misses in the list
that a *cheap* repair reaches: a duplicate-body predicate would flag them, at a false-alarm cost
this artifact fixes at 9 honest groups covering 20 of 64 items. That trade has not been evaluated
here and is not recommended on this evidence.

## What one free float does to the published verdict

`verdict_flip.py` first reproduces all six published numbers from the certificates using
`styxx.v8.distances`, the shipped Appendix B implementation, to nine decimal places:

```
exact  distance 0.062500000  floor 0.046875000  exceeds_floor
seqlp  distance 0.058564664  floor 0.036070694  exceeds_floor
topk   distance 1.407087824  floor 2.140233900  same
```

Then it changes **one float in one item** of the bf16 canonical fingerprint. Item
`a1f4a82542815cc4`, output `` ```xml\n<item>one</item>\n``` `` — 14 tokens generated, 8 recorded
distributions, 6 emitted tokens with none. The recorded prefix sum is −0.134172570; the honest
score is −1.223275343, sitting 1.089 nats below P8b's ceiling; the write puts it at −3.511959788,
3.378 nats below. P8b holds with room to spare.

```
seqlp  distance 0.022803969  floor 0.071090944  exceeds_floor -> same
internal battery : all predicates hold -- MISSED
digest challenge : 0 of 64 items flagged, P(catch) = 0.000 at k = 64
full-record      : 1 of 64 items flagged, P(catch) = 0.016 at k = 1, 0.125 at k = 8
```

Applying the same write to all three batch-1 runs keeps run 0, run 3 and run 4 byte-identical —
the structure entry 6 of the published log already records — and produces the identical distance
and floor. Applied to all 27 eligible partial items, the seqlp distance falls to 0.008512437
against the untouched published floor of 0.036070694, a ratio of 0.236; the floor is set by the
run1/run2 pair, which the move does not touch.

**The overall verdict does not flip on that alone**, and the reason is the useful part. `exact`
still reads `exceeds_floor` at 4 disagreeing items against a floor of 3. Removing one disagreement
means writing a token id — and a token id is exactly what a challenger reproduces. Copying the
fp16 body onto one of those four items, in the same three batch-1 runs, takes `exact` to
0.046875000 against a floor of 0.046875000 — which reads `same`, because the comparison is strict —
and the overall verdict becomes `same`. The whole forgery still passes detector B. Detector A now flags **1 of 64** items:
P(catch) = 0.0156 at k = 1, 0.1250 at k = 8, 0.5000 at k = 32, 1.000 at k = 64.

So the published verdict can be turned from `exceeds_floor` to `same` by two edits, and the two
edits have completely different exposures: the seqlp half is free at every k, and the exact half
is the one thing a challenge is for. **The channel a challenge can defend is the one currently
holding this verdict up.**

The knife-edge is not an accident of this script: `styxx/v8/floor.py` rounds both sides to
`ROUND_PLACES` (9) and returns `same` on `d <= f`. `verdict_flip.py` applies the same rounding and
the same `<=`, and every number above is unchanged by it. A forger therefore has to *reach* the
floor, not pass it, which is one item cheaper than it looks.

## What this changes about the challenge

1. **Fix the batch size, or do not run detector C.** 3 honest pairs at a matched batch size
   disagreed on 0 of 64 items on both comparisons; all 7 cross-batch pairs disagreed on 64 of 64
   for `seq_logprob` and `topk`. Off-batch, C is a false accusation generator.
2. **A digest-only challenge is half a detector.** It reaches every class that moves a token at
   k/64 per moved item, and nothing else at any k. The three total-miss classes are all fields it
   does not compare.
3. **k = 16 is a defensible default for what A can reach** — 0.694 against a four-item
   fabrication, 0.995 against sixteen — at a quarter of the cost of a battery, with a measured
   false-alarm rate of 0 when the batch size matches.
4. **The free battery should run first and the challenge should skip what it caught.** Four of
   fifteen classes are caught at rate 1.000 by a predicate costing nothing.
5. **H1 and H2 are format defects, not challenge defects,** and neither is repaired by sampling
   harder. H2 is closed by recording a distribution for every emitted token, or by recording the
   tail sum. H1 is closed by any predicate that ties the text to the ids.

## Limits, and what is not established

This measures a detector against a forger model, on one certificate, from one artifact, produced
by one lab on one machine in one day. It does not show the challenge mechanism works: **nobody has
run it.** Every number here is what a second party *would* find, computed from the published bytes
on the assumption that an honest re-run at a matched batch size reproduces them — which is
supported by 3 pairs and 192 item comparisons in this one artifact and by nothing else.

The mutation classes are a guess at what a dishonest issuer would do, made by the same session
that then measured them, and a class nobody thought of is missed by construction. The miss list is
a lower bound on what is missed, never an upper one. The residual rates for `copy_output` and
`swap_pairs` depend on this battery's particular duplicate structure (9 groups, 20 items) and do
not transfer to another battery. The uniform draw is the weakest challenger: an adversarial
challenger who picks the long-generation items would do better against the classes A reaches, and
no better at all against the three that miss at k = 64.

Detector C is proposed and measured but not implemented anywhere in the system.
