> ## THE CENTRAL MECHANISM WAS BROKEN. Corrected same day, after adversarial review.
>
> **The receipt returned `OK__path_could_have_failed` on measurements that could not
> have come out differently.** Re-scoped on the 58 drafts where the conscience fired
> every single time (value literally 1.0) and on the 25 where it never fired (0.0), it
> certified "the apparatus could have returned a different answer" at 37.5% and 25.5%
> live — while its own rows correctly recorded every terminal decision term as CONSTANT.
>
> The mechanism was **pooling**. Character-level tokenisation loops inside the instrument
> generate tens of thousands of live observations with no bearing on the verdict, and
> they outvoted the handful of terms that decide. The single heaviest "live" term was
> `' ' in phrase` at n=30,174 — a compile-time property of a lexicon file, invariant
> under every possible input, certified LIVE.
>
> That is a pass verdict decoupled from the question it claims to answer, inside the
> instrument built to detect exactly that, violating this lab's own standing rule that
> *a leg which cannot fail must not gate*. It is the worst defect of the day because it
> is the one artifact whose entire purpose **is** the verdict.
>
> **Two checks now stand in front of the verdict.** `mark_item()` per unit the value
> aggregates over, refusing outright when every item produced the same outcome; and a
> phi coefficient between each term's per-item value and the outcome, refusing when no
> live term tracks the result. Restricting to adjudicative terms alone does *not* fix it
> — a tokenisation loop's `if` is adjudicative too.
>
> **`RECEIPT_receipt_gate.json` was withdrawn and re-cut.** The original was
> unauditable: the source was edited 22 minutes after it was issued, five commits
> followed, the line numbers it cited matched no version on disk, and `live_terms` were
> never serialised at all — so the evidence *for* its OK verdict could not be recovered.
> Receipts now pin the subject commit, record whether the tree was dirty, and serialise
> the live terms with their phi.
>
> Numbers below are superseded by the re-cut receipts; the argument stands.

# Falsifiability receipts: attaching to a number the proof it could have been different

**Date:** 2026-08-13. **Code:** `falsifiability_receipt.py`, `certify_conscience_rate.py`.
**First real receipt:** `RECEIPT_conscience_fire_rate.json`.

## The gap

Three literatures answer three neighbouring questions, and none answers the one a
published measurement actually needs.

| asks | about | established |
|---|---|---|
| mutation testing | would a **test suite** notice a change in the code? | since the 1970s |
| execution provenance | where did a claim's **inputs** come from? | active, with a survey |
| PROBE E (today) | could a **gate** have decided otherwise on this population? | new, narrow |

What none of them provides: **of the decision terms that actually ran while producing
this specific number, how many could have gone the other way?**

That question is not the same as "is the instrument healthy." An instrument can be
perfectly well-tested in general and still, on the particular population that produced
the published figure, run entirely through terms that never varied. The resulting number
is not noisy or weak. It is a number the apparatus could not have failed to produce, and
it reports the apparatus's constants in the units of the thing it claims to measure.

The value gives no hint. That is the whole difficulty, and it is why the selftest asserts
it: the pinned fixture returns a rate of exactly `1.0` — a clean, quotable, entirely
uninformative number.

## The mechanism

`scope()` wraps a computation. Every boolean decision term evaluated inside it is
attributed to that measurement, and each gets a verdict from the counts observed **during
that computation** rather than from the module's lifetime history. The distinction is
load-bearing: a term that varies elsewhere in the program but was pinned throughout this
measurement is dead *for this number*, and lifetime statistics hide precisely that case.

Four verdicts, and the first two are meant to be obeyed rather than logged:

- `REFUSED__no_live_terms` — every adjudicable term on the path was pinned. The
  apparatus could not have produced a different answer.
- `REFUSED__nothing_adjudicable` — nothing ran often enough to say. The receipt cannot
  speak, which is *not* the same as the measurement being sound.
- `WEAK__mostly_pinned` — below the live-fraction floor.
- `OK__path_could_have_failed` — the apparatus could have returned a different answer on
  this population.

A receipt generator that has never refused anything would itself be an instrument that
cannot fail, so `--selftest` drives one pinned measurement and one contingent one and
asserts both outcomes, including that the pinned value looks unremarkable on its own.

## The first real receipt

Applied to a number this lab actually publishes — the conscience fire rate — over
darkflobi's own logged drafts rather than a synthetic battery, because a synthetic
population would let the receipt certify a number nobody reports.

```
conscience fire rate : 0.6914  (56/81)
terms on path        : 115
live / constant      : 44 / 69   (underpowered 2)
VERDICT              : OK__path_could_have_failed
```

**The rate survives.** 38.9% of the adjudicable decision terms on its path varied during
the measurement, so the apparatus could have returned a different answer.

**And 69 of 113 adjudicable terms — 61% — were pinned throughout it.** The receipt names
them. The heaviest sat at 29,455 evaluations without ever changing value. Several are
plainly benign (`if not text` never firing because no draft was empty); that is exactly
why the receipt lists rather than judges them. The point is that this list has never
before been printed next to the number it produced.

`MEMORY.md` already carries one correction to a rate of this kind — the `sensitivity 1.0`
strike, where a gate firing 6/6 on attacks *and* 6/6 on benign had its two numbers
counted once and named twice. That was caught by argument, days later. A receipt asks the
apparatus directly, at the moment of measurement.

## Second receipt: the gate shipped four hours earlier

`execution_receipt_gate.py` was extended today after returning zero claims on a status
report containing a fabricated work item. Its validation set went from four cases to six
and all six pass — which establishes it *can* fire and *can* stay quiet on hand-built
examples, and establishes nothing at all about its behaviour on real traffic. That
distinction is the original defect restated: the gate had been validated against
phrasings its author could think of.

```
receipt-gate fire rate : 0.1852  (15/81)
terms on path          : 13
live / constant        : 8 / 5   (underpowered 0)
VERDICT                : OK__path_could_have_failed   (61.5% live)
```

A non-degenerate rate on real drafts, with a majority-live decision path. But the receipt
also names something the six-case validation set could not have surfaced:

```
CONSTANT_FALSE n=11  review L246 [or] len(ev_terms) < TOPIC_POWER_FLOOR
```

**The `TOPIC_POWER_FLOOR` abstention never fires on real traffic.** That branch was added
hours earlier specifically to stop the topical-evidence check from vetoing on sources too
terse to discriminate — it is the fix that rescued a validated negative case. On 81 real
drafts, every evidence blob carried at least eight content words, so the branch that
makes the check safe is exercised only by the selftest that motivated it.

That is not a defect. It is a **coverage fact about a safety branch**, and there was no
way to learn it from a passing test suite: the suite exercises it by construction. Nobody
would have gone looking.

## What a receipt licenses

**Licensed:** the apparatus could (or could not) have produced a different value on this
population. For `REFUSED`, that is disqualifying on its own and no further argument is
needed.

**Not licensed:** that an `OK` number is correct. The receipt certifies *contingency*,
not accuracy — a measurement can be perfectly falsifiable and perfectly wrong.

The asymmetry is deliberate and is recorded in every receipt: attribution is by
observation delta, not data-flow taint, so a term evaluated inside the scope is credited
even if its result never reached the value. That bias runs toward making paths **look**
falsifiable. Therefore **`REFUSED` is strong evidence and `OK` is weak evidence** — the
error points away from false alarms and toward false comfort, which is the direction that
needs stating out loud.

## Why this is the part worth keeping

`PRIOR_ART_2026_08_13.md` deflated the day's novelty claim honestly: mutation testing is
the origin of the falsifiability idea, the provenance literature already types claims,
and most of what was built is reinvention. What survived that assessment was narrow —
falsifiability analysis applied to *measurement instruments* rather than test suites.

This is the first thing built on that residue that the neighbouring literatures do not
already do. Mutation testing certifies a suite; provenance certifies a source; a
falsifiability receipt certifies **a value** — and it is refusable, which is the only
property that makes a certificate worth anything.

## Reproduction

```
python falsifiability_receipt.py --selftest
python certify_conscience_rate.py --log <turn log jsonl> --json receipt.json
```

Instrumentation must be installed before the subject package is imported; otherwise the
gates run uninstrumented and the receipt certifies a path it never observed, producing a
confident `REFUSED__nothing_adjudicable` that would read as a finding.
