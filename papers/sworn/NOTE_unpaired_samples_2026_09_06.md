# NOTE — "same seed, same size" does not give paired samples across generator versions

Fathom Lab · 2026-09-06 · A correction to a claim this lab made in its own commit history, and the
trap behind it. Nothing here is a result; it is the reason one of yesterday's readings has to be
withdrawn.

## The claim being withdrawn

Commit `87581ccd` is titled *"the aperture closure made detection WORSE, and that is the finding"*
and reports:

| | caught | missed | rate |
|---|---|---|---|
| grammar v1 | 41 | 29 | 0.5857 |
| grammar v2 | 39 | 31 | 0.5571 |

It attributes the fall to the widening, offers dilution as the mechanism, and calls the five newly
missed mutations *displaced*. **The comparison does not support any of that**, and the arithmetic
that looked like evidence was comparing two different populations.

## Why

`differential.case(seed, index)` builds a document by drawing from a `random.Random` seeded on
`(seed, index)`. Every draw advances that stream. The aperture widening added new draws — a
`captured_at` choice where there had been a constant, an `if r.random() < 0.06` for uppercase
digests, a longer `harness` list, an `if r.random() < 0.10` block for `authored_sha256` — and **a
new draw re-randomises everything after it.**

Measured, not argued: of the first 500 cases at seed 20260905,

**only 51 produce the same document under both grammars. 449 of 500 are different documents.**
(Receipt: `conformance/sworn/generator_pairing.json`, which reloads both generators from git by
commit rather than from the working tree, so it re-derives in any checkout carrying the history.)

So "grammar v1 at seed 20260905" and "grammar v2 at seed 20260905" are not the same 150000 inputs
with some enriched. They are two samples from two distributions that happen to share a seed. A
before/after difference in detection rate over them is not a paired comparison and carries no
causal reading.

## What survives, and what does not

**Survives — the three closed misses.** *JCS emits a literal newline instead of the short escape*,
*manifest `core()` emits `authored_sha256` unsorted*, and *the JS manifest stops case-folding
`authored_sha256`* were missed at v1 for a **structural** reason, not a sampling one: the old
generator emitted no manifest string containing a newline, never more than one element in
`authored_sha256`, and never an uppercase digest. Those inputs did not exist at any case count.
They exist now. That is a statement about what the grammar can produce and it does not depend on
pairing.

**Survives — everything about the disagreements.** The 712 disagreements, the two defects and both
repairs were confirmed on minimal hand-built inputs that never touch the generator. Nothing there
rests on this comparison.

**Does not survive — the rate comparison.** 0.5857 against 0.5571 is two numbers from two
populations. Neither the direction nor the size of the difference is interpretable, and the
"dilution" mechanism offered for it was a rationalisation of an artefact. Dilution is real and was
measured — the payload pool went from 10 literals to 23 — but it is not what produced those five
losses, and at 200000 cases (forty times the guard's size) all five remain missed, which dilution
alone cannot explain either.

**Does not survive — "displaced".** The five are not the same mutations meeting a thinner slice of
the same inputs. They are mutations measured against different inputs entirely.

## The trap, stated for whoever hits it next

A seeded generator's output is a function of the **entire sequence of draws**, not of the values
any one draw produces. Therefore:

> **Any change to a seeded generator — including a purely additive one — changes every case the
> seed produces after the point of insertion. Fixing the seed does not fix the sample.**

This is not specific to this harness. It applies to every fuzzing or mutation study that iterates
on its generator and compares runs "at the same seed", which is the ordinary way such studies are
reported.

## The remedy

Draw each component from its **own** stream keyed by name, so adding a choice in one place cannot
move any other:

```python
def _rng_for(seed, index, part):
    return random.Random("%d:%d:%s" % (seed, index, part))
```

With per-part streams, adding a `captured_at` choice perturbs only `captured_at`; the document
body, the receipt ids and the needles keep the values they had. Runs across generator versions
become paired for every component that did not change, and a before/after detection rate becomes a
number worth reading.

That change is **not** made here. It would alter every case the harness produces and so invalidate
the receipts this leg just committed; it belongs in its own spec, with its own run, and with the
old receipts standing. This note exists so the next person does not read `87581ccd` and believe it.

---

*The measurement was fine. The comparison was not, and the difference between those two is the
whole subject of this leg.*
