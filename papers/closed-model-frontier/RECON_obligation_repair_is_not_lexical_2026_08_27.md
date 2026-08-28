# RECON — the obvious repair to the obligation predicate does not generalise

Fathom Lab · 2026-08-27 · **RECON. No preregistration was frozen and none should be.** Receipt:
`oath_obligation_repair_census.json`.

`RESULT_oath_verified_channel_internal_2026_08_27.md` put the obligation predicate at the top of
the repair queue: the miss rate is `0.4267` inside this laboratory and `0.4067` outside it, and
keeping the contract buys nothing against it. This is the reconnaissance that was supposed to
tell a preregistration what to freeze. It says: **not this.**

## The diagnosis, which is unambiguous

The blind panels of 2026-08-27 adjudicated every abstained token they were shown — the first
hand-labelled miss set this lane has ever had. Splitting the misses by *why* the verifier passed
over them:

* **External: 61 of 61** occurred because the trigger vocabulary never fired on the line at all.
* **Internal: 24 of 32.** The other 8 fired a trigger and were abstained downstream, 7 of them by
  the `spec-or-historical` clause.

So the miss is overwhelmingly a **coverage** problem in one list of words, not a downstream
suppression problem. Which makes the repair obvious: widen the vocabulary.

## In sample, widening looks like a triumph

Candidates defined mechanically — *add the K words most frequent among adjudicated misses and
rare among adjudicated non-claims* — scored against the null rule, which is **obligate every
number, no vocabulary test at all**:

| rule | misses caught | non-claims obligated | cost per catch |
|---|---|---|---|
| add top 5 words | 16 | 2 | 0.125 |
| add top 20 words | 36 | 4 | 0.111 |
| add top 40 words | 55 | 11 | 0.2 |
| **null: obligate every number** | **85** | **127** | **1.494** |

The top-20 rule retains `0.4235` of the null rule's catches for `0.0315` of its cost. Every
candidate beats the control on the deciding column. `styxx-discriminates` reports `holds: true`.
On this table the repair is ready to preregister.

## Held out, it collapses

The table above ranks the words on the same tokens it scores them on. Fitted instead on one half
of the **documents** and scored on the other — split by document, because tokens from one README
share vocabulary and a token-level split leaks the answer across folds:

| rule | misses caught | recall | share of the null's catches |
|---|---|---|---|
| add top 5 words | 0 | 0.0 | 0.0 |
| add top 20 words | **1** | **0.012** | **0.0118** |
| add top 40 words | 4 | 0.047 | 0.0471 |

The rule that looked like it caught two in five misses catches **one**. The words that separate
misses from non-claims in one set of documents do not transfer to another set, and the in-sample
table was measuring the fit, not the rule.

**Vocabulary widening is dead**, and it would have looked like a success to anyone who stopped at
the first table.

## The instrument built to prevent exactly this did not prevent it

`styxx.discriminates` shipped on 2026-08-26 to stop a cycle freezing a bar against a column that
cannot fail. Run here it reports **`holds: true` on both tables** — in-sample and held-out alike.
Two limits, now documented in the module and pinned by tests:

1. **A candidate can beat a control on a COST column by doing almost nothing.** The held-out
   top-20 rule costs `1` against the null's `127`, so it "beats" it — while catching one token in
   eighty-five. The module's own prior advice was *"declare the cost column as deciding in
   cost/benefit designs"*, and that advice is half a rule. A first attempt to flag this
   automatically flagged **every** candidate, including the good ones, because in a cost/benefit
   design the null rule wins the benefit column by construction. So the module now reports
   `share_of_control` per candidate instead of judging: `0.0118` beside `0.0079` says at a glance
   what a cost-column pass conceals. Reported, not enforced — how much benefit is enough is the
   caller's question.
2. **It cannot see overfitting, and no version of it will.** It asks whether a candidate beats
   doing nothing. Whether the candidate was fitted on the data it is scored on is a different
   question, and a discrimination check is not a substitute for a held-out split.

## A note on being wrong in both directions in one hour

The first look at this data was a crude frequency filter, and it concluded there was *"almost no
vocabulary that separates the two"*. The census then contradicted that flatly — in sample the
separation is excellent. The held-out split then vindicated the original conclusion, for a reason
the original had no evidence for.

Being right by accident is not being right. The eyeball filter was too strict and got the answer
for the wrong reason; the census was properly built and got the wrong answer because it scored a
fit; only the held-out split measured the thing. All three are in the record because the sequence
is the useful part.

## What is owed

The repair must be **structural rather than lexical**. Every instance in
`SYNTHESIS_mention_and_use_2026_08_26.md` is a marker standing in for a class, and a trigger word
list is exactly that — a marker on a line, standing in for *this sentence asserts a measurement*.
Widening the marker cannot repair a predicate whose defect is that it is a marker.

What a structural candidate might read instead — sentence position, grammatical role, the
presence of a comparative, whether the number is the object of a reporting verb — is not proposed
here and is not scored here. **This RECON licenses no design.** What it establishes is that the
cheapest available design is dead, measured, and must not be preregistered.

The 8 internal misses that *did* fire a trigger are a separate and much smaller problem, 7 of them
in one clause, and they are recorded rather than pursued.

---

*The obvious repair was tested before it was frozen, which is the only reason it cost a morning
instead of a cycle.*
