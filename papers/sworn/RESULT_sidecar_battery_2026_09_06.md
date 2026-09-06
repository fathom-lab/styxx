# RESULT — the sidecar layer refused nothing, and the verdict it produces means less than its name

Fathom Lab · 2026-09-06 · Spec: `SPEC_sidecar_battery_v01_2026_09_06.md`, frozen before the battery
was written. Catalogue: `conformance/sworn/sidecar_battery_catalogue.json`, committed before the
run. Receipts: `conformance/sworn/sidecar_battery.json` and
`papers/sworn/sidecar_battery_outcomes.json`. **This document is itself sworn.**

## What was asked

`RESULT_suite_power_2026_09_06.md` measured the sidecar layer at 6 killed of 13 and found both of
its injection boundaries undefended. That was a statement about the tests. This asks the question
about the code: **is there an attack?**

Six attackers took a surface each, an adversary hunted what they left alone, and every prediction
was recorded before anything was executed.

## The first result is the plainest one

<sworn r="path:papers/sworn/sidecar_battery_outcomes.json#/runnable" k="numeric">49 attacks ran,</sworn>
and
<sworn r="path:papers/sworn/sidecar_battery_outcomes.json#/refused_by_load_sidecar" k="numeric">load_sidecar refused 0 of them.</sworn>

Self-minted manifests, transplanted identities, forged commits, smuggled tags, receipt entries whose
shape `Manifest.add()` would reject — every one was accepted. `load_sidecar` is genuinely strict, but
it is strict about **shape**, and none of these attacks were the wrong shape. It also never crashed:
<sworn r="path:papers/sworn/sidecar_battery_outcomes.json#/crashed_not_refused" k="numeric">0 attacks produced anything other than a clean refusal or a clean acceptance,</sworn>
which is the promise its docstring makes and keeps.

By the criterion frozen in the runner before the battery existed — accepted **and** the rendered
document does not re-canonise to the span table that was validated —
<sworn r="path:papers/sworn/sidecar_battery_outcomes.json#/succeeded_by_frozen_criterion" k="numeric">10 succeeded.</sworn>
The round-trip guarantee `to_sidecar` asserts in the honest direction does not hold in this one, and
`text_smuggling` broke it five times out of seven.

## My criterion was too narrow, and the number it missed is the one that matters

I froze a round-trip test because the attack I had already confirmed was a round-trip attack. The
adversary attacked semantics instead, so its work is filed as *accepted and faithful* — technically
true and beside the point.
<sworn r="path:conformance/sworn/sidecar_battery.json#/counts/predictions_correct" k="numeric">7 of the 49 predictions matched the observed outcome,</sworn>
and most of that gap is my definition rather than their aim.

The number the definition missed:
<sworn r="path:papers/sworn/sidecar_battery_outcomes.json#/render_to_SWORN_HELD" k="numeric">42 of the 49 attacks render to a document that verifies SWORN-HELD,</sworn>
and
<sworn r="path:papers/sworn/sidecar_battery_outcomes.json#/render_to_SWORN_HELD_with_nothing_held" k="numeric">14 of those hold nothing at all</sworn>
— every sworn span UNRESOLVED, HELD zero.

## The finding

Confirmed on a minimal hand-built input rather than inferred from the battery:

```
<sworn r="r1" k="numeric">the audited loss is 0.</sworn>      with no manifest
  ->  span UNRESOLVED / manifest_absent,  HELD=0,  document_verdict = SWORN-HELD
```

The ladder is `elif counts["FAILED"] == 0 and counts["MALFORMED"] == 0: SWORN-HELD`. **`UNRESOLVED`
is never consulted.** A document in which nothing was checked carries the same headline as one in
which everything held.

That is the conflation this module's own doctrine refuses, four lines from the top of the same file:
*"a document that swore nothing is `UNSWORN`, never 'no failures'."* The principle is applied to
`sworn_total == 0` and not to `unresolved == sworn_total`.

**It is author-reachable with one string.** A manifest rung the verifier does not know makes every
span UNRESOLVED with reason `rung_unknown`, and that fires *before* the receipt id is looked up. Same
document, same receipt, a sentence saying the loss is 0 against a receipt saying 4200000:

| manifest rung | span | document |
|---|---|---|
| `L2` | FAILED — `value_mismatch` | SWORN-FAILED |
| `L3` | UNRESOLVED — `rung_unknown` | **SWORN-HELD** |

Nothing forged, nothing malformed, nothing `load_sidecar` would refuse.

## It catches honest documents too, which is how it was found

Verifying a real committed RESULT from this corpus with `--repo .` and no `--commit` resolves
nothing: held=0, unresolved=10, **SWORN-HELD**. With the commit its sidecar names: held=10,
unresolved=0, SWORN-HELD. The same document, verified two ways, prints the same verdict — one having
checked ten things and the other having checked nothing.

This is not an exotic attack. It is the ordinary way a reader would run the tool.

## What is repaired, and what is deliberately not

**Repaired:** the headline now warns, distinguishing *nothing was checked* from *N of M did not
hold*. The guard is watched to fail — remove the warning and 3 of its 6 tests fail, and the two
silence controls keep it from becoming noise a reader learns to ignore.

**Not repaired, and left to the operator:** renaming the verdict. `SWORN-HELD` should require
`UNRESOLVED == 0`, with a distinct state for *nothing failed but not everything resolved*. That is a
breaking change to a published vocabulary that every consumer of `document_verdict` depends on. The
blast radius was measured rather than guessed: of 39 committed sworn receipts in this corpus, **one**
is SWORN-HELD with unresolved spans, and it is a fixture named `sworn_action_sample.UNRESOLVED`. No
real document has both HELD and UNRESOLVED spans. The change is small and it is still not mine to
make.

## What this does not say

**That the sidecar layer is the problem.** The battery was aimed there and the worst finding is not
there; it is in the document verdict ladder, which every path reaches. The sidecar layer's own
failure is narrower: `load_sidecar` validates an object and `render` produces a document, and nothing
compares the second to the first.

**That 49 attacks is a security review.** They were written by agents in one pass, against a surface
list this lab chose, and the author of the repairs also commissioned the attacks. The adversary's
own gap list — published in the catalogue — names seven areas nobody attacked, including the numeric
comparison's precision being set by the span's own printed text, and `authored_sha256` being dead on
the `path:` and `prereg:` branches.

**That a document verifying SWORN-HELD is worthless.** The receipt has always carried `unresolved`,
and `verify`'s headline has always printed it. What was missing was any signal that the verdict word
does not account for it.

---

*The suite-power study said this layer was the least defended in the corpus. It was, and attacking it
turned up something larger: the word at the top of every verdict promises more than the ladder
underneath it delivers.*
