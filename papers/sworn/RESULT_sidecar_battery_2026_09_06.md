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
of the 49 predictions,
<sworn r="path:conformance/sworn/sidecar_battery.json#/counts/predictions_correct" k="numeric">7 matched the observed outcome,</sworn>
and most of that gap is my definition rather than their aim.

The number the definition missed:
of the forty-nine attacks,
<sworn r="path:papers/sworn/sidecar_battery_outcomes.json#/render_to_SWORN_HELD" k="numeric">42 render to a document that verifies SWORN-HELD,</sworn>
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

## A second finding, from the adversary's list of what nobody attacked

The rounding rule quantizes the receipt to the number of fractional digits **the author printed**.
That is deliberate and it is right: `DECISIONS["rounding"]` documents it, and demanding an exact
match would FAIL every honestly rounded figure in this corpus.

It has no floor. At zero fractional digits it stops rounding and starts erasing:

| sentence | receipt | compared against | verdict |
|---|---|---|---|
| "the A-share is 0.4211." | 0.4211 | 0.4211 | HELD |
| "the A-share is 0.42." | 0.4211 | 0.42 | HELD |
| **"the A-share is 0."** | **0.4211** | **0** | **HELD** |

Genuine harness-minted L2 receipt, correct digest, `complete`, nothing malformed, nothing forged.
The author chooses how much of the receipt's value to round away and may choose all of it.

The verdict is deliberately unchanged — refusing this would break the honest-rounding rule the
format needs, which would be worse than the problem. The headline counts these spans instead, on a
line nobody has to argue about: a non-zero receipt whose comparison was against zero carries no
information about the receipt at all. A receipt genuinely equal to zero erases nothing and is not
flagged, which is the control that keeps the signal from becoming noise.

**The first attempt at that signal was itself a defect, and this corpus caught it.** It added a
field to the span's `detail`, which is inside the digested core, so it moved the core digest of
every affected span and would have put the Python and JavaScript verifiers out of agreement. The
conformance generator refused the regeneration — *"a moved core is a finding about the verifier,
never a reason to rewrite the set"* — and the signal was moved to the headline, which is outside the
digest. A warning that changes what the format digests is a format change wearing a warning's
clothes.

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

---

## ERRATUM — 2026-09-06: the verdict was not the defect

**Withdrawn: the framing of the first finding.** This document said `SWORN-HELD` ignoring
`UNRESOLVED` "is the conflation this module's own doctrine refuses," and proposed that `SWORN-HELD`
should require `UNRESOLVED == 0`, leaving the rename to the operator. Preparing that rename was how
the error was found, and the measurement caught it before any code was written.

**It is a deliberate, documented decision, and it is right.** `tests/test_sworn.py` carries
`test_unresolved_only_is_held_and_the_count_travels_in_the_headline`, and the rationale is stated at
the `rung_unknown` branch of `_resolve`:

> v0.2 R6: a manifest claiming a rung this verifier cannot check (L3, or a string nobody defined).
> The verifier declines to see it; **it accuses nobody.**

`SWORN-FAILED` means the verifier **caught something wrong**. `UNRESOLVED` means the verifier
**could not look**. Refusing to conflate those is the same principle this corpus applies as *SKEW is
not DRIFT* and *a child that never ran is not a digest that moved*. Renaming the verdict would make
a document read worse because the verifier was unable — precisely what the doctrine refuses. The
doctrine line this document quoted ("a document that swore nothing is `UNSWORN`, never 'no
failures'") governs `sworn_total == 0`; it was not about unresolution, and citing it as though it
were was the overstatement.

**Withdrawn also: the rung-flip as an attack on the format.** The demonstration stands — rung `L2`
gives SWORN-FAILED and rung `L3` gives SWORN-HELD over the same contradicted receipt — but it
requires control of the manifest, which the format already declares a trust boundary in
`Manifest`'s own docstring: *"The turn manifest the HARNESS mints. Never the agent… The manifest is
only as trustworthy as the harness that wrote it, and every receipt says so."* An adversary who
writes the manifest is outside the model, and the model says so out loud.

**What stands, unchanged.** The reader-facing hazard is real and is not about attackers: running
`verify --repo .` without `--commit` gives an honest user SWORN-HELD over a document in which
nothing was checked. **The headline warning shipped in this leg is the correct and sufficient
repair**, and it needed no change to the verdict vocabulary. Everything else in this document — the
49 attacks, the 0 refusals, the round-trip gap between `load_sidecar` and `render`, the rounding
floor — is unaffected.

**The blast radius that was measured, and what it was measuring.** 68 of 2067 core conformance
vectors would have moved under the rename. That number was collected to size the change; read
correctly it is the corpus stating, 68 times over, a behaviour it had already decided on.

*Three times in two days a repair has been aimed at a decision this lab had already made and written
down — once in a spec's own errata, once in a comment three lines above an assertion, and once here
in a test's name. The tests are where the decisions live, and reading them is not optional.*
