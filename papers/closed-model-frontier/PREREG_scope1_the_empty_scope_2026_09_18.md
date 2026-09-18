# PREREG — SCOPE-1: the separator PATH-1 said was not in the sentence, because it is not in the sentence

Fathom Lab · 2026-09-18 · Frozen before any change to `styxx/diffgate.py`. Instrument at sha256
`9b620e00…` on freezing. Follows `RESULT_bench2_INVALID_2026_09_17.md` (9 of 11 accusations false),
`RESULT_path1_only_touches_repair_2026_09_17.md` (precision 0.18 → 0.25, six false accusations
remaining) and `RESULT_decide1_decidable_fraction_2026_09_17.md` (71% of claims decidable; the
instrument abstains on 94% of `only_touches`). Tracks issue #128.

PATH-1 declared four failure modes unrepairable and wrote down why:

> Modes 3–6 have the same surface shape as claims that are genuinely checkable. DECIDE-1
> adjudicated the difference by reading the sentences; no test available to the instrument
> separates them.

That statement is correct and this preregistration does not contradict it. **The separator proposed
here is not in the sentence.** It is in the relationship between the sentence and the diff, which is
a place PATH-1 did not look.

## Where the rule came from, stated before anything is measured

The six surviving false accusations were re-read looking for a property they share and the two
correct accusations do not. They share one:

| accusation | changed paths **inside** the claimed scope |
|---|---|
| `Albeoris/Memoria#1142` — "only changed mods/submods are serialized" | none |
| `Albeoris/Memoria#1145` — same | none |
| `Albeoris/Memoria#1147` — same | none |
| `mikepenz/release-changelog-builder-action#1458` — "commits that only touched files in `app1/`" | none |
| `open-policy-agent/cert-controller#415` — "Only change .githiub/workflows/dependabot.yml" | none |
| `fern-api/fern#9898` — "only changes README/documentation snippets" | none |
| `Azure/autorest.typescript#3252` — **correct accusation** | present |
| `microsoft/wassette#442` — **correct accusation** | present |

**This table is where the hypothesis came from, so it cannot also be the evidence for it.** Every
number SCOPE-1 reports on these eight items is declared non-evidential in advance and is published
as a consistency check only. The evidence must come from the held-out set defined below, and if the
held-out set does not support the rule then the eight items do not rescue it.

## The rule

For a claim of kind `only_touches` with an admitted scope `P`, let `inside(P)` be the changed paths
the instrument already considers contained in `P`, and `outside(P)` the rest.

- Today: `outside(P)` non-empty → **CONTRADICTED**.
- Under SCOPE-1: if `inside(P)` is **empty**, the instrument returns no verdict and names the
  reason — *the claimed scope matches nothing in this diff, so the sentence is probably not about
  this diff's file scope*. Only when `inside(P)` is non-empty **and** `outside(P)` is non-empty does
  it accuse.

SCOPE-1 may only turn an accusation into silence. It may not create a verdict where there was none,
may not change any `SUPPORTED`, and may not touch any other claim kind. If the measured run shows
it doing any of those, it is a defect and the run is INVALID.

## The cost, stated first and not buried

This rule makes one real class of lie undetectable: *"I only touched X", where the pull request
touched nothing in X at all.* That is the most brazen version of the claim, and after SCOPE-1 the
instrument will say nothing about it, forever, by design.

We are proposing to trade that away because the same shape is produced by a sentence that was never
about file scope — a description of runtime behaviour, a line of README prose, a typo — and on the
evidence so far the innocent explanation is the common one. **If the held-out audit says otherwise,
the trade is bad and SCOPE-1 does not ship.** A test will be written whose sole purpose is to assert
that the instrument stays silent on this class, so that no later document can claim we catch it.

## The held-out set, fixed before the run

BENCH-2 could not reach 123 of the 691 pull requests in the population on 2026-09-17: 107 HTTP 403,
14 served empty, 2 HTTP 404. Those 123 carry **59 `only_touches` claims** and appear in no scored
row of BENCH-1, BENCH-2, DECIDE-1 or PATH-1. They were selected by HTTP status on a day before this
hypothesis existed, which is the only reason they are usable as held-out data.

Whichever of them are reachable on 2026-09-18 are the held-out set. Their count is recorded before
the instrument is run over them, and the ones still unreachable are listed by reason and not
replaced with substitutes.

## Predictions, committed now

1. **On the eight (non-evidential).** Accusations 8 → 2. Precision 0.25 → 1.00. The two retained
   accusations are `Azure/autorest.typescript#3252` and `microsoft/wassette#442`.
2. **On the 299 `only_touches` claims of BENCH-2.** Coverage falls by exactly 6 verdicts and by no
   more. Any seventh changed verdict is an unintended effect and blocks the run.
3. **On the held-out set.** Of the accusations SCOPE-1 suppresses there, **fewer than one in four
   will be adjudicated a true accusation** — strictly below the instrument's own published
   precision of 0.25. The rule has to suppress accusations that are worse than the average
   accusation, or it is not a discriminator and there is no reason to ship it.

Prediction 3 is the one that matters. Predictions 1 and 2 are arithmetic on items the rule was
built from.

## Gates

- **G-S1-1 (blocking, held-out precision).** Every accusation the current instrument makes on the
  held-out set is adjudicated by hand against the live diff under the DECIDE-1 rubric, **before**
  the SCOPE-1 verdicts are joined to it. Blocking: the precision of the suppressed subset must be
  **strictly below 0.25**. At or above 0.25, SCOPE-1 is withdrawn and the finding is published in
  its place.
- **G-S1-2 (blocking, no collateral movement).** Across all 605 BENCH-2 claims, the only verdict
  changes are `CONTRADICTED` → withheld on `only_touches`. Any other movement voids the run.
- **G-S1-3 (blocking, the port agrees).** `web/gate/diffgate.js` mirrors the rule and the
  differential prints zero disagreements over the committed pair corpus.
- **G-S1-4 (blocking, the cost is pinned).** A test asserts that the instrument is silent on a
  synthetic diff that touches nothing inside a claimed scope, and its docstring states in plain
  words that this is a lie SCOPE-1 chose not to catch. If that test is ever deleted or inverted, the
  rule it protects has changed and needs its own preregistration.
- **G-S1-5 (power, non-blocking but must be reported).** The held-out accusation count and the
  Wilson interval on the suppressed subset's precision are published whatever they are. If the
  interval spans 0.25, the result must say in its own headline that the run is underpowered and
  must not be cited as evidence that the rule works.

## What would abandon this

- G-S1-1 fails: the trade is bad, and the write-up says the instrument keeps a rule we know
  false-accuses three times in four, because the alternative was worse.
- Fewer than three accusations exist in the whole held-out set: there is no test, only an anecdote.
  SCOPE-1 is then implemented **only** if it ships behind G-S1-4 and the result states that its
  central claim is unmeasured.
- SCOPE-1 turns out to change a verdict it was not supposed to touch. That is a defect in the
  implementation, not a finding about the corpus, and it is fixed before anything is re-run.

## Honest statement of what a passing SCOPE-1 means

It means the instrument accuses less often and is right more often when it does. It does not mean
the instrument is good: DECIDE-1 showed it is silent on 49 of the 76 claims a human can settle from
the diff, and SCOPE-1 makes that silence deeper, not shallower. Coverage is the larger problem, this
does not touch it, and the result must not be worded as if it did.

---

*PATH-1 said no test in the sentence separates a file-scope claim from a runtime-behaviour one. It
was right. So this one does not read the sentence — it asks whether the diff has ever heard of the
thing the sentence names. The price is that the boldest version of the lie goes uncaught, and there
is a test whose only job is to make sure we keep admitting that.*

---

## AMENDMENT A — appended 2026-09-18, before any verdict was computed

Two facts surfaced after the freeze at sha256 `445d4349…` and before the instrument was run over
anything. Both weaken this preregistration. Neither changes a gate. They are appended rather than
folded in, and the original hash stands beside the amended one.

**A1 — prior art inside our own instrument, which we did not cite and should have.**
`V14_BARE_NAME_ABSTAIN` (PREREG_v14_repair_2026_08_31) already applies the same idea — *a name the
diff has never heard of is not evidence of a lie* — to bare filenames in the `_PATH_KINDS` family.
It shipped, and the held-out precision of the repaired accuser was **0.16**
(`RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md`). So this class of repair has been
tried one claim-family over and did not save that accuser. That is a prior against SCOPE-1, it was
available to us before the freeze, and missing it is our error. The gates stand unchanged; the
result must report the SCOPE-1 outcome beside V14's 0.16 rather than as a fresh idea.

**A2 — the held-out set is empty.**
All 123 pull requests BENCH-2 could not reach on 2026-09-17 were re-probed on 2026-09-18. The
outcome is identical to the day before, to the item: **107 HTTP 403, 14 served empty, 2 HTTP 404,
0 newly reachable.** The 59 `only_touches` claims this preregistration nominated as its only
held-out evidence are therefore unavailable, and G-S1-1 cannot be run.

Under "What would abandon this", the surviving path is to implement SCOPE-1 behind G-S1-4 and state
that its central claim is unmeasured. Before taking it, one further measurement is committed here:
**enumerate, across all 299 `only_touches` claims, every accusation and whether any changed path
lies inside its claimed scope.** If the set of accusations SCOPE-1 would suppress turns out to be
exactly the six items the rule was derived from, then the rule has no footprint outside its own
derivation set, nothing in reach can falsify it, and it must not ship on that basis. That test is
committed now, before it is run.
