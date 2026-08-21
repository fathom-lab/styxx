# styxx — the next level

*Design plan, 2026-08-21. Written against measurements from the 7.36–7.43 audit
arc, not against ambition. Every bet below carries a kill criterion; a bet
without one is a wish.*

---

## what we actually established

Not opinions — things with receipts in this repo.

1. **One defect class dominates.** 74 defects fixed across three audit waves,
   and effectively all of them were *an absence presenting as a measurement*.
   The direction was never random: **the inert default and the flattering
   default are the same value** (risk 0.0, trust 1.0, gate "pass"), so the
   fallback chosen for safety is the one that reads as health.
2. **The second-order form is a closed loop.** One field derived from the
   system's own verdict corrupted six consumers, including the calibrator that
   retrained on it. Invisible per-file; obvious once you ask where the field
   comes from.
3. **Static analysis has a measured ceiling.** `absence` 9/20, `loops` 5/20,
   union 13/20 on SILENT-PASS. `absence` plateaus across the tolerance sweep;
   `loops` climbs, so part of its recall is the window.
4. **SP-6 is the hole: 1 of 5.** "Empty input produces a confident result" is
   usually a guard that was *never written*, and no pass over source can flag
   code that does not exist.
5. **Density is a weak proxy.** Fixing five confirmed defects moved candidate
   density 1.38 → 1.36. The screen reads shape, not semantics.
6. **Self-audit is necessary and not sufficient.** Our own tools committed the
   defect they hunt six times. What broke the circularity was external ground
   truth — our own git history — not more introspection.

Those six facts constrain everything below. In particular (3), (4) and (6) say
the next gain is **not** another static rule.

---

## the architecture this implies

styxx's actual subject is narrower and stranger than "AI honesty":

> **Did a measurement happen, and does its consumer know?**

Four quadrants follow, and we currently occupy two and a half:

| | **instance** (this defect) | **class** (this shape, anywhere) |
|---|---|---|
| **static** | `absence` ✅ | `loops` ✅ |
| **runtime** | `credits`, gates ◐ | **empty** ← the gap |
| **historical** | SILENT-PASS ✅ | case extractor ❌ |

The empty cell is where SP-6 lives, and SP-6 is exactly the 1-of-5 static
analysis cannot reach. That is not a coincidence: **you cannot see an absent
guard in source, but you can see its consequence at runtime** — an output that
is confident about an input that was not there.

---

## bet 1 — the runtime contract (highest leverage)

**Claim.** A measurement function that returns a confident value from a
degenerate input is detectable *at call time*, without reading its source.

**Shape.**

```python
@styxx.measured(inputs="trajectories", confident_when="confidence > 0.5")
def forecast(self, trajectories, n_tokens=None) -> ForecastResult: ...
```

The wrapper inspects the argument for emptiness / all-NaN / zero-variance, and
inspects the return for a confident-looking value. The *combination* is the
finding: it does not care why the guard is missing, only that the output claims
more than the input can support.

**Why this beats another static rule.** It converts SP-6 from "code that was
never written" (invisible) into "a call that happened" (observable). It also
covers the case static analysis will never reach: degeneracy that only appears
with real data.

**Kill criterion.** If, replayed against the six SP-6 cases in the corpus, it
catches fewer than 4, the wrapper's input/output heuristics are too weak to
carry the idea and the bet dies. We measure before we ship, and publish the
number the way we published 45%.

**Risk we are naming now.** Decorators that raise in production get removed by
the first on-call engineer. Default must be *record, not raise*, with an
explicit opt-in to strict mode. A guard that can break a deploy is a guard that
gets deleted.

---

## bet 2 — the case extractor (turns a receipt into a benchmark)

**Claim.** SILENT-PASS's cases have a recognizable *diff signature*, so they can
be mined from any repository's history rather than hand-labelled.

**Signature.** A fix commit whose diff (a) adds a guard, a `raise`, a `None`
return, or a provenance check, (b) immediately upstream of a return of a healthy
literal, and (c) whose message contains the vocabulary of this class. Candidates
go to a human; only confirmed cases enter the corpus.

**Why it matters.** The corpus is 20 cases from **one codebase written by us** —
stated in CORPUS.md as its central weakness. Cases from repositories we did not
write are worth more to the benchmark than anything we can add ourselves, and
hand-labelling does not scale to that.

**Kill criterion.** Run it over 5 large public repos. If human review confirms
fewer than 20% of its candidates, the signature is noise and we say so rather
than shipping a case generator whose output nobody can trust.

---

## bet 3 — diff-scoped CI (make the class non-recurring)

**Claim.** The cheap durable win is not finding old instances but refusing new
ones.

**Shape.** `styxx-absence --since origin/main` screens only changed lines and
posts findings on the PR. Never fails the build — a screen that blocks a merge
is a screen someone silences, and we have said that from the first commit.

**Kill criterion.** If, run against this repo's last 200 commits in
diff-scoped mode, it produces more than one candidate per commit on average, the
noise makes it unusable as a PR comment and it stays a manual tool.

---

## bet 4 — the counterfactual `credits` refuses to invent

`styxx.credits` reports what the gate **cost** and refuses to net without a
declared rework figure. That refusal is correct and it is also a gap: nobody
knows their rework cost.

**The honest way to get one is an experiment, not an estimate.** Randomize the
gate on/off across matched traffic, measure actual downstream correction cost on
both arms, and report the difference with its interval. That is a
preregisterable design with a real null: *the gate saves nothing measurable*.

**Kill criterion.** Pre-register it. If the confidence interval on savings
spans zero at the sample size we can actually collect, we publish that the
economic case is unproven at our scale — the way the ledger publishes 62 losses
out of 163 cycles.

---

## what we are explicitly NOT doing

Stated so they cannot creep back in as "quick wins":

- **No new static rules until bet 1 ships.** The measured ceiling says the next
  rule buys little; SP-6 is where the room is.
- **No LLM-judge scoring inside the detectors.** A judge is a measurement that
  can fail silently — precisely this class, added to the tool built to find it.
- **No accusations against named packages.** The census reports candidate
  density with a precision estimate and a confound. Any specific claim about
  another project requires hand-verification and disclosure to its maintainers
  first, unpublished until then.
- **No "AI-powered" repackaging of the screens.** They are AST passes. That is
  a feature: deterministic, auditable, no model in the loop that could itself
  return a confident answer from nothing.

---

## sequencing

1. **Bet 1**, measured against the corpus's SP-6 cases before any release.
2. **Bet 3**, which is small and makes 1's findings actionable in review.
3. **Bet 2**, once there is somewhere worth putting external cases.
4. **Bet 4** as a preregistration, on the operator's timing — it needs live
   traffic and a decision about spend.

## the one-line version

styxx measures whether a measurement happened. It does that statically today;
the next level is doing it **at the moment of the call**, because that is the
only place an absent guard becomes visible — and publishing the number it scores
against our own corpus, whatever that number turns out to be.
