# the absence census — 2.4M lines of the python AI stack, 2026-08-19

`styxx.absence` (7.39.0) was built from 41 defects found in **styxx itself** across
three adversarial audit waves. All one shape: a path that failed, or never ran,
and returned a value indistinguishable from a healthy measurement.

The question this census asks: is that shape peculiar to us, or endemic?

Reproduce: `python scripts/absence_census.py --verify-sample 14`

## the result

| package | KLOC | candidates | per KLOC | top rule |
|---|---:|---:|---:|---|
| **styxx (ours)** | 58.9 | 81 | **1.38** | SENTINEL_DEFAULT |
| requests | 4.1 | 2 | 0.49 | HEALTHY_ON_CRASH |
| transformers | 915.3 | 125 | 0.14 | UNDEFINED_AS_NUMBER |
| scipy | 260.2 | 16 | 0.06 | UNDEFINED_AS_NUMBER |
| pydantic | 35.4 | 2 | 0.06 | TRUTHY_GATE |
| torch | 634.3 | 32 | 0.05 | HEALTHY_ON_CRASH |
| sklearn | 163.5 | 7 | 0.04 | HEALTHY_ON_CRASH |
| datasets | 36.4 | 1 | 0.03 | HEALTHY_ON_CRASH |
| pandas | 199.3 | 2 | 0.01 | HEALTHY_ON_CRASH |
| openai · anthropic · numpy · httpx · fastapi · sentence_transformers | 250.7 | 0 | 0.00 | — |

**We score worst.** 1.38/KLOC, ~10x the densest large package. That is the
headline and it is not a humblebrag — see the confound below.

## precision: ~5 of 14 (36%), hand-verified, small sample

A density without a precision estimate is a fire-rate wearing the antibody's
name. 14 candidates were drawn at random (seed 0) and read by hand:

**Real instances of the class (5):** `weather.py:196` (a rate over an empty
window → 0.0), `fleet.py:201` and `probe.py:234` and `analytics.py:1032` (a mean
over an empty list → 0.0, the twin of the `check_health` defect fixed in
7.37.0), `preflight.py:403` (an absent composite → 0.0, the twin of the
`coherence` defect fixed in 7.37.0). **All five are ours.**

**False positives (9):** five instances of one transformers idiom
(`embed_scale = math.sqrt(d) if config.scale_embedding else 1.0` — a config flag
selecting a no-op scale factor, not a measurement); three display-only defaults
inside f-strings (`s.get('overconfidence', 0)` in a printed card); one
documented threshold default (`bars.get("D4_..._delta", 0.10)`).

14 is a small sample and the interval is wide. It is reported because a number
without one is worse than no number.

## the before/after that undercuts the headline

The five confirmed instances were then fixed (`weather.py` no longer computes a
trend direction from an empty window; `fleet.py` and `probe.py` stopped dropping
zero readings from their means; `preflight.py` treats an absent composite as
malformed input; `analytics.py` carries the counts its means were taken over, so
`n/a (unmeasured)` and a measured `0.000` print differently).

**Density moved 1.38 -> 1.36.** One candidate.

That is the most useful number in this document. The screen reads SHAPE, not
semantics: a `... if xs else 0.0` that now sits under an explicit guard, or
whose counts are disclosed alongside, looks identical to the fabricating version
it replaced. Five real defects were removed and the metric barely noticed.

So: **candidate density is a weak proxy for defect density**, demonstrated on
the one codebase where we know the ground truth. Any reading of the table above
— including our own last place — has to carry that. A package scoring 0.00 is
not clean, and a package scoring 1.38 is not five times worse than one scoring
0.28. The table measures how often a shape appears, and nothing more.

## the confound, stated plainly

**The rules were written from styxx's own defects, using styxx's own naming
vocabulary** (`composite`, `conf`, `cogn_*`, `rate`). A screen tuned on one
codebase will fire more on that codebase. Our 1.38 is therefore *partly* real
and *partly* an artifact of who wrote the rules, and this census cannot separate
the two. A clean separation needs rules derived from a corpus we did not write.

Note also what the verification found: **every confirmed instance in the sample
was ours.** Consistent with the confound, and equally consistent with us simply
having more of them. Both readings survive this data.

## the census's own failure, which is the point

The first run of this script reported **0.00 candidates for every external
package** — 2.4M lines of torch and transformers, perfectly clean.

It was wrong. `scan_path`'s default skip list contains `site-packages`, where
every installed package lives, so every file was skipped and the report printed
`candidates 0`. **The screen built to find measurements-that-never-ran had
produced one.** Third time this session the tool had the bug it hunts (after
`1 in {True}` flagging every CLI `return 1` as healthy).

Fixed structurally, not locally: `AbsenceReport.measured` is False when nothing
was scanned, `render()` prints SCANNED NOTHING — NOT A CLEAN RESULT, `__repr__`
says the same, and `scan_path` warns at the moment it happens. A report that
measured nothing can no longer be mistaken for a clean one.

## what this is not

Not a claim that any package named here is defective. The rules flag SHAPES.
A `dict.get("score", 0.0)` in a serializer is fine; the same line feeding a gate
is not, and only a reader tells them apart. No finding here was reported to any
maintainer, because none outside our own tree was verified.

## not done

The 9 false positives point at two obvious rule improvements (a config-flag
exclusion, a display-context exclusion). **They were not applied**, because
tuning on the same sample used to estimate precision is how a screen learns to
flatter itself. A tuning pass needs a fresh sample and a fresh estimate.
