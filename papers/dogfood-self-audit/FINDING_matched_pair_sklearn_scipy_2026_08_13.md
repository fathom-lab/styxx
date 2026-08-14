# A coverage-matched pair: sklearn and scipy differ by 2× in dead decision terms

**Date:** 2026-08-13. **Runs:** `probe_e_sklearn_v2_joined.json`,
`probe_e_scipy_joined.json`. **This is the first cross-repository comparison the tooling
permits** — every previous pairing was refused.

## Why this pair and not the others

`compare_probe_e.py` refuses to print rows whose exercised fractions differ by more than
1.5×, because `dead_rate` is computed over each suite's exercised minority and a suite
driving 82% of its decision terms is not answering the same question as one driving 37%.
styxx-vs-numpy-vs-sklearn spans 2.22× and is blocked.

sklearn and scipy are **matched to 1.5 points**:

| | exercised | terms | dead / decisions | value-position dead | no-population |
|---|---:|---:|---:|---:|---:|
| sklearn | **82.5%** | 9,484 | **16.2%** | 80 | 2/247 |
| scipy | **81.0%** | 18,251 | **28.1%** | 210 | 2/332 |

Two mature scientific Python libraries, both driving ~81% of their decision terms to the
observation floor under their own suites, differing by a **factor of 1.7** in the rate at
which those terms never take both values.

## Tested at module level, not term level

Terms cluster inside functions and modules, so a term-level test would treat thousands of
non-independent observations as independent and manufacture certainty. The comparison is
therefore between **per-module dead rates**, modules with at least 5 powered adjudicative
terms:

| | modules | median dead rate | Mann–Whitney |
|---|---:|---:|---|
| sklearn | 189 | **0.125** | U = 14,917.5 |
| scipy | 294 | **0.246** | **p < 1e-6**, rank-biserial **0.463** |

A rank-biserial of 0.46 is a large effect: pick a random sklearn module and a random
scipy module, and the scipy one has the higher dead rate about 73% of the time.

## It survives the covariate that explained the previous gap

The styxx-vs-numpy gap was largely attributable to suite design — numpy parametrises
7.5× more densely than styxx and 16.6% of its powered terms read exactly the dispatch
parameters those sweeps vary. That mechanism does **not** explain this pair.

sklearn parametrises **more** than scipy (20.9 vs 14.1 `@pytest.mark.parametrize` per
1,000 test lines, a factor of 1.48) — the right direction to explain a lower sklearn dead
rate. So the adjustment was applied directly: drop every term whose source reads dispatch
vocabulary (`dtype`, `axis`, `out`, `shape`, `order`, `casting`, `subok`, `copy`, `ndim`)
and re-run.

| | sklearn median | scipy median | rank-biserial |
|---|---:|---:|---:|
| all adjudicative terms | 0.125 | 0.246 | 0.463 |
| **excluding dispatch vocabulary** | **0.119** | **0.239** | **0.456** |

The gap barely moves. Whatever separates these two codebases, it is not the terms most
exposed to parametrised sweeping.

## What this licenses, and what it still does not

**Licensed:** the dead-decision rate is a **real, measurable property that varies
substantially between codebases of comparable maturity and comparable test coverage.**
That is the claim PROBE E was built to be able to make, and until this pair every
comparison had been refused for good reason. A metric that only ever differs when
coverage differs would be measuring coverage.

**Still not licensed:** that sklearn is *better engineered* than scipy. Coverage-matched
is not design-matched. The two libraries differ in domain, in the ratio of Python to
compiled code, in age, and in how much of their branching is defensive numerical guarding
against inputs their own tests never construct — and a guard that never fires may be
doing its job. Nothing here identifies which of those produces the gap.

**And the pre-registered null still binds.** Earlier the same day, this repository tested
whether dead instruments predict retracted claims and found they do not
(`FINDING_prereg_retraction_null_2026_08_13.md`, p = 0.248). So a higher dead rate is not
evidence that a library's *results* are less trustworthy. It is a statement about how much
of its decision logic its own suite can distinguish from a constant — no more.

## The honest one-sentence version

> Under their own test suites, and at near-identical decision-term coverage, scipy's
> Python-level decision terms are constant about twice as often as sklearn's — a
> difference that survives adjustment for parametrised dispatch sweeping, and whose cause
> is not identified here.

## Reproduction

```
python probe_e_runtime.py --pkg scipy --tests <site-packages>/scipy --chunked \
    --census census_scipy_broad.json --json probe_e_scipy.json
python compare_probe_e.py --runs probe_e_sklearn_v2_joined.json probe_e_scipy_joined.json
```
