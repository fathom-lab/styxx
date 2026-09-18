# RESULT — SCOPE-1 is ABANDONED: the repair's entire footprint is the six items it was read off

Fathom Lab · 2026-09-18 · Prereg `PREREG_scope1_the_empty_scope_2026_09_18.md`, sha256
`445d4349…` at freeze, `5b6986ee…` after Amendment A, both appended before anything was measured.
Receipts: `scope1_footprint.py`, `scope1_footprint.json`, `bench_reproduce.py --fetch`.
**The instrument is unchanged at sha256 `9b620e00…`. Nothing shipped.**

Read with `RESULT_bench2_INVALID_2026_09_17.md`, `RESULT_path1_only_touches_repair_2026_09_17.md`
and `RESULT_decide1_decidable_fraction_2026_09_17.md`. Tracks issue #128.

## The proposal

PATH-1 left six false accusations standing and said no test in the sentence separates a file-scope
claim from a runtime-behaviour one. SCOPE-1 proposed a test that does not read the sentence:

> For `only_touches P`, if **no changed path lies inside P**, say nothing. Accuse only when the
> diff has at least one path inside the claimed scope and at least one outside it.

The price was named in the preregistration before any measurement: the instrument would become
permanently unable to catch *"I only touched X"* where the pull request touched nothing in X at
all — the boldest version of that lie.

## Why it does not ship

**The rule was read off six items, and those six items are the only things it moves.**

Across all 299 `only_touches` claims of BENCH-2, re-fetched from source today:

| | |
|---|---|
| accusations by the shipped instrument | **8** |
| SCOPE-1 would suppress | **6** |
| SCOPE-1 would keep | **2** |
| suppressed items **outside** the set the rule was derived from | **0** |

The six suppressed are the six named in the preregistration. The two kept are the two already
adjudicated correct in BENCH-2. Precision would go 0.25 → 1.00 on a denominator of two, and every
item in the numerator and the denominator was known before the rule existed.

That is not a measurement. It is the hypothesis restated.

The scope column is the instrument's own normalised form — lowercased, trailing separator and
sentence-final period stripped — because that is the string the verdict was computed against, not
the author's spelling.

| PR | scope, as the instrument read it | inside | outside | SCOPE-1 |
|---|---|---|---|---|
| `Albeoris/Memoria#1142` | `mods/submods` | 0 | 2 | suppress |
| `Albeoris/Memoria#1145` | `mods/submods` | 0 | 1 | suppress |
| `Albeoris/Memoria#1147` | `mods/submods` | 0 | 2 | suppress |
| `mikepenz/release-changelog-builder-action#1458` | `app1` | 0 | 14 | suppress |
| `open-policy-agent/cert-controller#415` | `githiub/workflows/dependabot.yml` | 0 | 1 | suppress |
| `fern-api/fern#9898` | `readme/documentation` | 0 | 3 | suppress |
| `Azure/autorest.typescript#3252` | `packages/typespec-ts`, `packages/typespec-test` | 48 | 1 | keep |
| `microsoft/wassette#442` | `changelog.md` | 1 | 3 | keep |

The separation is perfect and it is perfect because it was drawn around these rows. The
changelog-builder line is the shape in miniature: the instrument names fourteen files as being
outside `app1/` in a pull request that never went near `app1/`.

## The held-out set that would have decided it does not exist

The preregistration nominated the only genuinely held-out data available: the 123 pull requests
BENCH-2 could not reach on 2026-09-17, carrying 59 `only_touches` claims, selected by HTTP status
on a day before this hypothesis was formed.

Re-probed on 2026-09-18, one day later, the outcome is identical to the item:

**107 HTTP 403 · 14 served empty · 2 HTTP 404 · 0 newly reachable.**

So G-S1-1, the only blocking gate that could have produced evidence, could not be run at all.

## What it would have cost

DECIDE-1 put the decidable fraction of `only_touches` claims at 52% [33.5%, 70.0%]. The shipped
instrument returns a verdict on 16 of 299 — **5.4%** [3.3%, 8.5%]. SCOPE-1 takes that to 10 of 299,
**3.3%** [1.8%, 6.0%]: a 37.5% cut in the only number DECIDE-1 identified as the larger problem.

PATH-1's anti-gaming condition, written before any of this, is explicit that a precision gain paid
for with coverage is a regression *regardless of the precision number*. SCOPE-1 is that trade in its
purest form: **+0.75 precision on n=2, −37.5% coverage, and no evidence from outside the derivation
set.** The 95% Wilson interval on a 2/2 precision is [34.2%, 100%], which is wider than the interval
on the 0.25 it was meant to replace.

## The prior we should have cited and did not

`V14_BARE_NAME_ABSTAIN` (`PREREG_v14_repair_2026_08_31`) already applies the same idea — a name the
diff has never heard of is not evidence of a lie — to bare filenames in the `_PATH_KINDS` family. It
shipped. The held-out precision of the repaired accuser afterwards was **0.16**
(`RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md`).

That was in our own repository, in a file we had open, and it was not in the preregistration when it
was frozen. It is recorded in Amendment A rather than folded into the original, and it is our error
rather than an unlucky break: the idea had been tried one claim-family over and had not saved that
accuser either.

## What did come out of this

**PATH-1 reproduces from scratch.** Every diff was re-fetched from `patch-diff.githubusercontent.com`
today, independent of the run that produced PATH-1: 566 of 568 match their published sha256, 2
mismatch because the pull request gained commits since, 0 unreachable. Both mismatches
(`microsoft/vscode#259001` and `#263514`) **are** `only_touches` rows and were scored against the
diff as it stands today rather than as BENCH-2 saw it; both read UNCHECKABLE under issue #110
(`prefix 'behavior' is not a path`, `prefix 'lines' is not a path`), so neither enters an
accusation count on either side of this comparison. Against those diffs the shipped instrument makes **8**
accusations at precision **2/8 = 0.25** — PATH-1's headline, re-derived end to end by a script that
shares no state with it.

**And the coverage number is now measured rather than inferred.** 16 verdicts on 299 claims:
8 VERIFIED, 8 CONTRADICTED, 281 UNCHECKABLE, 2 with no reading at all.

## Deviations

Amendment A was appended to the preregistration before any verdict was computed, adding the V14
prior and recording the empty held-out set, and committing the footprint measurement that decided
this. Both hashes are published. No gate was changed, relaxed, or reinterpreted after a number was
seen.

---

*We designed a repair that would have taken this instrument's precision from 0.25 to 1.00, and then
measured that the only pull requests it touches are the six we designed it from. The number it
produces is a restatement of the assumption. It stays out of the instrument, the six false
accusations stay in, and the gate keeps a behaviour we can demonstrate is wrong three times in four
— because the alternative was to ship something that could not be checked.*
