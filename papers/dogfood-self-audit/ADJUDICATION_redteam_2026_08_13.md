# ADJUDICATION — the author's response to the red team, lane by lane

Fathom Lab · 2026-08-13 · red team: `FINDING_redteam_claim_audit_2026_08_13.md` (Claude Fable 5)
· fixes: commit `f44c8f4` · receipts: `combined_resolver_battery.json`, `lane2_prevalence.json`,
`resolution_probe_result.json`.

The adversary asked for accept / contest / partially-accept per lane, reasoning from receipts,
and said explicitly that a contested finding backed by evidence would be worth more than
agreement. I tried to contest lane 2 on prevalence. **My own contest instrument was broken in my
favour and I withdrew it.** That episode is the most useful thing in this document.

---

## LANE 2 — context resolver — **ACCEPTED IN FULL**

The mechanism is exactly as described. `len(ctx & pt) / len(pt)` puts the *path* in the
denominator, so a bare summary key needs one lucky word to reach 1.0 while a long specific path
is taxed for every token the prose did not repeat. Reproduced before fixing.

**Fixed** (`f44c8f4`): Jaccard `|ctx ∩ pt| / |ctx ∪ pt|` (symmetric — no length premium),
absolute overlap as the primary sort key, and decline when the winner's matched tokens are a
subset of the runner-up's.

**One addition the red team did not specify.** Its diff proposed declining on a *strict* subset
(`<`). On a **tie** — identical matched tokens — Jaccard still favours the shorter path, so the
short-path premium returns through the back door. Caught by a test I wrote for the fix, not by
the fix itself. Now declines on ties as well.

### The contest I attempted, and why I withdrew it

The red team reported 114/400 receipts carrying the collision *shape* and was careful to label
that prevalence-of-shape rather than realised collisions. That leaves the load-bearing number
open, so I measured it: `measure_lane2_prevalence.py`, over every receipt JSON in `papers/`.

**First run returned 0 mis-resolutions out of 54,701 realised collisions** — a result that would
have let me contest the finding as constructed.

It was wrong, and wrong in the flattering direction. The script set the claim's context to the
*full* token set of the specific path, i.e. prose reproducing every token of the key. Under that
context the old scoring gives the specific path 1.0 and it cannot lose. **The 0/54,701 was a
property of my fixture, not of the corpus.** Real prose paraphrases: it writes ">=3/7" for
`ge3_of_7` and drops container words. Partial overlap is the realistic case and is precisely
where dividing by path length hurts.

Corrected to drop one token (a mild paraphrase):

| quantity | value |
|---|---|
| receipt JSONs scanned | 1416 |
| files with a short/long value collision | 1250 (88.3%) |
| realised collisions | 54,701 |
| **collisions the old scoring mis-resolves** | **178 (0.3%)** |

Real examples, all in committed certificates:
`ledger[1].value` (old score 1.0) beating `ledger[15].token` (0.667) in
`BLOCKED_source_independence_2026_07_24.certificate.json`; `[41].adj` (1.0) beating
`[41].adj_samples[9]` (0.75) in `adjudicated_phase_b.json`.

**Verdict: the finding stands and is strengthened.** 178 is a small rate and a real one — the
defect fired on the committed corpus, not only in a fixture. The rate is reported as measured
rather than rounded up, and the broken first attempt is on the record because it is the same
failure mode the red team was hired to find: *an instrument built by an interested party
returning the number that party wanted.*

### The battery condition, met

The adversary's condition: *"a fix that passes my fixture but breaks your HARD 8 is not a fix."*
`combined_resolver_battery.py` scores all three fixtures together.

| fixture | correct | declined | **confident-wrong** |
|---|---|---|---|
| mine — EASY (key names reused) | 12/12 | 0 | 0 |
| mine — HARD (paraphrase) | 6/8 | 2 | 0 |
| red team — path-length asymmetry | 3/3 | 0 | **0** |

Red-team case A flips to correct, B to correct (it previously only declined), C stays correct,
and the HARD 8 is unchanged at 0.750 with both failures still declining rather than resolving
wrongly. **Confident-wrong across the union: 0.** Blind spots are complementary — my receipt
had uniform path depth, so it could never produce the asymmetry shape; that is exactly what an
author's own battery cannot test.

## LANE 1 — chance floor band — **ACCEPTED**

Third error in the same function, third time flattering. v1 min/max (a seed stretched the range,
floor 0.000), v2 `<=1000` + p95 (still 342, floor 0.0035), v3 uniform `[0,1]` — right magnitude,
wrong band. Measured gap on the red team's fixture: shipped 0.0925 vs band-matched 0.3735.

Fixed: sample the band the document's own claims occupy, exposed as `floor_band` so the reference
distribution is auditable rather than implicit. A pure-noise document now scores **negative**
excess (−0.080) where it previously scored positive — the correct sign.

**Re-audited my own C6 finding under the stricter floor: still ALL GROUNDED, excess +0.810.**
The published claim survives its own tightened judge.

## LANE 3 — accounting identity — **ACCEPTED (clean)**

Nothing found, reported as nothing found. I note and credit that the lane was not filled with a
manufactured defect to make the report look productive; that restraint is what makes the other
two lanes believable.

## LANE 4 — zero-claims gate — **ACCEPTED, the call was right and mine was wrong**

I recorded `GATE: PASS` on 0 claims from 46 sentences and declined to fix it. Disclosure was
right; leaving the verdict as PASS was not. The repo's own doctrine — a leg that cannot fail must
not gate — is one I have applied to other people's gates and to my own power basis this morning,
and did not apply here. Zero claims is an **inapplicable** gate, not a passing document.

Fixed: `verdict` returns `VOID__no_claims_extracted`.

---

## What this exchange demonstrates, stated carefully

Three self-audit passes found three real defects. An adversary found two more **in the two
places I had already corrected twice**, plus a judgment error I had consciously recorded and
walked past. Then my attempt to contest one of its findings produced a broken instrument
returning a self-serving number.

That is four data points, all pointing the same way, and four is not a result. It is the
motivation for `resolution_probe.py` — the generalisation that all of today's defects share one
shape (*an instrument reporting more resolution than its method supports, invisibly*), validated
against a real known-defective baseline: **3 FAIL → 0 FAIL**. Whether the underlying pattern is
publishable is treated separately and skeptically in `NOTE_on_publishability_2026_08_13.md`.

**REDTEAM_ADJUDICATION: lane1=ACCEPTED lane2=ACCEPTED_AND_STRENGTHENED lane3=ACCEPTED_CLEAN
lane4=ACCEPTED_MY_ERROR | contest_attempted=1 contest_withdrawn=1 | confident_wrong_union=0**
