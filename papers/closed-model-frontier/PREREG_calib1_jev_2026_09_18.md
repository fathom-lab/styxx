# PREREG — CALIB-1: does Jev's confidence mean anything on our corpus?

Fathom Lab · 2026-09-18 · Frozen before a single Jev call is scored. Instrument at sha256
`9b620e00…` and **not modified by this run**. Follows
`RESULT_decide1_decidable_fraction_2026_09_17.md` (71% of claims decidable, the instrument silent on
most of them), `RESULT_path1_only_touches_repair_2026_09_17.md` and
`RESULT_scope1_ABANDONED_2026_09_18.md`.

## Why there is a model in this repository at all

PATH-1 closed a door and wrote down why:

> Modes 3–6 have the same surface shape as claims that are genuinely checkable. DECIDE-1
> adjudicated the difference by reading the sentences; no test available to the instrument
> separates them.

The difference between *"Only modifies CHANGELOG.md"* and *"Only changed mods/submods are
serialized"* is not in the tokens. It is in what the sentence is about. DECIDE-1 separated them by
reading; a regular expression cannot; and three quarters of this instrument's accusations are wrong
because of it.

**Jev** (TypeSafe, System One) answers one bounded question and returns a probability. That is
exactly the shape of the missing test. Before it is allowed anywhere near this repository, the thing
it is sold on gets measured.

## What is being measured, and what is not

**Measured:** whether Jev's `noul` separates *a sentence that claims a file scope* from *a sentence
that does not*, on claims this lab adjudicated by hand **before Jev was in the picture**; and
whether its probabilities are calibrated on that corpus; and whether the same input gives the same
answer twice.

**Not measured, and not licensed by any outcome here:** whether Jev should produce a verdict. It
should not, and `packages/styxx-js/src/triage.ts` is written so that it cannot. A hosted model is
not reproducible offline, cannot be sealed into a capsule, and can change under us without notice.
Every `VERIFIED` and `CONTRADICTED` styxx emits stays computable from the diff bytes by code a
stranger can run with no network and no key. **Triage widens what is read. It never changes what is
said.** Any later preregistration that proposes otherwise has to argue against this paragraph.

## Population

The 25 `only_touches` items of `decide1_adjudication.json` — the stratum triage exists for, each
carrying the claim sentence, the pull request's changed paths, a hand verdict and a written reason,
all recorded 2026-09-17.

Ground truth for the triage question, fixed here:

| DECIDE-1 label | CALIB-1 class | n |
|---|---|---|
| `decidable: true` | **POSITIVE** — is a file-scope claim | 13 |
| `runtime_behaviour`, `prose_or_documentation`, `other_refers_to_different_change` | **NEGATIVE** — is not | 8 |
| `ambiguous_scope` | **EXCLUDED** from scoring, reported separately | 4 |

The four excluded are excluded because a human called them genuinely ambiguous; scoring a model
against a label that says "unclear" measures the label. Their `noul` values are published anyway,
and **the refusal band is expected to contain them** — that is the one prediction about them.

**n = 21 scored. This is a pilot and is powered like one.** Every proportion is published with a
Wilson interval, and if the interval on the headline separation is wider than 30 points the result
says in its own headline that the run does not settle the question and that the decision is whether
to fund a larger hand adjudication.

The 11 BENCH-2 accusations are run as a second panel and are **declared non-evidential in advance**:
six of them are the set SCOPE-1's rule was read off, and all eleven have been read by us repeatedly.
They are a consistency check, never a number to cite.

## Predictions, committed now

1. **Separation.** Median `noul` on POSITIVE items exceeds median `noul` on NEGATIVE items by at
   least **0.35**.
2. **The four runtime-behaviour sentences** — the shape that produced three of the instrument's nine
   false accusations — all score below 0.5.
3. **Calibration will be worse than the marketing.** Expected calibration error over 5 equal-width
   bins will exceed **0.10**. TypeSafe's documentation declines to say how `confidence` is computed
   and publishes no calibration evidence, so the honest prior is that it is uncalibrated on a corpus
   it has never seen.
4. **Determinism is not promised and will not hold.** At least one of 20 items re-asked five times
   will return a `noul` spread wider than 0.05.

Predictions 3 and 4 are predictions *against* the tool. If they fail — if it is calibrated and
deterministic — that is a better outcome than we expect and must be reported as such rather than
buried.

## Gates

- **G-C1-1 (blocking, held-out thresholds).** Thresholds are chosen on a DEVELOPMENT split and
  scored once on a HELD-OUT split, using `v14_gates.bucket` on the first five URL segments —
  the existing convention in this repository, imported, not reimplemented. A threshold pair chosen
  and scored on the same items is not a measurement, and SCOPE-1 is what happens when that is
  forgotten.
- **G-C1-2 (blocking, separation).** Prediction 1 or the run ships no thresholds. Below 0.35 the
  answer is that Jev does not separate these sentences on our corpus and `triage.ts` stays unused.
- **G-C1-3 (calibration, reported whatever it is).** Reliability table over 5 bins, ECE published,
  plus the count in each bin. A bin with fewer than 3 items is reported as empty rather than
  averaged.
- **G-C1-4 (determinism, reported whatever it is).** 20 items × 5 repeats. The per-item spread is
  published. **Any spread above zero is a permanent argument against Jev on the verdict path** and
  the result must say so in those words.
- **G-C1-5 (cost and latency).** Total spend and median latency for the whole run, published to the
  cent. If a calibration study of 21 sentences costs more than a dollar, that is worth knowing too.
- **G-C1-6 (the refusal band is not free).** Whatever thresholds are chosen, the result reports how
  many of the 25 land inside the band — sentences triage declines to route, which the reader then
  drops exactly as it does today. A band that swallows most of the stratum is a rule that changes
  nothing, and must be reported as changing nothing.

## What would abandon this

- G-C1-2 fails: Jev does not separate these sentences and the module is not wired in. The write-up
  says the extraction gap is still open and that a bought model did not close it.
- Determinism is so poor that the same sentence routes differently on consecutive runs. Triage would
  then make the *set of claims read* unstable between runs, which makes the gate's output unstable
  even though each verdict is still computed from bytes. That is disqualifying on its own.
- The API key cannot be supplied without a human handling it. **This run cannot be executed by the
  agent that wrote this document**: the key belongs in the operator's environment or in a repository
  secret, alongside `PYPI_API_TOKEN`, and never in a transcript. The preregistration is frozen now
  so that whoever runs it cannot choose the thresholds afterwards.

## Honest statement of what a passing CALIB-1 means

It means one bounded question, asked of a hosted model, separates two kinds of sentence on 21
hand-labelled items well enough to be worth routing on. It does not mean the coverage problem is
solved, it does not make a single verdict more trustworthy, and it does not license a second
question. Coverage is 5.4% against a hand-adjudicated 52%; if triage moves that, the number gets its
own preregistration and its own held-out split.

---

*The product is that you do not have to trust us. Buying a model that reports its own confidence
does not change that — it just adds someone else you would have to trust, unless we measure them
first. This is the measurement, frozen before the first call.*


---

## Amendment A — 2026-09-18, appended before any Jev call was made

This preregistration was frozen at sha256
`7d550cd50473c6642770662149553eaf3db1e1a52308e8d80e1d518b3ad94aaa`. Everything below is appended;
nothing above it is edited. **No call to Jev had been made when this was written** — the runner had
been executed only against its own dry-run stub, whose answers come from a hash of the prompt.
Appending after a call would make this document worthless. Appending before one is the only thing
that keeps it worth anything.

Each item below either fixes an error in the frozen text or removes a freedom the frozen text left
open. None of them widens one.

**A1 — Prediction 2 names four runtime-behaviour sentences. There are five.**
`decide1_adjudication.json` carries five `only_touches` items with `reason_code:
runtime_behaviour`: ids **39, 46, 55, 70, 73**. Prediction 2 is restated over all five — *all five
score below 0.5* — and the ids are pinned here so the set cannot be chosen once the answers are in.

**A2 — the corpus caps `paths` at 25, and the model must not be told that the cap is the diff.**
`decide1_adjudication.json` records at most 25 paths per item, alongside the true `n_files`. Four of
the 25 `only_touches` items are capped: ids **45** (175 files), **61** (134), **73** (64), **95**
(92). Deriving the file count from the array would tell the model a 175-file pull request changes 25
files, which inverts the reason the paths are shown at all. `triageSentence` takes a `totalPaths`
argument for this and the runner passes `n_files`; the paths shown stay the 25 the corpus recorded.

**A3 — every item is asked five times, and G-C1-4 is still computed over twenty.**
The frozen text says 20 items x 5 repeats. The runner repeats all 25, because the marginal cost is
five calls. **G-C1-4 is computed over exactly the 20 lowest `id`s**, so the preregistered statistic
stays the preregistered statistic; the other five are published beside it and cannot move the gate.

**A4 — spend is published only if a price is supplied.**
G-C1-5 asks for spend "to the cent". No per-token price for Jev is recorded anywhere in this
repository. The scorer publishes token counts and median latency unconditionally, and spend only
when a price is passed on the command line. An estimated spend is not a measured one, and quietly
substituting one for the other is the defect `RESULT_decide1_decidable_fraction_2026_09_17.md`
names as a habit rather than a bug.

**A5 — G-C1-2 is read on the pooled scored items, and its interval is a concordance interval.**
Prediction 1 names no split, so the separation gate reads the pooled scored items; the per-split
figures are published beside it. The population section asks for a Wilson interval on "the headline
separation", and a difference of medians does not have one. The interval is therefore computed on
the **concordance proportion** — the share of POSITIVE/NEGATIVE pairs in which the POSITIVE scores
higher, ties counted as half. Those pairs are not independent, and the result must say so wherever
it prints the interval.

**A6 — G-C1-1 is unrunnable on this population, and is not being repaired.**

`v14_gates.bucket` on the first five URL segments splits the 25 items like this:

| split | POSITIVE | NEGATIVE | EXCLUDED |
|---|---|---|---|
| DEVELOPMENT (`bucket < 3`) | 3 | **0** | 4 |
| HELD-OUT (`bucket >= 3`) | 10 | 8 | 0 |

A development split holding no negative items cannot discriminate between threshold pairs: every
pair that reads everything scores identically on it. G-C1-1 is blocking, and it cannot be run.

The tempting repair is a different split — a different hash, a different segment count, a stratified
draw. **That repair is the thing G-C1-1 exists to forbid.** It does not become acceptable because
the person making it can see that this split is unlucky rather than unflattering; that is what every
such repair looks like from the inside. The split was fixed before the data existed and it stays
fixed.

So: **G-C1-1 is recorded as UNRUNNABLE, and CALIB-1 ships no thresholds under any outcome.**
`NO_DEFAULT_THRESHOLDS` stands, `triage.ts` stays unwired, and G-C1-6 reports band occupancy over a
grid of candidate bands rather than over a chosen one.

What the run is still worth: G-C1-2 through G-C1-5 do not depend on the split. They answer whether
Jev separates these sentences at all, whether its probabilities mean anything, and whether it
returns the same answer twice — which is exactly the question that decides whether a larger hand
adjudication is worth funding. The population section already named that decision as the one a wide
interval would force. The split forces it instead, and forces it earlier: **before a single call was
paid for.**

## Running it

```
node --experimental-strip-types papers/closed-model-frontier/calib1_ask.ts --dry-run
TYPESAFE_API_KEY=... node --experimental-strip-types papers/closed-model-frontier/calib1_ask.ts
python papers/closed-model-frontier/calib1_score.py
```

Node 22.6+ for the runner's type stripping, Node 20+ for the SDK. The key belongs in the operator's
environment or in a repository secret; `calib1_ask.ts` refuses one passed as an argument. A dry-run
raw file scores to `INVALID__DRY_RUN` and cannot be cited.
