# PREREG — PATH-1: repairing two of the six `only_touches` failure modes, and not pretending about the other four

Fathom Lab · 2026-09-17 · Frozen before any change to `styxx/diffgate.py`. Instrument at sha256
`4ba947a8…` on freezing. Follows `RESULT_bench2_INVALID_2026_09_17.md` (the instrument false-accuses
on 9 of 11) and `RESULT_decide1_decidable_fraction_2026_09_17.md` (71% of claims are decidable and
the instrument abstains on 94% of `only_touches`).

Publicly committed to on 2026-09-17: a preregistration, an audit by the method that caught the
defect, and a published before-and-after on the same eleven pull requests. This is that document.

## Scope — two modes, and no more

**In scope.**

1. **Basename containment.** A stated prefix that is a bare filename (`package.json`,
   `blobstorage_pdisk_impl.cpp`) currently anchors at the repository root, so
   `appservice/package.json` is read as outside `package.json`. It should mean "a file with this
   name, anywhere in the tree." Two false accusations
   (`microsoft/vscode-azuretools#2086`, `ydb-platform/ydb#25857`).
2. **Dotted identifiers.** `Assert.NotNull` satisfies the `name.ext` filename shape and is admitted
   as a path. The suffix after the final dot must be a real file extension drawn from a closed,
   committed list, or the token is not a path. One false accusation (`dotnet/runtime#117821`).

**Explicitly not attempted, and stated here so that no later document can imply otherwise.**

3. Runtime-behaviour sentences ("only changed mods/submods are serialized") — 3 false accusations.
4. Documentation prose describing a feature rather than the PR — 1.
5. Prose containing a slash (`README/documentation`) — 1.
6. Typos in the stated path (`.githiub/`) — 1.

Modes 3–6 have the same surface shape as claims that are genuinely checkable. DECIDE-1 adjudicated
the difference by reading the sentences; no test available to the instrument separates them. A
repair that guessed would convert false accusations into a different set of false accusations.

**Also not attempted: the coverage problem.** DECIDE-1 found 52% of `only_touches` claims decidable
[33.5%, 70.0%] while the instrument returns a verdict on 5.7%. That gap is larger and more damaging
than the precision gap this document repairs, and it is an extraction problem, not a containment
problem. It gets its own programme. PATH-1 does not touch it and must not be reported as progress
against it.

## Predicted outcome, committed before the change

Fixing modes 1 and 2 converts 3 of the 9 false accusations into abstentions. It does not convert
any of them into correct accusations.

- accusations on the known eleven: **11 → 8**
- correct accusations: **2 → 2**
- precision: **0.18 → 0.25**
- coverage on the 299 `only_touches` claims: **unchanged, or higher** — never lower

Recording the predicted numbers in advance is the point. If the measured result is better than
0.25, that is evidence the change did something beyond its stated scope and must be investigated
before it ships, not celebrated.

## The anti-gaming condition

Precision on an accusing tool can always be improved by accusing less. DECIDE-1 makes that move
unavailable: abstaining more is now known to be a failure, not a virtue. Therefore **every
precision figure in the PATH-1 result is published beside a coverage figure on the same rows**, and
a precision gain accompanied by a coverage loss is a regression regardless of the precision number.

## Gates — committed now

- **G-P1-1 (the eleven, per item).** Before-and-after verdict for each of the 11 known accusations,
  published individually with the instrument's verbatim reason string on both sides. Blocking:
  the three targeted items must become non-accusations and the two correct accusations must remain
  CONTRADICTED.
- **G-P1-2 (coverage beside precision).** Verdict counts across all 299 `only_touches` claims before
  and after. **Blocking: coverage must not fall.** Any claim whose verdict changes in an untargeted
  direction is listed individually.
- **G-P1-3 (fresh blind audit).** 25 `only_touches` items drawn with seed 20260921 from the 288 not
  among the eleven, hand-adjudicated against the live diff **before** the post-fix verdicts are
  joined, under the DECIDE-1 rubric. Blocking: no new false-accusation mode may appear that was not
  in the published six.
- **G-P1-4 (the port agrees).** `web/gate/diffgate.js` mirrors the change; the differential prints
  zero disagreements.
- **G-P1-5 (nothing else moves).** Claim kinds other than `only_touches` produce byte-identical
  readings, ledger to ledger, with the instrument as the only variable.
- **G-P1-6 (the extension list is closed and committed).** The file-extension list used by mode 2 is
  written into the repository as data before the run, not tuned afterwards against the eleven. Any
  later addition to it is a new preregistration.

## What would cause this to be abandoned

If the fresh audit (G-P1-3) turns up a seventh failure mode with more instances than modes 1 and 2
combined, PATH-1 ships nothing and the finding is published instead. Repairing the two cheapest
modes while a larger one sits unmeasured would be optimising the number we happen to have measured.

## Honest statement of what a passing PATH-1 means

Precision 0.25 on a tool whose entire pitch is that it does not accuse wrongly. Three quarters of
its accusations would still be false. PATH-1 is a bug fix, not a product, and the result must say
so in those words. The instrument should not be described as fixed, recommended, or ready on the
strength of it.

---

*Two modes repaired, four declared unrepairable by this method, and the larger problem — that we
stay silent on half the claims we could actually check — named and left standing. The numbers are
predicted before the change so that beating them counts as a warning rather than a win.*
