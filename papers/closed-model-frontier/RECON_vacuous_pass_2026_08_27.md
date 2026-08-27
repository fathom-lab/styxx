# RECON — the silent-pass detector does not see the silent passes in our own verification machinery

Fathom Lab · 2026-08-27 · **RECON. Licenses no claim, and after measuring, proposes no
detector.** Receipts: `oath_vacuous_pass_census.json` (does the detector see them?) and
`vacuous_pass_population_census.json` (what would a detector reach, and destroy?). Harnesses:
`oath_vacuous_pass_census.py`, `vacuous_pass_population_census.py` — both deterministic and
re-runnable.

---

## The two lanes finally touched

`SYNTHESIS_entitled_to_believe_2026_08_21.md` noticed that two lanes of this lab had been building
the same thing from opposite ends without either saying so. This is what happened when they met.

`styxx.absence` states its purpose in one line: *find the places where NOT MEASURING reads as a
good result.* Over 2026-08-25 and 26, four defects answering exactly that description were found in
this repository's own verification machinery. Every one was found by accident. None was found by
the detector.

| id | where | the silent pass | how it was actually found |
|---|---|---|---|
| VP-A | `tests/test_ledger.py` | skips itself on a shallow clone, and the workflow checks out shallow — so the LEDGER's regeneration guarantee has never run in CI. A skipped test reads green on a pull request. | reading the CI workflow for an unrelated reason |
| VP-B | `tests/test_certificate_reproduces.py` | yields only documents whose receipts all resolve; the rest are dropped rather than reported. A fifth of the corpus was invisible to the drift guard — the count is in that commit. | an adversarial program audit |
| VP-C | `styxx/corpus_audit.py` | a sha mismatch is classified as absent. Every receipt hash here was recorded on Windows and is a CRLF hash, so on Linux the documents simply vanished from the guard. | CI going red on a document that passes locally |
| VP-D | v0.11 battery, gate G4′b | `all()` over an empty list is true, so a panel artifact containing no fresh draw at all cleared the bar. | an adversarial audit of the battery, *after every gate passed* |

## What the detectors say

Both detectors were run on the real pre-fix source at each case's commit, driven exactly as
`benchmarks/silent_pass` drives them, counting a flag within six lines of the defect as a catch.

**0 of 4 caught.** `styxx.absence` returns no finding at all on any of the four files.
`styxx.loops` returns findings on three of them, none within the window — they are derivation and
trust sites elsewhere in the file, not the defect.

The controls matter more than usual here, so they are reported first. A **positive control** — a
textbook `except: return {"gate": "pass"}` — is flagged. A **negative control** stays clean. The
harness is being driven correctly and the detector works.

That control exists because the first version of this census reported zero everywhere, and the
reason was that `scan_source` returns a list while `scan_path` returns a report, so the harness
read `.findings` off a list and measured nothing. It was caught before publication by asking the
obvious question of a detector that finds nothing: *would it find something it should?* A census
that had skipped that step would have published a far more dramatic and completely false result —
which is, precisely, a silent pass in the instrument built to study silent passes.

## The shape they share, which is not the shape the detector hunts

None of the four is *a healthy value returned on a crash*. That is the family `absence` does catch
— SP-1 HEALTHY_ON_CRASH, SP-5 CRASH_TO_SENTINEL — and the positive control confirms it catches it.

All four are **success by empty population**:

> The thing to be checked is filtered away upstream — by a skip, by a generator guard, by a
> mismatch reclassified as absence — and a downstream check then passes over nothing, reporting
> the same green it would report on a full population.

The emptiness is not at the function boundary where a value-returning detector looks. It is
manufactured several steps earlier, and the check that passes never learns the difference. `all()`
over `[]` is not a bug in `all()`. `pytest.skip` is not a bug in pytest. In each case the code is
locally correct and the *composition* is what lies.

The existing taxonomy comes close — SP-6 UNMEASURED_AS_MEASURED is "an empty input produces a
full, confident result" — but SP-6 as instrumented looks at a function handed an empty input. In
all four cases here nothing is handed an empty input; the population is *made* empty by a filter
that had a good reason, and the check downstream is structurally incapable of noticing.

## A fifth instance, found while writing this — and the repair that was refused

`CAPSTONE_universal_mind_2026_06_10` cites `mind_v0_validation.json`. That file is present in the
tree and its content differs from what was certified. It is not a newline artifact. It is genuine
receipt drift, and the audit reports `receipt-drift 0`.

The reason is VP-C's shape exactly: the cross-directory branch of `_resolve_receipts` accepts only
a sha match and otherwise falls through to `missing`, so a changed receipt is reclassified as an
absent one and the whole document drops out of the drift guard. A guarantee prints a zero over it.

**The obvious repair was written, and then reverted.** Resolving the lone same-named candidate and
flagging it as drift keeps the document examinable and makes the drift visible — and it is wrong.
`tests/test_corpus_audit.py` already pins why, in a sentence written long before this note: *a
same-named file with DIFFERENT content must NOT satisfy the receipt — the search is stricter than
location-trust, not looser.* This repository is full of files called `*_result.json`. Resolving one
whose content does not match would certify a document against another experiment's data **while
reporting success**, which is a worse failure than invisibility and is the precise failure this
programme exists to prevent.

So the visibility defect stays open rather than being traded for a silent-pass of its own. The fix
it actually needs is a REPORTING change — a channel that distinguishes *no file of that name
anywhere* from *a file exists and has changed* — not a resolution change, and it is owed its own
cycle. The refusal is recorded here because the tempting repair took ten minutes to write and the
reason it is wrong was already in the repository.

## What this does not show

- **Not that `styxx.absence` is broken.** Its recall on the SILENT-PASS benchmark is published,
  and published as a plateau well short of complete — the figure is recorded in
  `benchmarks/silent_pass/__init__.py` and is not restated here, because restating a number whose
  receipt this note does not carry is the very habit `OATH_CONTRACT.md` rule 9 warns about. Its
  own `LIMITS` string says a clean run is not a certificate. Missing a family it was not built for
  is a measured blind spot, not a defect, and this note is not an accusation.
- **Not that the class is four.** Four is four. They were found in two days by people looking for
  other things, which suggests density rather than exhaustion, but suggestion is not measurement.
- **Not that a detector is feasible — and the next section shows it is not.** No detector was
  built. The rejected-design arithmetic that decides such things was done first this time, and it
  came back negative; the v0.11 and v0.12 cycles are both standing reminders of what happens when
  that arithmetic is skipped or taken from the wrong population.

## The arithmetic was done, and it says do not build the detector

The section below originally said SP-9 was owed *if the arithmetic supports one*. The arithmetic
has now been done first, for once, in `vacuous_pass_population_census.json`. It does not support
one.

Five naive syntactic candidates were scored over the whole repository — 1,482 Python files — on
what each REACHES among the catalogued instances and how many sites it fires on at all. Scored
honestly out of the three instances that have a syntactic defect site, because VP-C is a
classification decision and VP-E is a data fact and no AST pattern reaches either.

Every candidate reaches at most **one** of the three, and every one costs hundreds of sites per
reach. The cheapest, flagging every `skip()` call, reaches only the shallow-clone case and is a
lint rule rather than an instrument. Nothing here is a detector; the best of them is a
highlighter with a good excuse.

**So no SP-9 preregistration should be frozen on this evidence, and none is.** The class is real —
five instances, and the lab's own detectors blind to all four it could have caught — but it is not
*syntactically separable*, and the reason is the same fact that makes it dangerous: the emptiness
is manufactured upstream of the check that passes. A local pattern cannot see a composition. Any
future attempt should be runtime rather than static — instrument the check to record the size of
the population it examined — and that is a different instrument with a different census.

**Two measurement errors were made and corrected in producing that table, and both are recorded
because they are the same error this lab keeps making.** The first run scored reach against the
*current* tree while the recorded line numbers came from *pre-fix* code, so three repaired
defects read as zeroes — measuring against the wrong population, inside the census written to
stop v0.12's exact mistake happening again. The second: VP-D's line was copied from an audit
finding's header without being checked, and at the catalogued commit that line is
`if not PANEL.exists():`, not the vacuous bars, which sit forty lines further down. A census that
trusts a cited line number is measuring the citation.

## What is owed

1. **NOT a subtype detector.** That was the obvious next item and the arithmetic above killed it.
   What may still be owed is the SEED CORPUS: adding these instances to the SILENT-PASS benchmark
   records real cases the detectors miss, and will *lower* a published recall score, which is the
   correct direction — done deliberately under a preregistration, never as a side effect of a note
   like this one.
2. **A census at the population level, not the function level.** Every candidate detector for this
   shape must answer the same question v0.12 died on: what does it reach, and what does it
   destroy? A rule that flags every `if not xs: return` in this repository would flag hundreds of
   correct guards. That number should be measured before anything is designed, and measured over
   the population the rule will actually see.
3. **The cheapest partial fix, which is not a detector at all.** Three of the four would have been
   caught by a check that costs nothing: *assert the population is non-empty before asserting
   anything about it.* `G4'b` now does. `test_certificate_reproduces` still counts skipped
   documents without reporting them. That is a lint rule and a habit, not research.

---

*The instrument built to find checks that pass without looking was itself asked whether it had
looked, and the answer had to be forced out of it by a control.*
