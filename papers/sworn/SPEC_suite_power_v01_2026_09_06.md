# SPEC — suite power: where would a wrong change be caught, and where would it ship?

Fathom Lab · 2026-09-06 · **A spec, not a result.** Frozen in its own commit before any mutant is
run and before any catalogue is read. It makes no numeric claim; the gates below are written first
so the run cannot be scored after the fact.

## Why this exists

`RESULT_mutation_coverage_2026_09_05.md` sorted the differential harness's blind spots by cause and
found six that no generator can ever reach. They are not gaps in a grammar; they are whole layers
the comparison does not touch:

- the three tree handles — `MemoryTree`, `SnapshotTree`, `GitTree`, roughly 240 lines carrying the
  entire `UNRESOLVED` reason vocabulary, the blob-mode gate, the receipt size cap and the rule that
  a snapshot may not answer for a commit it was not taken at;
- the sidecar layer — `to_sidecar`, `render`, `load_sidecar`, whose refusals are the difference
  between a document and a forged one;
- the receipt layer — `issue_receipt`, `verify_receipt`, and R9's decision that the digest covers
  the core *without* `coverage`.

The JavaScript verifier has no repository at all, so the differential compares these against
silence. `RESULT_aperture_closure_2026_09_06.md` then showed what a blind spot can hide: two real
defects, one of them verdict-changing, sitting in a region two independent instruments called clean.

**These layers are in exactly that condition right now, and no second implementation exists to
compare them against.** The only instrument left is the test suite, and nobody has measured what it
would catch.

## The question, stated as an agent would need it

An agent changing `styxx/sworn.py` has no way to know which of its edits would be caught. This
measures that directly: apply one semantic change to a layer, run the suite, and record whether the
suite **kills** the mutant (some test fails) or the mutant **survives** (everything passes).

A survivor is not a bug. It is a place where a bug would ship silently, and the list of survivors is
the honest answer to *where can this repository not defend itself*.

## The rules, each with its attack

**S1 — the oracle is a named, committed set of tests, not "the suite".** The oracle is the sworn
family: `tests/test_sworn.py`, `test_sworn_attacks.py`, `test_sworn_dogfood.py`, `test_sworn_eol.py`
and `test_capsule_sworn.py`. Running the whole repository per mutant is minutes of wall clock times
dozens of mutants, and an unaffordable measurement does not get made. The set is named in the
receipt.
*Attack:* choosing a weak oracle so the survivor list looks alarming, or a strong one so it looks
clean. *Answer:* the oracle is fixed here, before the catalogue is read, and a survivor is reported
as *survived the sworn oracle* — never as *untested*, since a mutant this set misses may still be
killed elsewhere in the repository.

**S2 — a mutant must load and the oracle must actually run.** A mutant that will not import, or that
makes collection fail, tells us nothing about defence: the suite would "catch" it for a reason that
has nothing to do with the format. Such mutants are counted and excluded from the denominator.
*Attack:* padding with broken files to inflate the kill rate. *Answer:* the non-viable count is
published beside the rate, and a mutant that fails at collection is distinguished from one that
fails a test.

**S3 — every run carries controls, and a killed control voids it.** A control is a semantics-
preserving edit — a reworded comment, a consistently renamed local. A suite that fails on one is
detecting editing, not defect. The label is decided by `conformance/sworn/control_audit.py`, from
the edit, never taken from whoever proposed it: two of twelve claimed controls were mislabelled the
last time that trust was extended.
*Attack:* a test asserting on source text rather than behaviour. *Answer:* G-K voids the run.

**S4 — the survivor list is published in full, whichever way it falls.** Every survivor is named with
its layer, its anchor and what behaviour it changed. There is no bar on the kill rate and no
threshold that makes the run a success. A suite that turns out to defend little is a finding about
this repository, not a number to be reworded.
*Attack:* reporting "kill rate 80%" and omitting which 20%. *Answer:* the survivors are the result.

**S5 — which tests killed it is recorded.** For every killed mutant the failing test ids are stored.
A layer defended by one assertion in one file is a different fact from a layer defended by thirty,
and the difference does not appear in a rate.
*Attack:* a kill rate that hides a single load-bearing test. *Answer:* the per-mutant kill set is in
the receipt, and the census reports how many mutants each test file killed.

**S6 — the run writes one receipt and never rewrites it.** A second run is a second file, and the
receipt names the oracle, the catalogue and the digest of the file mutated.
*Attack:* re-running after strengthening a test so the number improves without a record.
*Answer:* the rule this corpus already pays — a receipt is history.

## The frozen gates

| gate | quantity | bar |
|---|---|---|
| G-N | viable mutants measured | ≥ 25 |
| G-K | controls killed | **exactly 0** — a killed control VOIDS the run |
| G-L | layers with at least one viable mutant | ≥ 3 of {tree handles, sidecar, receipt, coverage} |
| G-B | the unmutated oracle passes | **required** — a run over a red baseline measures nothing |
| G-S | kill rate, killed / viable | **no bar** — reported overall and per layer, with every survivor named |

A run failing G-N or G-L is under-powered and says so in its own headline. A run failing G-K or G-B
is VOID.

## What this spec does not say

That a killed mutant means the code is correct — it means a change there would be noticed. That a
survivor is a bug: it is a place where a bug would be invisible to this oracle, which is a statement
about the tests. That the sworn oracle is the whole repository's defence; it is not, and S1 says so.
That mutation score is a quality metric — it is a map of coverage in the only sense that matters
here, *would I be caught*, and it is being drawn because two instruments were recently clean over a
region that was not.

---

*The last two legs measured what an instrument could see. This one measures what the repository can
defend, over exactly the ground no instrument can reach — and the useful half of the answer is the
part where it cannot.*
