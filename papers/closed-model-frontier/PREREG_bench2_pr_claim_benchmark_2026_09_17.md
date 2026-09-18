# PREREG — BENCH-2: the PR-claim benchmark, re-frozen after its own audit voided BENCH-1

Fathom Lab · 2026-09-17 · Frozen before `bench2_oracle.py` is run and before anything is re-scored.
Supersedes `PREREG_bench1_pr_claim_benchmark_2026_09_17.md`, which failed its own blocking gate
G-B1-1 — 30 of 50 hand-audited items disagreed with the oracle against a limit of 2. That failure
and its cause are published in full in `RESULT_bench1_INVALID_2026_09_17.md`; this document does
not soften it. The instrument is unchanged and read-only again: `styxx/diffgate.py` sha256
`4ba947a8…`.

This is the third re-freeze in the programme (BC-1→BC-2, BIN-1→BIN-2, BENCH-1→BENCH-2). The
pattern is deliberate: a gate that cannot void the cycle it guards is decoration.

## What BENCH-1 got wrong, stated before the repair

Two defects, one disease. In both cases the oracle accepted the instrument's *extraction* of what
a sentence names and then treated it as a fact about the code, without ever asking whether the
extracted token was the kind of thing the rule could check.

**`only_touches`.** The oracle took the word following "only" as a path prefix. Across the corpus
that word is very often ordinary English — `the`, `with`, `are`, `one`, `3`. It then searched for
directories named `the/` and `with/`, found none, and labelled the PR CONTRADICTED. 279 of 299
decidable items — 93.3% — carried a non-path prefix.

**`symbol_added`.** The oracle accepted any token matching `[A-Za-z_]\w*` as a symbol name. From
"Added method **returning** the file descriptor for async resource tracking" it took `returning`,
looked for a definition of `returning` in the added lines, found none, and labelled the PR
CONTRADICTED.

The instrument abstained on both classes. That is the finding BENCH-1 produced and BENCH-2 must
not erase by quietly fixing the oracle and reporting a clean sheet.

## The repair, stated so a reader can audit it

Ground truth is still derived from the live diff by the three mechanical rules — `diff --git`
header count, the `b/` path of each header (or `a/` where deleted), and `+` lines that are not
`+++`. Those are unchanged. Two admissibility tests are added **in front** of them, and an
inadmissible claim is labelled UNDECIDABLE, counted, published, and never scored as either
outcome.

**PATH-SHAPED.** A stated prefix `p` is admissible only if at least one holds:

- (a) `p` contains `/`; or
- (b) `p` matches `[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8}` — a filename with an extension; or
- (c) `p` matches `[A-Za-z0-9_.-]{2,}` **and** occurs as a complete, slash-delimited segment of at
  least one path in the diff.

**Containment**, correspondingly: where `p` contains `/` or is filename-shaped, a path is inside it
only by exact match or `p + "/"` prefix. Where `p` is a bare directory name admitted by (c), a path
is inside it if `p` is any complete segment of that path — so "only touches docs" is satisfied by
`web/docs/api.md`. BENCH-1 used strict prefixing for both and would have mislabelled that.

**CODE-SHAPED.** A claimed symbol `s` is admissible only if at least one holds: `s` appears inside
backticks in the claim sentence; `s` is immediately followed by `(` in the claim sentence; `s`
contains `_` or a digit; or `s` contains an internal uppercase letter. `getAsyncId` is admissible;
`returning` is not.

`files_changed_count` is carried forward **unchanged**. It survived the BENCH-1 audit cleanly: 18
of the 50 audited items were of this kind and none disagreed, all 116 decidable sentences state a
file count in words, and the stated integers are genuine file-count claims. Changing a rule that
passed its audit in order to improve a score is the thing preregistration exists to prevent.

`tests_added` remains excluded from scoring for the reason BC-1 established: what counts as "a
test" is a judgement, and an oracle must not carry one.

## The known bias in clause (c), stated before the run

Clause (c) conditions admissibility on the diff. It can only move an item from CONTRADICTED to
UNDECIDABLE and never the reverse: if `p` occurs nowhere in the diff as a segment then no path is
inside `p`, so every path is outside and BENCH-1 would have scored it CONTRADICTED. The rule
therefore strictly removes contradictions from the scored set and biases the measured contradiction
rate **downward**. That is the conservative direction for a benchmark whose thesis is that
withholding beats accusing, and it is declared here rather than discovered later. A genuinely
false "only touches `src/`" on a PR that touches nothing under `src/` will be scored UNDECIDABLE
and lost. The cost is recall on the oracle's side, paid deliberately.

## Population

Unchanged and already frozen: `bench1_population.json`, 691 PRs drawn from the 71,016 eligible
AIDev PRs (HuggingFace `hao-li/AIDev`, CC-BY-4.0, Zenodo 10.5281/zenodo.16919272). The live diffs
are the same bytes already fetched and hashed — 568 reached, sha256 recorded per PR — so BENCH-2
re-labels the identical evidence and no PR enters or leaves the population as a result of the
repair. Re-fetching would let the population drift under the fix, which is exactly what a
re-freeze must not permit.

## Gates — committed now

- **G-B2-1 (the oracle survives audit).** 50 items hand-audited against the rendered PR, sampled
  with seed **20260919** — a different sample from BENCH-1's, so the repair is not graded on the
  items that exposed it. At most 2 disagreements. **Blocking**: failure voids BENCH-2 and forces a
  third freeze, with no third attempt permitted inside this programme without a different method.
- **G-B2-2 (the scored set is published, not chosen).** For every kind: total claims, admissible,
  and UNDECIDABLE broken out by which admissibility test rejected them. Shrinking the scored set
  until the instrument looks good is the obvious way to cheat this benchmark, so the size of the
  set it is scored on is reported beside every score, always.
- **G-B2-3 (both numbers, side by side).** Precision, recall, specificity and F1 per kind, **and**
  the false discovery rate implied at the population base rate. Never one without the other.
- **G-B2-4 (baseline on the same rows).** The string-similarity baseline is re-run on the BENCH-2
  scored set, with its threshold again swept after seeing the labels — an upper bound no
  deployable baseline could reach, reported as such. Where it beats the instrument, it is
  published beating the instrument.
- **G-B2-5 (BENCH-1's failure is carried, not buried).** Every publication of a BENCH-2 number
  links `RESULT_bench1_INVALID_2026_09_17.md`. The 279-to-1 result stands as a finding in its own
  right and is not restated as a BENCH-2 success.
- **G-B2-6 (circularity declared where it exists).** On `files_changed_count` the oracle counts
  `diff --git` headers and the instrument registers one file per header. They agree by
  construction on well-formed diffs. A high score there is evidence that two implementations of
  one rule agree, not evidence of discrimination, and is reported in those words.

## What would falsify the thesis

Unchanged from BENCH-1. The thesis is that a deterministic reading beats a judgement-based one on
specificity at a low base rate. It is falsified if the similarity baseline matches the
instrument's specificity, or if the instrument's false discovery rate at population base rate
exceeds 0.5. Either result is published as prominently as a passing one — and BENCH-1 is the
evidence that this sentence is not decorative.

---

*The first version of this benchmark called two hundred and seventy-nine honest pull requests
liars, and the instrument it was built to grade refused all but one of them. The repair is above,
the bias the repair introduces is declared above it, and the audit that could void this cycle too
runs on a sample the repair has never seen.*
