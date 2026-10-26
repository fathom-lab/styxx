# prereg-ci (v0)

Preregistration as CI. A PREREG seals one machine-readable decision: a ```gates table, or a ```locks block
naming a scorer program and every file it reads. Its outcome rows may name the sentences each row withdraws. On a
pull request, `prereg-ci check` re-derives the verdict of every receipt the pull request writes, or that a RESULT
it changes names, from the PREREG's sealed bytes, and reads it against what the receipt and the RESULT record. It
prints the seal, the time ladder, the outcome ladder with each gate's margin, the single-gate flip table, the
licences of the landed row, and what it did not examine. Every FAIL is a function of git objects at named commits;
no reading that needs a server decides one.

**Report-only by default.** The Action (`prereg-ci/action.yml`) writes the verdict, the JSON report and the job
summary, and exits 0 whatever the verdict, until a caller sets `report-only: "false"`. The command
`python -m styxx.prereg_ci check` always exits with its own code. The design runs it report-only on styxx through a
measurement window before any blocking is decided (its section 5); that window has not opened.

The design, its red team and its prior-art notes are in the lab's working tree, not in this repository; this
README names their sections as "the design" and quotes what it uses. Every number below names the file in this
repository that holds it.

## The commands

```
python -m styxx.prereg_ci lint    PREREG.md [--rev R]                 before data: grammar, totality, vacuity, licence coverage
python -m styxx.prereg_ci seal    PREREG.md [--commit C] [--append]   a seals manifest line (prereg-ci.seals.jsonl)
python -m styxx.prereg_ci seal    AMENDMENT.md --amends ID|PATH --seen LEVEL [--append]
python -m styxx.prereg_ci check   --base BASE [--head HEAD]           the merge gate
python -m styxx.prereg_ci ladder  RECEIPT.json --prereg P.md [--rev R]  the printout alone, for one receipt
python -m styxx.prereg_ci census  [--rev REV]                         every receipt at a revision re-derived
python -m styxx.prereg_ci stamp   PREREG.md --rfc3161 TSA_URL | --ots [--go]
```

`stamp` is the one command that opens a connection (an RFC 3161 time-stamp authority, or the OpenTimestamps
client); without `--go` it prints what it would send where, writes nothing and exits 2. Inside a checkout that
holds styxx itself, run the installed checker with `python -I -m styxx.prereg_ci`, so the checkout's own `styxx/`
is not imported. Exit codes (`styxx/prereg_ci/cli.py`'s docstring): 0 PASS or nothing to check, 1 FAIL, 2 an input
error, 3 UNCHECKABLE (a shallow clone, a missing commit, `openssl` absent under `require_time = "T3a"`), 4 an
internal error, which is no verdict.

The Action is used from a pinned commit (`uses: fathom-lab/styxx/prereg-ci@<sha>`), on `pull_request` and never
under `pull_request_target`, with `contents: read` and no secret in the job; its description says why for each.

## The four refusals a byte comparison of the plan does not give

A check that compares a PREREG's bytes, or its digest, with the registered ones (CODEOWNERS with required review on
the plan's path, honest-signal's registration rule, reproducible-science's `prereg check`) refuses an edit to the
plan. Each mutation below leaves the plan's bytes as they were sealed, and prereg-ci refuses it:

| mutation (the design's section 2.1) | what `check` prints | on styxx's history | in five minutes |
|---|---|---|---|
| 1. a RESULT prints a rung the receipt did not land on | `FAIL VERDICT_DIFFERS` (the RESULT prints Z) | A.1, `tests/test_prereg_ci_demo_224.py` | B.1 |
| 2. a new receipt, consistent with itself, scored under a bar edited outside the sealed file | `FAIL DECISION_DIGEST_UNSEALED`, and `VERDICT_DIFFERS` when the moved bar flips a gate | A.2 | B.2 |
| 3. a receipt whose recorded verdict is not the sealed table's reading of its numbers | `FAIL VERDICT_DIFFERS` (the receipt records Y) | vector `recorded/verdict-differs`, `conformance/prereg_ci/vectors_compare.json` | |
| 4. a shipped surface keeps a sentence the landed row withdraws | `FAIL LICENCE_WITHDRAWN` (file, line, column, row) | A.3, on a reconstructed table | B.4 |

Beside them, a committed receipt changed in place is `FAIL RECEIPT_REWRITTEN` (a re-run writes a new path; B.3), and
a bar moved in the sealed file itself is `FAIL DECISION_EDITED_AFTER_SEAL`. That last refusal is the one path
protection, honest-signal and reproducible-science already give; it is kept so a repository needs one tool, and the
report says so where it prints it.

**On styxx's history** (the A rows; `tests/test_prereg_ci_demo_224.py` builds each branch in a local clone and
holds the printouts to these numbers). The #224 merge passes: its receipt lands on
`BUDGET_CONFOUND__gap_does_not_survive_3x_steps` under the PREREG sealed at `3598e59d`, gates digest `578924f8...`,
and the flip table prints `G_B_long 0.1097 >= 0.15 (margin -0.0403)` among the gates whose flip moves the verdict.
A.2 re-runs the same receipt under `G_B_long` moved to 0.10 in the working copy only: the re-run records
`ROBUST__ROTATION_LOAD_BEARING__beyond_diversity` under gates `e513a1da...`, consistent with itself, and `check`
refuses it (`DECISION_DIGEST_UNSEALED`, `VERDICT_DIFFERS`) while the PREREG stays byte-identical to its seal. A.3
reads EXTERNAL-1's adjudication receipt under a table **reconstructed on 2026-10-24, after the result**
(`conformance/prereg_ci/reconstructed/external1_reconstructed.md`; not a catch): it lands on
`FAILURE__gate_claim_not_yet_true` with precision 0.23 (`conformance/prereg_ci/external1_rescore_ca347730.txt`)
against the table's bar of 0.95, and the five-word licence pattern refuses the sentence on
the 4 surfaces that still printed it at the RESULT's commit (`conformance/prereg_ci/licence_replay_ca347730.txt`).

**In five minutes** (the B rows): `examples/prereg-ci-demo/README.md`, a repository you build with `git init`, a
logistic regression on scikit-learn's bundled digits set and a three-row table; `tests/test_prereg_ci_example.py`
runs its commands.

## What a PASS does not say

Every PASS prints, at the same prominence, what it did not examine (the design's C9):

- **the run itself.** A gates PREREG locks no runner, so the receipt's numbers are read as written. A locks PREREG
  locks its scorer, not the run that wrote the receipt.
- **the order of data and seal.** Only the time ladder speaks (below), and git's order is the author's.
- **outcomes outside the decision block**, sentences outside sworn spans, result fences and licence patterns,
  discarded runs, and receipts never committed.
- **whether two equal digests mean the same thing.** A change of rounding or seeds in a runner leaves a gates
  digest where it was; a locks PREREG turns a scorer change into a moved digest.
- **data seen before the seal.** A seal bounds when the plan existed and orders nothing else. An amendment's
  `seen` field is the author's statement; the report prints it and does not check it.
- **a check an administrator bypasses.** `census` re-derives every receipt in a revision after merge, so a pair
  that would have failed is on the record.

## The time ladder (printed, never folded into PASS)

| rung | what holds | decides a FAIL? |
|---|---|---|
| T0 `NOT_ORDERED` | the seal's order commit is not a strict ancestor of the commit that added the receipt | yes (`SEAL_NOT_BEFORE_RECEIPT`) |
| T1 `AUTHOR_ORDER` | it is; the author set the commits and their dates, so this is order, not time | no |
| T2 `SERVER_ORDER` | the forge's merge times; v0 reads no server and prints "not read" | no |
| T3a `TSA_TIME` | an RFC 3161 token committed beside the sealed file verifies offline (`openssl ts -verify`) against the seal bytes and a certificate committed under `prereg-ci/tsa/` at the base side, and its time precedes the receipt's commit | under `require_time = "T3a"` |
| T3b `OTS_PROOF` | an OpenTimestamps proof parsed offline; its Bitcoin attestation is the proof's claim, which a Bitcoin node verifies, not this tool | no |
| T3c `STYXX_CLOCK` | lines of `styxx.clock`'s anchor format, read for shape, never on chain | no |

`preregistered_word`: a RESULT that calls itself preregistered while its seal reads below T3a prints
`NOTE PREREGISTERED_ON_AUTHOR_ORDER` (a FAIL under `"fail"`). The rungs are `styxx/prereg_ci/time.py`'s.

## Known misses

What v0 lets through is listed, with the reason for each, in [`KNOWN_MISSES.md`](KNOWN_MISSES.md), from the
design's step B8: a mutation battery written and run by a reader who wrote none of the code, and a fresh reading of
the branch by a second reader. Each class the fix round closed is pinned by a test in
`tests/test_prereg_ci_battery.py`. In short:

- **not refused:** a locked scorer whose verdict reads the clock, the entropy pool or the environment; every locks
  reading prints "the locked scorer's clock, entropy and environment: not watched". A landed locked scorer runs as
  code (`python -I -S`, a timeout, no network where a Linux runner permits it), which is not a sandbox; one that a
  pull request brings is not run (`SCORER_NOT_RUN`, exit 3) until it lands.
- **reached only in part:** a gates-like fence a renderer shows and the scanner does not count is refused by the
  check, the census and the lint (`FENCE_SHADOWED`), not by `core.rescore` alone; receipts are `.json` files; C0 and
  the tree's licence pass read every `.md` file and every path a receipt or a manifest line names.
- **gaps the design does not speak to:** numbers committed before the seal with the `prereg` key added after it;
  a receipt holding a key twice or a look-alike of it; look-alike metric paths and verdicts; the withdrawn sentence
  with a zero-width space, a soft hyphen or `&nbsp;` inside it; a receipt with a byte-order mark named by no RESULT.
- **from the fresh reading, kept:** an oversized `.json` a pull request touches makes the check UNCHECKABLE; the
  Action's own steps pin `actions/checkout` and `actions/setup-python` by tag until the operator pins them by sha.

## Measured limits

- **Licences match strings.** EXTERNAL-1's PREREG wrote the sentence a miss would withdraw. Registered as that
  verbatim quote, the licence matches 0 shipped surfaces, at the RESULT's commit and just before the pull request
  that withdrew the sentence (#221); `lint` refuses such a pattern before data (`LICENCE_PATTERN_FINDS_NOTHING`).
  The quote's last five words match 4 of the 4 surfaces a hand rule finds on the RESULT's day, and 3 of the 10 it
  finds just before #221; the other 7 were added 16 days after the RESULT, in other wordings
  (`conformance/prereg_ci/licence_replay_ca347730.txt`). A pattern registered before data does not reach a wording
  written after it. A pattern is matched case-insensitively over text with whitespace collapsed and curly
  apostrophes straightened; a withdrawal within 300 characters of a match (`_WINDOW` in `styxx/prereg_ci/core.py`, the window rule of
  `scripts/surface_lint.py`) makes the match allowed.
- **On the census's records** the landing rule re-derives each recorded verdict: 33 of 33 records naming a gates
  digest re-score to it (`conformance/prereg_ci/rescore_census_ca347730.txt`), and ordering the seal by its
  decision digest orders 32 of 32 distinct PREREG and receipt pairs strictly, where ordering it by the whole file
  orders 31, the miss being a PREREG whose prose moved after its smoke receipt while its gates did not
  (`conformance/prereg_ci/landing_order_ca347730.txt`). That is a check that the tool does no harm on the lab's
  records, not a measure of what it catches.
- **No value is measured yet.** Whether prereg-ci refuses anything on a pull request that a reviewer would have
  let through is what the design's measurement window and its outside frame are for; neither has run.

## Credit

prereg-ci follows neighbours that came to parts of this before it, and adopts from them by name
([`CREDITS.md`](CREDITS.md) says what each does, what prereg-ci adopts and what is still owed):
**reproducible-science** (Elliot Tower, 2026) for sealing a plan by digest with an outside timestamp, refusing a
changed plan, and amendments that state what had been seen; **Panossian (2026)** for a sealed claim matrix with the
sentence published under each verdict, and a deterministic evaluator that emits the registered verdict token;
**honest-signal** (Alessandro Cardurani, 2026) for merge-time strict ancestry with shallow clones and renames
failing closed; **preregister** (Nick Harris, 2026) for declared supports and branch reachability, adopted for v1;
**scienceverse** (Daniël Lakens and Lisa M. DeBruine, 2021) for criteria and evaluation rules written as data
before analysis.

What they do on the four mutations above is expected from their documentation and not yet tested against their
code; the run is owed, and its result goes to their authors before any sentence comparing the tools is published.

## Policies

Read from `prereg-ci.toml` at the work tree's top as the base commit holds it, so a pull request does not relax the
policy that judges it; a flag (or an Action input) wins over the file.

| key | default | the other value |
|---|---|---|
| `immutability` | `"gates"`: prose moved around a sealed fence prints `NOTE PROSE_AMENDED_IN_PLACE` | `"file"`: refused |
| `amend_after_data` | `"report"`: `NOTE AMENDED_AFTER_DATA` | `"fail"` |
| `receipt_rewrite` | `"fail"`: `RECEIPT_REWRITTEN` | `"note"`, for repositories whose receipts are build outputs |
| `require_time` | `"T1"` | `"T3a"` |
| `preregistered_word` | `"note"` | `"fail"` |
| `exclude` | none | path prefixes the pairing and the licence scan leave out |
| `surfaces`, `surfaces_exclude` | the `repo` profile of `scripts/surface_lint.py` | globs of the shipped surfaces |

## Files

`styxx/prereg_ci/` (the package: `core` is the pure half and its public names are `rescore`, `compare` and
`licences`; `git`, `scorer_run`, `time`, `report` and `cli` are not public API), `prereg-ci/action.yml` (the
composite Action), `conformance/prereg_ci/` (vectors and the copied receipts named above), `tests/test_prereg_ci_*.py`,
`examples/prereg-ci-demo/`.
