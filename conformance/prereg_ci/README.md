# The prereg-ci conformance set

Step B0 of the prereg-ci build: vectors for the two gates scorers the lab ships
(`styxx.research_capsule.score_frozen` and `styxx.protocol.Experiment.score`), for prereg-ci's
re-score (`styxx.prereg_ci.core.rescore` and `.compare`), and for the cube-split cover that reads a
table's totality with no gate cap. Label: conformance vectors for the gates scorers and the table
cover; no claim.

## What is here

| file | what it is |
|---|---|
| `vectors_score.json` | one vector per census record, per R2 replay and per synthetic case; each names its PREREG and receipt by sha256 and holds the outcome of every leg |
| `vectors_cover.json` | outcome tables (gate names and `when` rows) with the exact counts of holes and of landings per row |
| `blobs.json` | every byte object a score vector names, keyed by sha256; the lab's committed bytes as base64, synthetic ones as UTF-8 text |
| `replay.py` | the reference legs and the brute force; it imports nothing from `styxx.prereg_ci` |
| `gen_vectors.py` | builds the set from styxx's history at `REV` (in `__init__.py`), checks every vector, and writes the three JSON files |
| `vectors_compare.json` | added after the B0 review, beside the committed set: synthetic vectors for every code `compare` returns, the A.2 re-run through a fence the scanner does not count, and the readings the review asked for; its bytes are in its own `blobs` |
| `gen_vectors_compare.py` | builds `vectors_compare.json` from cases that state their outcome in advance; reads no history |

The JSON files are generated, never edited by hand. `tests/test_prereg_ci_score_vectors.py` and
`tests/test_prereg_ci_cover.py` replay every vector; only the checks that read history skip, when
`REV` is not in the object store, and they say so.

## What a score vector is

Inputs: the PREREG bytes, the receipt bytes (parsed with `json.loads`), and for a vector that
compares, what the record states (`recorded`), the digests a seal holds (`sealed`) and a RESULT's
verdict (`declared`). The id is the sha256 of the sorted, compact JSON of the name and the inputs.
The legs:

- `frozen`: `score_frozen` over `protocol._select_gates_block` of the bytes decoded as UTF-8, with
  no newline translation and no smoke branch;
- `experiment`: `Experiment(path).score(receipt, smoke=bool(receipt.get("smoke")))` over the PREREG
  written to disk and committed to a repository built for the run, which is how `styxx.seal` scores
  a pair;
- `rescore`: prereg-ci's reading, pinned on the keys the vector holds, so a key added later to its
  output moves no committed vector;
- `compare`: the findings of `compare`, pinned on code and kind.

An outcome is `{verdict, gates, gates_sha256}` or `{raises, at, message}`; the message is pinned for
`GateSpecError` only, since a Python error's text may move between interpreter versions and its
class does not. `agree` is true when `frozen` and `experiment` both refuse, or both give the same
verdict, gates and digest. Every vector on which they part ways says why, and the test pins the
list by name, so a new divergence is never silent. They fall in three groups: the differences of
DESIGN K1 (a bar that is not a finite number, a row naming an undeclared gate, a malformed row the
Experiment never reaches, a verdict that is not a string, an empty gates object, and the smoke
branch only `Experiment.score` has), the newline reading (a CRLF or lone-CR blob, which
`Experiment` reads through universal newlines and a raw decode does not), and prereg-ci's own smoke
readings.

## What a compare vector is

`vectors_compare.json` holds the same legs as a score vector (`frozen`, `experiment`, `rescore`,
`compare`) for 26 synthetic cases, and `compare` is run on every one of them (an absent record
reads as `{}`); a `compare` outcome is a list of `{code, kind}`, or `{raises: <class>}` where
`compare` refuses its arguments. Together the vectors reach every code `compare` returns:
`NOT_GATED`, `SCORER_REFUSES`, `SMOKE_AMBIGUOUS`, `DECISION_DIGEST_UNSEALED`, `VERDICT_DIFFERS`,
`RECORDED_DIGEST_ABSENT` and `RECORDED_VERDICT_ABSENT`. They pin:

- a receipt that records a gates digest against a PREREG reading `NOT_GATED` is
  `DECISION_DIGEST_UNSEALED` (DESIGN 3.4 C6(i): a PREREG with no decision holds no sealed digest);
  with no recorded digest it is a `NOT_GATED` NOTE and nothing recorded is read;
- DESIGN 4.A's A.2 (a re-run scored under a bar moved only in the working copy) through three
  fences the scanner does not count and CommonMark renders: a BOM before the opener on line 1, a
  fence in a block quote, a fence as a list item's content. `Experiment` (the reading `styxx.seal`
  uses) refuses each of the three PREREGs, and `compare` refuses the re-run's digest; beside them
  the visible fence, where `compare` gives `DECISION_DIGEST_UNSEALED` and `VERDICT_DIFFERS`;
- recorded gates read without a recorded verdict, a gates-only mismatch, an empty sealed list,
  one digest passed as a string (`TypeError`), and a table whose `receipt_pointer` is not a
  string, refused for a smoke receipt as for a scored one.

## What a cover vector is

A row matches an assignment when every literal of its `when` equals the fired value, the scorers'
own expression; the earliest matching row lands. `expect` holds the holes (assignments no row
matches), the landings per row, `total` and the unreachable rows, over `2 ** len(gates)`
assignments. With `brute` true, `expect` is brute force (`replay.brute_force`), recomputed live by
the test; with `brute` false (tables of 40, 64 and 100 gates), it is the closed form the generator
states for the table's shape.

## The numbers it reproduces

At `REV` = `ca3477308a4819b74e0d8011efcd6fd8849a5948`, the commit the prereg-ci design's census
receipts were made at. The scripts that made them belong to the design and are kept outside this
repository (its `census/rescore_census.py`, `census/replay_224.py`, `census/cover_check.py` and
`census/receipts.sh`); the generator re-derives each number from git objects and refuses to write
if one differs:

- R1: 33 committed records name a `gates_sha256`, over 32 distinct (PREREG, receipt) pairs; each,
  re-scored from the PREREG blob at the commit that added it, gives its recorded verdict and digest.
- R2: the #224 receipt under the blob sealed at `3598e59d` reads
  `BUDGET_CONFOUND__gap_does_not_survive_3x_steps` (gates digest `578924f8...`); with `G_B_long`'s
  bar moved from 0.15 to 0.10 it reads `ROBUST__ROTATION_LOAD_BEARING__beyond_diversity`
  (`e513a1da...`).
- R8: 52 gated tables; b41 is the one that is not total; no row is unreachable.
- R8b: the 15-gate leg-2 table (70 rows) is total over 32,768 assignments, where
  `research_capsule._table_total` returns None.

## Boundaries

- The census vectors seal at the commit that added the PREREG (R1's rule). The landing rule of the
  design (3.2(b)) is the git layer's, a later step; for every census PREREG the two rules give the
  same gates digest (the one PREREG where they do not, `PREREG_pmnist_untied_2026_09_02.md`, carries
  no census record).
- `newline/bom-on-the-fence-line`: a BOM before an opener on line 1 makes that line no fence for
  the scanner, so the PREREG reads `NOT_GATED`. A renderer that drops a leading BOM would show the
  fence. Pinned as the scanner reads it today, and left to the step that reads ```result and
  ```locks fences with the same rules. A fence in a block quote or a list item is the same class.
  Since the B0 review, `compare` refuses a receipt that records a gates digest against such a
  PREREG (`vectors_compare.json`, `a2/hidden-fence/*`); a receipt that records a verdict and no
  digest against it is still read as `NOT_GATED`, which the scanner step decides.
- `rescore` runs the cover under a work budget (`core._COVER_BUDGET`, 2 ** 20 subcubes): past it
  the verdict stands and `total`, `holes` and `unreachable_rows` stay None, with `cover_code`
  `COVER_UNCHECKABLE`. Every committed cover vector fits inside it (the largest visits 2,591).
- The vector files hold `gates_sha256` keys inside their inputs and outcomes; a receipt discovery
  that walks the tree for that key excludes `conformance/**`.
- The census and R2 vectors carry the lab's history as is: paths, verdict identifiers, gate names
  and the bytes. The generator scans the words it writes itself, never carried history.

## How to regenerate it

```
python -B conformance/prereg_ci/gen_vectors.py            # write the set where none is committed
python -B conformance/prereg_ci/gen_vectors.py --check    # regenerate in memory; exit 1 if a byte differs
python -B conformance/prereg_ci/gen_vectors_compare.py --check
```

Once committed, the set is never written again. A run that regenerates it byte for byte exits 0;
any other prints the vectors whose outcome moved and writes nothing. A moved outcome is a finding
about the code, never a reason to rewrite the set; new vectors go in new files.
