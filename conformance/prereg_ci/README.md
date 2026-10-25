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
| `rescore_census_ca347730.txt`, `seal_rule_ca347730.txt`, `landing_order_ca347730.txt`, `pairs_ca347730.txt` | added at step B1: the design's R1, RV1 and RV2 outputs and RV2's pair list, copied; see "The history the git layer reproduces" |
| `seals_manifest_from_SEALS_2026_09_13.jsonl` | added at step B2: the eight rows of `papers/checksum/SEALS_2026_09_13.md` as seals manifest lines; see "The SEALS record as manifest lines" |
| `gen_seals_manifest.py` | builds that file from the SEALS blob at `REV` and checks it; prints R10's `MATCH` lines |
| `vectors_fence.json` | added at step B3: the fence scanner with its info string as a parameter, for `gates`, `result` and `locks`, and what a RESULT declares; see "The fence scanner and the RESULT readers" |
| `gen_vectors_fence.py` | builds `vectors_fence.json` from cases that state their outcome in advance; reads no history |

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

## The history the git layer reproduces

Step B1 adds `styxx.prereg_ci.git`, and `tests/test_prereg_ci_history.py` re-derives three of the
design's receipts through it and `styxx.prereg_ci.core`, line by line, at `REV`. The receipts are the
outputs of the design's own probes, which live with the design outside this repository; the copies
here are those outputs with CRLF read as LF (the probes ran on Windows, whose console writes CRLF,
and a tracked file under `conformance/` holds no carriage return). Nothing else in them changed.

| copy | the design's file | its sha256 | the copy's sha256 |
|---|---|---|---|
| `rescore_census_ca347730.txt` (R1) | `census/rescore_census_ca347730.txt` | `42d3c6ba0123dc589d3e85a86ca428fe723488229eff69d29280a6cd26c85857` | `f901d7dea242bfba30d0e356a63ba53a8acb0622bbbc049f9b8035a54ed7c763` |
| `seal_rule_ca347730.txt` (RV1) | `revision_probes/seal_rule_ca347730.txt` | `ca880e9b2f016a93dc8b3d580b3226c79cb30c267c37e43769eaa3db41babebf` | `c59d1180ed033381aa1858294380bc37cf87c124ebe180d8c6f0e8ab88f4d6c9` |
| `landing_order_ca347730.txt` (RV2) | `revision_probes/landing_order_ca347730.txt` | `96e283ab57fb016c0b1ca3db6400fedac9dee54f1724b0030b765da19568ebfb` | `674e64cd01e8da324267ceb01bb2c575edd8f211eff9496432d0329e84e1e452` |
| `pairs_ca347730.txt` (RV2's input) | the 32 `PAIR` lines (lines 36-67) of `red_team_probes/distinct_pairs_ca347730.txt` | `7d2b4a0de249c2a85069fc200cd5a0955062f78dacf02d6c5f86e06c7afa83da` (the whole file) | `80c81a157ef6afe2125bd169753fe6cdeea58dc58dd6c4308c370467647c565b` |

What each copy pins, as the test reads it:

- R1: 33 records naming a gates digest, over 32 distinct (PREREG, receipt) pairs; each re-scored from
  the PREREG blob at its add commit gives the recorded verdict and digest (`TALLY {"RESCORES": 33}`),
  with the order of add commits, the count of PREREG blobs between seal and receipt (2 for #224, 1
  elsewhere) and whether the blob at `REV` is the sealed one (not for the realmodel smoke record).
- RV1: of 52 gated PREREGs, the add commit is the landing commit for 40, the file blob differs
  between the two rules for 2, and the gates digest for 1 (`PREREG_pmnist_untied_2026_09_02.md`,
  add `1cbc5997`, landing `320b3032`).
- RV2: over the 32 pairs, the order commit on the landed gates digest is strictly before the
  receipt's add commit for 32, the order commit on the landed file blob for 31 (the miss is
  `realmodel_bridge_v3_smoke.json`), the add commit for 32.

The copies carry the lab's history as is: paths, verdict identifiers and commit ids.

## Boundaries

- The census vectors seal at the commit that added the PREREG (R1's rule). The landing rule of the
  design (3.2(b)) is the git layer's (`styxx.prereg_ci.git`, step B1); for every census PREREG the
  two rules give the same gates digest (the one PREREG where they do not,
  `PREREG_pmnist_untied_2026_09_02.md`, carries no census record), and
  `tests/test_prereg_ci_history.py` shows every record reading as recorded under the landing rule.
- `newline/bom-on-the-fence-line`: a BOM before an opener on line 1 makes that line no fence for
  the scanner, so the PREREG reads `NOT_GATED`. A renderer that drops a leading BOM would show the
  fence. Pinned as the scanner reads it today, and left to the step that reads ```result and
  ```locks fences with the same rules. A fence in a block quote or a list item is the same class.
  Since the B0 review, `compare` refuses a receipt that records a gates digest against such a
  PREREG (`vectors_compare.json`, `a2/hidden-fence/*`); a receipt that records a verdict and no
  digest against it is still read as `NOT_GATED`, which the scanner step decides.
  Since step B3 the gates fence keeps this reading (sealed PREREGs depend on it), and the result and
  locks fences refuse the class: see "The fence scanner and the RESULT readers".
- `rescore` runs the cover under a work budget (`core._COVER_BUDGET`, 2 ** 20 subcubes): past it
  the verdict stands and `total`, `holes` and `unreachable_rows` stay None, with `cover_code`
  `COVER_UNCHECKABLE`. Every committed cover vector fits inside it (the largest visits 2,591).
- The vector files hold `gates_sha256` keys inside their inputs and outcomes, below their top
  level, so the git layer's record rule (a top-level `gates_sha256`, or a `protocol` entry holding
  one) reads none of them. Since the B1 review nothing is left out by default: a prefix to leave
  out is the caller's `exclude` argument, and the paths it leaves out are listed.
- The census and R2 vectors carry the lab's history as is: paths, verdict identifiers, gate names
  and the bytes. The generator scans the words it writes itself, never carried history.

## The SEALS record as manifest lines

Step B2 adds the seals manifest of DESIGN 3.2(b) (`styxx.prereg_ci.core._manifest`, the git layer's
`_manifest_*` readings, and the `seal` verb). `seals_manifest_from_SEALS_2026_09_13.jsonl` is the
lab's own record of seals in that format: `gen_seals_manifest.py` reads
`papers/checksum/SEALS_2026_09_13.md` as git stores it at `REV`, takes the table under "## the
digests" (lines 33-40), and writes one line per row. The design's receipt R10 is that each row
reproduces under `git show <commit>:<path> | sha256sum`; the generator prints R10's `MATCH` line
for each and refuses on any `DIFFERS`.

| id | kind | amends | decision | why |
|---|---|---|---|---|
| 2, 3, 6, 7 | `prereg` | null | `own` | a number alone |
| 3b, 6b, 7b | `amendment` | 3, 6, 7 | `inherits` | a number with a letter; the three CORRECTIONs hold no gates fence |
| 8 | `withdrawn` | 7 | `inherits` | its note says it "supersedes row 7" and was "Withdrawn"; its blob holds row 7's fence |

Every `seen` is null (the record states none) and every `time` is empty (no token exists). Lines
7, 7b and 8 are, byte for byte, the three example lines DESIGN 3.2(b) prints. Parsing the file
with `core._manifest` and rendering it with `core._manifest_bytes` gives its bytes back.
`tests/test_prereg_ci_manifest.py` re-runs the generator at `REV`, compares it with this file, and
re-derives each `blob_sha256` with `git show` itself. Under PLAN K10 as built, a null `seen` stands
on an amendment line only when the manifest held that line where it began and the line's commit was
on the default branch before the pull request that began it (the record a manifest imports when it
begins): a pull request that adds the manifest imports only what its base already holds, so the
record's four amendment and withdrawn lines stand at null, and an amendment that pull request
commits itself states its `seen`.

## The fence scanner and the RESULT readers

Step B3. `styxx.protocol._select_gates_block` names its info string in its code, and `styxx/protocol.py` does
not change (PLAN K4), so `styxx.prereg_ci.core._fence_at` holds the same scanner with the info string as a
parameter: the module's own fence-line pattern, the non-ASCII info string refused on any fence line, a fence a
renderer shows (at most 3 columns of indent) counted inside HTML comments too, exactly one, plain, outside every
comment, and closed. For `gates` its refusals carry the module's own words. `vectors_fence.json` holds 99
vectors, each with its outcome stated by the generator before the run:

- for `gates`, the scanner's outcome equals `_select_gates_block`'s on every vector (a text with no gates fence,
  which that module refuses with "prereg has no ... gates block", reads as `{none: true}`), and
  `tests/test_prereg_ci_result.py` shows the same over every PREREG blob the score and compare vectors name and
  over the 420 `PREREG_*.md` files in the tree at `REV` (52 gated, the 412 under `papers/` being R8's count);
- for `result` and `locks`, every attack class `tests/test_protocol_v4.py` pins for the gates fence refuses
  (a decoy in an HTML comment, a tilde fence, a case variant, the fence only in a comment, a confusable info
  string, a tab-indented opener, an unterminated comment, an unclosed fence, two fences, an indented opener),
  and so do duplicate and non-ASCII keys, read by `research_capsule`'s strict JSON;
- for `result` and `locks` only, `core._fence_json` also refuses a fence CommonMark renders and the scanner does
  not count: behind a byte-order mark on line 1, in a block quote, in a list item, or with a second word in its
  info string. The gates fence keeps the scanner's reading of these (it is `NOT_GATED`, and `compare` refuses a
  digest recorded against it, `vectors_compare.json`), since sealed PREREGs are read that way.

What a RESULT declares (`core._result_reading`, DESIGN 3.2(c)): its one result fence, a strict JSON object
naming `prereg`, `receipt` and `verdict` (three strings, and no other key), and every sworn span on a receipt's
verdict leaf (`path:<receipt>#/verdict`, read through `styxx.sworn.scan`), whose needle is the verdict. A span
is held to every check `styxx.sworn` makes from the bytes alone, in its order (the lexer's MALFORMED reasons,
including a tag in an HTML comment; a blank span; the length cap; the kind, which must be `quote`; the receipt
grammar; a directional override; one needle), and a document sworn reads as MALFORMED, or whose tags do not
lex, refuses when it names a verdict leaf. Any refusal is `RESULT_REFUSES`. A span naming its receipt by a
manifest id or a `prereg:` digest is printed (`RESULT_NOT_READ`), not paired. Two declarations for one receipt
in one RESULT that differ are `RESULT_FORMS_DIFFER` (`core._declared`).

The vectors are synthetic. The lab's own sworn spans are read by the test from git objects at `REV`: 20
markdown files name a `json#/verdict` leaf, 13 of them declare through 17 spans, and none is refused.

## How to regenerate it

```
python -B conformance/prereg_ci/gen_vectors.py            # write the set where none is committed
python -B conformance/prereg_ci/gen_vectors.py --check    # regenerate in memory; exit 1 if a byte differs
python -B conformance/prereg_ci/gen_vectors_compare.py --check
python -B conformance/prereg_ci/gen_seals_manifest.py --check
python -B conformance/prereg_ci/gen_vectors_fence.py --check
```

Once committed, the set is never written again. A run that regenerates it byte for byte exits 0;
any other prints the vectors whose outcome moved and writes nothing. A moved outcome is a finding
about the code, never a reason to rewrite the set; new vectors go in new files.
