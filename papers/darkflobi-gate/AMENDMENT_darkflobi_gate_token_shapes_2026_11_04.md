# AMENDMENT — token shapes: a ledger line stores the shape of an unbound numeral, never its digits

Fathom Lab · 2026-11-04 · Written by darkflobi (the build, the tests, the reading, on a copy) and the lead (the review,
the apply, the lock rows) · Amends `PREREG_darkflobi_gate_2026_10_23.md` and `RULES_darkflobi_gate_2026_10_23.md`, in
this directory, by the route the PREREG's section 12 closing paragraph provides: after the merge, a change to a rule
is a new AMENDMENT file in this folder, lock-hashed in its turn, and deliveries under the old rule are counted under
the old rule. The work was built and tested on a copy, then moved into `W:gate` on the operator's word ("Go",
2026-11-04; "The apply" below). The operator signs by merging the pull request that carries this file; "what is the
operator's and the lead's" lists what was theirs.

Receipt key: `W:` = `C:/Users/heyzo/clawd/`; `TS:` = `W:wt/_token_shapes_2026_11_04/`; `seq N` = a line of
`W:gate/ledger.jsonl`; `[R]` = a file or a run shows it; `[J]` = judgement. The live gate `W:gate` was read and its
locked files were never written (see "a stray write, recorded" below). Every change was built and tested on the copy
`TS:gate_copy/`, and the lead moves it into place at apply time.

## What the PREREG, the RULES and the ledger header say

- The ledger header, written by `W:gate/ledger.py` l.540-542 on the log's creation and present as line 1 of
  `W:gate/ledger.jsonl` (created 2026-10-24T15:09:56Z): `"certifies": "one line per outbound attempt and per gate
  event, chained; hashes and verdicts, never message text or chat ids; timestamps are the gate's statement, not evidence
  of order"`.
- `RULES` section 13 (l.552-580 at `origin/main` `310df453`): a `send` line's `numerals` carries `total`, `bound`,
  `unbound`, `uncovered`, `unbound_tokens`; the count-rule AMENDMENT (2026-10-25, item 9) adds `trailer_quotes`,
  `trailer_quote_numerals`, `trailer_quote_seqs`, `trailer_forms_unverified`.
- `RULES` l.76-78: a destination is stored as `dest_hash`, an HMAC under `W:gate/salt` "because a plain sha256 of a chat
  id is enumerable on a desktop". The salt never leaves the box.
- `W:gate/gate.py` `bind_numerals` (copy l.1048-1098 before the change) appends each unbound numeral's token as certify
  extracted it, in order, capped at 60: the raw digits of the message.

## What was observed

1. **The defect** [R, `TS:ledger_copy.jsonl`, a copy of `W:gate/ledger.jsonl` taken 2026-11-04 at seq 400; a counts-only
   scan whose output names seqs, days, classes and digit-run lengths, never a token]: the header promises no chat ids,
   and `numerals.unbound_tokens` carries the digits of the message, so a message that quotes an id puts the id on the
   line. Five lines carry a token holding a run of ten or more digits:

   | seq | day | dest_class | tokens with a run of 10+ digits | run length(s) | what the run is (the publisher's guard, 2026-11-04) |
   |---|---|---|---|---|---|
   | 129 | 2026-10-26 | operator_dm | 1 | 10 | a Telegram user id |
   | 131 | 2026-10-26 | operator_dm | 1 | 10 | a Telegram user id |
   | 136 | 2026-10-26 | operator_dm | 1 | 10 | a Telegram user id |
   | 138 | 2026-10-26 | operator_dm | 1 | 10 | a Telegram user id |
   | 351 | 2026-11-04 | operator_dm | 2 | 13, 13 | a supergroup chat id (`-100…`), twice |

   Of the 400 lines in the copy, 71 carry a non-empty `unbound_tokens`, and 40 carry a token with a run of three or more
   digits [R, same scan]. Each is the raw digits of a message: not an id in most cases, but message text all the same.
   The header's "never message text" holds only for the text as a whole.
2. **The cost** [R, the operator's brief of 2026-11-04 message 37136; the ledger page's publisher]: the publisher of
   `https://fathom-lab.github.io/styxx/ledger/` withholds those five lines. A stranger who recomputes the chain loses
   the run between the previous posted head and each withheld line.
3. **Why the digits were there** [J]: the replay (`gate_verify.py --rederive`) compared `unbound_tokens` byte for byte,
   so a stored token was the receipt that the replay counted the same numerals. A shape keeps that receipt for every
   count and every position, and drops only the digits.

## The amendment

**(a) `gate.py`.** A new function `token_shape(token)` returns the token with every run of digits replaced by the run's
length followed by `d`: `re.sub(r"\d+", lambda m: "%dd" % len(m.group(0)), token)`. So `402` → `3d`, `0044` → `4d`,
`3.1` → `1d.1d`, `026,964` → `3d,3d`, `-1004493221658` → `-13d`, and `[gate` → `[gate`. In `bind_numerals` and in
`_zero_numerals`, the key `unbound_tokens` is replaced by `unbound_shapes`. The new key holds the shape of each unbound
token, in the same order and under the same cap of 60. No other key of `numerals` changes. `total`, `bound`,
`unbound`, `uncovered` and the four trailer keys are computed exactly as before. In the patched `gate.py`, the string
`unbound_tokens` appears nowhere [R, `TS:make_patch.py` asserts it].

**(b) `gate_verify.py`.** `unbound_tokens` leaves `_COMPARE_NUMERALS`. `_diff_fields` then compares as follows:
- **A line that carries `unbound_shapes`** (written after the move) is compared on it, and a difference reads
  `numerals.unbound_shapes`.
- **A line that carries `unbound_tokens`** (written before the move) has the shapes of its stored tokens compared with
  the re-derived `unbound_shapes`, and a difference reads `numerals.unbound_tokens->shapes`.
- **A line that carries neither** is compared on the counts alone, as today.

This follows the same pattern as `_NUMERALS_SINCE_COUNT_RULE`: a key is compared only on a line that carries it.

**Why shapes and not nothing** [J]:
- **The replay keeps per-position evidence.** A line whose numerals moved from `4d` to `2d` at one position reads DRIFT
  even when every count agrees (the test of item (c)).
- **The census and the RESULT can still sort shapes.** Dotted, hyphenated and quoted forms survive the shape: `1d.1d`
  is a two-part number and `4d` a padded seq.
- **A shape is not a chat id.** A ten-digit id becomes `10d`, the same as every other ten-digit number. Enumeration
  needs the digits, and the line holds none.

**Rejected: hashing each token.** A keyed hash of a small number is enumerable from the salt holder's side and adds
nothing a stranger can check. An unkeyed hash of a ten-digit id is enumerable on a desktop: the reason RULES l.76-78
keys `dest_hash`.

**Rejected: dropping the key.** The replay would lose the per-position comparison, and a mismatch that keeps the counts
would go unseen.

## What does not change

- **Counted values:** the counts (`total`, `bound`, `unbound`, `uncovered`) and the four trailer keys.
- **What a sender sees:** the trailer text and where it is placed, and every decision, action, finding and reason code.
- **Rules:** RULES section 6's codes and section 7's matrix, the drop rule as signed, the resolver, certify and its
  scrubs, the lint, and the bars and gates of the PREREG.
- **The chain:** `ledger.py`, the header, the anchor and the census.
- **Every line already written.** Seqs 129, 131, 136, 138 and 351 keep their bytes, and so do the other lines that carry
  `unbound_tokens`. They are read under the old rule, as PREREG section 12 provides. The publisher's withholding of
  those five stays as the publisher decides; this file moves nothing that is already chained.
- **The plugin, the gateway and the installed copies.** They relay `content` and read nothing under `numerals`. Nothing
  in the decision, the trailer text or the drop rule reads tokens: in `gate.py` before the change, `unbound_tokens`
  appears at l.1006, l.1058 (docstring), l.1083, l.1092 and l.1095, all inside `_zero_numerals` and `bind_numerals`
  [R, a grep of `TS:orig/gate.py`].

## The tests

All in the copy, with the real legs of the pinned snapshot through the junctions `TS:gate_copy/snapshot_3437340b` and
`TS:gate_copy/wheel`, and a dummy salt (`os.urandom(32)`, never the live file) [R]:

- **Baseline, before any change:** `279 passed, 12 subtests passed in 665.30s (0:11:05)` [R, `TS:baseline_tests.txt`,
  last line]. An earlier attempt without a salt in the copy hung at the 25% mark and was stopped; its partial output is
  `TS:baseline_tests_nosalt_killed.txt`.
- **Rewritten to shapes**, keeping each test's intent:
  - `tests/test_gate.py` l.156;
  - `tests/test_gate_round2.py` l.96, l.108;
  - `tests/test_trailer_rule.py` l.62 (the `KEYS` set), l.178, l.189, l.224, l.259, l.288, l.359, l.374, l.391, l.397,
    l.402, l.406, l.419, l.426, l.449.

  Where two tokens collapse to one shape, the list is asserted, sorted where the original sorted. Three mentions of
  `unbound_tokens` remain in `test_trailer_rule.py`, all prose: the module docstring at l.9 and l.15, and the comment at
  l.218.
- **New tests in `tests/test_gate.py`, `TestTokenShapes`:**
  - `test_an_id_shaped_number_leaves_no_run_of_digits_in_the_line`: a message holding a ten-digit number and a
    `-100…` chat-id-shaped number gives `unbound` 2, and no string anywhere under `numerals` holds a run of three or more
    digits. The test numbers are composed at run time.
  - `test_token_shape_examples`: the six examples of item (a).
- **New tests in `tests/test_verify.py`, `TestTokenShapesCompare`:**
  - an old line with `unbound_tokens` `["0044", "7"]` re-derives SAME against `unbound_shapes` `["4d", "1d"]`, and DRIFT
    `numerals.unbound_tokens->shapes` against `["4d", "2d"]`;
  - a new line compares on its shapes;
  - a line with neither key compares on the counts alone.
- **After the change:** `285 passed, 12 subtests passed in 653.36s (0:10:53)` [R, `TS:after_tests.txt`, last line]: the
  baseline's 279 plus the 6 new tests.
- **The plugin harness was not run.** It touches the gateway log, and it is the lead's to run at apply.

## The re-derivation reading

`TS:gate_copy/gate_verify.py --rederive` was run over `TS:ledger_copy.jsonl` (400 lines, last seq 400), with absolute
paths throughout:
- `--sources` points at a copy of `W:gate/sources` (`TS:verify_src/sources`, 182 files, kept in the work folder and
  never committed), with a copy of `W:gate/config.json` beside it, same sha256.
- `STYXX_GATE_VERIFY_HOME` is set to `TS:verify_home`, and `PYTHONDONTWRITEBYTECODE=1`.
- `--snapshot-root` is `TS:gate_copy`, and `--repo` is `W:styxx`, read-only.

Reading [R, `TS:rederive_after.txt`, `TS:rederive_after.json`]: 400 lines, chain re-derives, head `656f05132c37…`;
anchor lines 11, problems 0. Re-derived: SAME 186, SAME_UNDER_SKEW 0, UNAVAILABLE_THEN 0, DRIFT 1, SKEW 0,
UNRESOLVED 0, TAMPER 0 (exit 1, from the DRIFT). The one non-SAME line is seq 46, with diffs `sent_sha256`, `numerals.total`,
`numerals.unbound`, `numerals.unbound_tokens->shapes` and no module skew. That is the lab's standing reading, the count rule's known
effect (count-rule AMENDMENT item 11, whose `numerals.unbound_tokens` diff now reads through the shapes compare). No
other line differs. In particular, seqs 129, 131, 136, 138 and 351 re-derive SAME: the shapes of their stored tokens
equal the re-derived `unbound_shapes`.

## A stray write, recorded

At 07:56:58–07:57:06 EST on 2026-11-04, an earlier attempt at the re-derivation passed `--sources` the live
`W:gate/sources`. `gate_verify._verify_env` homes itself at `sources.parent / "_verify"` unless
`STYXX_GATE_VERIFY_HOME` is set, so it rewrote 14 files that already existed under `W:gate/_verify/`:
- 7 files under `sources/*.md`;
- 7 files named `_runs/<hash>/payload.md`.

Each file's sha256 equals its name, so the bytes are unchanged and only the modification times moved. No file was
created. The run was stopped after about 10 seconds. `_verify` is scratch and not a lock row: the lead's
`lock_hashes.py check --strict` with the six amendments (relayed by the operator) read 88 rows, 0 changed, exit 0 —
after the lead removed one stray `scripts/__pycache__/surface_lint.cpython-312.pyc` from the pinned snapshot, written
2026-11-02 15:16 local by an earlier run of the pinned lint without `-B`, which had the snapshot's tree row reading
CHANGED (6165 files for a locked 6164) [R, the lead's runs, 2026-11-04 ~13:05Z].

[J] A verifier that defaults its home to a sibling of the live sources is a footgun. The lead may want
`_verify_env` to refuse a default home inside the live gate folder. That would be a separate change, not made here.

## The apply

On the operator's "Go" (2026-11-04), the lead moved the six files from `TS:patched/` into `W:gate` by staged atomic
renames (`os.replace`, each file copied beside its target as `<name>.new` first): the six renames took 2.69 ms
together at 2026-11-04T13:15:12Z; the ledger's last line was seq 410 (`delivered`, 13:08:45Z) before the move and
after it, so no line was written while the files were in mixed state; every file read, after the move, the digest of
the copy's patched file; no `.new` file remained [R, `TS:apply_move.json`]. No gateway restart: `gate.py` runs as a
subprocess per send, and the plugin's own files did not move. Then, in place:

- **The plugin harness:** `node plugin/tests/run_tests.mjs`, 85 passed, 0 failed, 71.0 s, 2026-11-04T13:17:10Z [R,
  `TS:plugin_harness.txt`]. The plugin reads nothing under `numerals`; the harness says so by passing unchanged.
- **The live re-derivation:** `gate_verify.py --rederive` over `W:gate/ledger.jsonl` (410 lines) with `--sources`
  `W:gate/sources`, `--snapshot-root W:gate`, `--repo W:styxx`, `STYXX_GATE_VERIFY_HOME` set to a folder under `TS:`
  (never the gate folder), `PYTHONDONTWRITEBYTECODE=1`: chain re-derives, head `25db4c719300…`, anchor lines 11,
  problems 0; re-derived SAME 191, SAME_UNDER_SKEW 0, UNAVAILABLE_THEN 0, DRIFT 1, SKEW 0, UNRESOLVED 0, TAMPER 0; the
  DRIFT is seq 46 [R, `TS:rederive_live.txt`, `TS:rederive_live.json`]. Five more SAME than the copy's reading over
  the 400-line copy: the lines written in between.
- **The suite in place:** `py -3.12 -B -m pytest tests -q` in `W:gate`, with the live salt and config: 285 passed,
  12 subtests passed in 637.66 s (0:10:37), exit 0 [R, `TS:inplace_tests.txt`]; the strict lock check run again after
  it reads the same 88 rows, 0 changed, unlocked 0, bytecode 0 — the suite left nothing in the folder.
- **The lock check:** `lock_hashes.py check --strict` (the tool of this pull request) with the seven amendment files,
  `--w-root C:/Users/heyzo/clawd --home C:/Users/heyzo`: seven `SUPERSEDED` lines (the declarations below), rows 88,
  changed 0, built since 0, absent 0, unreadable 0, unlocked 0, bytecode 0, exit 0. The same check with the tool as
  merged reads the six moved paths CHANGED and, with this file given, a CLASH — which is why the tool moves with this
  file. The repository's `tests/test_darkflobi_gate_prereg.py` with `STYXX_GATE_W_ROOT` naming the box: 40 passed,
  the two supersession tests among them.
- **The new `gate_sha256` stretch:** over the 20 files `gate.py` hashes (every `.py` of `W:gate` and `W:gate/helpers`,
  every file of the installed plugin folder), the digest after the move is
  `f321b1bcde60a62e7be990883d82243e5c5cd2a41fef036381e0a8b24c4a01ff`, computed by the lead with `gate.py`'s own
  recipe; the first ledger line that carries it is named in the addendum.

## The lock rows that move

Read at apply from the files moved into place (sha256 of the LF bytes on the box; each equals the copy's patched
file, `TS:apply_move.json`). For the lock tool as merged (#243, #244) a path two amendment files name with two
digests is a CLASH — "each path must carry one digest across the amendment files" — and five of these six paths
already carry a row in the count-rule or the posting-skill AMENDMENT. This file therefore comes with the tool's
ordered supersession (D9), `lock_hashes.py` as built and reviewed on the substrate branch (commit `be538fa4`),
carried here byte for byte: amendment files are read by the date their names end with, then by name; a later file
declares, per earlier file whose row it replaces, the path, that file and that row's digest, in a five-cell line that
is never a lock row; the tool prints `SUPERSEDED <path> <earlier> <digest> -> <later> <digest>` and reads the later
row, and refuses as a CLASH, before any row is read, when a declaration is missing, names another digest, or names a
file read later. The declarations stand under the table.

| what | path | sha256 | note |
|---|---|---|---|
| gate file | `W:gate/gate.py` | `c17e3d06c52ac929a1bfdc673451b04661fc70c1f24992b21d9f3deb99b239cc` | 96086 bytes; `token_shape`; `unbound_shapes` in `bind_numerals` and `_zero_numerals`; before it `4416692c…` (the count-rule and posting-skill AMENDMENTs' row) |
| gate file | `W:gate/gate_verify.py` | `5ab8454c15918734a7946bf493959506ed37f65e4234b2d499d9977c9a404ee3` | 29842 bytes; the shapes compare in `_diff_fields`; before it `049d870b…` (the same two files' row) |
| gate file | `W:gate/tests/test_gate.py` | `5a1e29fb73649f85bb4c1cf62cf3c7deab53b7c350edd589287d7f341b1b566a` | 25205 bytes; l.156 to shapes; `TestTokenShapes`; before it `626d5e9e…` (the posting-skill AMENDMENT's row) |
| gate file | `W:gate/tests/test_gate_round2.py` | `2a542e33918ef844f58974a51b067a85007ee65c4faa7dbe1d8b3ba1e73394c3` | 27529 bytes; l.96, l.108 to shapes; before it `ddb8e988…` (the posting-skill AMENDMENT's row) |
| gate file | `W:gate/tests/test_trailer_rule.py` | `d636bafc4578a976870c94d1528c6b3d3276f7508b03c2083a8f53816321d0e6` | 29201 bytes; the `KEYS` set and 15 assertions to shapes; before it `f5a19f6a…` (the count-rule AMENDMENT's row) |
| gate file | `W:gate/tests/test_verify.py` | `0fe01e730a9e6e87d3934bb5283bc47711d6a09a586ad7996f86535ded0f446b` | 13872 bytes; `TestTokenShapesCompare`; before it `669100d4…` (the PREREG's row, §11; no declaration needed) |

| | path | earlier file | its row's digest | note |
|---|---|---|---|---|
| supersedes | `W:gate/gate.py` | `AMENDMENT_darkflobi_gate_count_rule_2026_10_25.md` | `4416692cfc983dbb9bc7d0f31342b3eaaf30c3beb7e27f31acdaa991c778dfbf` | the count rule's row |
| supersedes | `W:gate/gate.py` | `AMENDMENT_darkflobi_gate_posting_skill_2026_10_25.md` | `4416692cfc983dbb9bc7d0f31342b3eaaf30c3beb7e27f31acdaa991c778dfbf` | the posting skill's row, the same digest |
| supersedes | `W:gate/gate_verify.py` | `AMENDMENT_darkflobi_gate_count_rule_2026_10_25.md` | `049d870bcdbf4726d961bc4b58d5b3134c63f06511d398c11c0f7ed57dd3487b` | the count rule's row |
| supersedes | `W:gate/gate_verify.py` | `AMENDMENT_darkflobi_gate_posting_skill_2026_10_25.md` | `049d870bcdbf4726d961bc4b58d5b3134c63f06511d398c11c0f7ed57dd3487b` | the posting skill's row, the same digest |
| supersedes | `W:gate/tests/test_gate.py` | `AMENDMENT_darkflobi_gate_posting_skill_2026_10_25.md` | `626d5e9eb98dd3b0ddead29151ace4c003419b4f3139be9e91ee26210c8c97e5` | the posting skill's row |
| supersedes | `W:gate/tests/test_gate_round2.py` | `AMENDMENT_darkflobi_gate_posting_skill_2026_10_25.md` | `ddb8e988bc1e59636c1f280a7ebb6dc8b24fef5fa227bf333ac1af220574bfff` | the posting skill's row |
| supersedes | `W:gate/tests/test_trailer_rule.py` | `AMENDMENT_darkflobi_gate_count_rule_2026_10_25.md` | `f5a19f6a801d5ce2cd31b06b50b1a4746703debb2c63ca58d5f87145b74b50ba` | the count rule's row |

This file carries 6 rows in `lock_hashes.py`'s row form.

This file declares 7 supersessions in `lock_hashes.py`'s declaration form.

This file has no row of its own: the tool prints its sha256 on the `AMENDMENT` line of every check that reads it.
`gate.py` and `gate_verify.py` both enter `gate_sha256`, so the move opens a new stretch; its digest and the seq of
the first line under it are named in the addendum.

## The public-page consequence

Lines written after the move need no withholding for this cause. Their `numerals` hold shapes, and a shape holds no
run of digits from the message. A stranger can then recompute the chain across every new line the publisher posts.
The five lines above, and any line written before the move that the publisher withholds, keep their gap. The page
should state the gap where it stands.

## What is the operator's and the lead's

1. The shape grammar (digits to `<n>d`, everything else kept). Two alternatives were considered and rejected above:
   hashing the tokens, and dropping the key.
2. `gate_verify.py` moving with it. Without the shapes compare, an old line would re-derive as DRIFT on a key it no
   longer shares.
3. The relock and the move, with the suite run in place, `check --strict` with this file among the amendments, the
   plugin harness, and `--rederive` against the live log.
4. Whether to restate `RULES` section 13's key list in a later restatement of the RULES. This file names the change
   and edits no frozen text.

Operator sign-line: the merge of the pull request that carries this file. Apply date: 2026-11-04 (13:15:12Z, the move).

## Addendum — the first line under the new stretch (filled 2026-11-08 by the lead)

- **seq 411**, `send`, 2026-11-04T14:18:13Z, `public`, `telegram_action`, decision `allow`, finding `clean`,
  `gate_sha256` `f321b1bcde60a62e7be990883d82243e5c5cd2a41fef036381e0a8b24c4a01ff` — the digest computed with
  `gate.py`'s recipe in "The apply" above, byte for byte. Its `numerals` carry `unbound_shapes` and no
  `unbound_tokens` (`total` 0: a digit-free sentence darkflobi posted about this change, Telegram message 10191).
- No line sits between seq 410 (the last under `445ac94c…`, 13:08:45Z) and seq 411; no line of the log carries a
  digest other than the five stretches' (`387fc063…`, `ff34b52d…`, `c5084df2…`, `445ac94c…`, `f321b1bc…`).
- Read at 2026-11-08 ~19:00Z over 734 lines: 324 lines under `f321b1bc…`, of them 157 send lines, every one with
  `unbound_shapes` and none with `unbound_tokens`. The public page's publisher, which withholds a line whose string
  carries an identifier's shape, withheld nothing new in its three nightly builds since the move (2026-11-05, -06, -08):
  the withheld set is still the five lines of this file's defect table [R, `exhibit` branch `ledger/withheld.json`,
  `ledger/SNAPSHOTS.jsonl`].

## What this amendment does not say

- That lines written before the move are repaired. They are not: they keep their bytes, and the five keep their ids.
- That a shape is anonymous in every case. A shape keeps lengths and separators, so a message of a known form can
  still be told apart by its shape.
- That the header's promise held before the move. It did not, on those five lines.
- That the gate reads what a sentence says. It still does not.
