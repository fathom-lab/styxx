# NOTE — PATH-2, eleventh pass: the licensed-difference rule moves to the verdict

2026-09-28. Branch `fix/diffgate-path-resolution` (pull request #161), head `d93dccce` before this round;
`origin/main` `2a6ce0a3` is merged in, and `main`'s `styxx/diffgate.py` there is the file 7.48.0 ships and the
scorer's baseline (sha256 `9b620e00…`, LF); `main`'s `web/gate/diffgate.js` is sha256 `06688702…`.

This note sits on top of `PREREG_path2_resolution_2026_09_17.md`, `AMENDMENT_path2_resolution_2026_09_17.md`,
`ERRATUM_path2_amendment_2026_09_17.md`, `NOTE_path1_pairs_repinned_by_path2_2026_09_25.md` and the third to
tenth-pass notes. **None of them is edited.** Where one is wrong, the correction is here (section E).

**How this round is worked.** Unlike the ninth and tenth passes, this note is written **before** the code and
committed alone, ahead of it. It states the design, the guarantee and what it does not cover, how the round-10
findings fare, and the recall cost this round **expects**. The measurements are taken afterwards, on the commits
that follow, and are recorded in `web/gate/README.md` and the CHANGELOG entry, not here; where they differ from an
expectation below, those records say so.

---

## The operator's decision

On 2026-09-28 the operator decided to move the licensed-difference rule from the **reading** level to the
**verdict** level. Ten passes compared this branch's reading of a diff with `main`'s reading of it, rule by rule
(Z-1 to Z-5, K-1 to K-5), and each round a review found a place where the comparison was incomplete: a line, a
status or a key that `main` read one way and the branch another, which no reading-level rule had listed. The
comparison now happens where the answer is given. Per claim, `main`'s own **verdict** is computed by `main`'s own
code, and the branch may differ from it only where a named repair, on its own precondition, explains the
difference.

The merge bar is unchanged: (1) the #97, #121 and #101 reproductions are fixed; (2) no claim reads worse than
`origin/main` on any door — Python `gate_diff_text`, Python `gate_diff`, the JavaScript port — under any supported
Python (3.9 to 3.14), and no new Python/JavaScript disagreement, an identical abstention in both ports allowed;
(3) the scorer cannot admit a planted defect.

---

## A. The architecture

### A.1 The reference

`main`'s reader is **vendored unchanged**:

- `styxx/_diffgate_ref.py` is byte-identical to `origin/main`'s `styxx/diffgate.py` (sha256 `9b620e00…`). Its
  module name is the file name; no byte of it is changed, because its one relative import (`from .declare import
  declaration_pass`) resolves inside the `styxx` package as it does on `main`, and `styxx/declare.py` is the same
  file on `main` and here.
- `web/gate/diffgate_ref.js` is byte-identical to `origin/main`'s `web/gate/diffgate.js` (sha256 `06688702…`). The
  port loads it with `require` in Node; the bookmarklet build wraps it, unchanged, in a function scope of its own
  (so its top-level names cannot collide with the port's and its two export lines write to a local `module` and to
  no global) and embeds it before the port.

Both files are pinned by sha256 in the tests, in `web/gate/differential/py_side.py` and in the scorer, and
`.gitattributes` marks both `-text` so that a checkout keeps the LF bytes. The reference is never edited: a change
to `main`'s reader reaches this branch only by vendoring the new file under a new pin.

### A.2 The switches

The branch's reader can be evaluated with any of the three licensed repairs switched off. The switch set is an
**explicit parameter** (`_Repairs` in the Python, the `rp` argument in the port), passed from the door down to every
function that reads it; there is no module-level state, so two evaluations with different switches can run side by
side. Each switch turns off one repair's own code and nothing else:

| repair | switched off |
|---|---|
| #97 (tiered resolution) | a path claim resolves by `main`'s loop: the earliest entry in diff order that the claim matches exactly, by suffix or by basename (the scorer's `#97` revert) |
| #121 (the dot-keeping key) | every key is `main`'s key, `p.replace("\\", "/").lstrip("./").lower()` — the status map, the sides, the header pending its pair, the claimed path, the `only_touches` prefixes (the scorer's `#121` revert) |
| #101 (one-to-one pairing) | nothing is paired: `_changed_test_defs` counts 0 and `_definition_only_changed` answers no (the scorer's `#101` revert) |

Every other layer of the branch's reader — W-1's hunk walk, F-2's split, Y-1 to Y-5, Z-1 to Z-5, K-1 to K-4 —
stays on in every evaluation. The parity layers remain needed: they keep the Python and the port reading alike
(the guard does not, A.4), and #101's pairing is read through them.

### A.3 The guard

Per claim, in both ports, and on both doors — `gate_diff_text` against the reference's `gate_diff_text`, the port's
`gateDiffText` against the reference port's `gateDiffText`, and the git door's `gate_diff` against the reference's
`gate_diff` on the same repository and range:

1. **The branch's reading** is evaluated with every repair on. This is the gate the tenth pass returned.
2. **The reference's verdict** is computed by the vendored reader on the same input (in the port, by `main`'s port).
   A branch claim is paired with the reference claim of the same kind and the same text (the sentence it was read
   from) and the same occurrence among such claims. The two readers share the claim templates, the sentence split
   and the extraction filters, so on every corpus this branch has, the two claim lists are the same list; where the
   port's extraction of a symbol's name differs from `main`'s port (a name running into U+00B2 or U+2160), the pairing
   is still by sentence and occurrence.
3. **If the branch's verdict equals the reference's, the branch's claim is kept**, with its own reason. An
   UNCHECKABLE branch claim is always kept: an abstention is never a false verdict.
4. **If it differs**, the branch is evaluated again with each single repair switched off. The difference is
   **licensed by repair R** only if switching R off gives the reference's verdict on this claim **and** R's own
   precondition holds on this claim:
   - **#97**: the claim is a path claim, the tiered resolution took its entry by the exact or the suffix tier, and
     `main`'s loop over the same file list takes another entry;
   - **#121**: a key the claim reads keeps a leading dot that `main`'s key drops — a key of the file list or of the
     per-file sides, the claimed path's key, or an `only_touches` prefix's key;
   - **#101**: a removed definition of the same name in the same file was paired — for `tests_added`, the pairing
     paired at least one test; for `symbol_added`, some file that is not created (`A`) both adds and removes a
     definition of the claimed name.

   Licensed: the branch's claim is kept. Otherwise the claim is **UNCHECKABLE**, with a reason that names the
   reference's verdict and says that no named repair explains the difference.
5. **Where the reference raises** (`main` raises `AttributeError` on a `+++ /dev/null` with no `---` line before it,
   the port a `TypeError`), or where a branch claim has no reference counterpart, there is no reference verdict to
   license a difference from: every such claim the branch decides is UNCHECKABLE.
6. **The gate's verdict**, and `--strict`, are recomputed from the final claims. The gate-level fields (the
   never-read sentences, `measured`, `why_unmeasured`, `base`, `head`) are the branch's own.

`tests_pass` is the one claim whose verdict reads something besides the diff and the summary. Its function
(`_tests_pass_verdict`, `_evidence_leg`, `_run_leg`) is `main`'s, byte for byte, and a test pins that. So the
reference is evaluated without `--run` and `--evidence` — a `--run` command is never executed twice — and the
reference's verdict for a `tests_pass` claim is that same function's result wherever the reference is measured, and
UNCHECKABLE where it is not (the diff parsed to nothing on `main`).

The switched evaluations run only when a decided claim differs from the reference, and each runs once per gate
call, so on a diff where the branch and `main` agree the guard costs one evaluation of `main`'s reader. The git door's
reference runs `git diff --name-status` and `git diff` again (`main`'s `gate_diff` reads the repository itself).

### A.4 K-5 at the sentence level

The guard holds each port to its own reference; it does not make the two ports agree. Where `main`'s Python and
`main`'s port agree and the branch's Python and port read a sentence differently, a difference the guard licenses
in one port and not the other — or a branch abstention in one port only — is a **new** Python/JavaScript
disagreement. Round 10 found exactly that (R0.1, below). So the tenth pass's K-5 moves into the guard and widens to
the sentence: **a claim read from a sentence holding any character at or past U+0080, or any of U+001C to U+001F,
reads as `main`'s same port read it** — the reference's claim, its verdict, reason and detail — in each port. Those
are the characters on which the two ports' templates can disagree: Python's `\w`, `\s`, `\b` and `re.I` against
JavaScript's ASCII `\w` and `\b`, its `\s` (U+001C to U+001F and U+0085 are Python's only, U+FEFF JavaScript's
only) and its ASCII-only case folding. Where the reference makes no such claim of the sentence, or raises, the
claim is UNCHECKABLE. The tenth pass's narrower K-5 (a non-ASCII character in the path or just before it) is removed
from the reader: every claim it reached is in such a sentence. A `tests_pass` claim is not replaced (A.3).

---

## B. The guarantee, and what it does not cover

**Stated.** For every claim, on every door, in both ports: the final verdict is the reference's verdict, or
UNCHECKABLE, or the branch's own verdict licensed by one named repair — #97, #121 or #101 — whose precondition holds
on that claim and whose switch, alone, gives the reference's verdict back. **No claim the reference decides can come
out with a different verdict unless a named repair on its own precondition explains it.** The three repairs are
therefore the only surface where a new false verdict can arise.

**Tested** (commits after this note): a property test over every corpus this branch has — the differential
corpora, every reviewer harness set, the Unicode grid, and at least 10,000 fresh randomised diffs — asserting the
statement above claim by claim; and mutation tests that plant a defect in the branch's reader outside the three
repairs and show that it can produce only abstentions (or the reference's own verdict), never a verdict the clean
branch does not give.

**Not covered**, stated so that no reader takes the guarantee for more than it is:

1. **Python/JavaScript agreement.** The guard holds each port to its own reference. Agreement between the ports
   stays an implementation property of the parity layers and of K-5 (A.4), measured by the differential, not
   guaranteed by the guard.
2. **The soundness of a licensed difference.** #121 licenses a file-list difference wherever a dotted key is
   involved; if `main`'s answer there was right only because a second error balanced its merging of dotfile twins,
   the branch's licensed answer is wrong where `main`'s was right. The reader's defence is Z-3's list of doubts
   `main`'s reading also held (a header read for its shape, a replaced `diff --git` file, an unreadable header, a
   line no reading places; this round adds two, section C). That list is not provably complete; it is the open
   surface the differential probes.
3. **A defect whose trigger needs a repair's effect.** A defect outside the three repairs acts in every switched
   evaluation too, so it cannot pass as licensed — unless it only fires on what a repair produces (a dotted key, a
   paired definition, a tiered match). Switching that repair off then also removes the defect's trigger, and the
   defect's verdict passes as the repair's. Only the tests of each repair's own reading and the scorer's oracles
   see such a defect.
4. **`main`'s own errors.** Where the reference is wrong and no repair licenses a difference, the branch gives the
   reference's wrong verdict or abstains. That is the bar: not worse than `main`.
5. **The git door reads the repository twice.** The reference's `gate_diff` runs git again; if the repository
   changes between the two reads, the two readings are of different ranges.

---

## C. The round-10 findings under the guard

The round-10 review (`round10_diffgate.json`, both reviewers) found four regression classes and four scorer gaps.
Under the guard, none of the four regressions is moot: the two verdict regressions pass through #121's licence —
which is the surface the guarantee names — and the other two are Python/JavaScript disagreements, which the guard
does not cover. Each is fixed in the reader or the guard; each scorer gap in the scorer.

| finding | under the guard alone | this round |
|---|---|---|
| R0.0 `git diff --no-prefix` / `diff.noprefix=true`: a directory named `b/` (or `a/`) is stripped as git's prefix, and a trailing-whitespace twin is merged by `strip()`; beside #121's dotted split the count moves from `main`'s right answer to a wrong one (R10-NP1, R10-NP2: 7 of 7 WORSE) | licensed by #121 (a dotted key is involved; #121 off gives `main`'s count) — **not caught** | **fixed in the reader**: a `---`/`+++` pair under a `diff --git` header neither reading can read is a Z-3 doubt `main`'s reading also held, so beside #121 the file-list claims abstain (both ports) |
| R0.1 K-5's precondition narrower than the sentences the two ports read apart (a non-ASCII character after the path, glued to the verb, in the verb's tail, U+001C to U+001F or U+0085 or U+FEFF as the separator: 335 and 405 new raw-door disagreements on the reviewer's targeted set) | a Python/JavaScript disagreement — **not covered** | **fixed**: K-5 at the sentence level, in the guard (A.4) |
| R0.2 `_hunk_is_exact` accepts any `---`/`+++` pair after an exact hunk as the next file's header; a header-shaped content pair then counts a phantom file in both readings, which balanced `main`'s dotfile merge (R10-UNDER1, R10-UNDER2; 25 WORSE cells per door) | licensed by #121 — **not caught** | **fixed in the reader**: a `---`/`+++` pair read as a header without a header's shape (no `@@` after it, or two different paths) is a Z-3 doubt `main`'s reading also held |
| R0.3 two reasons printed runtime-lowercased keys through `repr()`/`pyRepr`: #121's `only_touches` dot-miss reason and Z-5's refused-file reason (reason-only disagreements) | not a verdict — **not covered** | **fixed**: both print their keys through `_shown` (fold, then `ascii()`), in both ports |
| R1.0 the strict gates' `to_dict()`, `base`, `head` and details never compared | — | the scorer compares each strict gate's `to_dict()` with its strict-off gate's, key for key except `verdict`, and runs the report check on the strict gates, in both branches |
| R1.1 no door canary for U+2029 | — | `SPLIT_ONLY_BY_PYTHON` gains U+2029; a third canary |
| R1.2 where `main` raises, nothing anchors `why_unmeasured` | — | the scorer derives the unmeasured reason with its own code, and requires `why_unmeasured == ""` wherever `measured` |
| R1.3 corpus mode cannot reach rules whose trigger `reconstruct` cannot produce | — | raw-door canaries scored in both modes (`main` raising, a K-3 GNU `+++ /dev/null<TAB>` with no `---`, a Y-4 space after `/dev/null`, an unreadable header beside dotted twins, an under-counted hunk before a header-shaped pair) |

The four regression reproductions must read, on every door each has, the reference's verdict or UNCHECKABLE: R10-NP1
and R10-NP2 (both doors), R10-UNDER1 and R10-UNDER2, R10-K5A to R10-K5D (the reference's verdict, K-5), R10-WHY1 and
R10-WHY2 (UNCHECKABLE, one reason in both ports). Each becomes a pinned pair or, where the two ports extract different
claims from one sentence as they do on `main`, a test.

**What the guard does catch.** Every regression rounds 1 to 9 found outside the three repairs — round 9's W-1 licence
(877 and 828 WORSE cells), round 8's unlicensed file-list moves, the fifth to seventh passes' parse differences — is a
difference no named repair explains, and would now abstain on every door.

**The scorer.** The guard is a rule the scorer re-implements: `main`'s verdict from its own baseline (the same bytes,
read from `git show`), the switched readings from its own reverts of #97, #121 and #101, the three preconditions
from its own code, and the final claim, reason included, compared with the instrument's (a gate of its own). The
reading-level oracles (G-C7) move to the branch's reading before the guard, which the instrument exposes. A planted
defect in the guard — licensing without the precondition, skipping the reference, comparing with the wrong claim,
not recomputing the verdict — must fail a gate, each proved by a committed test.

---

## D. The recall cost this round expects

Measured against `main`, the recall cost is the claims `main` decides that the branch leaves UNCHECKABLE. The guard
adds to it exactly the claims the branch decided **differently** from `main` where no named repair explains the
difference; the tenth pass's own abstentions stay; K-5 at the sentence level gives `main`'s verdict back on every
claim of a sentence holding a non-ASCII character or U+001C to U+001F, which lowers the cost there.

- **On the committed differential corpus** (3,475 pairs, 7,396 claims the two readers pair; `main` raises on 2
  records) the tenth-pass branch, before any guard, decides 25 claims differently from `main`: 9 path claims
  (`main` UNCHECKABLE, the branch VERIFIED: #97's reproductions and #121's dotted names), 10 `files_changed_count`
  and 4 `only_touches` on dotted twins (#121), and 2 on one pinned pair whose `diff --git` header holds a line
  separator (F-2's split, no named repair). The expectation: 23 licensed, the 2 abstain, and the tenth pass's 454
  abstentions stay, less those K-5 gives back.
- **On the round-8 and round-9 generator sets**, built around shapes W-1, F-2 and Y-4 read differently from `main`
  (a `-- x`/`++ y` content pair inside an exact hunk, a lone CR, a GNU `/dev/null<TAB>` header), the branch reads
  right a large number of claims `main` reads wrong (41,438 cells under Python 3.12 on eight of those sets at the
  tenth pass). Those are exactly the differences no named repair explains, so most of them are expected to become
  abstentions: a real loss of those repairs' benefit, taken knowingly, because the same mechanism is the one every
  earlier regression passed through.
- **Round-10's two new doubts** (C) abstain the file-list claims beside #121's split only, as K-4 does.

The measured numbers, by kind and by door, are recorded after the code (`web/gate/README.md`, CHANGELOG).

---

## E. Corrections to earlier notes

1. **Tenth pass, section B, K-5**: "a path claim the two ports' templates may read apart reads as `main` read it",
   with the precondition "a character at or past U+0080 in the path, or just before it". The ports read more
   sentences apart than that (R0.1); K-5 is now the sentence (A.4).
2. **Tenth pass, section E.2**: "0 WORSE" and "0 new Python/port disagreement". Round 10's regressions reviewer
   measured WORSE cells on real `diff.noprefix` output (R0.0) and on under-counted hunks (R0.2), and 335 and 405 new
   raw-door disagreements on a targeted set (R0.1).
3. **Tenth pass, section C, R1.5**: "`report_violations` holds ... each gate's `to_dict()`". Not the strict gates
   (round 10, R1.0).
4. **Tenth pass, section C, R1.2**, and the scorer's docstring: the door canaries named U+0085 and U+2028 as the
   code points `str.splitlines()` breaks on and git does not; U+2029 is a third (round 10, R1.1).
5. **Tenth pass, section C, R1.1**: where `main` raises, the unmeasured reason was held only to the gate's own
   `why_unmeasured` (round 10, R1.2).

---

## Protocol changes the operator is asked to accept

The architecture itself was the operator's decision (2026-09-28). What this note adds within it:

- **The pairing** of a branch claim with a reference claim: same kind, same text, same occurrence.
- **The preconditions** as written in A.3, the #97 one read over the branch's own file list (switching #97 off is
  `main`'s loop over that list).
- **`tests_pass`**: the reference is evaluated without `--run` and `--evidence`, and its `tests_pass` verdict is the
  shared function's result wherever it is measured.
- **K-5 at the sentence level**, as the guard's rule for Python/JavaScript agreement (A.4), replacing the tenth
  pass's reading-level K-5.
- **Two Z-3 doubts**: a `---`/`+++` pair under an unreadable `diff --git` header, and a pair read as a header without
  a header's shape.
- **The scorer**: the guard re-implemented as a gate; the reading-level oracles held to the reading before the guard;
  the strict gates' reports compared; a U+2029 canary; `why_unmeasured` anchored by the scorer's own code; raw-door
  canaries in both modes.
