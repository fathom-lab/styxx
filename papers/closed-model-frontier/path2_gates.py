"""PATH-2 gates (PREREG_path2_resolution_2026_09_17, as amended by AMENDMENT_path2_resolution_2026_09_17):
the instrument before the repair against the instrument after it, claim by claim, with every moved
record attributed to #97, #121 or #101.

    python path2_gates.py differential [--out FILE]
        G-P4's attribution half, over web/gate/differential's corpus files (build_corpus.py and
        fuzz_corpus.py run beforehand; the pinned pair files are committed). Record ids are named.

    python path2_gates.py corpus --shelf DIR/external1_shelf.sqlite [--limit N] [--out FILE]
        G-C0 to G-C6 over the EXTERNAL-1 shelf. Counts only: no ledger is written and no PR is named
        in the output file (the ids of any violating PR go to stderr for the operator, nowhere else).

Baseline: `styxx/diffgate.py` at BASE_COMMIT, read from this checkout's object store with `git show`
and REFUSED unless it hashes to BASE_SHA256 (LF). Repaired: this checkout's `styxx/diffgate.py`, whose
sha256 is recorded. Both run in one process on the same bytes. The shelf is opened `immutable=1`
(read only, no lock, no WAL index), so scoring a live checkout's shelf cannot write to it.

Provenance (G-C0): every payload records this file's sha256, `external1_harness.py`'s sha256 and the
git HEAD, and whether those two files, `styxx/diffgate.py` and the files both instruments import or the oracle reads
(`styxx/_xid.py`, `styxx/_fold.py`, `styxx/declare.py`, `path1_extensions.txt`; NOTE_path2_tenth_pass) are unmodified
against HEAD. A payload written from a modified tree fails G-C0, so a number cannot be cited without the bytes that
made it.

`styxx.claimdetect` is blocked for both instruments: it feeds `unparsed_claims` only, `_gate`
swallows its absence by design, and nothing here reads that field.

Reconstruction and eligibility are EXTERNAL-1's (`external1_harness._fold_statuses` / `reconstruct`,
imported unchanged): an empty body, no file records, or a reconstruction whose parse differs from the
implied status map excludes the PR -- evaluated per instrument, with the implied map keyed by that
instrument's own `_norm`.

The attribution rules are the amended table, written out below rather than imported: the tiered and
any-tier resolutions (`tiered`, `any_tier`), the one-to-one definition pairing (`_pairs`,
`_test_counts`, `defined`, `ident_at`, `symbol_def_changed`), the dot-miss / dotted-prefix exception
(`only_touches_new_accusation_allowed`) and the hunk walk behind `raw_paths` and `parse_differs`
(`hunk_walk`, `header_pair`) are this file's own code.

WHAT IS NOT INDEPENDENT (NOTE_path2_third_pass_2026_09_25, R-5; the earlier wording overstated this).
The scorer calls the repaired module for its inputs, and a defect in any of these would be attributed
to #121 by the scorer that is supposed to catch it:

    new._norm                    every key comparison: key_moved, moved_keys, collision,
                                 path_claim_by121, resolutions_differ, the only_touches prefix test
                                 and the dotted-prefix / dotdot exception
    new.parse_unified_diff       the repaired status map (the baseline's comes from BASE)
    new.parse_unified_diff_sides the added/removed sides the definition rules read (both of these
                                 are `new._read_diff` since NOTE_path2_sixth_pass W-1)
    new._header_paths            the `diff --git` half of raw_paths
    new.gate_diff_text           the repaired verdicts themselves, necessarily

`_norm` is the one of these the attribution leans on hardest, so G-C4 now carries a blocking
key-shape check: a key may move ONLY by dropping leading dots, i.e. the repaired key with `./`
stripped from its front is the baseline key. A `_norm` that stopped lower-casing, or that moved a key
any other way, fails `G-C4_key_moved_not_by_a_dot` instead of being attributed to #121. The unit
tests remain the guard for what the check cannot see. (NOTE_path2_seventh_pass: `new._norm`,
`new._read_diff` -- behind both parsers -- and `new._find_path` are now also held, record by record, to
this file's own reading of them: G-C7 below. `new._header_paths` is main's BIN-1 code, which this branch
does not change, and the oracle reads it through BASE.)

#121 is attributed per claim wherever the claim names something: a path claim by its own key or the
entry it resolves to, a compatibility claim by the paths in its detail; `only_touches` reads every
path of the PR, so the PR's moved keys are the claim's.

Inputs (G-C0): in differential mode every corpus file is hashed into the payload and a missing one is
a FAILURE, not a note on stderr -- the payload is meant to be the RESULT's receipt, and a receipt that
cannot say which corpus it read is not one. Corpus mode records the shelf's file name, its byte size
and the row counts of its `pr` and `f` tables (NOTE_path2_fourth_pass_2026_09_25, P-1); the shelf is
not hashed, because it is large and opened immutable.

COUNTERFACTUAL ATTRIBUTION (NOTE_path2_fifth_pass_2026_09_25, V-3). The fourth pass attributed F-2
and F-3 per RECORD: one stray separator anywhere in a pull request excused every move on it, of any
kind, in any direction -- the round-3 blocker itself would have passed on such a record. Every moved
claim is now attributed per CLAIM, by the repaired instrument with one rule reverted: this file loads
its own copy of the repaired module from the same bytes (`CF`), and `REVERTS` gives, as this file's own
code, what each rule's code was before it. A rule is credited with a move only if the copy with that
rule reverted gives back the baseline claim -- verdict, reason and, for a compatibility claim, detail:

    #97   `_find_path` back to the any-tier loop           #101  the pairing counts nothing
    #121  `_norm` back to lstrip("./").lower()             F-2   `_diff_lines` back to str.splitlines()
    F-3   `got` back to `^\\uFEFF?\\s*def test_`             R-1   the U+FEFF out of a reverted `got`
    V-1   the definition reading, `got`, `hit` and the claimed name back to their fourth-pass forms
    V-4   `_parent_prefix` reads nothing and `_could_lie_under` is the fourth pass's
    W-1   the two parsers back to their fifth-pass forms: no hunk counts, no U+FEFF dropped at line 1
    W-2   the definition reading, `got` and the claimed name back to their fifth-pass (V-1) forms
    A-1   an added `async def test_` no longer abstains the test count (NOTE_path2_seventh_pass)
    Y-1   the file list is always sure (NOTE_path2_eighth_pass): no header after lines no count placed, and no
          two paths one key in case, makes a count, a scope or a path claim abstain
    Y-2   no U+FEFF dropped outside the counts, none read behind an indent, none noted before a test at line 1
    Y-3   no code point is in the skew set (Unicode 13.0 to 16.0 against the table's 15.0.0)
    Y-4   `/dev/null` is the whole header path again (GNU's TAB and timestamp not cut)
    Y-5   the #101 pairing verifies `net` again (a count equal to it after a pairing reads VERIFIED)
    Z-1   `got` and BC-1 are not asked against main's reading (NOTE_path2_ninth_pass, the licensed-difference rule)
    Z-2   `hit` and BC-1 are not asked against main's reading
    Z-3   the file list is not asked against main's (an unlicensed difference, main's ports apart, a doubt main held)
    Z-4   a claim with a directory component that only the basename tier resolves verifies again
    Z-5   no added definition line is refused as a whole file

(NOTE_path2_seventh_pass: W-1's and W-2's code changed again this round -- a hunk read by its counts
only when exact; one pinned name table; a claimed name that runs on past the identifier names none --
and their reverts still give back the fifth pass, so they revert this round's code with the rest.)

(NOTE_path2_sixth_pass_2026_09_25: R-1's own code -- `got` taking a U+FEFF on any line -- is gone: W-1's
parse drops a U+FEFF at line 1 of a file and nowhere else, so W-1's revert is where a move R-1 used to
explain is given back. R-1 stays in the table only to take the U+FEFF out of a `got` that F-3, V-1 or W-2
reverted, so alone it reverts nothing and is never credited alone. V-1 and W-2 patch the same names; the
older rule's code wins when both are reverted, so V-1 reverted means the fourth pass's reading.)

Rules C-1, C-2, C-3, R-2, R-3, F-1 and F-4 have no entry of their own: each acts only through a
reading one of these rules introduced (a dotted key, or the pairing), so reverting #121 or #101
reverts it too. When no single revert gives the baseline back, the smallest set of two or three
that does is the attribution; when none does, the move is a violation whatever the table says.

Every rule in the attribution must admit the move. #97, #121 and #101 admit it only if the amended
table above does (its preconditions and its directions, unchanged). F-3, V-1 and W-2 admit moves of
the kinds they read (tests_added; V-1 and W-2 symbol_added too); A-1 a tests_added move to UNCHECKABLE and
nothing else. F-2 and W-1 read the diff's lines and
admit a move of any kind -- but only on a diff where they can act at all: F-2 where str.splitlines()
and the git split differ (`split_differs`), W-1 where the hunk counts decide a line the fifth pass read
otherwise, or a U+FEFF opens line 1 of a side (`parse_differs`). V-4 admits an only_touches move to the
off-tree abstention. A compat2_candidate flip is admitted only when F-2 alone or W-1 alone explains it,
under the same precondition (G-C6); a G-C2 eligibility move only when reverting #121 (with a key moved),
F-2 (splits differ) or W-1 (parses differ) gives the baseline's eligibility back. Every move and new
accusation a post-amendment rule explains is counted under `attribution` in the payload, by rule, kind
and transition, and moves toward VERIFIED are counted apart: none is silent.

G-C3 IS WAIVED for a move attributed only to post-amendment rules (F-2, F-3, V-1, V-4, W-1, W-2, A-1, Y-1 to Y-5;
R-1 while it was in the table; A-1, Y-1, Y-3 and Y-5 can make no accusation, and Y-2 and Y-4 only where their
precondition holds). G-C3 ("no new accusation") is asked only through the amended table, i.e.
only when #97, #121 or #101 is in the attribution. A new accusation a post-amendment rule explains is
admitted by `admits` and counted under `attribution.new_accusations_admitted` (also printed as
`G-C3_no_accusation_added.waived_for_post_amendment_rules`); it is never refused by G-C3. A reader of a
corpus payload reads that count before reading G-C3's pass.

What the counterfactual cannot see (limit 12, restated in NOTE_path2_sixth_pass). (a) A defect planted
inside a rule's own code reverts with that rule, and is caught only by that rule's admission test and
by the pinned pairs. The round-3 blocker is such a case -- it lives in #121's reading of a dotted prefix
-- and the table's direction test catches it. (b) A defect in code no rule touches, whose TRIGGER needs
a rule's effect: when the text a claim reads is text F-2, F-3, V-1, W-1 or W-2 newly exposes (a form
feed on the line the claim counts, a line W-1 now reads as content), reverting that rule also removes
the defect's trigger, so the move is credited to the rule and admitted if the rule admits its kind. A
record without such text still exposes the defect; the per-claim credit is what can be wrong.

THE ORACLES (NOTE_path2_seventh_pass_2026_09_25, G-C7, blocking). Limit 12 is closed for every rule this
file re-implements: on EVERY record, the scorer computes what the rule reads with its own code and the
repaired instrument must read the same, so a defect inside a rule's code fails a gate instead of being
credited to the rule. Written out here, not borrowed from the repair:

    F-2        new._diff_lines(diff)                  == git_lines(diff)
    W-1        new._read_diff(diff)                   == own_read(diff): the status map in diff order, the
                                                         added lines, and every file's sides, by this
                                                         file's own hunk walk (`hunk_exact`, `hunk_walk`)
    V-1, W-2   new._defined_name / new._test_name      == defined / test_name, for every added and removed
                                                         line, both sides' readings (`ident_at` reads the
                                                         name table with this file's own decoder)
    F-3, R-1   new._added_tests(blob)                 == the count of the lines `test_name` reads
    #121       new._norm(token)                       == own_key(token), for every path and every claimed
                                                         path or prefix
    #97        new._find_path(status, claimed)         == own_find_path(status, claimed): exact, suffix,
                                                         basename, each tier over every entry
    claims     every tests_added and symbol_added claim's verdict and reason == this file's own reading
               of it (`expected_tests`, `expected_symbol`): the pairing (#101, C-1), `got`, the claimed name
               and its runs-past and middle-dot rules, BC-1's no-Python abstention and BC-2's nouns

NOTE_path2_eighth_pass (round-7 protocol lens, blocker and majors). A declared claim (DECLARE-1) IS re-read, through
its canonical sentence ("Adds function X.", "Added N tests."): the seventh pass skipped it, so a defect inside a
rule's code whose trigger came through a ```styxx block was admitted. The oracles also read every
files_changed_count, only_touches and path claim (`expected_count`, `expected_only_touches`, `expected_path`:
PATH-1's containment and path shape, BC-2's second prefix, C-3, R-3, V-4's written parent and F-4's could-lie-under,
each written out here, the extension list read from its committed data file), COMPAT's C-2 surface flags and
languages on the undotted key (`compat_violations`), and the eighth pass's rules: Y-1's and Y-2's notes
(`own_read`'s fourth value against `new._diff_notes`), Y-2's U+FEFF (`own_bom_hidden`, and the note of a U+FEFF
dropped before a test at line 1), Y-3's skew set (`own_skew_test`, `claimed_name`), Y-4's /dev/null
(`own_dev_null`) and Y-5's withdrawal (`expected_tests`: a count equal to `net` after a pairing is UNCHECKABLE).
W-1's exactness rule here is the ports' (an added line opens a stretch only a context line closes), not the one
the seventh note wrote.

THE GATE (G-C1, extended, blocking). The gate-level fields -- measured, why_unmeasured,
uncovered_sentences, sentences_total -- must be the baseline's, and each instrument's verdict (strict off)
must be the one its own claims give: FAIL when a claim is CONTRADICTED, else PASS. A verdict that moves
therefore moves only with an attributed claim.

THE GIT DOOR (G-C8, blocking). Records whose every hunk is exact and whose paths are safe to write are rebuilt
as two-commit repositories in a temporary directory (NOTE_path2_ninth_pass: bare, by `git fast-import`); `gate_diff` (the git door) is run by both instruments and
scored as a door of its own, with the counterfactual's reverts acting through `gate_diff`, and the repaired git
door must read exactly what the repaired raw door reads on git's bytes (except a file-list claim the raw door
abstains on by Y-1 and git's list answers; G-C7 reads that claim from git's list). NOTE_path2_eighth_pass:
differential mode tries EVERY record (a `--git-sample` smoke run fails the gate); corpus mode every PR with a
tests_added, symbol_added, file-list (NOTE_path2_ninth_pass) or compat claim (NOTE_path2_tenth_pass) and 1 in
`--git-every` (25) of the others, up to
`--git-sample` (3,000) scored;
and the counts are part of the verdict -- every record tried accounted for, none failed to rebuild, some scored.

THE NAME TABLE AND THE SKEW SET (NOTE_path2_eighth_pass). Their sha256s are pinned in this file as literals; the
scorer refuses to run on a Python whose Unicode is not 15.0.0 (the table is checked against that database, code
point by code point); the skew set is re-derived from web/gate/xid_versions.json.

NOTE_path2_ninth_pass. The licensed-difference rules Z-1 to Z-5 have reverts and admissions (each only abstains, on
the kinds it reads) and oracles: main's reading of the diff in both of main's spellings is this file's own
(`own_main_status`, `OwnMain`; main's Python is this interpreter, main's port is spelled out), and every claim's
verdict and reason re-derives the rule. The round-8 scorer findings: G-C7 holds Y-1's git-door reading
(`_status_notes`) to this file's on every record's paths and a case-folded variant; the git door's repositories are
bare and written by `git fast-import`, so dot-led segments and case twins rebuild, and a record that deletes one file
and creates another is scored a second time with git's rename detection on; `compat_violations` re-derives C-2's
whole reading (the removed and signature_changed lists and the reason); corpus mode sends every PR with a file-list
or definition claim through the git door; G-C1 compares `uncovered_texts` and scores every record a second time with
strict on, for both instruments.

NOTE_path2_tenth_pass. K-1 to K-5 in the oracles: `own_file_list_differs` reads main's file list with and without an
exact hunk's lines (K-1, W-1 licenses no file-list difference); `own_read` and `own_status_notes` compare header paths
by this file's own decoding of styxx/_fold.py (K-2, pinned here, refused at import unless sound against this
interpreter's str.lower()), read a `+++ /dev/null` with no `---` line as no file (K-3) and hold the unplaced-line doubt
(K-4); `expected_path` reads a path the two ports may read apart by main's own resolution over main's file list (K-5).
The round-9 scorer findings: `compat_violations` has no guard (every COMPAT_EXTRAS key is required); where the baseline
raises, the gate verdict, the strict verdict and the summary-only fields are scored too; every run scores two door
canaries (DOOR_CANARIES: a U+0085 or U+2028 path, core.quotePath off, where main's --name-status split cuts the path
and Z-3 abstains at the git door); a PR the repair excludes, and any eligibility move before it is credited, is held to
the parse oracles (`parse_violations`); corpus mode sends every PR with a compat claim through the git door too;
base, head and `to_dict()` are compared (`report_violations`); and styxx/declare.py, styxx/_fold.py and
path1_extensions.txt are provenance.

NOTE_path2_eleventh_pass. The licensed-difference rule moves to the verdict: the repaired gate's claims are the GUARD's
(styxx/diffgate.py `_guard`), this reading held per claim to main's verdict, a difference kept only where one repair
switched off gives main's verdict back on the repair's own precondition. The guard is a rule this file re-implements
(G-C9, blocking): main's claims from BASE (the bytes the instrument vendors as styxx/_diffgate_ref.py, whose sha256 G-C0
holds to BASE_SHA256), the switched readings from this file's own reverts of #97, #121 and #101 on its copy of the
instrument, the pairing, the three preconditions, K-5's sentence reading (the characters the two ports' templates read
apart, by this file's own decoding of the name table) and every reason the guard prints -- and the instrument's own
switches are held to those reverts on every record (`new._Repairs` against `REVERTS`). The reading-level oracles (G-C7)
read the branch's reading BEFORE the guard (`new._evaluate_text`, `new._evaluate_git`), and the tenth pass's reading-level
K-5 leaves them with the reader; G-C8 compares the two doors' readings before the guard (their guards read two different
main readings). The round-10 scorer findings: each strict gate's `to_dict()` is compared with its strict-off gate's, key
for key but the verdict, and the report check runs on the strict gates, in both branches; U+2029 joins the git door's
canaries; the unmeasured reason is derived by this file's own code and `why_unmeasured` is "" on a measured gate; and
raw-door canaries (RAW_CANARIES: main raising, K-3's GNU `+++ /dev/null<TAB>` with no `---`, Y-4's `/dev/null` and a space,
an unreadable header beside dotted twins, an under-counted hunk before a pair without a header's shape) are scored in
both modes, for rules whose trigger `external1_harness.reconstruct` cannot produce.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import types
import unicodedata
from bisect import bisect_right
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DIFFERENTIAL = ROOT / "web" / "gate" / "differential"
PREREG = "PREREG_path2_resolution_2026_09_17.md"
AMENDMENT = "AMENDMENT_path2_resolution_2026_09_17.md"
NOTE = ["NOTE_path2_third_pass_2026_09_25.md", "NOTE_path2_fourth_pass_2026_09_25.md",
        "NOTE_path2_fifth_pass_2026_09_25.md", "NOTE_path2_sixth_pass_2026_09_25.md",
        "NOTE_path2_seventh_pass_2026_09_25.md", "NOTE_path2_eighth_pass_2026_09_27.md",
        "NOTE_path2_ninth_pass_2026_09_27.md", "NOTE_path2_tenth_pass_2026_09_28.md",
        "NOTE_path2_eleventh_pass_2026_09_28.md"]
# The baseline is "the instrument before THIS repair". The preregistration named `87dded26`, the
# origin/main this branch was cut from; the branch has since been rebased onto `98a5c368`, and
# PATH-1 (#127), the COMPAT-2 port (#126) and DECLARE-1 (#129/#130) landed in between. Scored
# against `87dded26` the gates read those three changes as PATH-2's and fail: 9 G-C1 claim-set
# differences, 4 unattributed `only_touches` moves and 2 unattributed `tests_added` moves, none of
# them this branch's. The baseline therefore moves to the rebase target, so that every move the
# gates see is a move THIS branch makes. This is a change to the protocol the preregistration set
# out, and it is recorded in NOTE_path2_third_pass_2026_09_25 (R-6) rather than made quietly. The
# preregistered baseline is kept beside it and written into every payload.
PREREG_BASE_COMMIT = "87dded26377a2d0cee1d872a7af9584646db5199"
PREREG_BASE_SHA256 = "473a7dd7c2dce7b1fefd07eaba27291090dd351a0c108b28f4812c7dc77f536d"
BASE_COMMIT = "98a5c368ba9ffa242c6862e021df7f8bad2ed8e6"
BASE_SHA256 = "9b620e00a19464589308a987819894ae7cc3c111c66a5f8a457a84b8a6c604eb"
# Every pinned-pair file the differential harness reads, so the scorer sees the same corpus the
# differential does. compat2/path1/declare1 arrived on main while this branch was open.
DIFFERENTIAL_FILES = ("corpus_real.json", "corpus_fuzz.json", "bc1_pairs.json", "compat_pairs.json",
                      "bin1_pairs.json", "compat2_pairs.json", "path1_pairs.json", "declare1_pairs.json",
                      "path2_pairs.json")
PATH_KINDS = ("file_created", "file_deleted", "file_touched")
COMPAT_EXTRAS = ("removed", "languages", "surface_removed", "signature_changed", "compat2_candidate")
# NOTE_path2_tenth_pass (round-9 scorer lens, minor): and the bytes both instruments import or the oracle reads besides --
# styxx/declare.py (DECLARE-1, `from .declare import declaration_pass`), styxx/_fold.py (K-2) and path1_extensions.txt
PROVENANCE_FILES = ("papers/closed-model-frontier/path2_gates.py", "papers/closed-model-frontier/external1_harness.py",
                    "styxx/diffgate.py", "styxx/_diffgate_ref.py", "styxx/_xid.py", "styxx/_fold.py", "styxx/declare.py",
                    "papers/closed-model-frontier/path1_extensions.txt")
# AMENDMENT C-1 as NOTE_path2_fifth_pass V-1 and NOTE_path2_sixth_pass W-2 read it, written out:
# CPython's indentation (space, tab, form feed), keywords separated by the same class, then a Python
# identifier (by the pinned name table since NOTE_path2_seventh_pass, one character at a time), and the
# character after it ASCII or the end of the line. No U+FEFF: W-1's parse drops one at line 1 of a file
# and nowhere else. No \s, \w or \b.
DEF_INDENT = r"^[ \t\f]*"
DEF_SEP = r"[ \t\f]+"
DEF_HEAD = re.compile(DEF_INDENT + r"(def|class)" + DEF_SEP)
# NOTE_path2_third_pass R-2: the REMOVED side alone accepts `async`, as the instrument does.
DEF_HEAD_REMOVED = re.compile(DEF_INDENT + r"(?:async" + DEF_SEP + r")?(def|class)" + DEF_SEP)
# The fifth pass's name end, kept for its revert (W-2).
NAME_END_V1 = r"(?=[\x00-\x2f\x3a-\x40\x5b-\x5e\x60\x7b-\x7f]|$)"
HUNK = re.compile(r"^@@ -([0-9]+)(?:,([0-9]+))? \+([0-9]+)(?:,([0-9]+))? @@")
# NOTE_path2_third_pass R-3: a dotfile prefix is one dot then a name character; `..` is not.
_DOTFILE_PREFIX = re.compile(r"^\.[^./\\]")
# NOTE_path2_fourth_pass F-2, written out here rather than borrowed from the repair.
GIT_LINE_BREAK = re.compile(r"\r\n|\r|\n")
OFF_TREE_WHY = "is relative to a directory the diff does not name (#121)"

sys.path.insert(0, str(ROOT))  # the checkout FIRST: the repaired instrument is this tree's
import styxx.diffgate as new  # noqa: E402
if not Path(new.__file__).resolve().is_relative_to(ROOT):
    sys.exit("path2_gates: styxx.diffgate resolved to an installed package, not this checkout")
sys.modules["styxx.claimdetect"] = None  # type: ignore[assignment]  -- observer blocked, see docstring
sys.path.insert(0, str(HERE))
from external1_harness import _fold_statuses, reconstruct  # noqa: E402

# NOTE_path2_seventh_pass: the name table both ports read (styxx/_xid.py). The data is the repair's; the
# decoder below is this file's own, the table must hash to the sha256 its own block pins, and on a Python
# whose Unicode database is the table's version it must equal that database, code point by code point.
# NOTE_path2_eighth_pass (round-7 protocol lens, minor): the sha256s are pinned HERE, as literals, not read from
# the block they check; and the database check is blocking -- the scorer refuses to run on a Python whose
# Unicode is not the table's (run it with py -3.12). The skew set (Y-3) is pinned the same way and, besides,
# re-derived here from web/gate/xid_versions.json, the table and `unicodedata.ucd_3_2_0` with this file's code.
XID_TABLE_SHA256_PINNED = "8df68f217cca495ab8a38ced9096213aabac4cf23927068d61397d2c9074d4cb"
XID_SKEW_SHA256_PINNED = "0b7134fd20e249f7ea69f8fcfcc1993bac0ebd5b7fe9e2e41d507594b097bfd3"
XID_UNICODE_VERSION = "15.0.0"
_XID = sys.modules.get("styxx._xid") or __import__("styxx._xid", fromlist=["TABLE"])
_XID_ENDS, _XID_MORE = "0123456789abcdefghijklmnopqrstu", "vwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _decode_name_table(table: str) -> tuple:
    starts, masks, cp, value = [], [], 0, 0
    for ch in table:
        if ch in _XID_MORE:
            value = value * 31 + _XID_MORE.index(ch)
            continue
        value = value * 31 + _XID_ENDS.index(ch)
        starts.append(cp)
        masks.append(value % 8)
        cp += value // 8
        value = 0
    if cp != 0x110000:
        sys.exit(f"path2_gates: the name table covers {cp:#x} code points, not 0x110000")
    return starts, masks


def _per_code_point(table: str) -> list:
    starts, masks = _decode_name_table(table)
    out: list = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else 0x110000
        out.extend([masks[i]] * (end - start))
    return out


def _derive_skew(xid) -> list:
    """Y-3's set, re-derived: a code point is in it when 16.0.0 gives it other bits than the table, when the
    table gives bits to one Unicode's Age places after 13.0 or 14.0, or when its general category, read as the
    three bits, moved since Unicode 3.2 (the generator's margin). Sources: web/gate/xid_versions.json."""
    src = json.loads((ROOT / "web" / "gate" / "xid_versions.json").read_text(encoding="utf-8"))["versions"]
    t15 = _per_code_point(xid.TABLE)
    m16 = _per_code_point(src["16.0.0"]["table"])
    p13 = _per_code_point(src["13.0.0"]["present_in"])
    p14 = _per_code_point(src["14.0.0"]["present_in"])

    def bits(cat: str) -> int:
        opens = cat[0] == "L" or cat == "Nl"
        return (1 if opens else 0) | (2 if opens or cat in ("Nd", "Mn", "Mc", "Pc") else 0) | (4 if cat[0] in "LN" else 0)
    old = unicodedata.ucd_3_2_0
    return [1 if (m16[c] != t15[c] or (t15[c] and not (p13[c] and p14[c]))
                  or (old.category(chr(c)) != "Cn" and bits(old.category(chr(c))) != bits(unicodedata.category(chr(c)))))
            else 0 for c in range(0x110000)]


def check_tables(xid, table_pin: str, skew_pin: str, unidata: str | None = None) -> tuple:
    """(starts, masks, skew) of styxx/_xid.py's table and skew set, or exit: this Python's Unicode must be the
    table's; each string must hash to the sha256 pinned in THIS file (not the one its own block states); the
    table must equal this Python's database code point by code point; and the skew set must be the one its
    sources give. Called at import with the literal pins; the tests call it with a planted table."""
    version = unicodedata.unidata_version if unidata is None else unidata
    if version != XID_UNICODE_VERSION:
        sys.exit(f"path2_gates: this Python reads Unicode {version}; the name table is checked "
                 f"against a {XID_UNICODE_VERSION} database only, so the scorer runs under py -3.12")
    for what, text, pin in (("table", xid.TABLE, table_pin), ("skew set", xid.SKEW, skew_pin)):
        if hashlib.sha256(text.encode("ascii")).hexdigest() != pin:
            sys.exit(f"path2_gates: styxx/_xid.py's {what} does not hash to the sha256 this file pins")
    starts, masks = _decode_name_table(xid.TABLE)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else 0x110000
        for c in range(start, end):
            ch = chr(c)
            want = (ch.isidentifier() * 1) | (("a" + ch).isidentifier() * 2) | ((ch.isalnum() or ch == "_") * 4)
            if want != masks[i]:
                sys.exit(f"path2_gates: the name table reads U+{c:04X} as {masks[i]}, this Python's "
                         f"Unicode {version} as {want}")
    derived = _derive_skew(xid)
    if derived != _per_code_point(xid.SKEW):
        sys.exit("path2_gates: styxx/_xid.py's skew set is not the one its sources give")
    return starts, masks, derived


XID_STARTS, XID_MASKS, SKEW = check_tables(_XID, XID_TABLE_SHA256_PINNED, XID_SKEW_SHA256_PINNED)
XID_TABLE_SHA256 = XID_TABLE_SHA256_PINNED
XID_CHECKED_AGAINST_DATABASE = True


def xid_mask(ch: str) -> int:
    return XID_MASKS[bisect_right(XID_STARTS, ord(ch)) - 1]


def skew(ch: str) -> bool:
    return bool(SKEW[ord(ch)])


# NOTE_path2_tenth_pass (K-2): the case fold both ports compare two header paths with (styxx/_fold.py). The data is the
# repair's; the decoder below is this file's own; the table must hash to the sha256 pinned HERE; and it must be sound
# against this interpreter's own str.lower() -- a pair of paths this Python keys alike folds alike: fold(c.lower()) ==
# fold(c) for every code point, a fold is its own fold, and Sigma's two lowercases fold alike.
FOLD_SHA256_PINNED = "a52cda82375292f71230e7e781acc24760084994812e083870b7419bdc5d5fd6"
FOLD_UNICODE_VERSION = "16.0.0"
_FOLD_MOD = sys.modules.get("styxx._fold") or __import__("styxx._fold", fromlist=["FOLD"])


def _decode_fold(table: str) -> dict:
    out: dict = {}
    for entry in table.split(","):
        head, sep, rest = entry.partition("=")
        if sep:
            out[int(head, 36)] = "".join(chr(int(x, 36)) for x in rest.split("."))
            continue
        start, count, delta, step = (int(x, 36) for x in entry.split(":"))
        for i in range(count):
            out[start + i * step] = chr(start + i * step + delta)
    return out


def check_fold(fold_mod, pin: str) -> dict:
    """{code point: fold} of styxx/_fold.py's table, or exit: it must hash to the sha256 pinned in THIS file and be
    sound against this Python's str.lower(). Called at import with the literal pin; the tests call it with a planted
    table."""
    if fold_mod.UNICODE_VERSION != FOLD_UNICODE_VERSION:
        sys.exit(f"path2_gates: styxx/_fold.py reads Unicode {fold_mod.UNICODE_VERSION}, not {FOLD_UNICODE_VERSION}")
    if hashlib.sha256(fold_mod.FOLD.encode("ascii")).hexdigest() != pin:
        sys.exit("path2_gates: styxx/_fold.py's table does not hash to the sha256 this file pins")
    m = _decode_fold(fold_mod.FOLD)
    for c in range(0x110000):
        if 0xD800 <= c <= 0xDFFF:
            continue
        ch = chr(c)
        f = ch.translate(m)
        if ch.lower().translate(m) != f or f.translate(m) != f:
            sys.exit(f"path2_gates: the case fold is not sound at U+{c:04X} against this Python's str.lower()")
    sig = chr(0x3A3)
    if len({x.translate(m) for x in (sig, chr(0x3C3), chr(0x3C2), ("A" + sig).lower()[1:])}) != 1:
        sys.exit("path2_gates: the case fold does not fold Sigma's two lowercases alike")
    return m


FOLD_MAP = check_fold(_FOLD_MOD, FOLD_SHA256_PINNED)


def own_fold(text: str) -> str:
    return text.translate(FOLD_MAP)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def _git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, timeout=60)
    if r.returncode != 0:
        sys.exit(f"path2_gates: git {' '.join(args)} failed: {r.stderr.decode('utf-8', 'replace')[:200]}")
    return r.stdout.decode("utf-8", "replace")


def _module_from(src: bytes, name: str, where: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__file__ = where
    # The file carries `from .declare import declaration_pass` (DECLARE-1, on main before this branch).
    # `styxx.declare` is byte-identical on main and on this branch -- the branch touches one file under
    # styxx/ -- so resolving the relative import against this checkout's package gives every copy the
    # same reader main has.
    mod.__package__ = "styxx"
    sys.modules[mod.__name__] = mod
    exec(compile(src.decode("utf-8"), where, "exec"), mod.__dict__)  # noqa: S102
    return mod


def load_base() -> types.ModuleType:
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:styxx/diffgate.py"],
                       capture_output=True, timeout=60)
    if r.returncode != 0:
        sys.exit(f"path2_gates: git show {BASE_COMMIT[:8]}:styxx/diffgate.py failed: "
                 f"{r.stderr.decode('utf-8', 'replace')[:200]}")
    if _sha(r.stdout) != BASE_SHA256:
        sys.exit(f"path2_gates: the baseline hashes to {_sha(r.stdout)[:16]}, not {BASE_SHA256[:16]}")
    return _module_from(r.stdout, "styxx_diffgate_base", f"<git show {BASE_COMMIT[:8]}:styxx/diffgate.py>")


BASE = load_base()
NEW_SHA256 = _sha(Path(new.__file__).read_bytes())
# NOTE_path2_fifth_pass V-3: this file's own copy of the repaired instrument, from the same bytes as
# `new`, which the counterfactual reverts patch and restore. `new` itself is never patched.
_CF_BYTES = Path(new.__file__).read_bytes()
CF = _module_from(_CF_BYTES, "styxx_diffgate_counterfactual", f"<copy of {new.__file__}>")
CF_SHA256 = _sha(_CF_BYTES)


def provenance() -> dict:
    """G-C0: the bytes that produced a payload, and whether the tree matched HEAD when they ran."""
    dirty = [line for line in _git("status", "--porcelain", "--", *PROVENANCE_FILES).splitlines() if line.strip()]
    return {"scorer_sha256": _sha(Path(__file__).read_bytes()),
            "harness_sha256": _sha((HERE / "external1_harness.py").read_bytes()),
            "declare_sha256": _sha((ROOT / "styxx" / "declare.py").read_bytes()),
            "fold_sha256": _sha((ROOT / "styxx" / "_fold.py").read_bytes()),
            "xid_sha256": _sha((ROOT / "styxx" / "_xid.py").read_bytes()),
            "path1_extensions_sha256": _sha((HERE / "path1_extensions.txt").read_bytes()),
            "repaired_sha256": NEW_SHA256,
            "counterfactual_copy_sha256": CF_SHA256,
            # NOTE_path2_eleventh_pass: the guard's reference must be the baseline, byte for byte
            "reference_sha256": _sha((ROOT / "styxx" / "_diffgate_ref.py").read_bytes()),
            "reference_is_the_baseline": _sha((ROOT / "styxx" / "_diffgate_ref.py").read_bytes()) == BASE_SHA256,
            "baseline_commit": BASE_COMMIT,
            "prereg_baseline_commit": PREREG_BASE_COMMIT,
            "baseline_moved_from_prereg": BASE_COMMIT != PREREG_BASE_COMMIT,
            "git_head": _git("rev-parse", "HEAD").strip(),
            "unmodified_against_head": not dirty,
            "modified": dirty}


# ── the counterfactual reverts: each rule's code as it was before the rule, this file's own ─────────

def _find_path_any_tier(m: types.ModuleType):
    """#97 reverted: the entry clearing any tier, in diff order (the baseline's loop)."""
    def find_path(status: dict, claimed: str):
        c = m._norm(claimed)
        for p, st in status.items():
            if p == c or p.endswith("/" + c) or Path(p).name == Path(c).name:
                return p, st
        return None, None
    return find_path


def _could_lie_under_fourth_pass(path: str, pref: str, raw: str = "") -> bool:
    """V-4 reverted: NOTE_path2_fourth_pass F-4's test, which dropped every dots-only segment."""
    want = [seg.lstrip(".") for seg in pref.split("/") if seg.strip(".")]
    have = [seg.lstrip(".") for seg in path.split("/")]
    if not want:
        return True
    return any(have[i:i + len(want)] == want for i in range(len(have) - len(want) + 1))


def _symbol_line_fourth_pass(name: str) -> re.Pattern:
    return re.compile(r"^\uFEFF?[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+" + re.escape(name) + r"(?=[ \t(:]|$)")


def _symbol_hit_fourth_pass(name: str, added_blob: str) -> bool:
    return bool(re.search(r"^\s*(?:def|class)\s+" + re.escape(name) + r"\b", added_blob, re.M))


def _defines_fourth_pass(line: str, name: str, removed: bool = False) -> bool:
    """V-1 reverted: the fourth pass read one pattern, `async` included, on both sides."""
    return bool(_symbol_line_fourth_pass(name).match(line))


def _defines_fifth_pass(line: str, name: str, removed: bool = False) -> bool:
    """W-2 reverted: the fifth pass's V-1 patterns, a U+FEFF anywhere and the ASCII name end."""
    head = r"^\uFEFF?[ \t\f]*" + (r"(?:async[ \t\f]+)?" if removed else "") + r"(?:def|class)[ \t\f]+"
    return bool(re.match(head + re.escape(name) + NAME_END_V1, line))


def _test_name_by(added_rx: re.Pattern, removed_rx: re.Pattern):
    def test_name(line: str, removed: bool = False):
        mm = (removed_rx if removed else added_rx).match(line)
        return mm.group(1) if mm else None
    return test_name


TEST_FOURTH_PASS = (re.compile(r"^\uFEFF?[ \t]*def (test_[^ \t(:]*)"),
                    re.compile(r"^\uFEFF?[ \t]*(?:async[ \t]+)?def (test_[^ \t(:]*)"))
TEST_FIFTH_PASS = (re.compile(r"^\uFEFF?[ \t\f]*def (test_[^ \t(:]*)"),
                   re.compile(r"^\uFEFF?[ \t\f]*(?:async[ \t\f]+)?def (test_[^ \t(:]*)"))


def _claimed_by_template(sentence: str, m) -> tuple:
    """W-2 (and V-1) reverted: the claimed name is the template's `name` group, and no claimed name is
    refused (the seventh pass's runs-past and middle-dot rules are W-2's, and revert with it)."""
    return m.group("name"), None


def _got_pattern(rules: frozenset) -> str:
    """`got` with F-3, V-1 and W-2 each reverted or not: F-3 moved the indent from `\\s*` to `[ \\t]*`
    (its round-3 form is `^\\uFEFF?\\s*def test_`), V-1 moved it to `[ \\t\\f]*`, and W-2 made `got` the
    pairing's own reading. (R-1's U+FEFF is in every form before W-1; W-1 reads it at line 1 only.)"""
    bom = "" if "R-1" in rules else r"\uFEFF?"
    indent = r"\s*" if "F-3" in rules else (r"[ \t]*" if "V-1" in rules else r"[ \t\f]*")
    return "^" + bom + indent + "def test_"


def _parsers_fifth_pass(m: types.ModuleType) -> dict:
    """W-1 reverted: the fifth pass's two parsers, this file's own copy -- no hunk counts, so a `--- ` or
    `+++ ` line is a header wherever it stands, the sides parser forgets its file at a `--- ` line, and no
    U+FEFF is dropped at line 1. They call the module's `_diff_lines`, `_Pending` and `_norm`, so F-2's and
    #121's reverts still act through them -- and (NOTE_path2_ninth_pass) its `_dev_null`, Y-4's reading, a later rule's
    code the fifth pass did not have: W-1's revert no longer reverts Y-4 with it, and Y-4's acts through these."""
    def parse_unified_diff(diff_text: str):
        status: dict = {}
        added: list = []
        old_path = None
        pending = None

        def flush() -> None:
            if pending is not None and pending.path() and pending.path() not in status:
                status[pending.path()] = pending.status

        for line in m._diff_lines(diff_text):
            if line.startswith("diff --git "):
                flush()
                pending = m._Pending(line)
            elif line.startswith("--- "):
                old_path = line[4:].strip()
            elif line.startswith("+++ "):
                new_ = line[4:].strip()
                if m._dev_null(new_):
                    status[m._norm(old_path[2:] if old_path.startswith("a/") else old_path)] = "D"
                elif old_path is None or m._dev_null(old_path):
                    status[m._norm(new_[2:] if new_.startswith("b/") else new_)] = "A"
                else:
                    status[m._norm(new_[2:] if new_.startswith("b/") else new_)] = "M"
                pending = None
            elif line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:])
            elif pending is not None:
                pending.note(line)
        flush()
        return status, "\n".join(added)

    def parse_unified_diff_sides(diff_text: str) -> dict:
        sides: dict = {}
        old_path = None
        cur = None
        pending = None

        def flush() -> None:
            if pending is not None and pending.path():
                sides.setdefault(pending.path(), ([], []))

        for line in m._diff_lines(diff_text):
            if line.startswith("diff --git "):
                flush()
                pending = m._Pending(line)
                cur = None
            elif line.startswith("--- "):
                old_path = line[4:].strip()
                cur = None
            elif line.startswith("+++ "):
                new_ = line[4:].strip()
                if m._dev_null(new_):
                    raw = old_path[2:] if old_path and old_path.startswith("a/") else (old_path or "")
                else:
                    raw = new_[2:] if new_.startswith("b/") else new_
                cur = m._norm(raw)
                sides.setdefault(cur, ([], []))
                pending = None
            elif cur is not None and line.startswith("+") and not line.startswith("+++"):
                sides[cur][0].append(line[1:])
            elif cur is not None and line.startswith("-") and not line.startswith("---"):
                sides[cur][1].append(line[1:])
            elif pending is not None:
                pending.note(line)
        flush()
        return sides

    return {"parse_unified_diff": parse_unified_diff, "parse_unified_diff_sides": parse_unified_diff_sides}


REVERTS = {
    "#97": lambda m: {"_find_path": _find_path_any_tier(m)},
    "#121": lambda m: {"_norm": lambda p: p.replace("\\", "/").lstrip("./").lower()},
    "#101": lambda m: {"_changed_test_defs": lambda sides, status=None: 0,
                       "_definition_only_changed": lambda name, sides, status=None: False},
    "R-1": lambda m: {},                                 # `got`'s U+FEFF, with F-3, V-1 or W-2: _got_pattern
    "F-2": lambda m: {"_diff_lines": lambda text: text.splitlines()},
    "F-3": lambda m: {},                                 # `got` only: see _got_pattern
    "V-1": lambda m: {"_test_name": _test_name_by(*TEST_FOURTH_PASS), "_defines": _defines_fourth_pass,
                      "_symbol_hit": _symbol_hit_fourth_pass, "_claimed_name": _claimed_by_template},
    "V-4": lambda m: {"_parent_prefix": lambda raw: "", "_could_lie_under": _could_lie_under_fourth_pass},
    "W-1": _parsers_fifth_pass,
    "W-2": lambda m: {"_test_name": _test_name_by(*TEST_FIFTH_PASS), "_defines": _defines_fifth_pass,
                      "_claimed_name": _claimed_by_template},
    # NOTE_path2_seventh_pass: an added `async def test_` abstains the count; reverted, it counts nothing.
    "A-1": lambda m: {"_async_tests_added": lambda sides, status=None: 0},
    # NOTE_path2_eighth_pass: Y-1 the file list is always sure; Y-2 no U+FEFF dropped outside the counts, none read
    # behind an indent, and none noted before a test at line 1; Y-3 no code point is in the skew set; Y-4 /dev/null
    # is the whole header path again; Y-5 the pairing verifies `net` again.
    "Y-1": lambda m: {"_files_unsure": lambda notes: None},
    "Y-2": lambda m: {"_line_one_bom": lambda text, at_one: text, "_bom_hidden": lambda line: None,
                      "_bom_test_note": lambda raw, text: None},
    "Y-3": lambda m: {"_skew": lambda ch: False},
    "Y-4": lambda m: {"_dev_null": lambda path: path == "/dev/null"},
    "Y-5": lambda m: {"_pairing_withdraws": lambda chg: False},
    # NOTE_path2_ninth_pass: the licensed-difference rule. Z-1 `got` and BC-1 are not asked against main's reading;
    # Z-2 nor is `hit`; Z-3 the file list is not asked against main's; Z-4 the basename tier verifies again; Z-5 no
    # definition line is refused as a whole file.
    "Z-1": lambda m: {"_tests_differ": lambda got, status, main: None},
    "Z-2": lambda m: {"_symbol_differs": lambda hit, name, name_py, name_js, status, main: None},
    "Z-3": lambda m: {"_files_differ": lambda notes: None},
    "Z-4": lambda m: {"_basename_only": lambda status, claimed, before="": None},
    "Z-5": lambda m: {"_whole_file_tests": lambda blob, sides: None,
                      "_whole_file_symbol": lambda name, blob, sides: None},
}
RULES = tuple(REVERTS)
TABLE_RULES = ("#97", "#121", "#101")
NINTH_PASS_RULES = ("Z-1", "Z-2", "Z-3", "Z-4", "Z-5")
# Where two reverts patch the same name (V-1 and W-2: the definition reading), the OLDER rule's code wins:
# W-2 was written over V-1, so V-1 reverted means the fourth pass's reading whether or not W-2 is.
PRECEDENCE = ("Z-5", "Z-4", "Z-3", "Z-2", "Z-1", "Y-5", "Y-4", "Y-3", "Y-2", "Y-1", "A-1", "W-2", "W-1", "V-4", "V-1",
              "F-3", "F-2", "R-1", "#101", "#121", "#97")
# NOTE_path2_eighth_pass: the kinds whose verdict reads the file list (Y-1 abstains them where it is unsure).
FILE_LIST_KINDS = ("files_changed_count", "only_touches") + PATH_KINDS


@contextlib.contextmanager
def reverted(m: types.ModuleType, rules):
    """`m` with every rule in `rules` reverted, restored on exit."""
    rules = frozenset(rules)
    patch: dict = {}
    for r in PRECEDENCE:
        if r in rules:
            patch.update(REVERTS[r](m))
    if rules & {"F-3", "V-1", "W-2"}:
        rx = _got_pattern(rules)
        patch["_added_tests"] = lambda blob, rx=rx: len(re.findall(rx, blob, re.M))
    missing = [k for k in patch if not hasattr(m, k)]
    if missing:
        sys.exit(f"path2_gates: the repaired module has no {missing}; the counterfactual cannot revert it")
    saved = {k: getattr(m, k) for k in patch}
    try:
        for k, v in patch.items():
            setattr(m, k, v)
        yield m
    finally:
        for k, v in saved.items():
            setattr(m, k, v)


def signature(c) -> tuple:
    """What the counterfactual must give back: verdict and reason, and a compatibility claim's detail."""
    return (c.kind, c.verdict, c.why, json.dumps(c.detail, sort_keys=True) if c.kind == "compat_claim" else "")


def admits(rule: str, k: str, vb: str, vn: str, why: str, diff: str = "", summary: str = "") -> bool:
    """Whether a post-amendment rule may make this move. The table rules are asked through the table.
    F-2 and W-1 read the diff's lines, so they may move a claim of any kind -- but only on a diff where
    they can act at all (NOTE_path2_sixth_pass): F-2 where git's split and str.splitlines() differ, W-1
    where the hunk counts decide a line the fifth pass read otherwise (`parse_differs`)."""
    if rule in ("R-1", "F-3"):
        return k == "tests_added"
    if rule == "A-1":
        return k == "tests_added" and vn == "UNCHECKABLE"
    if rule in ("V-1", "W-2"):
        return k in ("tests_added", "symbol_added")
    if rule == "F-2":
        return split_differs(diff)
    if rule == "W-1":
        return parse_differs(diff)
    if rule == "V-4":
        return k == "only_touches" and vn == "UNCHECKABLE" and why.endswith(OFF_TREE_WHY)
    # NOTE_path2_eighth_pass. Y-1, Y-3 and Y-5 only abstain: Y-1 on the file-list kinds, Y-3 on the two definition
    # kinds, Y-5 on tests_added. Y-2 reads a U+FEFF where the diff shows line 1 (a verdict can move either way) and
    # abstains elsewhere, on the definition kinds, and only on a diff with a +/- line a U+FEFF opens. Y-4 moves any
    # kind, but only on a diff whose header names /dev/null followed by a TAB.
    # A rule whose code is idle on the record as the repair reads it -- Y-1 with nothing it is unsure of (by this
    # file's own reading), Y-3 with no skew code point in the summary or the diff -- shapes no verdict there; it is
    # in an attribution only because another rule's revert wakes it (F-2 reverted splits a line on U+2028 and
    # exposes a header Y-1 then doubts), and is admitted as that.
    if rule == "Y-1":
        return (vn == "UNCHECKABLE" and k in FILE_LIST_KINDS) or not own_read(diff)[3].get("files")
    if rule == "Y-3":
        return ((vn == "UNCHECKABLE" and k in ("tests_added", "symbol_added"))
                or not any(skew(ch) for ch in summary + diff))
    if rule == "Y-2":
        return k in ("tests_added", "symbol_added") and bom_line_in(diff)
    if rule == "Y-4":
        return gnu_null_in(diff)
    if rule == "Y-5":
        return k == "tests_added" and vn == "UNCHECKABLE"
    # NOTE_path2_ninth_pass. Each licensed-difference rule only abstains, on the kinds it reads: Z-1 tests_added, Z-2
    # symbol_added, Z-3 the file-list kinds, Z-4 the path claims, Z-5 the two definition kinds. As for Y-1, a rule whose
    # abstention the repaired claim does not show shaped nothing there (another rule's revert woke it) and is admitted
    # as that; G-C7 re-derives every one of them with this file's own code either way.
    if rule in NINTH_PASS_RULES:
        kinds = {"Z-1": ("tests_added",), "Z-2": ("symbol_added",), "Z-3": FILE_LIST_KINDS, "Z-4": PATH_KINDS,
                 "Z-5": ("tests_added", "symbol_added")}[rule]
        if rule == "Z-3" and not own_read(diff)[3].get("differs"):
            return True
        return (vn == "UNCHECKABLE" and k in kinds) or abstention_owner(why, k) != rule
    raise ValueError(rule)


def abstention_owner(why: str, k: str = ""):
    """The eighth- or ninth-pass rule (or A-1) whose abstention a reason is, by its words, else None. NOTE_path2_twelfth_pass:
    and "guard", the eleventh pass's guard abstaining where no licence holds (its three reasons, which G-C9 re-derives)."""
    if any(why.startswith(x.split("{", 1)[0]) for x in (GUARD_DIFFERS, GUARD_RAISES, GUARD_ABSENT)):
        return "guard"
    if any(x in why for x in (Z3_DIFFERS, Z3_APART)):
        return "Z-3"
    if Z1_WHY in why or ((Z12_BC1 in why or Z12_RAISES in why) and k == "tests_added"):
        return "Z-1"
    if Z2_WHY in why or Z2_SKEW in why or ((Z12_BC1 in why or Z12_RAISES in why) and k == "symbol_added"):
        return "Z-2"
    if Z4_WHY in why:
        return "Z-4"
    if Z5_WHY in why:
        return "Z-5"
    if why.startswith(NOT_SURE) or any(x in why for x in (LOOSE_WHY, COLLIDE_WHY, UNCOUNTED_WHY)):
        return "Y-1"
    if Y2_WHY in why or Y2_TEST in why:
        return "Y-2"
    if "Unicode 13.0 to 16.0" in why:
        return "Y-3"
    if Y5_WHY in why:
        return "Y-5"
    if "async test functions, which this template does not count" in why:
        return "A-1"
    return None


def bom_line_in(diff: str) -> bool:
    """Y-2's precondition: an added or removed line whose text a U+FEFF opens (after any indent)."""
    return any(x[:1] in ("+", "-") and re.match("^[ \t\f]*\ufeff", x[1:]) for x in git_lines(diff))


def gnu_null_in(diff: str) -> bool:
    """Y-4's precondition: a `---` or `+++` line naming /dev/null followed by a TAB."""
    return any(x.startswith(("--- ", "+++ ")) and x[4:].strip().startswith("/dev/null\t") for x in git_lines(diff))


# ── attribution, written out independently of the repair ─────────────────────────────────────

def git_lines(diff: str) -> list:
    """The diff's lines as git writes them: split on \\r\\n, \\r and \\n, no trailing empty line."""
    lines = GIT_LINE_BREAK.split(diff)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def split_differs(diff: str) -> bool:
    """str.splitlines() and the git split differ: the only diffs on which F-2 can act, so F-2 admits a
    move only here (NOTE_path2_sixth_pass; the fifth pass had made this informational)."""
    return diff.splitlines() != git_lines(diff)


def counts(mm) -> tuple:
    """(old start, old count, new start, new count) of a hunk header; an omitted count is one line."""
    return (int(mm.group(1)), int(mm.group(2)) if mm.group(2) is not None else 1,
            int(mm.group(3)), int(mm.group(4)) if mm.group(4) is not None else 1)


def hunk_exact(lines: list, k: int) -> bool:
    """This file's own reading of NOTE_path2_seventh_pass's W-1: whether the hunk headed at `lines[k]`
    carries exactly what it declares. Walk its counts; a `--- ` line, a `+++ ` line and a line opening
    with `@@` in a row end the walk (a file header and its hunk); so does a `--- ` line right after an
    added one (git, `diff -u` and difflib write removed lines before added ones), the end of the diff, or
    a line the counts do not allow. The hunk is exact only if the counts close and the next line (past any
    `\\` marker, and, unless it is a hunk header, past blank lines) can end a hunk: the end of the diff,
    `diff --git`, a `---`/`+++` pair, a hunk header at or past this hunk's end on both sides with no
    blank line before it, `-- ` (a format-patch signature) before a line no hunk carries, or any line
    that opens with none of `+`, `-`, space, `\\`."""
    a, b, c, d = counts(HUNK.match(lines[k]))
    left_old, left_new, j, n = b, d, k + 1, len(lines)
    # NOTE_path2_eighth_pass (round-7 protocol lens, minor): one exactness rule with the ports. An added line
    # opens a stretch that only a context line closes; a `--- ` line anywhere in it ends the walk (generators
    # write a change's removed lines before its added ones, so no removed line follows an added one there).
    in_added = False
    while left_old > 0 or left_new > 0:
        if j >= n:
            return False
        x = lines[j]
        if x.startswith("--- ") and j + 2 < n and lines[j + 1].startswith("+++ ") and lines[j + 2].startswith("@@"):
            return False
        if in_added and x.startswith("--- "):
            return False                  # a `--- ` line after an added one, no context between: no generator writes that
        if x[:1] == "+" and left_new > 0:
            left_new, in_added = left_new - 1, True
        elif x[:1] == "-" and left_old > 0:
            left_old -= 1
        elif x[:1] in (" ", "") and left_old > 0 and left_new > 0:
            left_old, left_new, in_added = left_old - 1, left_new - 1, False
        elif x[:1] != "\\":
            return False
        j += 1
    while j < n and lines[j][:1] == "\\":
        j += 1
    after_close = j
    while j < n and lines[j] == "":
        j += 1
    if j == n:
        return True
    x = lines[j]
    if x.startswith("diff --git "):
        return True
    if x.startswith("--- ") and j + 1 < n and lines[j + 1].startswith("+++ "):
        return True
    if x.startswith("@@"):
        mm = HUNK.match(x)
        if not mm or j != after_close:
            return False
        a2, _b2, c2, _d2 = counts(mm)
        return a2 >= a + b and c2 >= c + d
    if x == "-- ":
        return j + 1 == n or lines[j + 1][:1] not in ("+", "-", " ", "@", "\\")
    return x[:1] not in ("+", "-", " ", "\\")


def hunk_walk(diff: str, split=None):
    """This file's own walk of a diff by its hunk counts: yields (line, where) with `where` one of
    "added" (with the new-side line number), "removed" (old-side number), "context" or "outside".
    Only an exact hunk (`hunk_exact`) is walked by its counts; any other is "outside", read as main read
    it (NOTE_path2_seventh_pass). `split` is git's split unless another is given."""
    lines = (split or git_lines)(diff)
    old_left = new_left = old_no = new_no = 0
    for k, line in enumerate(lines):
        if old_left or new_left:
            head = line[:1]
            if head == "+" and new_left:
                yield line, ("added", new_no)
                new_left, new_no = new_left - 1, new_no + 1
                continue
            if head == "-" and old_left:
                yield line, ("removed", old_no)
                old_left, old_no = old_left - 1, old_no + 1
                continue
            if (head == " " or line == "") and old_left and new_left:
                old_left, new_left, old_no, new_no = old_left - 1, new_left - 1, old_no + 1, new_no + 1
                yield line, ("context", 0)
                continue
            if head == "\\":
                yield line, ("context", 0)
                continue
            old_left = new_left = 0
        mm = HUNK.match(line)
        if mm and hunk_exact(lines, k):
            old_no, old_left, new_no, new_left = counts(mm)
        yield line, ("outside", 0)


def parse_differs(diff: str) -> bool:
    """The only diffs on which W-1 can act: a line inside a counted hunk that opens with `---` or `+++`
    (which the fifth pass read as a header, or dropped), or a U+FEFF opening line 1 of either side --
    under git's split or under str.splitlines(), because a move F-2 and W-1 explain only jointly acts
    through a line F-2's revert would cut."""
    for split in (git_lines, str.splitlines):
        for line, (where, no) in hunk_walk(diff, split):
            if where in ("added", "removed") and (line.startswith(("---", "+++")) or (no == 1 and line[1:2] == "\uFEFF")):
                return True
    return False


def raw_paths(diff: str) -> list:
    """Every path a header names. Read by `hunk_walk` (NOTE_path2_sixth_pass W-1), so a removed `-- a/x`
    or an added `++ b/x` inside a hunk is content, not a path."""
    out = []
    for line, (where, _no) in hunk_walk(diff):   # NOTE_path2_fourth_pass F-2: the git split
        if where != "outside":
            continue
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            out.append(line[6:].strip())
        elif line.startswith("diff --git "):
            a, b = new._header_paths(line)
            out += [x for x in (a, b) if x]
        elif line.startswith("rename from ") or line.startswith("rename to "):
            out.append(line.split(" ", 2)[2])
    return list(dict.fromkeys(out))


def key_moved(paths) -> bool:
    return any(BASE._norm(p) != new._norm(p) for p in paths)


def moved_keys(paths) -> tuple[set, set]:
    """(baseline keys, repaired keys) of the filenames whose key moved."""
    moved = [p for p in paths if BASE._norm(p) != new._norm(p)]
    return {BASE._norm(p) for p in moved}, {new._norm(p) for p in moved}


def key_shape_violations(tokens) -> list:
    """NOTE_path2_third_pass R-5, blocking. #121 widened `_norm` in exactly one way: a leading run of
    `/` and `./` segments is dropped instead of every leading dot and slash. So for ANY token, the
    repaired key with `./` stripped from its front must be the baseline key. A `_norm` that moved a
    key any other way -- stopped lower-casing, dropped a segment, changed a separator -- would have
    its moves attributed to #121 by every rule below, because those rules ask only whether the key
    moved. This asks HOW."""
    return [p for p in tokens if new._norm(p).lstrip("./") != BASE._norm(p)]


def collision(paths) -> bool:
    by_old: dict = {}
    for p in paths:
        by_old.setdefault(BASE._norm(p), set()).add(new._norm(p))
    return any(len(v) > 1 for v in by_old.values())


def _name(p: str) -> str:
    return p.rstrip("/").rsplit("/", 1)[-1]


def any_tier(status: dict, c: str):
    """The resolution before #97: the entry clearing any tier, in diff order."""
    return next((p for p in status if p == c or p.endswith("/" + c) or _name(p) == _name(c)), None)


def tiered(status: dict, c: str):
    """The resolution after #97: exact, then suffix, then basename, each over every entry."""
    return (next((p for p in status if p == c), None)
            or next((p for p in status if p.endswith("/" + c)), None)
            or next((p for p in status if _name(p) == _name(c)), None))


def resolutions_differ(status: dict, claimed: str) -> bool:
    c = new._norm(claimed)
    return any_tier(status, c) != tiered(status, c)


def path_claim_by121(claimed: str, base_status: dict, status: dict, moved_old: set, moved_new: set) -> bool:
    """AMENDMENT G-C4, per claim: the claim's key moved, or the entry it resolves to is a moved key --
    under the baseline, the any-tier resolution over the baseline map; under the repair, the tiered
    resolution over the repaired map."""
    if BASE._norm(claimed) != new._norm(claimed):
        return True
    return (any_tier(base_status, BASE._norm(claimed)) in moved_old
            or tiered(status, new._norm(claimed)) in moved_new)


def _pairs(sides: dict, status: dict, rx_for, rx_for_removed=None) -> list:
    """Per file whose repaired status is not `A`: (added count, removed count). The removed side may
    read a different pattern (NOTE_path2_third_pass R-2: `async` is accepted there and not on the
    added side, because the added-blob count does not read `async def test_`)."""
    out = []
    for path, (added, removed) in sides.items():
        if status.get(path) == "A":
            continue
        out.append((rx_for(added), (rx_for_removed or rx_for)(removed)))
    return out


def ident_at(text: str, i: int) -> str:
    """This file's own reading of a Python identifier at `text[i]` (NOTE_path2_sixth_pass W-2), by the
    pinned name table (NOTE_path2_seventh_pass) through this file's own decoder: the opening character
    may open an identifier (bit 1), each later one continue it (bit 2)."""
    if i >= len(text) or not xid_mask(text[i]) & 1:
        return ""
    j = i + 1
    while j < len(text) and xid_mask(text[j]) & 2:
        j += 1
    return text[i:j]


MIDDLE_DOTS = ("\u00b7", "\u0387")     # the punctuation XID_Continue holds in Unicode 15.0.0
Y2_WHY = "opens with U+FEFF where the diff does not show it is line 1 of its file, the one place CPython reads one"
Y2_TEST = "an added test definition opens with U+FEFF, which main's Python counted as no test and its port as one"
Y5_WHY = ("a count left after pairing changed tests away is not verified, since a line this template reads may be "
          "one Python does not define (#101)")
Y3_VERSIONS = "the Pythons this package supports (Unicode 13.0 to 16.0)"
# NOTE_path2_ninth_pass: each licensed-difference rule's abstention, by its words.
Z1_WHY = "(`^\\s*def test_` over main's line split); no repair licenses the difference"
Z2_WHY = "(`^\\s*(?:def|class)\\s+NAME\\b` over main's line split); no repair licenses the difference"
Z2_SKEW = "through `\\b` before U+"
Z12_BC1 = "(BC-1 read on main's keys); no repair licenses the difference"
Z12_RAISES = "main raises on this diff (`+++ /dev/null` with no `---` line before it)"
Z3_DIFFERS = "this reading's file list differs from main's"
Z3_APART = "main's Python and its port read the file list apart"
Z4_WHY = "only a file with the same name in another directory is in the diff"
Z5_WHY = "is one this reading refuses and CPython may refuse too"


def own_wide(text: str, i: int) -> str:
    """Y-3, written out: the table's identifier at `text[i]`, widened by every skew code point."""
    if i >= len(text) or not (xid_mask(text[i]) & 1 or skew(text[i])):
        return ""
    j = i + 1
    while j < len(text) and (xid_mask(text[j]) & 2 or skew(text[j])):
        j += 1
    return text[i:j]


def own_skew_test(line: str):
    """Y-3, written out: the earliest skew code point in a test definition's widened name, or None."""
    text = own_bom_hidden(line) or line
    mm = DEF_HEAD_REMOVED.match(text)
    if mm is None or mm.group(1) != "def":
        return None
    wide = own_wide(text, mm.end())
    return next((ch for ch in wide if skew(ch)), None) if wide.startswith("test_") else None


def claimed_name(text: str, start: int) -> tuple:
    """This file's own reading of a symbol claim's name (NOTE_path2_seventh_pass): (identifier, None), or
    (identifier, the UNCHECKABLE reason) when the name runs on into a word character (bit 4) no identifier
    holds, or ends in a middle dot."""
    name = ident_at(text, start)
    end = start + len(name)
    met = next((ch for ch in text[start:end + 1] if skew(ch)), None)      # NOTE_path2_eighth_pass (Y-3)
    if met is not None:
        return name, (f"the claimed name '{name}' meets U+{ord(met):04X}, which {Y3_VERSIONS} read differently; "
                      "no definition is read for it")
    if end < len(text) and xid_mask(text[end]) & 4:
        ch = text[end]
        return name, (f"the claimed name runs past '{name}' into '{ch}' (U+{ord(ch):04X}), which no Python "
                      "identifier holds; no definition is read for it")
    if name[-1:] in MIDDLE_DOTS and name:
        return name, (f"the claimed name '{name}' ends in '{name[-1]}' (U+{ord(name[-1]):04X}), which prose "
                      "also writes after a word; no definition is read for it")
    return name, None


def defined(line: str, removed: bool = False):
    """(`def` or `class`, name) a diff line defines, as W-2 reads it, or None."""
    mm = (DEF_HEAD_REMOVED if removed else DEF_HEAD).match(line)
    if mm is None:
        return None
    name = ident_at(line, mm.end())
    end = mm.end() + len(name)
    if not name or (end < len(line) and ord(line[end]) > 0x7F):
        return None
    return mm.group(1), name


def test_name(line: str, removed: bool = False):
    d = defined(line, removed)
    return d[1] if d is not None and d[0] == "def" and d[1].startswith("test_") else None


def _test_counts(lines) -> Counter:
    return Counter(t for t in map(test_name, lines) if t)


def _test_counts_removed(lines) -> Counter:
    return Counter(t for t in (test_name(x, True) for x in lines) if t)


def test_def_changed(sides: dict, status: dict) -> bool:
    """#101 for tests_added: a non-`A` file where one test name is defined by an added and a removed line."""
    return any(set(a) & set(r) for a, r in _pairs(sides, status, _test_counts, _test_counts_removed))


def test_def_excess(sides: dict, status: dict) -> bool:
    """G-C5: a non-`A` file with a changed test name defined in more added lines than removed lines."""
    return any(a[n] > r[n] for a, r in _pairs(sides, status, _test_counts, _test_counts_removed)
               for n in set(a) & set(r))


_SYMBOL_CLAIM = re.compile(r"(?:function|class|method)\s+(?:(?:named|called)\s+)?" + "[`\"']?", re.I)


def claimed_identifier(text: str, name: str) -> str:
    """The identifier a symbol claim names, read as W-2 reads it: from where the template's `name` group
    starts in the claim's text (found here by this file's own pattern), else the template's name."""
    for mm in _SYMBOL_CLAIM.finditer(text):
        if text.startswith(name, mm.end()):
            return ident_at(text, mm.end()) or name
    return name


def symbol_def_changed(sides: dict, status: dict, name: str) -> bool:
    """#101 for symbol_added, as V-1 and W-2 read it: the added side without `async`, the removed side
    with it, the name a whole identifier."""
    def count(lines, removed):
        return sum(1 for x in lines if (defined(x, removed) or ("", ""))[1] == name)
    return any(a and r for a, r in _pairs(sides, status, lambda lines: count(lines, False),
                                          lambda lines: count(lines, True)))


def only_touches_new_accusation_allowed(detail: dict, prefixes: list, paths) -> bool:
    """AMENDMENT G-C3 / C-3: a prefix key carrying a leading dot, or a changed path whose repaired key
    begins with `..` -- the two shapes whose old VERIFIED came from the old key dropping the dot."""
    # NOTE_path2_third_pass R-3: exactly ONE leading dot. A prefix key opening with `..` is not a
    # repo path and the repaired gate abstains on it, so it can no longer be a new accusation; the
    # exception is narrowed to the shape the amendment actually argued for.
    dotted_prefix = any(_DOTFILE_PREFIX.match(new._norm(x).rstrip("/.")) and BASE._norm(x) != new._norm(x)
                        for x in prefixes)
    dotdot = any(new._norm(p).startswith("..") and BASE._norm(p) != new._norm(p) for p in paths)
    return dotted_prefix or dotdot


def off_tree_key(key: str) -> bool:
    """NOTE_path2_third_pass R-3, written out: a prefix key opening with a dot that is not a dotfile name."""
    return key.startswith(".") and not _DOTFILE_PREFIX.match(key)


def compat_detail_paths(detail: dict) -> list:
    return ([x.get("path") for x in detail.get("removed", []) or []]
            + [x.get("path") for x in detail.get("signature_changed", []) or []])


def core(c) -> tuple:
    d = {k: v for k, v in c.detail.items() if not (c.kind == "compat_claim" and k in COMPAT_EXTRAS)}
    return (c.kind, c.text, json.dumps(d, sort_keys=True))


# ── the oracles (NOTE_path2_seventh_pass, G-C7): every re-implemented rule's reading, this file's own ──

def own_key(p: str) -> str:
    """#121's key, written out: backslashes read as slashes, then a leading run of `/` and `./` segments
    dropped (a dotfile, `..env` and `../x` keep their dots), then lower case."""
    p = p.replace("\\", "/")
    while p.startswith("/") or p.startswith("./"):
        p = p[1:] if p.startswith("/") else p[2:]
    return p.lower()


def own_find_path(status: dict, claimed: str) -> tuple:
    """#97, written out: an entry equal to the claim, else one ending in "/" + claim, else one with the
    claim's basename -- each tier over every entry, diff order deciding only within a tier."""
    c = own_key(claimed)
    for tier in (lambda p: p == c, lambda p: p.endswith("/" + c), lambda p: Path(p).name == Path(c).name):
        for p, st in status.items():
            if tier(p):
                return p, st
    return None, None


_BINARY = re.compile(r"^Binary files (?P<a>.+?) and (?P<b>.+?) differ$")


def _note(pend: list, line: str) -> None:
    """BIN-1's header reading (main's, unchanged by this branch), on a [a, b, status] list."""
    if line.startswith("new file mode"):
        pend[2] = "A"
    elif line.startswith("deleted file mode"):
        pend[2] = "D"
    elif line.startswith("rename from "):
        pend[0] = line[len("rename from "):]
    elif line.startswith("rename to "):
        pend[1] = line[len("rename to "):]
    else:
        mm = _BINARY.match(line)
        if mm and mm.group("a") == "/dev/null":
            pend[2] = "A"
        elif mm and mm.group("b") == "/dev/null":
            pend[2] = "D"


def own_dev_null(p) -> bool:
    """Y-4, written out: /dev/null, or /dev/null then a TAB (GNU diff's timestamp)."""
    return p == "/dev/null" or (p or "")[:10] == "/dev/null\t"


def own_shape(p: str) -> str:
    """Y-1, written out: a header path cut at its earliest TAB and stripped, then `a/` or `b/` dropped."""
    p = p.split("\t")[0].strip()
    return p[2:] if p[:2] in ("a/", "b/") else p


def own_clean_header(lines: list, k: int) -> bool:
    """Y-1, written out: `--- X`, `+++ Y`, a line opening `@@`, with X and Y one path or one of them /dev/null."""
    if k + 2 >= len(lines) or not lines[k + 1].startswith("+++ ") or not lines[k + 2].startswith("@@"):
        return False
    x, y = own_shape(lines[k][4:]), own_shape(lines[k + 1][4:])
    return x == y or x == "/dev/null" or y == "/dev/null"


def own_shaped_pair(lines: list, k: int) -> bool:
    """NOTE_path2_eleventh_pass (round 10, R0.2), written out: a `--- ` line read as a header outside any count opens a pair
    with a header's shape -- Y-1's clean header, or a pair naming one file once quotes, `a/` and `b/` are dropped (git's
    quoted path) -- a `+++ ` line then a line opening `@@`."""
    if own_clean_header(lines, k):
        return True
    return (k + 2 < len(lines) and lines[k + 1].startswith("+++ ") and lines[k + 2].startswith("@@")
            and own_pair_names(lines[k][4:]) == own_pair_names(lines[k + 1][4:]))


def own_bom_hidden(line: str):
    """Y-2, written out: the line with every U+FEFF of its leading run of space, tab, form feed and U+FEFF
    dropped, when that run holds one; else None."""
    j = 0
    while j < len(line) and line[j] in " \t\f\ufeff":
        j += 1
    return line[:j].replace("\ufeff", "") + line[j:] if "\ufeff" in line[:j] else None


LOOSE_WHY = ("a `---` or `+++` line after lines no hunk count holds may be content (a SQL or Lua comment, "
             "a `++` line) or a file header")
COLLIDE_WHY = "two header paths that differ only in case are one key"
UNCOUNTED_WHY = ("a line names a changed file no header pair counts (GNU's `Binary files ... differ`, "
                 "`Only in ...` and the like)")
UNCOUNTED = re.compile(r"^(?:(?:Binary files|Files|Symbolic links) .+ and .+ differ|Only in .+: .+|File .+ is a .+ while file .+ is a .+)$")


def own_case_kept(p: str) -> str:
    """NOTE_path2_twelfth_pass (A.2), written out: backslashes read as slashes, a leading run of `/` and `./` segments
    dropped, and the case kept."""
    p = p.replace("\\", "/")
    while p.startswith("/") or p.startswith("./"):
        p = p[1:] if p.startswith("/") else p[2:]
    return p


def own_git_writes(line: str):
    """NOTE_path2_twelfth_pass (A.1), written out: a `diff --git` header's paths as git writes them after it -- the `---`
    path, the `+++` path, the `rename from` path, the `rename to` path, each in quotes where the header has the path in
    quotes -- or None where main's header reading (BIN-1's, which the branch does not change) reads no path."""
    a, b = BASE._header_paths(line)
    if not a or not b:
        return None
    body = line[len("diff --git "):]
    half = (len(body) - 1) // 2
    if len(body) % 2 and body[half] == " " and body[:2] == "a/" and body[half + 1:half + 3] == "b/" \
            and body[2:half] == body[half + 3:]:
        quoted_a = quoted_b = False
    else:
        mm = BASE._DIFF_GIT.match(line)
        quoted_a, quoted_b = mm.group("qa") is not None, mm.group("qb") is not None
    return ('"a/' + a + '"' if quoted_a else "a/" + a, '"b/' + b + '"' if quoted_b else "b/" + b,
            '"' + a + '"' if quoted_a else a, '"' + b + '"' if quoted_b else b)


GIT_NAMES = (("rename from ", 2), ("copy from ", 2), ("rename to ", 3), ("copy to ", 3))


def own_read(diff: str, facts: dict | None = None) -> tuple:
    """W-1, written out: (status map, added lines, per-file sides, notes) by this file's own hunk walk. Inside an
    exact hunk a line is content by its counts and a U+FEFF opening line 1 of a side is dropped; every other
    line is read as main read it (a `---`/`+++` line a header, a `+`/`-` line added/removed), with the eighth
    pass's Y-1 note (a header read after lines no count placed, unless it has a header's shape; two paths one
    key in case; a GNU line naming a changed file no header pair counts), Y-2's line-1 U+FEFF outside the counts and its note (a U+FEFF dropped from an added line that
    then defines a test), and Y-4's /dev/null. NOTE_path2_twelfth_pass: `facts`, when given, receives what the guard's
    tightened licences read -- whether the diff is git's own rendering, whether a Z-3 doubt was read, and each key's
    paths as written, case kept (A.1, A.2)."""
    status: dict = {}
    added: list = []
    sides: dict = {}
    notes: dict = {}
    old_path = cur = pend = None
    lines = git_lines(diff)
    loose, plus_ok = False, -1
    lead_old = lead_new = False
    counted: set = set()          # NOTE_path2_ninth_pass (Z-3): the lines an exact hunk's counts read
    soft: list = []               # Z-3: the doubts main's reading of the file list also held
    in_binary = False             # NOTE_path2_tenth_pass (K-4): inside a `GIT binary patch` block
    outside_git = False           # NOTE_path2_twelfth_pass (A.1): a line git's own rendering would not hold
    any_header = False
    as_git = None                 # the pending header's paths as git writes them
    minus_read = False            # a `---` line under the pending header
    kept: dict = {}

    def pend_key() -> str:
        raw = pend[0] if pend[2] == "D" else pend[1]
        return own_key(raw) if raw else ""

    def flush() -> None:
        if pend is not None and pend_key():
            register(pend[0] if pend[2] == "D" else pend[1])
            status.setdefault(pend_key(), pend[2])
            sides.setdefault(pend_key(), ([], []))
        elif pend is not None:
            soft.append(Z3_UNREAD)

    def unsure(*keys_why) -> None:
        for key, why in keys_why:
            notes.setdefault(key, why)

    forms: dict = {}

    def register(raw_path: str) -> None:
        form = raw_path.replace("\\", "/")
        while form.startswith("/") or form.startswith("./"):
            form = form[1:] if form.startswith("/") else form[2:]
        seen_as = kept.setdefault(own_key(raw_path), [])
        if form not in seen_as:
            seen_as.append(form)
        if forms.setdefault(own_fold(form), form) != form:              # NOTE_path2_tenth_pass (K-2)
            unsure(("files", COLLIDE_WHY))

    def bom_dropped(raw: str, text: str) -> None:
        if text != raw and test_name(text):
            unsure(("bom", Y2_TEST))

    for k, (line, (where, no)) in enumerate(hunk_walk(diff)):
        if where in ("added", "removed"):
            counted.add(k)
            text = line[1:]
            if no == 1 and text[:1] == "\ufeff":
                text = text[1:]
            if where == "added":
                added.append(text)
                bom_dropped(line[1:], text)
            if cur is not None:
                sides[cur][0 if where == "added" else 1].append(text)
            continue
        if where == "context":
            counted.add(k)
            continue
        mm = HUNK.match(line)
        if line.startswith("diff --git "):
            flush()
            pend, cur = [*BASE._header_paths(line), "M"], None
            loose = lead_old = lead_new = in_binary = False
            any_header, as_git, minus_read = True, own_git_writes(line), False
            outside_git = outside_git or as_git is None
        elif line.startswith("--- "):
            named = line[4:].split("\t")[0]
            if pend is None or as_git is None or (named != "/dev/null" and named != as_git[0]):
                outside_git = True
            minus_read = True
            if loose:
                if own_clean_header(lines, k):
                    plus_ok, loose = k + 1, False
                    soft.append(Z3_SHAPED)
                else:
                    unsure(("files", LOOSE_WHY))
            elif not own_shaped_pair(lines, k):
                soft.append(Z3_UNSHAPED)                  # NOTE_path2_eleventh_pass (round 10, R0.2)
            old_path, cur = line[4:].strip(), None
            lead_old = lead_new = False
        elif line.startswith("+++ "):
            named = line[4:].split("\t")[0]
            if pend is None or as_git is None or not minus_read or (named != "/dev/null" and named != as_git[1]):
                outside_git = True
            as_git, minus_read = None, False
            if loose and k != plus_ok:
                unsure(("files", LOOSE_WHY))
            nw = line[4:].strip()
            if pend is not None and pend_key():
                named = own_pair_names((old_path or "") if own_dev_null(nw) else nw)
                if named not in (own_key(pend[0]), own_key(pend[1])):
                    soft.append(f"main's reading also dropped the `diff --git` file {shown(pend_key())} for the next "
                                "`---`/`+++` pair, which names another")
            elif pend is not None:
                soft.append(Z3_UNREAD_PAIR)                # NOTE_path2_eleventh_pass (round 10, R0.0)
            if own_dev_null(nw) and old_path is None:
                cur, pend = None, None                     # NOTE_path2_tenth_pass (K-3): names no file
            else:
                if own_dev_null(nw):
                    status[own_key(old_path[2:] if old_path.startswith("a/") else old_path)] = "D"
                    raw = old_path[2:] if old_path.startswith("a/") else old_path
                else:
                    raw = nw[2:] if nw.startswith("b/") else nw
                    status[own_key(raw)] = "A" if old_path is None or own_dev_null(old_path) else "M"
                cur, pend = own_key(raw), None
                register(raw)
                sides.setdefault(cur, ([], []))
            lead_new = old_path is not None and own_shape(old_path) == "/dev/null"
            lead_old = own_shape(nw) == "/dev/null"
        elif mm:
            if hunk_exact(lines, k):
                lead_old = lead_new = False
            else:
                loose = True
                lead_old = lead_old or int(mm.group(1)) == 1
                lead_new = lead_new or int(mm.group(3)) == 1
        elif line[:1] in ("+", "-") and line[:3] not in ("+++", "---"):
            loose = True
            text = line[1:]
            at_one = lead_new if line[0] == "+" else lead_old
            if at_one and text[:1] == "\ufeff":
                text = text[1:]
            if line[0] == "+":
                lead_new = False
                added.append(text)
                bom_dropped(line[1:], text)
            else:
                lead_old = False
            if cur is not None:
                sides[cur][0 if line[0] == "+" else 1].append(text)
            elif line[0] == "-" and pend is not None:
                _note(pend, line)
        else:
            if line.startswith("@@") or line[:1] == " " or line == "":
                loose = True
                if not line.startswith("@@"):
                    lead_old = lead_new = False
                in_binary = in_binary and line != ""
            elif UNCOUNTED.match(line) and not (pend is not None and _BINARY.match(line)):
                unsure(("files", UNCOUNTED_WHY))
                outside_git = True
            elif line == "GIT binary patch" or (line.split(" ")[0] in ("literal", "delta") and len(line.split(" ")) == 2
                                                 and line.split(" ")[1].isascii() and line.split(" ")[1].isdigit()):
                in_binary = True
            elif not (in_binary or line[:1] == "\\" or line.startswith(GIT_EXTENDED)
                      or (pend is not None and _BINARY.match(line))):
                soft.append(Z3_UNPLACED)                  # NOTE_path2_tenth_pass (K-4)
                outside_git = True
            for prefix, at in GIT_NAMES:
                if line.startswith(prefix) and (pend is None or as_git is None or line[len(prefix):] != as_git[at]):
                    outside_git = True
            if pend is not None:
                _note(pend, line)
    flush()
    why = own_file_list_differs(diff, status, counted, soft)           # NOTE_path2_ninth_pass (Z-3)
    if why:
        notes["differs"] = why
    if facts is not None:
        facts.update(rendered=any_header and not outside_git, soft=bool(soft), forms=kept)
    return status, added, sides, notes


def own_git_licence(text: str, status_paths: list) -> dict:
    """NOTE_path2_twelfth_pass (A.1, A.2), written out, at the git door: git's diff text read for its rendering; no Z-3
    doubt, since the file list is `--name-status`; each key's `--name-status` paths as written, case kept."""
    facts: dict = {}
    own_read(text, facts)
    kept: dict = {}
    for path in status_paths:
        seen_as = kept.setdefault(own_key(path), [])
        if own_case_kept(path) not in seen_as:
            seen_as.append(own_case_kept(path))
    return {"rendered": bool(facts["rendered"]), "soft": False, "forms": kept}


def own_name_status(text: str) -> dict:
    """The git door's status map, written out: `git diff --name-status` lines, the status letter and the
    last tab-separated field, keyed by `own_key`."""
    status: dict = {}
    for line in git_lines(text):
        parts = line.split("\t")
        if len(parts) >= 2:
            status[own_key(parts[-1])] = parts[0][:1]
    return status


# ── NOTE_path2_ninth_pass: main's own reading, written out -- the other side of the licensed-difference rule ──
# main's Python is this interpreter (the scorer runs under 3.12, whose Unicode is the table's): str.splitlines(),
# str.strip(), `re`'s `\s`, `\w` and `\b`. main's port is JavaScript, spelled out here: its `\s` (which trim() strips),
# its `.` (no line terminator), its multiline `^` (after \n, \r, U+2028, U+2029), its ASCII `\w` and `\b`.
LS_PS = chr(0x2028) + chr(0x2029)
JS_SPACE = "".join(map(chr, (9, 10, 11, 12, 13, 32, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F,
                            0x205F, 0x3000, 0xFEFF)))
JS_WS = "[" + re.escape(JS_SPACE) + "]"
JS_DOT = "[^\n\r" + LS_PS + "]"
JS_LINE_START = "(?:(?<![\\s\\S])|(?<=[\n\r" + LS_PS + "]))"
DIFF_GIT_JS = re.compile(r'^diff --git (?:"a/(?P<qa>(?:[^"\\]|\\' + JS_DOT + r')*)"|a/(?P<a>' + JS_DOT + r'*?)) '
                         r'(?:"b/(?P<qb>(?:[^"\\]|\\' + JS_DOT + r')*)"|b/(?P<b>' + JS_DOT + r'*))$')
BINARY_JS = re.compile("^Binary files (?P<a>" + JS_DOT + "+?) and (?P<b>" + JS_DOT + "+?) differ$")


def main_key(p: str) -> str:
    """main's `_norm`: `lstrip("./")` after the backslashes turn, lower case."""
    return p.replace("\\", "/").lstrip("./").lower()


def own_main_header_paths(line: str, js: bool) -> tuple:
    if not js:
        return BASE._header_paths(line)                  # main's BIN-1 code, which this branch does not change
    body = line[len("diff --git "):]
    if len(body) % 2 == 1:
        mid = len(body) // 2
        if body[mid] == " " and body[:mid].startswith("a/") and body[mid + 1:].startswith("b/") and body[2:mid] == body[mid + 3:]:
            return body[2:mid], body[mid + 3:]
    mm = DIFF_GIT_JS.match(line)
    if not mm:
        return "", ""
    return (mm.group("qa") if mm.group("qa") is not None else (mm.group("a") or ""),
            mm.group("qb") if mm.group("qb") is not None else (mm.group("b") or ""))


def own_main_status(lines: list, js: bool, skip=(), forms: list | None = None):
    """main's `parse_unified_diff` status map over `lines`, keyed by `main_key`, in main's Python's spelling or its
    port's; a line whose index is in `skip` (W-1's exact hunks) is not read. None where main raises.
    NOTE_path2_eleventh_pass: `forms`, when given, receives every path main keys, as written."""
    strip = (lambda s: s.strip(JS_SPACE)) if js else str.strip
    status: dict = {}
    old_path = pend = None

    def main_key(raw: str) -> str:
        if forms is not None:
            forms.append(raw)
        return _MAIN_KEY(raw)

    def flush() -> None:
        if pend is not None:
            raw = pend[0] if pend[2] == "D" else pend[1]
            if raw and _MAIN_KEY(raw) not in status:
                status[main_key(raw)] = pend[2]
    for i, line in enumerate(lines):
        if i in skip:
            continue
        if line.startswith("diff --git "):
            flush()
            pend = [*own_main_header_paths(line, js), "M"]
        elif line.startswith("--- "):
            old_path = strip(line[4:])
        elif line.startswith("+++ "):
            nw = strip(line[4:])
            if nw == "/dev/null":
                if old_path is None:
                    return None
                status[main_key(old_path[2:] if old_path.startswith("a/") else old_path)] = "D"
            else:
                status[main_key(nw[2:] if nw.startswith("b/") else nw)] = "A" if old_path in ("/dev/null", None) else "M"
            pend = None
        elif line.startswith("+") and not line.startswith("+++"):
            pass
        elif pend is not None:
            if js and not line.startswith(("new file mode", "deleted file mode", "rename from ", "rename to ")):
                mm = BINARY_JS.match(line)
                if mm and mm.group("a") == "/dev/null":
                    pend[2] = "A"
                elif mm and mm.group("b") == "/dev/null":
                    pend[2] = "D"
            else:
                _note(pend, line)
    flush()
    return status


_MAIN_KEY = main_key


class OwnMain:
    """main's reading of one door's text: its added lines in each spelling, its `got` in each, BC-1 on its file
    list(s), and whether it raises."""

    def __init__(self, text: str, maps: tuple):
        self.added_py = [x[1:] for x in text.splitlines() if x.startswith("+") and not x.startswith("+++")]
        self.added_js = [x[1:] for x in git_lines(text) if x.startswith("+") and not x.startswith("+++")]
        self.tests = (len(re.findall(r"^\s*def test_", "\n".join(self.added_py), re.M)),
                      len(re.findall(JS_LINE_START + JS_WS + "*def test_", "\n".join(self.added_js))))
        self.raises = any(m is None for m in maps)
        self.python = [m is not None and any(p.lower().endswith((".py", ".pyi")) for p in m) for m in maps]
        self.maps = maps                  # NOTE_path2_tenth_pass (K-5): main's Python's file list at index 0


def own_main_raw(diff: str) -> OwnMain:
    return OwnMain(diff, (own_main_status(diff.splitlines(), False), own_main_status(git_lines(diff), True)))


def own_tests_differ(got: int, main: OwnMain):
    """Z-1, written out."""
    if main.raises:
        return Z12_RAISES
    if not all(main.python):
        return ("this reading finds a Python file in the diff's file list where main's reading of it found none "
                + Z12_BC1)
    py, js = main.tests
    if got == py == js:
        return None
    return f"this reading counts {got} added test definitions where main's Python counted {py} and its port {js} {Z1_WHY}"


def own_symbol_differs(hit: bool, name: str, text: str, start: int, main: OwnMain):
    """Z-2, written out: main's Python's name is its template's (this interpreter's `\\w`), its port's the ASCII run."""
    if main.raises:
        return Z12_RAISES
    if not all(main.python):
        return ("this reading finds a Python file in the diff's file list where main's reading of it found none "
                + Z12_BC1)
    template_name = re.match(r"[A-Za-z_]\w*", text[start:], re.I)
    name_py = template_name.group(0) if template_name else ""
    ascii_run = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[start:])
    name_js = ascii_run.group(0) if ascii_run else ""
    py, skew_cp = False, None
    if name_py:
        blob = "\n".join(main.added_py)
        for mm in re.finditer(r"^\s*(?:def|class)\s+" + re.escape(name_py), blob, re.M):
            nxt = blob[mm.end():mm.end() + 1]
            if nxt and skew(nxt):
                skew_cp = skew_cp if skew_cp is not None else ord(nxt)
            elif not nxt or not re.match(r"\w", nxt):
                py = True
    js = bool(name_js) and re.search(JS_LINE_START + JS_WS + "*(?:def|class)" + JS_WS + "+" + re.escape(name_js)
                                     + "(?![A-Za-z0-9_])", "\n".join(main.added_js)) is not None
    if not py and skew_cp is not None:
        return f"main's Python read a definition of '{name_py}' {Z2_SKEW}{skew_cp:04X}, which {Y3_VERSIONS} read differently"
    if py == hit and js == hit:
        return None
    return (f"this reading finds {'an' if hit else 'no'} added definition of '{name}' where main's Python "
            f"{'did' if py else 'did not'} and its port {'did' if js else 'did not'} {Z2_WHY}")


def own_refused(line: str) -> bool:
    """Z-5, written out: a definition head read loosely (any whitespace either of main's spellings reads, and U+FEFF),
    a name opening there (by the table, or the skew set), and W-2's reading refusing the line."""
    mm = re.match("^[\\s" + chr(0xFEFF) + "]*(?:async[\\s" + chr(0xFEFF) + "]+)?(?:def|class)[\\s" + chr(0xFEFF) + "]+", line)
    if mm is None or mm.end() >= len(line):
        return False
    ch = line[mm.end()]
    return bool(xid_mask(ch) & 1 or skew(ch)) and defined(line, True) is None


def own_strays(added: list, sides: dict) -> list:
    left = Counter(x for a, _r in sides.values() for x in a)
    out = []
    for x in "\n".join(added).split("\n"):
        if left[x] > 0:
            left[x] -= 1
        else:
            out.append(x)
    return out


def own_refused_files(added: list, sides: dict) -> dict:
    out: dict = {}
    for path, (a, _r) in sides.items():
        if own_undotted(path).lower().endswith((".py", ".pyi")):
            hit = next((x for x in a if own_refused(x)), None)
            if hit is not None:
                out[path] = hit
    stray = next((x for x in own_strays(added, sides) if own_refused(x)), None)
    if stray is not None:
        out[None] = stray
    return out


def own_refused_why(path) -> str:
    where = "outside any file" if path is None else f"in {shown(path)}"      # NOTE_path2_eleventh_pass (R0.3)
    return (f"an added definition line {where} {Z5_WHY}, and a file CPython refuses defines nothing; this reading "
            "reads it line by line")


def own_whole_symbol(name: str, added: list, sides: dict):
    refused = own_refused_files(added, sides)
    for path, (a, _r) in sides.items():
        if path in refused and any((defined(x) or ("", ""))[1] == name for x in a):
            return own_refused_why(path)
    if None in refused and any((defined(x) or ("", ""))[1] == name for x in own_strays(added, sides)):
        return own_refused_why(None)
    return None


def shown(key: str) -> str:
    """NOTE_path2_tenth_pass: a key as Z-3's reasons print it, written out -- folded by the one table, escaped as ascii()
    escapes it."""
    return ascii(own_fold(key))


def own_licensed(status: dict, main: dict):
    """Z-3, written out: each of this reading's keys, undotted, is one of main's, every one of main's is, and main's
    status for it is one of theirs -- else the earliest difference, in words."""
    groups: dict = {}
    for k, st in status.items():
        groups.setdefault(own_undotted(k), []).append((k, st))
    for k, st in main.items():
        if k not in groups:
            return f"main reads {shown(k)} ({st!r}), which this reading does not"
    for u, ks in groups.items():
        if u not in main:
            return f"this reading reads {shown(ks[0][0])} ({ks[0][1]!r}), which main does not"
        if main[u] not in [st for _k, st in ks]:
            return f"main reads {shown(u)} as {main[u]!r}, this reading {shown(ks[0][0])} as {ks[0][1]!r}"
    return None


def own_folds_apart(forms: list) -> bool:
    """NOTE_path2_eleventh_pass, written out: two paths main keys differ as written (a leading run of `.` and `/` dropped,
    backslashes read as slashes) and fold alike by this file's decoding of the fold."""
    seen: dict = {}
    for raw in forms:
        form = raw.replace("\\", "/").lstrip("./")
        if seen.setdefault(own_fold(form), form) != form:
            return True
    return False


Z3_FOLDS = ("main's reading holds two paths that differ only in case, which the runtimes this package supports key "
            "apart or together")


def own_file_list_differs(diff: str, status: dict, inside: set, soft: list):
    """Z-3 at the raw door, written out."""
    forms: list = []
    py, js = own_main_status(diff.splitlines(), False, forms=forms), own_main_status(git_lines(diff), True, forms=forms)
    if py is None or js is None:
        return f"{Z3_DIFFERS}: main raises on it (`+++ /dev/null` with no `---` line before it)"
    if own_folds_apart(forms):                          # NOTE_path2_eleventh_pass: before main's two lists are compared
        return f"{Z3_DIFFERS}: {Z3_FOLDS}"
    if py != js:
        odd = next(((k, st) for k, st in py.items() if js.get(k) != st), None)
        if odd is None:
            k, st = next((k, st) for k, st in js.items() if k not in py)
            where = f"main's port reads {shown(k)} ({st!r}), which its Python does not"
        elif odd[0] not in js:
            where = f"main's Python reads {shown(odd[0])} ({odd[1]!r}), which its port does not"
        else:
            where = f"main's Python reads {shown(odd[0])} as {odd[1]!r}, its port as {js[odd[0]]!r}"
        return f"{Z3_APART} (str.splitlines() breaks lines JavaScript does not): {where}"
    # NOTE_path2_tenth_pass (K-1): W-1's exact hunk licenses no file-list difference -- main's reading of the file list
    # with the lines an exact hunk's counts hold must be its reading without them
    full, skipped = own_main_status(git_lines(diff), False), own_main_status(git_lines(diff), False, inside)
    if full is None:
        return f"{Z3_DIFFERS}: main raises on it (`+++ /dev/null` with no `---` line before it)"
    if skipped != full:
        if skipped is None:
            moved = "without those lines main raises (`+++ /dev/null` with no `---` line before it)"
        else:
            odd = next(((k, st) for k, st in full.items() if skipped.get(k) != st), None)
            if odd is None:
                k, st = next((k, st) for k, st in skipped.items() if k not in full)
                moved = f"main reads {shown(k)} ({st!r}) only without them"
            elif odd[0] not in skipped:
                moved = f"main reads {shown(odd[0])} ({odd[1]!r}) from them"
            else:
                moved = f"main reads {shown(odd[0])} as {odd[1]!r} with them, as {skipped[odd[0]]!r} without"
        return (f"{Z3_DIFFERS}: main read the file list from lines an exact hunk's counts hold, which W-1 reads as "
                f"content ({moved}); a changed file neither reading counts may have balanced it, so W-1 licenses no "
                "file-list difference")
    why = own_licensed(status, full)
    if why:
        return f"{Z3_DIFFERS} where no repair accounts for it: {why}"
    if soft and set(status) != set(js):
        repair = ("#121 keeps a dotfile's dot" if {own_undotted(k) for k in status} == set(js)
                  else "W-1 reads an exact hunk's `---`/`+++` line as content")
        return f"{Z3_DIFFERS} by a repair ({repair}), and {soft[0]}; the repair may have balanced that error"
    return None


def own_main_name_status(text: str) -> dict:
    """main's git-door file list: `--name-status` split by str.splitlines(), keyed by `main_key`."""
    out: dict = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            out[main_key(parts[-1])] = parts[0][:1]
    return out


def own_status_differs(name_status: str, status: dict):
    why = own_licensed(status, own_main_name_status(name_status))
    return f"{Z3_DIFFERS} where no repair accounts for it: {why}" if why else None


def own_basename_only(status: dict, claimed: str):
    """Z-4, written out: a claim with a directory component that only the basename tier resolves."""
    c = own_key(claimed)
    if "/" not in c:
        return None
    p, st = own_find_path(status, claimed)
    if p is None or p == c or p.endswith("/" + c):
        return None
    return (f"{claimed!r}: {Z4_WHY} ({shown(p)}, status {st!r}); #97 licenses the exact and suffix tiers only")


Z3_SHAPED = ("main's reading also took a `---`/`+++` pair after lines no hunk count holds for a header because it has "
             "a header's shape, and it may be content (a SQL `-- ` comment beside a `++` line)")
Z3_UNREAD = "main's reading also dropped a `diff --git` file whose header paths neither reading can read"
# NOTE_path2_eleventh_pass (round 10, R0.0 and R0.2): a pair under a `diff --git` header neither reading can read (`git diff
# --no-prefix`), and a pair read as a header without a header's shape, written out.
Z3_UNREAD_PAIR = ("main's reading also read a `---`/`+++` pair under a `diff --git` header neither reading can read "
                  "(`git diff --no-prefix`), where a directory named `a/` or `b/` is taken for git's prefix")
Z3_UNSHAPED = ("main's reading also took a `---`/`+++` pair without a header's shape (no `@@` after it, or two different "
               "paths) for a file header, and it may be content (a SQL `-- ` comment beside a `++` line)")
# NOTE_path2_tenth_pass (K-4): a line no reading places, written out: outside every header, hunk, context line and blank,
# not a `\` marker, not a GNU line Y-1 already doubts, not a binary line under a git header, not inside a `GIT binary
# patch` block, and none of git's extended header lines (git-diff(1)).
Z3_UNPLACED = ("main's reading also passed over a line no reading places, which may name a changed file neither reading "
               "counts (git's `Submodule` line, svn's and hg's binary notices, and the like)")
GIT_EXTENDED = ("index ", "old mode ", "new mode ", "deleted file mode ", "new file mode ", "copy from ", "copy to ",
                "rename from ", "rename to ", "similarity index ", "dissimilarity index ")


def own_pair_names(raw: str) -> str:
    p = raw.split("\t", 1)[0].strip()
    if len(p) >= 2 and p[0] == '"' and p[-1] == '"':
        p = p[1:-1]
    return own_key(p[2:] if p[:2] in ("a/", "b/") else p)


BC1_WHY = "no Python file in the diff; this template counts `def` lines (#110)"
NO_EVIDENCE_WHY = "the diff carries no file statuses and no added lines"


def own_unmeasured(raw_input_len) -> str:
    """The reason a gate with no evidence gives (main's, unchanged), written out: the fixed sentence, and where the door
    counted its input and it was not empty, that it parsed to nothing."""
    why = NO_EVIDENCE_WHY
    if raw_input_len:
        why += (f"; {raw_input_len} characters of input parsed to nothing, which is a parse failure, not an empty "
                "change")
    return why
NOT_FUNCTIONS = {"case": "case", "cases": "case", "file": "file", "files": "file", "scenario": "scenario",
                 "scenarios": "scenario", "suite": "suite", "suites": "suite", "class": "class", "classes": "class"}
SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
SYMBOL_TEMPLATE = next(rx for kind, rx in BASE._TEMPLATES if kind == "symbol_added")


def touches_python(status: dict) -> bool:
    """BC-1 repair 1 on the undotted key (AMENDMENT C-2)."""
    return any(p.lstrip("./").lower().endswith((".py", ".pyi")) for p in status)


def changed_tests(sides: dict, status: dict) -> int:
    """#101 / C-1 for tests, written out: per non-`A` file, per test name, min(added, removed)."""
    n = 0
    for path, (a, r) in sides.items():
        if status.get(path) == "A":
            continue
        gone = Counter(t for t in (test_name(x, True) for x in r) if t)
        if gone:
            fresh = Counter(t for t in (test_name(x) for x in a) if t)
            n += sum(min(k, gone[t]) for t, k in fresh.items())
    return n


def only_changed(name: str, sides: dict, status: dict) -> bool:
    """#101 / C-1 for symbols, written out: some file adds and removes `name`, none adds it more often."""
    paired = False
    for path, (a, r) in sides.items():
        na = sum(1 for x in a if (defined(x) or ("", ""))[1] == name)
        nr = 0 if status.get(path) == "A" else sum(1 for x in r if (defined(x, True) or ("", ""))[1] == name)
        if na > nr:
            return False
        paired = paired or bool(na and nr)
    return paired


def async_tests_added(sides: dict, status: dict) -> int:
    """A-1, written out: per file, the `async def test_` definitions added beyond those the removed lines
    define (none removed from an `A` file), summed."""
    n = 0
    for path, (a, r) in sides.items():
        fresh = Counter(t for t in (test_name(x, True) for x in a if not test_name(x)) if t)
        if fresh:
            gone = Counter() if status.get(path) == "A" else Counter(t for t in (test_name(x, True) for x in r) if t)
            n += sum(max(0, k - gone[t]) for t, k in fresh.items())
    return n


def own_removed_lines(sides: dict) -> list:
    return [x for _a, r in sides.values() for x in r]


def own_test_doubt(added: list, sides: dict, notes: dict | None = None):
    """Y-2 and Y-3 for tests, written out: a U+FEFF the parse dropped before a test at line 1, then the added
    lines in order, then every file's removed lines."""
    if (notes or {}).get("bom"):
        return notes["bom"]
    for line in "\n".join(added).split("\n") + own_removed_lines(sides):
        hidden = own_bom_hidden(line)
        if hidden is not None and test_name(hidden, True):
            return f"a test definition {Y2_WHY}"
        ch = own_skew_test(line)
        if ch is not None:
            return f"a test definition's name holds U+{ord(ch):04X}, which {Y3_VERSIONS} read differently"
    return None


def expected_tests(c, status: dict, added: list, sides: dict, notes: dict | None = None,
                   main: "OwnMain | None" = None) -> tuple:
    """A tests_added claim's (verdict, reason), by this file's own code."""
    n, noun = int(c.detail["n"]), c.detail.get("noun", "").lower()
    if not touches_python(status):
        return "UNCHECKABLE", BC1_WHY
    got = sum(1 for x in "\n".join(added).split("\n") if test_name(x))
    chg = min(changed_tests(sides, status), got)
    net = got - chg
    note = f" ({chg} changed, not added: #101)" if chg else ""
    unread = async_tests_added(sides, status)
    if unread:
        return "UNCHECKABLE", f"diff adds {unread} async test functions, which this template does not count; claim says {n}"
    doubt = own_test_doubt(added, sides, notes)
    if doubt:
        return "UNCHECKABLE", f"{doubt}; claim says {n}"
    z1 = own_tests_differ(got, main) if main is not None else None          # NOTE_path2_ninth_pass (Z-1)
    if z1:
        return "UNCHECKABLE", f"{z1}; claim says {n}"
    refused = own_refused_files(added, sides)                                # Z-5
    if refused:
        return "UNCHECKABLE", f"{own_refused_why(next(iter(refused)))}; claim says {n}"
    if net == n and chg:                                   # Y-5: the pairing withdraws, it does not verify
        return "UNCHECKABLE", f"diff adds {net} test functions and changes {chg}, claim says {n}; {Y5_WHY}"
    if net == n:
        return "VERIFIED", f"diff adds {net} test functions, claim says {n}{note}"
    if noun in NOT_FUNCTIONS:
        return "UNCHECKABLE", (f"counts test {noun}, diff adds {net} test functions; a {NOT_FUNCTIONS[noun]} is "
                               f"not a function (#110){note}")
    if chg and net < n <= got:
        return "UNCHECKABLE", (f"diff adds {net} test functions and changes {chg}, claim says {n}; a changed test "
                               "is not an added one (#101)")
    return "CONTRADICTED", f"diff adds {net} test functions, claim says {n}{note}"


def expected_symbol(c, sentence: str, start: int, status: dict, added: list, sides: dict,
                    main: "OwnMain | None" = None) -> tuple:
    """A symbol_added claim's (verdict, reason), by this file's own code."""
    if not touches_python(status):
        return "UNCHECKABLE", BC1_WHY
    name, why = claimed_name(sentence, start)
    if why:
        return "UNCHECKABLE", why
    for line in "\n".join(added).split("\n") + own_removed_lines(sides):     # Y-2
        hidden = own_bom_hidden(line)
        if hidden is not None and (defined(hidden, True) or ("", ""))[1] == name:
            return "UNCHECKABLE", f"a definition of '{name}' {Y2_WHY}"
    kind = c.detail["kind"]
    hit = any((defined(x) or ("", ""))[1] == name for x in "\n".join(added).split("\n"))
    z2 = own_symbol_differs(hit, name, sentence, start, main) if main is not None else None   # NOTE_path2_ninth_pass
    if z2:
        return "UNCHECKABLE", z2
    z5 = own_whole_symbol(name, added, sides) if hit else None                                  # Z-5
    if z5:
        return "UNCHECKABLE", z5
    if hit and only_changed(name, sides, status):
        return "UNCHECKABLE", (f"added lines define {kind} '{name}' only where the removed lines of the same file "
                               "define it too; a changed definition is not an added one (#101)")
    return ("VERIFIED" if hit else "CONTRADICTED"), f"added lines {'do' if hit else 'do NOT'} define {kind} '{name}'"


# ── the file-list claims, written out (NOTE_path2_eighth_pass: the round-7 protocol lens's V-4/F-4 finding) ──
# Every only_touches verdict and reason is re-derived below with this file's own code: PATH-1's containment and
# path-shape test (the extension list read from its committed data file), BC-2's second prefix, C-3's dot miss,
# R-3's off-tree key, V-4's written parent and F-4's could-lie-under. A files_changed_count and a path claim are
# re-derived too, so Y-1's abstention on them is read, not trusted.

PATH1_EXT = frozenset(w for line in (HERE / "path1_extensions.txt").read_text(encoding="utf-8").splitlines()
                      if not line.startswith("#") for w in line.split())
NOT_SURE = "the diff's file list is not certain: "
NO_PATHS_WHY = "the diff carries no file paths, so scope cannot be checked"


def own_undotted(key: str) -> str:
    return key.lstrip("./")


def own_real_extension(token: str) -> bool:
    return "." in token and token.rsplit(".", 1)[-1].strip().lower() in PATH1_EXT


def own_inside(path: str, pref: str) -> bool:
    """PATH-1 mode 1: a slashless prefix with a real extension names a file anywhere, by basename."""
    if "/" not in pref and own_real_extension(pref):
        return path == pref or path.endswith("/" + pref)
    return path == pref or path.startswith(pref + "/")


def own_path_shaped(prefix: str, status: dict) -> bool:
    raw = prefix.strip("`\"'").rstrip(".")
    if not raw:
        return False
    if "/" in raw or "\\" in raw:
        return True
    if "." in raw and own_real_extension(raw.rstrip("/")):
        return True
    low = own_undotted(own_key(raw)).rstrip("/").lower()
    return any(low in [seg.lower() for seg in own_undotted(p).split("/")] for p in status)


def own_parent(raw: str) -> str:
    """V-4, written out: the prefix as written when its last named segment is `..` (trailing `/` and `.`
    segments dropped), else ""."""
    segs = raw.split("/")
    while segs and segs[-1] in ("", "."):
        segs.pop()
    return "/".join(segs) if segs and segs[-1] == ".." else ""


def own_could_hold(path: str, pref: str, raw: str) -> bool:
    """F-4 and V-4, written out: some reading of an off-tree prefix could hold `path`."""
    if own_parent(raw):
        return True
    seen_name = False
    for seg in pref.split("/"):
        if seg.strip("."):
            seen_name = True
        elif seen_name and seg not in ("", "."):
            return True
    want = [seg.lstrip(".") for seg in pref.split("/") if seg.strip(".")]
    have = [seg.lstrip(".") for seg in path.split("/")]
    return not want or any(have[i:i + len(want)] == want for i in range(len(have) - len(want) + 1))


def own_dot_miss(path: str, prefs: list) -> bool:
    return (path[:1] == "." and path[:2] != ".."
            and any(not x.startswith(".") and own_inside(path[1:], x) for x in prefs))


def expected_only_touches(d: dict, status: dict, notes: dict) -> tuple:
    """An only_touches claim's (verdict, reason), by this file's own code."""
    if not status:
        return "UNCHECKABLE", NO_PATHS_WHY
    if notes.get("files") or notes.get("differs"):
        return "UNCHECKABLE", NOT_SURE + (notes.get("files") or notes["differs"])
    p2 = d.get("prefix2")
    prefs = [own_key(d["prefix"]).rstrip("/.")] + ([own_key(p2).rstrip("/.")] if p2 else [])
    parent2 = bool(p2) and bool(own_parent(p2.replace("\\", "/")))
    if p2 and not parent2 and not own_path_shaped(p2, status):
        prefs = prefs[:1]
    raw_prefs = [d["prefix"]] + ([p2] if len(prefs) == 2 else [])
    not_paths = [own_key(x).rstrip("/.") for i, x in enumerate(raw_prefs)
                 if not (i == 1 and parent2) and not own_path_shaped(x, status)]
    outside = [p for p in status if not any(own_inside(p, x) for x in prefs)]
    dot_miss = [p for p in outside if own_dot_miss(p, prefs)]
    real = [p for p in outside if p not in dot_miss]
    off_pairs = [(x, own_key(r)) for x, r in zip(prefs, raw_prefs) if off_tree_key(x) or own_parent(own_key(r))]
    off = [own_parent(r) or x for x, r in off_pairs]
    beside = bool(off) and len(off) < len(prefs)
    if beside:
        real = [p for p in real if not any(own_could_hold(p, x, r) for x, r in off_pairs)]
    if not_paths:
        return "UNCHECKABLE", f"prefix {not_paths[0]!r} is not a path (#110)"
    if off and not (beside and real):
        return "UNCHECKABLE", f"prefix {off[0]!r} {OFF_TREE_WHY}"
    if dot_miss and not real:
        # NOTE_path2_eleventh_pass (round 10, R0.3): the keys print as Z-3's do, folded and ascii()-escaped
        miss = "[" + ", ".join(shown(p) for p in dot_miss[:3]) + "]"
        if len(prefs) == 1:
            return "UNCHECKABLE", f"paths outside {shown(prefs[0])} differ from it only by a leading dot: {miss} (#121)"
        return "UNCHECKABLE", (f"paths outside {' and '.join(shown(x) for x in prefs)} differ from them only by a "
                               f"leading dot: {miss} (#121)")
    if not real:
        return "VERIFIED", "all changed paths under prefix"
    if len(prefs) == 1:
        return "CONTRADICTED", f"paths outside {prefs[0]!r}: {real[:3]}"
    return "CONTRADICTED", f"paths outside {' and '.join(repr(x) for x in prefs)}: {real[:3]}"


def expected_count(n: int, status: dict, notes: dict) -> tuple:
    if not status:
        return "UNCHECKABLE", NO_PATHS_WHY
    if notes.get("files") or notes.get("differs"):
        return "UNCHECKABLE", f"{NOT_SURE}{notes.get('files') or notes['differs']}; claim says {n}"
    return ("VERIFIED" if n == len(status) else "CONTRADICTED"), f"diff changes {len(status)} files, claim says {n}"


def expected_path(kind: str, claimed: str, status: dict, notes: dict) -> tuple:
    """A path claim (accusation withheld, V14 repair 2): main's branches and strings, over this file's #97. (The tenth
    pass's K-5 is the guard's since NOTE_path2_eleventh_pass, and G-C9 reads it.)"""
    if notes.get("files") or notes.get("differs"):
        return "UNCHECKABLE", NOT_SURE + (notes.get("files") or notes["differs"])
    only_name = own_basename_only(status, claimed)                 # NOTE_path2_ninth_pass (Z-4)
    if only_name:
        return "UNCHECKABLE", only_name
    p, st = own_find_path(status, claimed)
    want = {"file_created": "A", "file_deleted": "D"}.get(kind)
    if p is None and "/" not in claimed and "\\" not in claimed:
        return "UNCHECKABLE", (f"{claimed!r} is a bare name absent from the diff — ambiguous between a file and a "
                               "library, so no accusation is made (V14 repair 2, a deliberate recall sacrifice)")
    if p is None:
        return "UNCHECKABLE", (f"{claimed!r} does not appear in the diff — accusation WITHHELD: this class failed "
                               "EXTERNAL-1 precision (0.23 vs 0.95 floor), disabled pending repair")
    if want and st != want:
        return "UNCHECKABLE", (f"{claimed!r} is status {st!r}, claim wants {want!r} — accusation WITHHELD pending the "
                               "EXTERNAL-1 repair")
    return "VERIFIED", f"diff status {st!r} for {p!r}"


# COMPAT-2's scaffold reading and the language suffixes (main's), read on the undotted key (AMENDMENT C-2).
COMPAT_SCAFFOLD = re.compile(
    r"(?:^|/)(?:tests?|testing|specs?|__tests__|examples?|samples?|demos?|docs?|scripts?|tools?|bench|"
    r"benchmarks?|fixtures?|internal|_internal|private|vendor|third_party|migrations?|cmd|e2e|integration|"
    r"mocks?|stories|storybook|playground|sandbox|experiments?|dev|build)/"
    r"|(?:^|/)(?:test_[^/]*|[^/]*_test\.(?:go|py)|[^/]*\.(?:test|spec)\.[^/]+|conftest\.py|setup\.py)$")
COMPAT_SUFFIXES = (("python", (".py",)), ("js/ts", (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts")),
                   ("go", (".go",)), ("rust", (".rs",)), ("java", (".java", ".kt")))


def own_compat_reading(sides: dict) -> tuple:
    """C-2, written out whole (NOTE_path2_ninth_pass, round-8 scorer lens): COMPAT's verdict, reason and detail with
    every language suffix and the scaffold test read on the undotted key, and main's definition patterns, parameter
    reading and `\\b` test (main's COMPAT code, which this branch does not change, read through BASE)."""
    empty = {"removed": [], "languages": [], "surface_removed": 0, "signature_changed": [], "compat2_candidate": False}
    if not sides:
        return ("UNCHECKABLE", "compatibility claimed; no per-file diff available to read (behaviour beyond names not "
                               "checked)", dict(empty))
    rx_of = {lang: rx for lang, (_sufs, rx) in BASE._COMPAT_LANGS.items()}
    added_by: dict = {}
    langs: list = []
    for path, (a, _r) in sides.items():
        for lang, sufs in COMPAT_SUFFIXES:
            if own_undotted(path).endswith(sufs):
                added_by.setdefault(lang, []).extend(a)
                if lang not in langs:
                    langs.append(lang)
    dropped: list = []
    changed: list = []
    for path, (_a, r) in sides.items():
        for lang, sufs in COMPAT_SUFFIXES:
            if not own_undotted(path).endswith(sufs):
                continue
            a_lines = added_by.get(lang, [])
            blob = "\n".join(a_lines)
            for line in r:
                mm = rx_of[lang].match(line)
                if not mm or mm.group("name").startswith("_"):
                    continue
                name = mm.group("name")
                if re.search(r"\b" + re.escape(name) + r"\b", blob):
                    before = BASE._compat_params(line, mm.end("name"))
                    if before is not None:
                        afters = [BASE._compat_params(x, am.end("name")) for x in a_lines
                                  for am in [rx_of[lang].match(x)] if am and am.group("name") == name]
                        afters = [x for x in afters if x is not None]
                        if afters and before not in afters and (path, lang, name) not in [x[:3] for x in changed]:
                            changed.append((path, lang, name, before, afters[0]))
                    continue
                if (path, lang, name) not in [x[:3] for x in dropped]:
                    dropped.append((path, lang, name, not COMPAT_SCAFFOLD.search(own_undotted(path))))
    if not langs:
        return ("UNCHECKABLE", "compatibility claimed; no language this reading covers in the diff (python, js/ts, go, "
                               "rust, java)", dict(empty))
    sig = f"; {len(changed)} signature(s) changed" if changed else ""
    detail = {"removed": [{"path": p, "language": lg, "name": n, "surface": sf} for p, lg, n, sf in dropped],
              "languages": langs, "surface_removed": sum(1 for x in dropped if x[3]),
              "signature_changed": [{"path": p, "language": lg, "name": n, "before": b, "after": a_}
                                    for p, lg, n, b, a_ in changed],
              "compat2_candidate": any(x[3] for x in dropped)}
    if not dropped:
        return ("UNCHECKABLE", f"compatibility claimed; no public top-level definition removed ({', '.join(langs)} "
                               f"read; behaviour beyond names not checked){sig}", detail)
    surface = [x for x in dropped if x[3]]
    scaffold = [x for x in dropped if not x[3]]
    named = surface or scaffold
    shown = ", ".join(f"{p}: {n}" for p, _lg, n, _s in named[:5])
    more = f" (+{len(named) - 5} more)" if len(named) > 5 else ""
    if surface:
        rest = f"; {len(scaffold)} more in test/example/internal code" if scaffold else ""
        return ("UNCHECKABLE", f"compatibility claimed; the diff removes {len(surface)} public definition(s) from the "
                               f"surface, not re-defined in the added lines: {shown}{more}{rest}{sig}", detail)
    return ("UNCHECKABLE", f"compatibility claimed; {len(scaffold)} public definition(s) removed, all in "
                           f"test/example/internal code: {shown}{more}{sig}", detail)


def compat_violations(c, sides: dict) -> list:
    """C-2, written out: NOTE_path2_ninth_pass, the whole reading -- the verdict, the reason and every field of the
    detail (the removed and signature_changed lists, languages, counts, candidate) -- not only the languages and the
    surface flags, so a defect inside C-2's own code is refused rather than attributed to #121."""
    d = c.detail or {}
    # NOTE_path2_tenth_pass (round-9 scorer lens, blocker): no guard. Every measured compat claim carries every
    # COMPAT_EXTRAS key (the unmeasured one returns before this in oracle_violations), so a missing key is a defect,
    # not a reason to skip the claim: C-2's code dropping "languages" on a leading-dot path was admitted by that guard.
    verdict, why, want = own_compat_reading(sides)
    if (not set(COMPAT_EXTRAS) <= set(d) or (c.verdict, c.why) != (verdict, why)
            or {k: d[k] for k in COMPAT_EXTRAS if k in d} != want):
        return ["G-C7_oracle:C-2_compat_surface"]
    return []


def own_status_notes(status_paths: list) -> dict:
    """Y-1 at the git door, written out: two name-status paths that differ only in case are one key (compared by the
    one fold both ports carry, NOTE_path2_tenth_pass K-2)."""
    seen: dict = {}
    for p in status_paths:
        form = p.replace("\\", "/")
        while form.startswith("/") or form.startswith("./"):
            form = form[1:] if form.startswith("/") else form[2:]
        seen.setdefault(own_fold(form), set()).add(form)
    return {"files": COLLIDE_WHY} if any(len(v) > 1 for v in seen.values()) else {}


def parse_violations(diff: str, paths, with_reading: bool = False):
    """G-C7's parse-level oracles: F-2's split, W-1's status, added lines and sides, the notes (Y-1, Y-2, Z-3), Y-1's
    git-door reading on the record's paths (and a case-folded variant), Y-4's /dev/null on every header path, #121's
    key on every path. NOTE_path2_tenth_pass: also asked, alone, of a PR the repaired instrument excludes (G-C2)."""
    out: list = []
    if new._diff_lines(diff) != git_lines(diff):
        out.append("G-C7_oracle:F-2_split")
    own_facts: dict = {}
    own = own_read(diff, own_facts)
    got = new._read_diff(diff)
    if list(got[0].items()) != list(own[0].items()):
        out.append("G-C7_oracle:W-1_status")
    if got[1] != own[1]:
        out.append("G-C7_oracle:W-1_added_lines")
    if got[2] != own[2]:
        out.append("G-C7_oracle:W-1_sides")
    got_facts: dict = {}
    if new._diff_notes(diff, got_facts) != own[3]:
        out.append("G-C7_oracle:Y_notes")
    if got_facts != own_facts:                            # NOTE_path2_twelfth_pass: what the licences read (A.1, A.2)
        out.append("G-C7_oracle:licence_facts")
    raw_list = list(paths)
    folded = next(([*raw_list, p.upper()] for p in raw_list if p.upper() != p), None)
    for variant in (raw_list, folded):
        if variant is not None and new._status_notes(variant) != own_status_notes(variant):
            out.append("G-C7_oracle:Y-1_status_notes")
            break
    heads = [x[4:].strip() for x in git_lines(diff) if x.startswith(("--- ", "+++ "))]
    if any(new._dev_null(h) != own_dev_null(h) for h in heads):
        out.append("G-C7_oracle:Y-4_dev_null")
    if any(new._norm(p) != own_key(p) for p in raw_list):
        out.append("G-C7_oracle:#121_key")
    return (out, own) if with_reading else out


def oracle_violations(summary: str, diff: str, paths, g, status_override: dict | None = None,
                      status_paths: list | None = None, name_status: str | None = None,
                      git_text: str | None = None) -> list:
    """G-C7: the repaired instrument against this file's own reading of every rule it re-implements, on one
    record. `g` is the repaired gate's result for the record; `status_override` is the git door's status.
    NOTE_path2_eighth_pass: declared (DECLARE-1) claims are re-read too, through their canonical sentences, and
    the file-list kinds (files_changed_count, only_touches, the path claims) and COMPAT's C-2 reading join the
    claim oracles. NOTE_path2_ninth_pass: the licensed-difference rule is re-derived against main's reading as this
    file reads it (`OwnMain`, `own_file_list_differs`); at the git door, main's reading is of the text `gate_diff`
    reads (`git_text`, git's bytes through universal newlines) and of git's `--name-status` (`name_status`); and
    Y-1's git-door reading (`_status_notes`) is held to this file's on every record's paths and a case-folded
    variant, whether or not the record can be rebuilt."""
    out, own = parse_violations(diff, paths, with_reading=True)
    status, added, sides, notes = own
    main = own_main_raw(diff)
    if status_override is not None:
        status = status_override
        listed = own_status_notes(status_paths or [])        # the git door's file list is git's
        differs = own_status_differs(name_status or "", status)
        notes = {k: v for k, v in (("files", listed.get("files")), ("bom", notes.get("bom")),
                                   ("differs", differs)) if v}
        main = OwnMain(git_text if git_text is not None else diff, (own_main_name_status(name_status or ""),))
        if new._status_differs(new._main_name_status(name_status or ""), status) != differs:
            out.append("G-C7_oracle:Z-3_status_differs")
    lines = set(added)
    for a, r in sides.values():
        lines.update(a)
        lines.update(r)
    for x in sorted(lines):
        if any(new._defined_name(x, rm) != defined(x, rm) for rm in (False, True)):
            out.append("G-C7_oracle:W-2_defined_name")
            break
        if any(new._test_name(x, rm) != test_name(x, rm) for rm in (False, True)):
            out.append("G-C7_oracle:W-2_test_name")
            break
        if new._bom_hidden(x) != own_bom_hidden(x):
            out.append("G-C7_oracle:Y-2_bom_hidden")
            break
        if new._skew_test(x) != own_skew_test(x):
            out.append("G-C7_oracle:Y-3_skew_test")
            break
    blob = "\n".join(added)
    if new._added_tests(blob) != sum(1 for x in blob.split("\n") if test_name(x)):
        out.append("G-C7_oracle:F-3_got")
    tokens = list(paths) + [v for c in g.claims for k, v in (c.detail or {}).items() if k in ("path", "prefix", "prefix2") and v]
    if any(new._norm(p) != own_key(p) for p in tokens):
        out.append("G-C7_oracle:#121_key")
    for c in g.claims:
        if c.kind in PATH_KINDS:
            if new._find_path(status, c.detail["path"]) != own_find_path(status, c.detail["path"]):
                out.append("G-C7_oracle:#97_tiers")
                break
    sites = [(s, mm) for s in SENTENCES.split(summary) for mm in SYMBOL_TEMPLATE.finditer(s)]
    measured = bool(status) or bool(blob)
    if measured != g.measured:
        out.append("G-C7_oracle:measured")
    # NOTE_path2_eleventh_pass (round-10 scorer lens, blocker): the unmeasured reason, by this file's own code -- where
    # main raises too -- and none on a measured gate. `gate_diff_text` counts the characters of its input; `gate_diff` does
    # not (the git door, `status_override`).
    unmeasured = "" if measured else own_unmeasured(None if status_override is not None else len(diff or ""))
    if g.why_unmeasured != unmeasured:
        out.append("G-C7_oracle:why_unmeasured")
    read = ("tests_added", "symbol_added", "files_changed_count", "only_touches", "compat_claim") + PATH_KINDS
    for c in g.claims:
        if c.kind not in read:
            continue
        declared = bool((c.detail or {}).get("declared"))
        if not measured:
            if (c.verdict, c.why) != ("UNCHECKABLE", unmeasured):
                out.append(f"G-C7_oracle:{c.kind}_claim")
            continue
        if c.kind == "compat_claim":
            out += compat_violations(c, sides)
            continue
        if c.kind == "tests_added":
            want = expected_tests(c, status, added, sides, notes, main)
        elif c.kind == "files_changed_count":
            want = expected_count(int(c.detail["n"]), status, notes)
        elif c.kind == "only_touches":
            want = expected_only_touches(c.detail, status, notes)
        elif c.kind in PATH_KINDS:
            want = expected_path(c.kind, c.detail["path"], status, notes)
        elif declared:
            # a declared symbol claim's text is its canonical sentence ("Adds function X."): read the name there
            mm = next((x for x in SYMBOL_TEMPLATE.finditer(c.text) if x.group("name") == c.detail["name"]), None)
            if mm is None:
                out.append("G-C7_oracle:symbol_claim_unplaced")
                continue
            want = expected_symbol(c, c.text, mm.start("name"), status, added, sides, main)
        else:
            k = next((i for i, (s, mm) in enumerate(sites) if s.strip()[:160] == c.text
                      and mm.group("name") == c.detail["name"] and mm.group("kind") == c.detail["kind"]), None)
            if k is None:
                out.append("G-C7_oracle:symbol_claim_unplaced")
                continue
            want = expected_symbol(c, sites[k][0], sites[k][1].start("name"), status, added, sides, main)
            del sites[k]
        if (c.verdict, c.why) != want:
            out.append(f"G-C7_oracle:{c.kind}_claim")
    return list(dict.fromkeys(out))


# NOTE_path2_ninth_pass (round-8 scorer lens, major): the never-read sentences themselves, not only their count.
GATE_FIELDS = ("measured", "why_unmeasured", "uncovered_sentences", "sentences_total", "uncovered_texts")


def gate_violations(gb, gn) -> list:
    """G-C1, extended: each gate's verdict (strict off) is the one its own claims give. (The fields in
    GATE_FIELDS are held to the baseline's in `Tally.pair`, by counterfactual.)"""
    out = []
    for g in (gb, gn):
        if g.verdict != ("FAIL" if any(c.verdict == "CONTRADICTED" for c in g.claims) else "PASS"):
            out.append("G-C1_gate_verdict_not_from_its_claims")
    return list(dict.fromkeys(out))


SUMMARY_FIELDS = ("uncovered_sentences", "sentences_total", "uncovered_texts")
REPORT_FIELDS = ("verdict", "base", "head", "uncovered_sentences", "sentences_total", "uncovered_texts",
                 "unparsed_claims", "measured", "why_unmeasured")


def report_violations(gb, gn) -> list:
    """NOTE_path2_tenth_pass (round-9 scorer lens, minor): the report users read. Both instruments name the same base
    and head and the same never-parsed claims (the observer is blocked for both); and each gate's serialised report
    (`to_dict`, which the CLI and the Action print) is its own verdict, fields and claims, key for key. The fields
    that may move with a rule (GATE_FIELDS, the verdict) are scored where they are compared."""
    out = []
    if (gb.base, gb.head, gb.unparsed_claims) != (gn.base, gn.head, gn.unparsed_claims):
        out.append("G-C1_base_head_differ")
    for g in (gb, gn):
        d = g.to_dict()
        if (set(d) != {"diffgate", "claims", *REPORT_FIELDS} or d["diffgate"] != "v0"
                or any(d[f] != getattr(g, f) for f in REPORT_FIELDS)
                or d["claims"] != [dict(kind=c.kind, text=c.text, detail=c.detail, verdict=c.verdict, why=c.why)
                                   for c in g.claims]):
            out.append("G-C1_report_differs_from_its_gate")
    return list(dict.fromkeys(out))


def report_self_violations(g) -> list:
    """A gate's serialised report (`to_dict`) is its own verdict, fields and claims, key for key (`report_violations`'s
    second half, for one gate)."""
    d = g.to_dict()
    if (set(d) != {"diffgate", "claims", *REPORT_FIELDS} or d["diffgate"] != "v0"
            or any(d[f] != getattr(g, f) for f in REPORT_FIELDS)
            or d["claims"] != [dict(kind=c.kind, text=c.text, detail=c.detail, verdict=c.verdict, why=c.why)
                               for c in g.claims]):
        return ["G-C1_report_differs_from_its_gate"]
    return []


def strict_violations(*pairs) -> list:
    """G-C1 under --strict (NOTE_path2_ninth_pass): for each instrument's (strict off, strict on) gates, strict moves
    no claim, and the strict verdict is FAIL exactly when a claim is CONTRADICTED or UNCHECKABLE.
    NOTE_path2_eleventh_pass (round-10 scorer lens, blocker): and the strict report is the strict-off report, key for
    key, but its verdict -- the claims with their details, the never-read sentences and their counts, `measured`,
    `why_unmeasured`, `base`, `head`, the never-parsed claims -- and it is its own gate's report."""
    out = []
    for g, gs in pairs:
        if [(c.kind, c.text, c.verdict, c.why) for c in gs.claims] != [(c.kind, c.text, c.verdict, c.why) for c in g.claims]:
            out.append("G-C1_strict_moves_a_claim")
        if gs.verdict != ("FAIL" if any(c.verdict in ("CONTRADICTED", "UNCHECKABLE") for c in gs.claims) else "PASS"):
            out.append("G-C1_strict_verdict_not_from_its_claims")
        d, ds = g.to_dict(), gs.to_dict()
        d.pop("verdict", None)
        ds.pop("verdict", None)
        if json.dumps(d, sort_keys=True, default=str) != json.dumps(ds, sort_keys=True, default=str):
            out.append("G-C1_strict_report_differs")
        out += report_self_violations(gs)
    return list(dict.fromkeys(out))


# ── NOTE_path2_eleventh_pass: THE GUARD (G-C9), this file's own ───────────────────────────────────────────────────────
#
# The repaired gate's claims are the guard's: this reading's (every repair on), each held to main's -- BASE, the bytes the
# instrument vendors as styxx/_diffgate_ref.py -- a difference kept only where one repair switched off gives main's verdict
# back and that repair's own precondition holds on the claim. Written out here: the pairing, the switched readings (this
# file's own reverts of #97, #121 and #101, on its copy of the instrument), the three preconditions, K-5's sentence reading
# and every reason the guard prints.

GUARD_DIFFERS = ("main's reading gives {main} and this one {this}; no named repair (#97's exact and suffix tiers, "
                 "#121's dotted key, #101's pairing) explains the difference on this claim, so it abstains")
GUARD_RAISES = ("main's reading raises on this diff and gives no verdict; this one gives {this}, and no named repair "
                "licenses a verdict where main gives none")
GUARD_ABSENT = ("main's reading makes no such claim of this sentence; this one gives {this}, and no named repair "
                "licenses a verdict where main gives none")
K5_WHY = ("the two ports' templates may read this sentence apart (it holds a character at or past U+0080, or one of "
          "U+001C to U+001F), so the claim reads as main read it, and {}")
K5_ABSENT = "main's reading makes no such claim of it"
K5_RAISES = "main's reading raises on this diff"
# The characters the two ports' templates read apart besides the word characters: Python's `\s` alone holds U+001C to
# U+001F and U+0085, JavaScript's alone U+FEFF; JavaScript's multiline `^` starts a line after U+2028 and U+2029.
APART_MARKS = frozenset(map(chr, (0x1C, 0x1D, 0x1E, 0x1F, 0x85, 0xFEFF, 0x2028, 0x2029)))
K5_KINDS = PATH_KINDS + ("files_changed_count", "tests_added", "only_touches")


def claim_keys(claims) -> list:
    """(kind, text, occurrence) of each claim: how a claim is paired with main's."""
    seen: Counter = Counter()
    out = []
    for c in claims:
        out.append((c.kind, c.text, seen[(c.kind, c.text)]))
        seen[(c.kind, c.text)] += 1
    return out


def own_claim_sentences(summary: str, declared: bool = True) -> list:
    """(kind, sentence) of each claim, in the order both instruments extract them, replayed with main's own extraction
    (the baseline's templates and filters, which the repair does not change); a declared claim's sentence is its
    canonical one, and a declaration that reads as no sentence (an unverifiable key, a problem) has none."""
    out: list = []
    for sent in SENTENCES.split(summary):
        for kind, rx in BASE._TEMPLATES:
            for mm in rx.finditer(sent):
                # NOTE_path2_twelfth_pass (round-11 scorer lens): `kind` itself is rebound, as main's loop rebinds it (V13
                # repair 1 writes `kind = "file_touched"`), so a later match of the same template in the sentence is read
                # as file_touched too -- main's leak, which both instruments carry
                if kind in PATH_KINDS and BASE._names_without_claiming(sent, mm):
                    continue
                if kind in PATH_KINDS and BASE._is_non_file_noun(mm.group("path")):
                    continue
                if kind in ("file_created", "file_deleted") and BASE._demoted_by_containment(sent, mm):
                    kind = "file_touched"
                if BASE.V14_CONTAINMENT_TOUCH and kind == "file_touched" and BASE._demoted_by_containment(sent, mm):
                    continue
                if (BASE.BC1_BY_CONSTRUCTION and kind == "symbol_added"
                        and mm.group("name").lower() in BASE._SYMBOL_WORDS):
                    continue
                out.append((kind, sent))
    if declared:
        dtext, drep = BASE._declaration_pass(summary)
        if drep["declared"]:
            if dtext:
                out += own_claim_sentences(dtext, declared=False)
            out += [(u["key"], None) for u in drep["unverifiable"]]
            out += [("declaration_problem", None) for _p in drep["problems"]]
    return out


def own_apart_readings(sentence: str) -> tuple:
    """K-5 at the sentence, written out: (the two ports' templates may read this sentence's claims apart, they may read
    its symbol_added claims apart). A mark -- U+001C to U+001F, U+0085, U+FEFF, U+2028, U+2029, a CR with a character after
    it -- reads both apart; a non-ASCII word character (the table's word bit, or the skew set) reads the claims apart, and a
    symbol claim only where it lies outside every name the symbol template reads (a name runs, from where it starts, over
    the table's word and identifier-continuing characters and the skew set)."""
    words = []
    for i, ch in enumerate(sentence):
        if ch < "\x80":
            if "\x1c" <= ch <= "\x1f" or (ch == "\r" and i + 1 < len(sentence)):
                return True, True
        elif ch in APART_MARKS:
            return True, True
        elif xid_mask(ch) & 4 or skew(ch):
            words.append(i)
    if not words:
        return False, False
    spans = []
    for mm in SYMBOL_TEMPLATE.finditer(sentence):
        j = mm.start("name")
        while j < len(sentence) and (xid_mask(sentence[j]) & 6 or skew(sentence[j])):
            j += 1
        spans.append((mm.start("name"), j))
    return True, any(not any(a <= i < b for a, b in spans) for i in words)


def own_k5(kind: str, sentence) -> bool:
    if sentence is None or (kind not in K5_KINDS and kind != "symbol_added"):
        return False
    anyk, symk = own_apart_readings(sentence)
    return symk if kind == "symbol_added" else anyk


def own_precondition(repair: str, c, status: dict, sides: dict, licence: dict | None = None) -> bool:
    """The three preconditions, written out, on this file's own reading of the diff (every repair on):
    #97 a path claim the tiered resolution resolved by the exact or the suffix tier, where main's loop over the same file
    list takes another entry; #121 a key the claim reads keeps a leading dot (a key of the file list or of the sides, the
    claimed path's, an only_touches prefix's); #101 a removed definition of the same name in the same file was paired.
    NOTE_path2_twelfth_pass (A.1, A.2), `licence` this file's own facts (`own_read`, `own_git_licence`): #97's match must
    also hold on every path its entry was read from, case kept, and no Z-3 doubt may be read; #121 needs git's own
    rendering. With no facts, neither licenses."""
    d = c.detail or {}
    facts = licence or {}
    if repair == "#97":
        if c.kind not in PATH_KINDS or not isinstance(d.get("path"), str):
            return False
        key = own_key(d["path"])
        p, _st = own_find_path(status, d["path"])
        if p is None or not (p == key or p.endswith("/" + key)):
            return False
        if facts.get("soft"):
            return False
        claimed = own_case_kept(d["path"])
        read_as = (facts.get("forms") or {}).get(p, [])
        if not read_as or any(f != claimed and not f.endswith("/" + claimed) for f in read_as):
            return False
        return any_tier(status, key) != p
    if repair == "#121":
        if not facts.get("rendered"):
            return False
        own = [d[k] for k in ("path", "prefix", "prefix2") if isinstance(d.get(k), str)]
        return (any(k.startswith(".") for k in status) or any(k.startswith(".") for k in sides)
                or any(own_key(x).startswith(".") for x in own))
    if repair == "#101":
        if c.kind == "tests_added":
            return changed_tests(sides, status) > 0
        if c.kind == "symbol_added" and isinstance(d.get("name"), str):
            return any(status.get(p) != "A" and any((defined(x) or ("", ""))[1] == d["name"] for x in a)
                       and any((defined(x, True) or ("", ""))[1] == d["name"] for x in r) for p, (a, r) in sides.items())
        return False
    raise ValueError(repair)


def expected_guard(summary: str, before: list, main, switched, status: dict, sides: dict, tally=None,
                   facts: dict | None = None) -> list:
    """The final claims, by this file's own guard: `before` the repaired reading's claims before the guard, `main` the
    baseline's claims (None where it raises), `switched(repair)` the reading's claims with that repair reverted by this
    file's own revert (None where that copy raises). Each claim is (kind, text, verdict, why, detail).
    NOTE_path2_twelfth_pass: `facts` what the tightened licences read, by this file's own code (`own_read`,
    `own_git_licence`); an abstention where some repair's precondition held (its switch did not give main's verdict
    back) is counted apart, since only such a record can refuse a licence granted without the switch."""
    theirs = dict(zip(claim_keys(main), main)) if main is not None else {}
    sentences = own_claim_sentences(summary)
    memo: dict = {}

    def switched_verdict(repair: str, key):
        if repair not in memo:
            got = switched(repair)
            memo[repair] = dict(zip(claim_keys(got), got)) if got is not None else {}
        other = memo[repair].get(key)
        return None if other is None else other.verdict

    def count(what: str) -> None:
        if tally is not None:
            tally.n[what] += 1

    out = []
    for i, (key, c) in enumerate(zip(claim_keys(before), before)):
        r = theirs.get(key)
        kind, sentence = sentences[i] if i < len(sentences) and sentences[i][0] == c.kind else (c.kind, None)
        if c.kind != "tests_pass" and own_k5(kind, sentence):
            count("guard_k5_read_as_main")
            if r is not None:
                out.append((r.kind, r.text, r.verdict, r.why, r.detail))
            else:
                out.append((c.kind, c.text, "UNCHECKABLE", K5_WHY.format(K5_RAISES if main is None else K5_ABSENT),
                            c.detail))
            continue
        mv = None if r is None else r.verdict
        if c.verdict == "UNCHECKABLE" or mv == c.verdict:
            out.append((c.kind, c.text, c.verdict, c.why, c.detail))
            continue
        licence = next((repair for repair in TABLE_RULES if mv is not None
                        and own_precondition(repair, c, status, sides, facts) and switched_verdict(repair, key) == mv),
                       None)
        if licence is not None:
            count(f"guard_licensed_by_{licence}")
            out.append((c.kind, c.text, c.verdict, c.why, c.detail))
            continue
        count("guard_abstained")
        if mv is not None and any(own_precondition(repair, c, status, sides, facts) for repair in TABLE_RULES):
            count("guard_abstained_where_a_precondition_held")
        why = (GUARD_RAISES.format(this=c.verdict) if main is None else
               GUARD_ABSENT.format(this=c.verdict) if r is None else GUARD_DIFFERS.format(main=mv, this=c.verdict))
        out.append((c.kind, c.text, "UNCHECKABLE", why, c.detail))
    return out


def guard_violations(summary: str, main, before, final, switched, instrument_switched, status: dict, sides: dict,
                     tally=None, facts: dict | None = None) -> list:
    """G-C9: the repaired gate's final claims are this file's own guard's, claim for claim (verdict, reason, detail); its
    verdict is the one its final claims give (G-C1); and the instrument's own switches read as this file's reverts of the
    same rules (`instrument_switched(repair)` against `switched(repair)`)."""
    out = []
    want = expected_guard(summary, before.claims, main, switched, status, sides, tally, facts)
    got = [(c.kind, c.text, c.verdict, c.why, c.detail) for c in final.claims]
    if len(want) != len(got):
        out.append("G-C9_guard_claims")
    for w, g in zip(want, got):
        if json.dumps(w, sort_keys=True, default=str) != json.dumps(g, sort_keys=True, default=str):
            out.append(f"G-C9_guard:{g[0]}")
    for repair in TABLE_RULES:
        a, b = switched(repair), instrument_switched(repair)
        sig = (lambda cl: None if cl is None else [(c.kind, c.text, c.verdict, c.why) for c in cl])
        if sig(a) != sig(b):
            out.append(f"G-C9_switch_is_not_the_revert:{repair}")
    return list(dict.fromkeys(out))


# NOTE_path2_twelfth_pass (round-11 scorer lens): what a reverted copy may raise and still read as "the copy raises" --
# main's `AttributeError` on a `+++ /dev/null` with no `---` line before it (K-3), which a revert can restore. Nothing
# else: RuntimeError (git failing), OSError and subprocess.TimeoutExpired are the environment's, and fail the run.
REVERTED_RAISES = (AttributeError,)


class Counterfactual:
    """One record's claims under the repaired copy with a set of rules reverted, computed on demand, through
    the raw door or, given a `GitDoor`, through `gate_diff` (NOTE_path2_seventh_pass, G-C8)."""

    def __init__(self, summary: str, diff: str, door=None):
        self.summary, self.diff, self.door = summary, diff, door
        self.cache: dict = {}

    def gate(self, rules):
        """The gate with `rules` reverted, or None where that copy raises. NOTE_path2_tenth_pass (K-3): a revert may
        restore code that raised (the fifth pass's parsers on a `+++ /dev/null` with no `---` line before it); a
        copy that raises gives nothing back. NOTE_path2_twelfth_pass (round-11 scorer lens): only what reverted code
        raises reads so -- `AttributeError`, as main raises on K-3 -- and a failure of git, the operating system or a
        timeout propagates and fails the run, instead of reading as an instrument defect."""
        key = frozenset(rules)
        if key not in self.cache:
            with reverted(CF, key):
                try:
                    self.cache[key] = (CF.gate_diff_text(self.summary, self.diff, run=None, strict=False)
                                       if self.door is None else self.door.gate(CF, self.summary))
                except REVERTED_RAISES:
                    self.cache[key] = None
        return self.cache[key]

    def claims(self, rules) -> list:
        g = self.gate(rules)
        return [] if g is None else g.claims

    def reading(self, rules):
        """NOTE_path2_eleventh_pass: the copy's reading BEFORE the guard (every switch on) with `rules` reverted -- the
        claims the guard reads for a switched repair when `rules` is one of #97, #121, #101 -- or None where it raises."""
        key = ("reading", frozenset(rules))
        if key not in self.cache:
            with reverted(CF, frozenset(rules)):
                try:
                    g = (CF._evaluate_text(self.summary, self.diff, CF._ALL_ON) if self.door is None
                         else self.door.reading(CF, self.summary))
                    self.cache[key] = g.claims
                except REVERTED_RAISES:
                    self.cache[key] = None
        return self.cache[key]

    def attribute_fields(self, gb):
        """G-C1, extended: the rules whose single revert gives back the baseline's gate-level fields, else
        the smallest set of two that does, else None."""
        want = tuple(getattr(gb, f) for f in GATE_FIELDS)

        def gives_back(rules) -> bool:
            g = self.gate(rules)
            return g is not None and tuple(getattr(g, f) for f in GATE_FIELDS) == want
        single = tuple(r for r in RULES if gives_back({r}))
        if single:
            return single
        return next((combo for combo in itertools.combinations(RULES, 2) if gives_back(combo)), None)

    def attribute(self, i: int, cb):
        """(rules, how) for claim `i`: every rule whose single revert gives back the baseline claim;
        else the smallest set of two or three that does; else (None, "none") or (None, "all rules")."""
        target = signature(cb)

        def gives_back(rules) -> bool:
            got = self.claims(rules)
            return i < len(got) and signature(got[i]) == target

        single = tuple(r for r in RULES if gives_back({r}))
        if single:
            return single, "single"
        for size in (2, 3):
            for combo in itertools.combinations(RULES, size):
                if gives_back(combo):
                    return combo, "joint"
        if not gives_back(RULES):
            return None, "none"
        # NOTE_path2_ninth_pass: a claim four rules abstain on at once (a U+FEFF-led test: W-1's line 1, Y-2, Z-1 and
        # Z-5) needs all four reverted. The rules whose revert is needed with every other one reverted are tried as a
        # set; if they give the baseline back they are the attribution, and each must still admit the move.
        needed = tuple(r for r in RULES if not gives_back(set(RULES) - {r}))
        if needed and gives_back(needed):
            return needed, "joint"
        return None, "all rules"


# ── the git door (NOTE_path2_seventh_pass, G-C8): a record rebuilt as a two-commit repository ─────────

# NOTE_path2_ninth_pass (round-8 scorer lens, blocker): the door's repositories are written by `git fast-import`
# into a bare repository -- no working tree, no file on disk -- so a path segment may open with a dot (#121's dotted
# keys: `.github/x.yml`, `.pr_agent.toml`, `src/.env`) and two paths may differ only in case (Y-1's collision); only
# `.`, `..` and `.git` (in any case) are refused, which git itself refuses in a tree.
# NOTE_path2_tenth_pass (round-9 scorer lens, major): and U+0085 and U+2028 inside a segment, which str.splitlines()
# breaks on and git's split does not. With core.quotePath off (below) git writes them raw in `--name-status` and in
# the diff, so main's git door cut such a path and Z-3's git-door reading abstains: the one shape a rebuilt record can
# carry that makes it fire (DOOR_CANARIES score one in both modes).
# NOTE_path2_eleventh_pass (round-10 scorer lens, blocker): and U+2029, the third code point str.splitlines() breaks on and
# git's split does not; it makes a third canary.
SPLIT_ONLY_BY_PYTHON = chr(0x85) + chr(0x2028) + chr(0x2029)
SAFE_PATH = re.compile("^[A-Za-z0-9_.][A-Za-z0-9_." + SPLIT_ONLY_BY_PYTHON + "-]*(?:/[A-Za-z0-9_.][A-Za-z0-9_."
                       + SPLIT_ONLY_BY_PYTHON + "-]*)*$")
PLACEHOLDER = "rebuilt_placeholder.txt"


def _safe(path: str) -> bool:
    return (bool(SAFE_PATH.match(path)) and len(path) < 180 and path != PLACEHOLDER
            and not any(seg in (".", "..") or seg.lower() == ".git" for seg in path.split("/")))


def rebuild(diff: str):
    """(before, after) -- {path: text or None} -- that a diff describes, or None when it cannot be rebuilt
    faithfully: every file a `---`/`+++` pair naming one safe path (a `diff --git`, `index`, `new file mode`
    or `deleted file mode` line may precede it; nothing else may), every hunk exact by its counts and
    followed by a header, a hunk or the end, hunks moving forward with the same gap on both sides. The
    lines between hunks are written as the same filler on both sides; a `\\` marker drops the newline."""
    lines = diff.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    before: dict = {}
    after: dict = {}
    k, n = 0, len(lines)
    while k < n:
        x = lines[k]
        if x.startswith(("diff --git ", "index ", "new file mode ", "deleted file mode ")):
            k += 1
            continue
        if not (x.startswith("--- ") and k + 2 < n and lines[k + 1].startswith("+++ ") and HUNK.match(lines[k + 2])):
            return None
        old, nw = x[4:], lines[k + 1][4:]
        pa = None if old == "/dev/null" else (old[2:] if old.startswith("a/") else old)
        pb = None if nw == "/dev/null" else (nw[2:] if nw.startswith("b/") else nw)
        path = pb if pb is not None else pa
        if path is None or (pa is not None and pb is not None and pa != pb) or not _safe(path):
            return None
        if path in before or path in after:
            return None
        k += 2
        old_lines: list = []
        new_lines: list = []
        eol = {"old": True, "new": True}
        while k < n and HUNK.match(lines[k]):
            a, b, c, d = counts(HUNK.match(lines[k]))
            gap_old = (a if b else a + 1) - 1 - len(old_lines)
            gap_new = (c if d else c + 1) - 1 - len(new_lines)
            if gap_old != gap_new or gap_old < 0 or not (eol["old"] and eol["new"]):
                return None
            fill = [f"# rebuilt line {len(old_lines) + i + 1}" for i in range(gap_old)]
            old_lines += fill
            new_lines += fill
            k += 1
            last = None
            while (b or d) and k < n:
                y = lines[k]
                if y[:1] == " " and b and d:
                    old_lines.append(y[1:])
                    new_lines.append(y[1:])
                    b, d, last = b - 1, d - 1, "both"
                elif y[:1] == "-" and b:
                    old_lines.append(y[1:])
                    b, last = b - 1, "old"
                elif y[:1] == "+" and d:
                    new_lines.append(y[1:])
                    d, last = d - 1, "new"
                elif y[:1] == "\\" and last:
                    for side in (("old", "new") if last == "both" else (last,)):
                        eol[side] = False
                else:
                    return None
                k += 1
            if b or d:
                return None
            while k < n and lines[k][:1] == "\\" and last:
                for side in (("old", "new") if last == "both" else (last,)):
                    eol[side] = False
                k += 1
        if pa is None and old_lines:
            return None
        if pb is None and new_lines:
            return None
        before[path] = None if pa is None else "\n".join(old_lines) + ("\n" if old_lines and eol["old"] else "")
        after[path] = None if pb is None else "\n".join(new_lines) + ("\n" if new_lines and eol["new"] else "")
    if not after:
        return None
    paths = list(after)
    if any(q.startswith(p + "/") for p in paths for q in paths):      # a file and a directory of one name
        return None
    return before, after


def _rmtree(path: str) -> None:
    def onerror(func, p, _exc):
        with contextlib.suppress(OSError):
            os.chmod(p, stat.S_IWRITE)
            func(p)
    shutil.rmtree(path, onerror=onerror)


def _fast_import_path(path: str) -> bytes:
    return path.encode("utf-8")                       # `_safe` paths need no quoting in a fast-import `M` command


class GitDoor:
    """A record rebuilt as a two-commit repository in a temporary directory (G-C8). `gate(m, summary)`
    runs m.gate_diff on it; git's answers are memoised per module, so a counterfactual's reverts re-read
    the same bytes without re-running git. Use as a context manager: the directory is removed on exit.

    NOTE_path2_ninth_pass (round-8 scorer lens, blocker): the repository is BARE and both commits are written with
    `git fast-import` (core.ignorecase off), so no file touches the disk: a dot-led segment and two paths that differ
    only in case rebuild. `renames` turns git's rename detection on (the second pass, for a record that deletes one
    file and creates another); the plain pass keeps it off, because the record says "deleted" and "created"."""

    def __init__(self, before: dict, after: dict, renames: bool = False):
        self.dir = tempfile.mkdtemp(prefix="path2_gitdoor_")
        self.memo: dict = {}
        self.renames = renames

        def git(*args, stdin: bytes | None = None):
            r = subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", *args], cwd=self.dir,
                               capture_output=True, timeout=60, input=stdin)
            if r.returncode != 0:
                raise RuntimeError(f"git {args[0]}: {r.stderr.decode('utf-8', 'replace')[:200]}")
            return r.stdout.decode("utf-8", "replace")

        def commit(ref: str, files: dict, parent: str | None) -> bytes:
            out = bytearray(b"commit refs/heads/" + ref.encode("ascii") + b"\n")
            out += b"committer t <t@t> 1700000000 +0000\ndata 0\n"
            if parent:
                out += b"from refs/heads/" + parent.encode("ascii") + b"\n"
            out += b"deleteall\n"
            for p, text in sorted(files.items()):
                data = text.encode("utf-8")
                out += b"M 100644 inline " + _fast_import_path(p) + b"\n" + b"data %d\n" % len(data) + data + b"\n"
            return bytes(out) + b"\n"

        try:
            git("init", "-q", "--bare")
            git("config", "core.ignorecase", "false")
            git("config", "core.quotePath", "false")          # NOTE_path2_tenth_pass: U+0085, U+2028 written raw
            git("config", "diff.renames", "true" if renames else "false")
            placeholder = {PLACEHOLDER: "rebuilt by path2_gates.py\n"}
            old = {**placeholder, **{p: t for p, t in before.items() if t is not None}}
            new_files = {**placeholder, **{p: t for p, t in after.items() if t is not None}}
            git("fast-import", "--quiet", stdin=commit("before", old, None) + commit("after", new_files, "before"))
            self.base = git("rev-parse", "refs/heads/before").strip()
            self.head = git("rev-parse", "refs/heads/after").strip()
            self.diff = git("diff", f"{self.base}..{self.head}")
            # `gate_diff` reads git's output through text mode (universal newlines); main's reading of it is of that text
            self.text = self.diff.replace("\r\n", "\n").replace("\r", "\n")
            self.name_status = git("diff", "--name-status", f"{self.base}..{self.head}")
            self.status = own_name_status(self.name_status)
            self.status_paths = [x.split("\t")[-1] for x in git_lines(self.name_status) if len(x.split("\t")) >= 2]
        except Exception:
            _rmtree(self.dir)
            raise

    @contextlib.contextmanager
    def _memo_git(self, m: types.ModuleType):
        """`m._git` (and, for the repaired instrument, its reference's: NOTE_path2_eleventh_pass) memoised per module."""
        mods = [m] + ([m._REF] if hasattr(m, "_REF") else [])
        saved = [(mod, mod._git) for mod in mods]
        for mod, real in saved:
            memo = self.memo.setdefault(id(real), {})

            def git(repo, *args, real=real, memo=memo):
                if args not in memo:
                    memo[args] = real(repo, *args)
                return memo[args]
            mod._git = git
        try:
            yield
        finally:
            for mod, real in saved:
                mod._git = real

    def gate(self, m: types.ModuleType, summary: str, strict: bool = False):
        with self._memo_git(m):
            return m.gate_diff(summary, self.dir, self.base, self.head, run=None, strict=strict)

    def reading(self, m: types.ModuleType, summary: str, rp=None):
        """NOTE_path2_eleventh_pass: the repaired git door's reading BEFORE the guard, with the switches `rp` (every
        repair on by default), on git's own bytes for this repository."""
        with self._memo_git(m):
            name_status = m._git(Path(self.dir), "diff", "--name-status", f"{self.base}..{self.head}")
            text = m._git(Path(self.dir), "diff", f"{self.base}..{self.head}")
            return m._evaluate_git(summary, name_status, text, rp if rp is not None else m._ALL_ON, repo=Path(self.dir),
                                   base=self.base, head=self.head)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        _rmtree(self.dir)


def sample_key(pid) -> int:
    """A record's place in the git door's sample: its id's sha256, so the sample is the same on every run."""
    return int(hashlib.sha256(str(pid).encode("utf-8")).hexdigest()[:12], 16)


# NOTE_path2_tenth_pass (round-9 scorer lens, major): records every run scores through both doors, whatever its input,
# for rules no rebuildable record of a shelf reaches. Z-3 at the git door fires only where main's str.splitlines() cuts
# a `--name-status` path git's split does not, which needs a U+0085 or U+2028 in a path; no shelf PR carries one, so a
# defect that stopped Z-3 abstaining there was admitted in both modes.
def _canary_diff(ch: str) -> str:
    return (f"--- a/docs/a{ch}b.md\n+++ b/docs/a{ch}b.md\n@@ -1 +1 @@\n-x\n+y\n"
            "--- a/src/c.py\n+++ b/src/c.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n")


def _git_file(path: str, st: str, old: str = "a", new: str = "b") -> str:
    """One file of a canary, in git's own rendering (NOTE_path2_twelfth_pass: #121 licenses only there)."""
    head = f"diff --git a/{path} b/{path}\n"
    if st == "A":
        return head + f"new file mode 100644\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+{new}\n"
    if st == "D":
        return head + f"deleted file mode 100644\n--- a/{path}\n+++ /dev/null\n@@ -1 +0,0 @@\n-{old}\n"
    return head + f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-{old}\n+{new}\n"


DOOR_CANARIES = tuple((f"canary:z3-git-door-u{ord(ch):04x}",
                       "2 files changed. 3 files changed. Only touches docs/. Only touches docs/ and src/. Modified src/c.py.",
                       _canary_diff(ch)) for ch in SPLIT_ONLY_BY_PYTHON) + (
    # NOTE_path2_twelfth_pass (round-11 scorer lens, blocker): the guard's outcomes, on both doors, in every run. K-5: a
    # sentence holding a non-ASCII word character reads as main read it (main UNCHECKABLE, the reading #97's VERIFIED).
    ("canary:guard-k5", "Created integrations/git/README.md, voil\u00e0.",
     _git_file("README.md", "M") + _git_file("integrations/git/README.md", "A")),
    # a reading that differs from main's with no licence: #97's suffix match holds only once lower-cased (R11.3), so the
    # guard abstains where main abstained and the reading verified
    ("canary:guard-a-case-only-match-abstains", "Created src/README.md. Deleted src/Config.py.",
     _git_file("docs/README.md", "M") + _git_file("lib/src/readme.md", "A") + _git_file("a/Config.py", "M")
     + _git_file("lib/src/config.py", "D")),
    # dotted keys (round-11 scorer lens, minor): created, deleted and touched claims on a dotted path whose status does not
    # match, and a count and a scope beside a lone dotfile -- a reader defect that fires only on a dotted key passes as
    # #121's licence, and the reading oracle (G-C7) refuses it here
    ("canary:dotted-status-mismatch", "Created .github/ci.yml. Deleted .github/ci.yml. Modified .github/ci.yml. "
     "Created .eslintrc.json. 2 files changed.",
     _git_file(".github/ci.yml", "M") + _git_file(".eslintrc.json", "M")),
    ("canary:a-lone-dotfile", "1 file changed. 2 files changed. Only touches .github/. Only touches github/. "
     "Modified .github/ci.yml.", _git_file(".github/ci.yml", "M")),
)


# NOTE_path2_eleventh_pass (round-10 scorer lens, minor): records every run scores through the raw door, in both modes, for
# rules whose trigger `external1_harness.reconstruct` cannot produce (it always writes `diff --git a/f b/f` and a `---` line
# before `+++`, so main never raises there and the shapes below never occur on a shelf).
RAW_CANARIES = (
    ("canary:raw-main-raises", "t\n\nAdded 1 test. 1 file changed. Refactored the loader.",
     "diff --git a/t.py b/t.py\n x\x0c+++ /dev/null\n+def test_a():\n"),
    ("canary:raw-main-raises-unmeasured", "t\n\nAdded 1 test. 1 file changed.", "x\x0c+++ /dev/null\n"),
    ("canary:raw-k3-gnu-dev-null", "t\n\n1 file changed. 2 files changed. Only touches src/.",
     "+++ /dev/null\t2024-01-01 00:00:00\n@@ -1 +0,0 @@\n-a\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a\n+b\n"),
    ("canary:raw-y4-dev-null-and-a-space", "t\n\nDeleted docs/old.md. 1 file changed.",
     "--- a/docs/old.md\n+++ /dev/null 2024-01-01\n@@ -1 +0,0 @@\n-a\n"),
    ("canary:raw-unreadable-header-beside-dotted-twins", "2 files changed. 3 files changed. 4 files changed.",
     "diff --git .env .env\nnew file mode 100644\n--- /dev/null\n+++ .env\n@@ -0,0 +1 @@\n+A=1\n"
     "diff --git b/x.py b/x.py\n--- b/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-y = 1\n+y = 2\n"
     "diff --git env env\nnew file mode 100644\n--- /dev/null\n+++ env\n@@ -0,0 +1 @@\n+B=1\n"
     "diff --git x.py x.py\n--- x.py\n+++ x.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"),
    ("canary:raw-an-unshaped-pair-after-an-exact-hunk", "3 files changed. 4 files changed.",
     "diff --git a/.env b/.env\nnew file mode 100644\n--- /dev/null\n+++ b/.env\n@@ -0,0 +1 @@\n+A=1\n"
     "diff --git a/db/q.sql b/db/q.sql\n--- a/db/q.sql\n+++ b/db/q.sql\n@@ -1,1 +1,1 @@\n SELECT 1;\n--- users\n+++ x\n"
     "diff --git a/env b/env\nnew file mode 100644\n--- /dev/null\n+++ b/env\n@@ -0,0 +1 @@\n+B=1\n"),
    # NOTE_path2_twelfth_pass (round-11 scorer lens, blocker): an abstention whose #121 precondition holds (git's own
    # rendering, a dotted key) but whose #121 switch does not give main's verdict back -- F-2's split of a header holding
    # U+2028 moves the count -- so a licence granted without asking the switch keeps a verdict here
    ("canary:raw-guard-a-precondition-without-its-switch", "1 file changed. Only touches docs/.",
     "diff --git a/docs/a\u2028b.png b/docs/c.png\nBinary files a/docs/a\u2028b.png and b/docs/c.png differ\n"
     + _git_file(".env", "A")),
    # the same shapes as the door's guard canaries, in a plain rendering (the raw door and the port read only the text)
    ("canary:raw-guard-k5", "Created integrations/git/README.md, voil\u00e0.",
     "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-a\n+b\n--- /dev/null\n+++ b/integrations/git/README.md\n"
     "@@ -0,0 +1 @@\n+hello\n"),
    ("canary:raw-guard-a-case-only-match-abstains", "Created src/README.md.",
     "--- a/docs/README.md\n+++ b/docs/README.md\n@@ -1 +1 @@\n-a\n+b\n--- /dev/null\n+++ b/lib/src/readme.md\n"
     "@@ -0,0 +1 @@\n+x\n"),
    # main's `kind` leak (round-11 scorer lens, minor): a containment demotion rebinds the template's kind for the rest of
    # the sentence; the scorer's replay of main's extraction must read it so, or G-C9 flags a correct instrument
    ("canary:raw-mains-kind-leak", "Delet\u00e9d the parser in lib/x.css and removed the file ci.yml.",
     _git_file(".github/ci.yml", "M") + _git_file("lib/x.css", "M")),
    # the tightened licences themselves (NOTE_path2_twelfth_pass, A.1 and A.2), each where dropping it would keep a verdict:
    # a plain rendering (round 11's R11.0, a directory b/ beside dotted twins) licenses no #121 difference; a rename written
    # under --no-prefix is not git's rendering; #97 licenses nothing in a reading holding a Z-3 doubt (a Submodule line)
    ("canary:raw-121-a-plain-rendering-licenses-nothing", "3 files changed.",
     "--- b/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n--- x.py\n+++ x.py\n@@ -1 +1 @@\n-a\n+b\n"
     "--- .env\n+++ .env\n@@ -1 +1 @@\n-A=1\n+A=2\n--- env\n+++ env\n@@ -1 +1 @@\n-B=1\n+B=2\n"),
    ("canary:raw-121-a-rename-under-no-prefix", "3 files changed.",
     "diff --git a/x.py b/x.py\nsimilarity index 100%\nrename from a/x.py\nrename to b/x.py\n"
     + _git_file(".env", "M", "A=1", "A=2") + _git_file("env", "M", "B=1", "B=2")),
    ("canary:raw-97-beside-a-z3-doubt", "Created integrations/git/README.md.",
     _git_file("README.md", "M") + _git_file("integrations/git/README.md", "A")
     + "Submodule vendor/lib 1234567..89abcde:\n"),
)

# NOTE_path2_twelfth_pass (round-11 scorer lens, blocker): the guard outcomes the canaries must reach in every run, by
# door -- else a guard defect only those outcomes expose (the git door's guard skipped, main's verdict ignored, a licence
# without its switch) is admitted in both modes.
GUARD_OUTCOMES = {"raw": ("guard_k5_read_as_main", "guard_abstained", "guard_abstained_where_a_precondition_held"),
                  "git": ("guard_k5_read_as_main", "guard_abstained")}


def score_canaries() -> tuple:
    """(report, violations): each canary through the raw door and the git door, in tallies of their own so a shelf's
    numbers stay the shelf's. A canary the door cannot rebuild or score is a violation, since the gate it exists for
    would then be vacuous again."""
    t, t_git, counter = Tally(name_prs=True), Tally(name_prs=True), Counter()
    for cid, summary, diff in DOOR_CANARIES:
        t.pair(cid, summary, diff, raw_paths(diff))
        before = counter["scored"]
        git_door_pair(t_git, cid, summary, diff, counter)
        if counter["scored"] != before + 1:
            t_git.violate("G-C8_canary_not_scored", cid)
    # NOTE_path2_eleventh_pass: the raw-door canaries, in the same tally; one the scorer cannot score is a violation
    for cid, summary, diff in RAW_CANARIES:
        try:
            t.pair(cid, summary, diff, raw_paths(diff))
        except Exception:
            t.violate("G-C8_raw_canary_not_scored", cid)
    # NOTE_path2_twelfth_pass: each door's tally must show the guard outcomes the canaries exist for
    for door, tally in (("raw", t), ("git", t_git)):
        for outcome in GUARD_OUTCOMES[door]:
            if not tally.n[outcome]:
                tally.violate(f"G-C8_canaries_reach_no_{outcome}_at_the_{door}_door", "canaries")
    violations = t.violations + t_git.violations
    return ({"records": len(DOOR_CANARIES) + len(RAW_CANARIES), "door": dict(counter),
             "raw_door_canaries": len(RAW_CANARIES), "moved_records": t.n["records_moved"],
             "guard": {k: v for k, v in t.n.items() if k.startswith("guard_")},
             "guard_git_door": {k: v for k, v in t_git.n.items() if k.startswith("guard_")},
             "violations": dict(violations), "pass": not violations}, t.violating + t_git.violating)


def git_door_pair(t, pid, summary: str, diff: str, counter: Counter) -> None:
    """Rebuild one record and score it through the git door (G-C8); counts what could not be rebuilt."""
    counter["tried"] += 1
    try:
        files = rebuild(diff)
    except (UnicodeError, ValueError):
        files = None
    if files is None:
        counter["not_rebuildable"] += 1
        return
    try:
        door = GitDoor(*files)
    except (RuntimeError, OSError, UnicodeError, subprocess.TimeoutExpired):
        counter["rebuild_failed"] += 1
        return
    with door:
        if not door.diff.strip():
            counter["rebuilt_to_no_change"] += 1
            return
        counter["scored"] += 1
        if any(seg.startswith(".") for p in door.status for seg in p.split("/")):
            counter["scored_with_a_dotted_path"] += 1
        t.pair(f"{pid}#git", summary, door.diff, raw_paths(door.diff), door=door)
    # NOTE_path2_ninth_pass (round-8 scorer lens, blocker): a second pass with git's rename detection on, for a record
    # that deletes one file and creates another -- `--name-status` then writes `R<n> old new`, and `gate_diff` keys
    # the entry by its last field. Scored as a door of its own (the counterfactual and G-C7 act through it); the raw
    # door is not compared with it (git's `R` against the rendered text's `M` is main's reading too).
    before, after = files
    if any(t_ is None for t_ in after.values()) and any(t_ is None for t_ in before.values()):
        counter["renames_tried"] += 1
        try:
            door = GitDoor(before, after, renames=True)
        except (RuntimeError, OSError, UnicodeError, subprocess.TimeoutExpired):
            counter["renames_rebuild_failed"] += 1
            return
        with door:
            renamed = any(x.startswith("R") for x in git_lines(door.name_status))
            counter["renames_scored"] += 1
            counter["renames_detected"] += renamed
            t.pair(f"{pid}#git-renames", summary, door.diff, raw_paths(door.diff), door=door)


class Tally:
    def __init__(self, name_prs: bool):
        self.name_prs = name_prs
        self.n = Counter()
        self.transitions = Counter()
        self.reason_only = Counter()
        self.claims_by_verdict = {"baseline": Counter(), "repaired": Counter()}
        self.accusations_by_kind = {"baseline": Counter(), "repaired": Counter()}
        self.violations = Counter()
        self.violating = []
        self.moved_records = []
        self.new_accusations = Counter()
        self.compat2_flips = Counter()
        self.fold_exposed_new_verified = 0
        self.excess_new_verified = 0
        # NOTE_path2_fifth_pass V-3: what the counterfactual attributed, by rule set; and every move,
        # new accusation, new VERIFIED and compat2 flip a post-amendment rule explains.
        self.attribution = {"records_whose_split_differs": 0, "records_whose_parse_differs": 0,
                            "attributed_by": Counter(),
                            "joint_attributions": 0, "moves_admitted_by_rule": Counter(),
                            "new_accusations_admitted": Counter(), "new_verified_admitted": Counter(),
                            "compat2_flips_admitted": Counter(), "f4_withdrawals": 0,
                            "gate_fields_moved_by": Counter(), "table_direction_left_to_an_abstention": Counter()}

    def violate(self, rule: str, pid) -> None:
        self.violations[rule] += 1
        if len(self.violating) < 50:
            self.violating.append((rule, pid))

    def pair(self, pid, summary: str, diff: str, paths, *, fold_repeats: bool = False, door=None) -> None:
        """One record, scored through the raw door, or through the git door when `door` is a `GitDoor`
        (whose `diff` is git's own bytes for the rebuilt repository). Returns the repaired gate."""
        if door is None:
            try:
                gb = BASE.gate_diff_text(summary, diff, run=None, strict=False)
            except Exception:
                # NOTE_path2_ninth_pass: `main` raises on a diff whose Python split cuts a `+++ /dev/null` out of a
                # line with no `---` before it. It gave no verdict there, so the repair may give none either: every
                # claim must abstain (Z-1 to Z-3 read main's reading as raising), and the oracles still read it.
                self.n["baseline_raises"] += 1
                gn = new.gate_diff_text(summary, diff, run=None, strict=False)
                gu = new._evaluate_text(summary, diff, new._ALL_ON)          # NOTE_path2_eleventh_pass: before the guard
                if any(c.verdict != "UNCHECKABLE" for c in gn.claims):
                    self.violate("G-C1_a_verdict_where_the_baseline_raises", pid)
                # NOTE_path2_tenth_pass (round-9 scorer lens, blocker): the gate verdict, the strict verdict and the
                # fields that read only the summary are scored here too. The baseline raised, so its gate on the same
                # summary is read over no diff: the claims it extracts and the sentences it leaves unread do not read
                # the diff, and must be the repair's.
                gns = new.gate_diff_text(summary, diff, run=None, strict=True)
                for v in gate_violations(gn, gn) + strict_violations((gn, gns)):
                    self.violate(v, pid)
                gb0 = BASE.gate_diff_text(summary, "", run=None, strict=False)
                if [core(c) for c in gb0.claims] != [core(c) for c in gn.claims]:
                    self.violate("G-C1_claims_differ", pid)
                if any(getattr(gb0, f) != getattr(gn, f) for f in SUMMARY_FIELDS):
                    self.violate("G-C1_gate_fields_differ", pid)
                for v in report_violations(gb0, gn):
                    self.violate(v, pid)
                for v in oracle_violations(summary, diff, paths, gu):
                    self.violate(v, pid)
                facts: dict = {}
                own = own_read(diff, facts)
                cf = Counterfactual(summary, diff)
                for v in guard_violations(summary, None, gu, gn, lambda r: cf.reading({r}),
                                          lambda r: new._evaluate_text(summary, diff, new._Repairs({r})).claims,
                                          own[0], own[2], self, facts):
                    self.violate(v, pid)
                return gn
            gn = new.gate_diff_text(summary, diff, run=None, strict=False)
            gu = new._evaluate_text(summary, diff, new._ALL_ON)              # NOTE_path2_eleventh_pass: before the guard
        else:
            gb, gn = door.gate(BASE, summary), door.gate(new, summary)
            gu = door.reading(new, summary)
        self.n["gated_under_both"] += 1
        if [core(c) for c in gb.claims] != [core(c) for c in gn.claims]:
            self.violate("G-C1_claims_differ", pid)
            return gn
        # NOTE_path2_seventh_pass: the gate-level fields (G-C1, extended) and the oracles (G-C7). Fields that
        # moved must be given back by reverting F-2 or W-1 (the only rules that change what a diff parses
        # to), on a diff where that rule can act.
        cf = Counterfactual(summary, diff, door)
        for v in gate_violations(gb, gn) + report_violations(gb, gn):
            self.violate(v, pid)
        # NOTE_path2_ninth_pass (round-8 scorer lens, major): each record is scored a second time with strict on, for
        # both instruments -- the verdict the GitHub Action reads under STYXX_STRICT. Strict moves no claim, and each
        # strict verdict is FAIL exactly when a claim is CONTRADICTED or UNCHECKABLE.
        if door is None:
            gbs = BASE.gate_diff_text(summary, diff, run=None, strict=True)
            gns = new.gate_diff_text(summary, diff, run=None, strict=True)
        else:
            gbs, gns = door.gate(BASE, summary, strict=True), door.gate(new, summary, strict=True)
        for v in strict_violations((gb, gbs), (gn, gns)):
            self.violate(v, pid)
        if any(getattr(gb, f) != getattr(gn, f) for f in GATE_FIELDS):
            rules = cf.attribute_fields(gb)
            # NOTE_path2_tenth_pass (K-3): Y-4 too, on a diff whose header names /dev/null followed by a TAB -- a
            # `+++ /dev/null<TAB>...` with no `---` line before it names no file (this reading raised there), so a
            # diff holding only that reads as holding nothing, where main read a created file 'dev/null<TAB>...'
            if rules is None or not all(r in ("F-2", "W-1", "Y-4") and admits(r, "gate", "", "", "", diff) for r in rules):
                self.violate("G-C1_gate_fields_differ", pid)
            else:
                self.attribution["gate_fields_moved_by"]["+".join(rules)] += 1
        for v in oracle_violations(summary, diff, paths, gu, None if door is None else door.status,
                                   None if door is None else door.status_paths,
                                   None if door is None else door.name_status,
                                   None if door is None else door.text):
            self.violate(v, pid)
        # NOTE_path2_eleventh_pass (G-C9): the guard, this file's own, against the repaired gate's final claims
        facts = {}
        own = own_read(diff if door is None else door.text, facts)
        own_status = own[0] if door is None else door.status
        if door is not None:                   # NOTE_path2_twelfth_pass: the git door's facts (A.1, A.2)
            facts = own_git_licence(door.text, door.status_paths)
        switched_new = ((lambda r: new._evaluate_text(summary, diff, new._Repairs({r})).claims) if door is None
                        else (lambda r: door.reading(new, summary, new._Repairs({r})).claims))
        for v in guard_violations(summary, gb.claims, gu, gn, lambda r: cf.reading({r}), switched_new, own_status,
                                  own[2], self, facts):
            self.violate(v, pid)
        if door is not None and not door.renames:
            # G-C8: the repaired git door reads exactly what the repaired raw door reads on git's bytes -- except
            # where the raw door, whose file list is its own parse, is not sure of it (NOTE_path2_eighth_pass,
            # Y-1: a lone CR git prints inside a line splits it, and a line after it may read as a header) and
            # abstains on a file-list claim that git's `--name-status` answers. G-C7 re-reads that claim from
            # git's list; the gate verdicts are compared only where no claim was so excused.
            # NOTE_path2_eleventh_pass: the two doors' readings BEFORE the guard -- each door's guard reads its own main
            # reading (main's raw door and main's git door), which may differ; G-C9 holds each guard on its own door
            raw = new._evaluate_text(summary, diff, new._ALL_ON)

            def sig(c) -> tuple:
                return (core(c), c.verdict, c.why, json.dumps(c.detail, sort_keys=True))
            excused = [i for i, (a, b) in enumerate(zip(raw.claims, gu.claims))
                       if a.kind in FILE_LIST_KINDS and a.verdict == "UNCHECKABLE" and a.why.startswith(NOT_SURE)
                       and core(a) == core(b)]
            if (len(raw.claims) != len(gu.claims)
                    or any(sig(a) != sig(b) for i, (a, b) in enumerate(zip(raw.claims, gu.claims)) if i not in excused)
                    or (not excused and raw.verdict != gu.verdict)):
                self.violate("G-C8_git_door_differs_from_the_raw_door", pid)
        moved_pr = key_moved(paths)
        moved_old, moved_new = moved_keys(paths)
        # NOTE_path2_third_pass R-5: every path and every claimed path or prefix, before any rule
        # below is allowed to attribute a move to #121.
        tokens = list(paths)
        for c in gb.claims:
            tokens += [v for k, v in (c.detail or {}).items() if k in ("path", "prefix", "prefix2") and v]
        if key_shape_violations(tokens):
            self.violate("G-C4_key_moved_not_by_a_dot", pid)
        coll = collision(paths)
        self.n["key_moved_prs"] += moved_pr
        self.n["collision_prs"] += coll
        base_status = BASE.parse_unified_diff(diff)[0]
        status = new.parse_unified_diff(diff)[0]
        sides = new.parse_unified_diff_sides(diff)
        self.attribution["records_whose_split_differs"] += split_differs(diff)
        self.attribution["records_whose_parse_differs"] += parse_differs(diff)
        record_moved = False
        for i, (cb, cn) in enumerate(zip(gb.claims, gn.claims)):
            self._claim(pid, i, cb, cn, paths, fold_repeats, coll, moved_pr, moved_old, moved_new,
                        base_status, status, sides, cf, diff, summary)
            if (cb.verdict, cb.why) != (cn.verdict, cn.why) or (cb.kind == "compat_claim" and cb.detail != cn.detail):
                record_moved = True
        if record_moved:
            self.n["records_moved"] += 1
            if self.name_prs:
                self.moved_records.append(pid)
        return gn

    def _table(self, k, cb, cn, vb, vn, paths, coll, moved_pr, moved_old, moved_new,
               base_status, status, sides, fold_repeats) -> list:
        """The amended table, as written: what it would call a violation for this moved claim."""
        pending: list = []
        prefixes = [cb.detail.get("prefix", "")] + ([cb.detail["prefix2"]] if cb.detail.get("prefix2") else [])
        ot_exception = (k == "only_touches" and vb == "VERIFIED" and vn == "CONTRADICTED"
                        and only_touches_new_accusation_allowed(cb.detail, prefixes, paths))
        # NOTE_path2_fourth_pass F-4: withdrawing an accusation because an off-tree prefix could hold the
        # path is the safe direction; admitted when the reason says so AND a prefix key is off-tree by
        # this file's own test.
        f4_withdrawal = (k == "only_touches" and vb == "CONTRADICTED" and vn == "UNCHECKABLE"
                         and cn.why.endswith(OFF_TREE_WHY)
                         and any(off_tree_key(new._norm(x).rstrip("/.")) for x in prefixes))
        if vn == "CONTRADICTED" and vb != "CONTRADICTED":
            if not ((k == "files_changed_count" and coll) or ot_exception):
                pending.append(f"G-C3_new_accusation:{k}")
        if k in PATH_KINDS:
            claimed = cb.detail.get("path", "")
            by121 = path_claim_by121(claimed, base_status, status, moved_old, moved_new)
            by97 = resolutions_differ(status, claimed)
            if not (by97 or by121):
                pending.append(f"G-C4_unattributed:{k}")
            elif "CONTRADICTED" in (vb, vn):
                pending.append(f"G-C4_direction:{k}")
            elif k == "file_touched" and vb != vn and not by121:
                pending.append(f"G-C4_direction:{k}")
        elif k == "files_changed_count":
            if not coll:
                pending.append(f"G-C4_unattributed:{k}")
        elif k == "only_touches":
            if not (moved_pr or any(BASE._norm(x) != new._norm(x) for x in prefixes)):
                pending.append(f"G-C4_unattributed:{k}")
            elif vb != vn and not ((vb == "VERIFIED" and vn == "UNCHECKABLE" and cn.why.endswith("(#121)"))
                                   or ot_exception or f4_withdrawal):
                pending.append(f"G-C4_direction:{k}")
        elif k == "tests_added":
            if not test_def_changed(sides, status):
                pending.append(f"G-C4_unattributed:{k}")
            elif vb != vn and (vb, vn) not in {("VERIFIED", "UNCHECKABLE"), ("CONTRADICTED", "VERIFIED"),
                                               ("CONTRADICTED", "UNCHECKABLE"), ("UNCHECKABLE", "VERIFIED")}:
                pending.append(f"G-C4_direction:{k}")
        elif k == "symbol_added":
            name = cb.detail.get("name", "")
            if not (symbol_def_changed(sides, status, name)
                    or symbol_def_changed(sides, status, claimed_identifier(cb.text, name))):
                pending.append(f"G-C4_unattributed:{k}")
            elif not (vb == "VERIFIED" and vn == "UNCHECKABLE" and cn.why.endswith("(#101)")):
                pending.append(f"G-C4_direction:{k}")
        elif k == "compat_claim":
            if not (any(p in moved_old for p in compat_detail_paths(cb.detail))
                    or any(p in moved_new for p in compat_detail_paths(cn.detail))):
                pending.append(f"G-C4_unattributed:{k}")
            elif vb != "UNCHECKABLE" or vn != "UNCHECKABLE":
                pending.append(f"G-C4_direction:{k}")
        else:                                   # tests_pass, and any kind the table does not name
            pending.append(f"G-C4_unattributed:{k}")
        return pending, f4_withdrawal

    def _claim(self, pid, i, cb, cn, paths, fold_repeats, coll, moved_pr, moved_old, moved_new,
               base_status, status, sides, cf, diff: str = "", summary: str = "") -> None:
        k = cb.kind
        self.claims_by_verdict["baseline"][cb.verdict] += 1
        self.claims_by_verdict["repaired"][cn.verdict] += 1
        if cb.verdict == "CONTRADICTED":
            self.accusations_by_kind["baseline"][k] += 1
        if cn.verdict == "CONTRADICTED":
            self.accusations_by_kind["repaired"][k] += 1
        flip = None
        if k == "compat_claim":
            a, b = cb.detail.get("compat2_candidate"), cn.detail.get("compat2_candidate")
            if a != b:
                flip = f"{a}->{b}"
                self.compat2_flips[flip] += 1
        moved = (cb.verdict, cb.why) != (cn.verdict, cn.why) or (k == "compat_claim" and cb.detail != cn.detail)
        if not moved:
            return
        vb, vn = cb.verdict, cn.verdict
        if vb == vn:
            self.reason_only[k] += 1
        else:
            self.transitions[f"{k}: {vb} -> {vn}"] += 1
        if vn == "CONTRADICTED" and vb != "CONTRADICTED":
            self.new_accusations[k] += 1
        if k == "tests_added" and vn == "VERIFIED" and vb != "VERIFIED":
            if fold_repeats:
                self.fold_exposed_new_verified += 1
            if test_def_excess(sides, status):
                self.excess_new_verified += 1
        pending, f4_withdrawal = self._table(k, cb, cn, vb, vn, paths, coll, moved_pr, moved_old, moved_new,
                                             base_status, status, sides, fold_repeats)
        # NOTE_path2_fifth_pass V-3: the counterfactual decides WHICH rule made the move; every rule in the
        # attribution must then admit it -- the table rules through the table, the others by `admits`.
        rules, how = cf.attribute(i, cb)
        if rules is None:
            self.violate(f"G-C4_unattributed_counterfactual:{k}" + (":diffuse" if how == "all rules" else ""), pid)
            if flip is not None:
                self.violate("G-C6_compat2_candidate_flipped", pid)
            return
        refused = []
        # NOTE_path2_eighth_pass: when the repaired claim is the abstention of an eighth-pass rule (or A-1) that is in
        # the attribution and admits it, a table rule beside it only shaped what the claim would have read had that
        # rule not abstained (#121's keys listed in a reason, say); the table's DIRECTION test is not asked of it --
        # the abstention is the move -- while its other conditions are. G-C7 re-reads the claim either way.
        owner = abstention_owner(cn.why, k) if vn == "UNCHECKABLE" else None
        # NOTE_path2_ninth_pass: Z-3's doubt that main's reading also held counts only beside a licensed difference,
        # and #121's dotted keys are one: reverting #121 removes Z-3's precondition, so #121 alone gives main's claim
        # back although the move is Z-3's abstention. There the table's conditions do not apply at all -- #121 moved
        # no count and no scope -- and G-C7 re-derives the claim (Z-3 with this file's own code) either way.
        woken = owner == "Z-3" and "#121" in rules
        # NOTE_path2_twelfth_pass: the guard's abstention (no licence holds on a difference from main's verdict) is the move
        # itself, as Z-3's is beside #121: the table's conditions shaped only the verdict the guard withheld, so they are
        # not asked of it (G-C9 re-derives every such claim, reason included, with this file's own guard); every other rule
        # in the attribution must still admit the move on this diff.
        guarded = owner == "guard"
        abstained = guarded or ((owner in rules or woken) and admits(owner, k, vb, vn, cn.why, diff, summary))
        waived = ("G-C4_direction:", "G-C4_unattributed:") if (woken or guarded) else ("G-C4_direction:",)
        for r in rules:
            if r in TABLE_RULES:
                refused += [x for x in pending if not (abstained and x.startswith(waived))]
            elif not admits(r, k, vb, vn, cn.why, diff, summary):
                refused.append(f"G-C4_direction:{k}:{r}")
        if abstained and any(r in TABLE_RULES for r in rules):
            self.attribution["table_direction_left_to_an_abstention"][f"{owner} {k}: {vb} -> {vn}"] += 1
        # G-C6: a compat2_candidate flip is admitted only when F-2 alone or W-1 alone explains it, on a diff
        # where that rule can act at all (`admits` above has already asked; NOTE_path2_sixth_pass).
        if flip is not None and tuple(rules) not in (("F-2",), ("W-1",)):
            refused.append("G-C6_compat2_candidate_flipped")
        if refused:
            for v in dict.fromkeys(refused):
                self.violate(v, pid)
            return
        label = "+".join(rules)
        self.attribution["attributed_by"][label] += 1
        self.attribution["joint_attributions"] += how == "joint"
        if f4_withdrawal and any(r in TABLE_RULES for r in rules):
            self.attribution["f4_withdrawals"] += 1
        post = [r for r in rules if r not in TABLE_RULES]
        if post:
            tag = "+".join(post)
            self.attribution["moves_admitted_by_rule"][f"{tag} {k}: {vb} -> {vn}"] += 1
            if vn == "CONTRADICTED" and vb != "CONTRADICTED":
                # G-C3 is not asked for a post-amendment rule (NOTE_path2_sixth_pass): each new accusation
                # such a rule explains is admitted by `admits` and counted here, not refused.
                self.attribution["new_accusations_admitted"][f"{tag} {k}"] += 1
            if vn == "VERIFIED" and vb != "VERIFIED":
                self.attribution["new_verified_admitted"][f"{tag} {k}"] += 1
            if flip is not None:
                self.attribution["compat2_flips_admitted"][f"{tag} {flip}"] += 1

    def report(self, prov: dict) -> dict:
        if not prov.get("reference_is_the_baseline", False):
            self.violate("G-C0_reference_is_not_the_baseline", "(provenance)")
        if not prov["unmodified_against_head"]:
            self.violate("G-C0_modified_tree", "(provenance)")
        blocking = {r: v for r, v in self.violations.items()}
        att = self.attribution
        return {
            "provenance": prov,
            "counts": dict(self.n),
            "transitions": dict(sorted(self.transitions.items())),
            "reason_only_moves_by_kind": dict(sorted(self.reason_only.items())),
            "new_accusations_by_kind": dict(sorted(self.new_accusations.items())),
            "claims_by_verdict": {k: dict(v) for k, v in self.claims_by_verdict.items()},
            "accusations_by_kind": {k: dict(sorted(v.items())) for k, v in self.accusations_by_kind.items()},
            "compat2_candidate_flips": dict(self.compat2_flips),
            "attribution": {"method": "counterfactual, per claim (NOTE_path2_fifth_pass V-3)",
                            "rules": list(RULES),
                            "records_whose_split_differs": att["records_whose_split_differs"],
                            "records_whose_parse_differs": att["records_whose_parse_differs"],
                            "attributed_by": dict(sorted(att["attributed_by"].items())),
                            "joint_attributions": att["joint_attributions"],
                            "moves_admitted_by_rule": dict(sorted(att["moves_admitted_by_rule"].items())),
                            "new_accusations_admitted": dict(sorted(att["new_accusations_admitted"].items())),
                            "new_verified_admitted": dict(sorted(att["new_verified_admitted"].items())),
                            "compat2_flips_admitted": dict(sorted(att["compat2_flips_admitted"].items())),
                            "f4_withdrawals": att["f4_withdrawals"],
                            "gate_fields_moved_by": dict(sorted(att["gate_fields_moved_by"].items())),
                            "table_direction_left_to_an_abstention":
                                dict(sorted(att["table_direction_left_to_an_abstention"].items()))},
            "new_verified_tests_added_on_prs_whose_rows_repeat_a_filename": self.fold_exposed_new_verified,
            "new_verified_tests_added_where_a_file_adds_a_changed_name_more_often_than_it_removes_it":
                self.excess_new_verified,
            "violations": blocking,
            "G-C0_provenance": {"pass": not any(r.startswith("G-C0") for r in blocking)},
            "G-C1_same_claims": {"pass": not any(r.startswith("G-C1") for r in blocking)},
            "G-C3_no_accusation_added": {"pass": not any(r.startswith("G-C3") for r in blocking),
                                         "waived_for_post_amendment_rules":
                                             dict(sorted(att["new_accusations_admitted"].items()))},
            "G-C4_every_move_attributed": {"pass": not any(r.startswith("G-C4") for r in blocking)},
            "G-C6_compat2_candidate_does_not_flip": {"pass": not any(r.startswith("G-C6") for r in blocking)},
            "G-C9_the_guard_reads_as_this_file_reads_it": {
                "pass": not any(r.startswith("G-C9") for r in blocking),
                "outcomes": {k: v for k, v in sorted(self.n.items()) if k.startswith("guard_")}},
            "G-C7_every_rule_reads_as_this_file_reads_it": {"pass": not any(r.startswith("G-C7") for r in blocking),
                                                            "name_table_sha256": XID_TABLE_SHA256,
                                                            "name_table_checked_against_this_python":
                                                                XID_CHECKED_AGAINST_DATABASE},
        }

    def door_report(self, sample: Counter) -> dict:
        """The git door's own section (G-C8): what was sampled and rebuilt, what moved, what failed.
        NOTE_path2_eighth_pass (round-7 protocol lens): the counts are part of the verdict -- every record the
        door tried is accounted for (scored, not rebuildable, or rebuilt to no change), none failed to rebuild,
        and at least one was scored; otherwise the door fails."""
        att = self.attribution
        tried = sample["tried"]
        accounted = sample["scored"] + sample["not_rebuildable"] + sample["rebuilt_to_no_change"] + sample["rebuild_failed"]
        count_violations = []
        if accounted != tried:
            count_violations.append("G-C8_records_unaccounted")
        if sample["rebuild_failed"] or sample["renames_rebuild_failed"]:
            count_violations.append("G-C8_rebuild_failed")
        if tried and not sample["scored"]:
            count_violations.append("G-C8_nothing_scored")
        for v in count_violations:
            self.violate(v, "(git door)")
        return {"sample": dict(sample), "counts": dict(self.n),
                "transitions": dict(sorted(self.transitions.items())),
                "attributed_by": dict(sorted(att["attributed_by"].items())),
                "new_accusations_admitted": dict(sorted(att["new_accusations_admitted"].items())),
                "new_verified_admitted": dict(sorted(att["new_verified_admitted"].items())),
                "violations": dict(self.violations),
                "pass": not self.violations}


def run_differential(out: Path, git_sample: int | None = None) -> int:
    # NOTE_path2_third_pass R-5: the inputs are hashed into the payload and a missing one FAILS.
    # Before this, a run with no generated corpora printed two lines on stderr, scored 67 pinned
    # pairs, wrote "all_attribution_gates_pass": true and exited 0 -- a receipt with no corpus.
    items = []
    inputs: dict = {}
    missing: list = []
    for name in DIFFERENTIAL_FILES:
        p = DIFFERENTIAL / name
        if not p.exists():
            missing.append(name)
            print(f"(missing {name}: run build_corpus.py / fuzz_corpus.py for the full corpus)", file=sys.stderr)
            continue
        raw = p.read_bytes()
        rows = json.loads(raw.decode("utf-8"))
        items += [(name, it) for it in rows]
        inputs[name] = {"sha256": _sha(raw), "bytes": len(raw), "items": len(rows)}
    prov = provenance()
    t = Tally(name_prs=True)
    for name in missing:
        t.violate("G-C0_missing_corpus_input", name)
    for name, it in items:
        t.pair(it["id"], it["summary"], it["diff"], raw_paths(it["diff"]))
    # NOTE_path2_seventh_pass (G-C8): the git door. NOTE_path2_eighth_pass: on EVERY record that can be rebuilt,
    # in the order of their ids' sha256 (`--git-sample N` stops after N scored, for a smoke run only; the
    # payload says so).
    t_git = Tally(name_prs=True)
    sample = Counter()
    for name, it in sorted(items, key=lambda x: sample_key(x[1]["id"])):
        if git_sample is not None and sample["scored"] >= git_sample:
            break
        git_door_pair(t_git, it["id"], it["summary"], it["diff"], sample)
    sample["records"] = len(items)
    sample["every_record_tried"] = int(sample["tried"] == len(items))
    if not sample["every_record_tried"]:
        t_git.violate("G-C8_not_every_record_tried", f"{sample['tried']} of {len(items)}")
    canaries, canary_violating = score_canaries()
    rep = t.report(prov)
    pre = [i for n, i in items if n != "path2_pairs.json"]
    payload = {"prereg": PREREG, "amendment": AMENDMENT, "note": NOTE, "mode": "differential",
               "baseline_sha256": BASE_SHA256, "repaired_sha256": NEW_SHA256,
               "inputs": inputs, "missing_inputs": missing,
               "pairs": len(items), "pairs_before_path2": len(pre),
               "moved_record_ids": t.moved_records, **rep,
               "G-C8_git_door": t_git.door_report(sample), "G-C8_canaries": canaries}
    payload["all_attribution_gates_pass"] = not t.violations and not t_git.violations and canaries["pass"]
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "moved_record_ids"}, indent=1))
    print(f"-> {out}")
    for rule, pid in t.violating + t_git.violating + canary_violating:
        print(f"VIOLATION {rule} {pid}", file=sys.stderr)
    return 0 if payload["all_attribution_gates_pass"] else 1


def eligible(mod: types.ModuleType, diff: str, net: dict) -> bool:
    """EXTERNAL-1's eligibility for one instrument: its parse of the reconstruction equals the implied
    status map, keyed by that instrument's own `_norm`."""
    code = {"added": "A", "removed": "D"}
    implied = {mod._norm(fn): code.get(st, "M") for fn, st in net.items()}
    return mod.parse_unified_diff(diff)[0] == implied


def run_corpus(shelf: Path, limit: int | None, out: Path, git_sample: int = 3000, git_every: int = 25) -> int:
    if not shelf.exists():
        sys.exit(f"path2_gates: no shelf at {shelf}")
    prov = provenance()
    con = sqlite3.connect(f"{shelf.resolve().as_uri()}?immutable=1", uri=True)
    # NOTE_path2_fourth_pass P-1: which shelf, by size and row counts, not by file name alone.
    shelf_input = {"name": shelf.name, "bytes": shelf.stat().st_size,
                   "pr_rows": con.execute("SELECT COUNT(*) FROM pr").fetchone()[0],
                   "f_rows": con.execute("SELECT COUNT(*) FROM f").fetchone()[0]}
    t = Tally(name_prs=False)
    t_git = Tally(name_prs=False)
    sample = Counter()
    excl = {"baseline": Counter(), "repaired": Counter()}
    elig_moves = Counter()
    seen = 0
    t0 = time.time()
    q = "SELECT id, title, body FROM pr" + (f" LIMIT {int(limit)}" if limit else "")
    for pid, title, body in con.execute(q):
        seen += 1
        if seen % 5000 == 0:
            print(f"  seen {seen}  gated under both {t.n['gated_under_both']}  {time.time() - t0:.0f}s", flush=True)
        if not body or not body.strip():
            excl["baseline"]["empty_body"] += 1
            excl["repaired"]["empty_body"] += 1
            continue
        files = con.execute("SELECT filename, status, patch FROM f WHERE pr_id=?", (pid,)).fetchall()
        if not files:
            excl["baseline"]["no_file_records"] += 1
            excl["repaired"]["no_file_records"] += 1
            continue
        diff, _implied = reconstruct(files)
        net = _fold_statuses(files)
        ok = {"baseline": eligible(BASE, diff, net), "repaired": eligible(new, diff, net)}
        for tag in ok:
            if not ok[tag]:
                excl[tag]["reconstruction_mismatch"] += 1
        names = [fn for fn in net]
        # NOTE_path2_tenth_pass (round-9 scorer lens, major): a PR the repaired instrument cannot score is excluded
        # from both, so no claim oracle reads the rules that parsed it. Before an eligibility move is credited (and on
        # every PR the repair excludes), its parse is held to this file's own: F-2's split, W-1's status, added lines
        # and sides, the notes, #121's key.
        parse_bad = parse_violations(diff, names) if (ok["baseline"] != ok["repaired"] or not ok["repaired"]) else []
        for v in parse_bad:
            t.violate(v, pid)
        if ok["baseline"] != ok["repaired"]:
            elig_moves["baseline_only" if ok["baseline"] else "repaired_only"] += 1
            # NOTE_path2_fifth_pass V-3: an eligibility move is attributed as a claim is -- by the copy with
            # one rule reverted giving the baseline's eligibility back. #121 needs a moved key besides
            # (the table's own test); F-2 and W-1 are admitted and counted where they can act at all
            # (NOTE_path2_sixth_pass: F-2 on a diff whose splits differ, W-1 on one whose parses differ).
            back = [r for r in RULES if _elig_reverted(r, diff, net) == ok["baseline"]]
            if "#121" in back and key_moved(names):
                credit = "attributed_to_#121"
            elif "F-2" in back and split_differs(diff):
                credit = "attributed_to_F-2"
            elif "W-1" in back and parse_differs(diff):
                credit = "attributed_to_W-1"
            elif "Y-4" in back and gnu_null_in(diff):
                credit = "attributed_to_Y-4"
            else:
                credit = None
                t.violate("G-C2_eligibility_moved_without_a_rule", pid)
            if credit is not None:
                # NOTE_path2_tenth_pass: a rule is credited only where its own reading of the PR is this file's
                elig_moves["refused_by_a_parse_oracle" if parse_bad else credit] += 1
        if not (ok["baseline"] and ok["repaired"]):
            continue
        rows = Counter(fn for fn, _s, _p in files if fn)
        gn = t.pair(pid, f"{title or ''}\n\n{body}", diff, names, fold_repeats=any(v > 1 for v in rows.values()))
        # NOTE_path2_seventh_pass (G-C8). NOTE_path2_ninth_pass (round-8 scorer lens, major; the eighth pass's rationale
        # was wrong): every PR with a claim `gate_diff` reads through code of its own -- a tests_added or symbol_added
        # claim (its own added blob and sides) AND a file-list claim (its own `--name-status` parse, `_status_notes`,
        # the Z-3 comparison with main's list and the notes dict) -- and every `git_every`-th other PR by its id's
        # sha256 (default 1 in 25), up to `git_sample` scored (default 3,000).
        definitional = gn is not None and any(c.kind in ("tests_added", "symbol_added") for c in gn.claims)
        file_list = gn is not None and any(c.kind in FILE_LIST_KINDS for c in gn.claims)
        # NOTE_path2_tenth_pass (round-9 scorer lens, minor): and a compat claim, which reads gate_diff's own sides
        compat = gn is not None and any(c.kind == "compat_claim" for c in gn.claims)
        if sample["scored"] < git_sample and (definitional or file_list or compat
                                              or sample_key(pid) % max(git_every, 1) == 0):
            sample["tried_for_a_definition_claim"] += definitional
            sample["tried_for_a_file_list_claim"] += file_list
            sample["tried_for_a_compat_claim"] += compat
            git_door_pair(t_git, pid, f"{title or ''}\n\n{body}", diff, sample)
    con.close()
    canaries, canary_violating = score_canaries()
    rep = t.report(prov)
    payload = {"prereg": PREREG, "amendment": AMENDMENT, "note": NOTE, "mode": "corpus",
               "shelf": shelf.name, "shelf_input": shelf_input, "limit": limit,
               "baseline_sha256": BASE_SHA256, "repaired_sha256": NEW_SHA256,
               "claimdetect": "blocked for both instruments (unparsed_claims only; never a verdict)",
               "prs_seen": seen, "excluded": {k: dict(v) for k, v in excl.items()},
               "G-C2_eligibility": {"moves": dict(elig_moves),
                                    "pass": not any(r.startswith("G-C2") for r in t.violations)},
               **rep, "G-C8_git_door": t_git.door_report(sample), "G-C8_canaries": canaries,
               "seconds": round(time.time() - t0)}
    payload["all_blocking_gates_pass"] = not t.violations and not t_git.violations and canaries["pass"]
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))
    print(f"-> {out}")
    for rule, pid in t.violating + t_git.violating + canary_violating:
        print(f"VIOLATION {rule} pr_id={pid}", file=sys.stderr)
    return 0 if payload["all_blocking_gates_pass"] else 1


def _elig_reverted(rule: str, diff: str, net: dict) -> bool:
    with reverted(CF, {rule}):
        return eligible(CF, diff, net)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("differential")
    d.add_argument("--out", type=Path, default=HERE / "path2_differential_gates.json")
    d.add_argument("--git-sample", type=int, default=None,
                   help="stop the git door (G-C8) after N records scored; default: every rebuildable record")
    c = sub.add_parser("corpus")
    c.add_argument("--shelf", type=Path, default=HERE / "external1_shelf.sqlite")
    c.add_argument("--limit", type=int, default=None, help="score only the leading N PRs (a smoke run)")
    c.add_argument("--out", type=Path, default=HERE / "path2_corpus_gates.json")
    c.add_argument("--git-sample", type=int, default=3000, help="PRs scored through the git door at most (G-C8)")
    c.add_argument("--git-every", type=int, default=25,
                   help="besides every PR with a tests_added, symbol_added, file-list or compat claim, a PR is sampled "
                        "when sha256(id) %% N == 0")
    a = ap.parse_args()
    return (run_differential(a.out, a.git_sample) if a.mode == "differential"
            else run_corpus(a.shelf, a.limit, a.out, a.git_sample, a.git_every))


if __name__ == "__main__":
    sys.exit(main())
