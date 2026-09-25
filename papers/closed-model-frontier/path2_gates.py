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
git HEAD, and whether those two files and `styxx/diffgate.py` are unmodified against HEAD. A payload
written from a modified tree fails G-C0, so a number cannot be cited without the bytes that made it.

`styxx.claimdetect` is blocked for both instruments: it feeds `unparsed_claims` only, `_gate`
swallows its absence by design, and nothing here reads that field.

Reconstruction and eligibility are EXTERNAL-1's (`external1_harness._fold_statuses` / `reconstruct`,
imported unchanged): an empty body, no file records, or a reconstruction whose parse differs from the
implied status map excludes the PR -- evaluated per instrument, with the implied map keyed by that
instrument's own `_norm`.

The attribution rules are the amended table, written out below rather than imported: the tiered and
any-tier resolutions (`tiered`, `any_tier`), the one-to-one definition pairing (`_pairs`,
`_test_counts`, `DEF_TEST`, `symbol_def_changed`) and the dot-miss / dotted-prefix exception
(`only_touches_new_accusation_allowed`) are this file's own code.

WHAT IS NOT INDEPENDENT (NOTE_path2_third_pass_2026_09_25, R-5; the earlier wording overstated this).
The scorer calls the repaired module for its inputs, and a defect in any of these would be attributed
to #121 by the scorer that is supposed to catch it:

    new._norm                    every key comparison: key_moved, moved_keys, collision,
                                 path_claim_by121, resolutions_differ, the only_touches prefix test
                                 and the dotted-prefix / dotdot exception
    new.parse_unified_diff       the repaired status map (the baseline's comes from BASE)
    new.parse_unified_diff_sides the added/removed sides the definition rules read
    new._header_paths            the `diff --git` half of raw_paths
    new.gate_diff_text           the repaired verdicts themselves, necessarily

`_norm` is the one of these the attribution leans on hardest, so G-C4 now carries a blocking
key-shape check: a key may move ONLY by dropping leading dots, i.e. the repaired key with `./`
stripped from its front is the baseline key. A `_norm` that stopped lower-casing, or that moved a key
any other way, fails `G-C4_key_moved_not_by_a_dot` instead of being attributed to #121. The unit
tests remain the guard for what the check cannot see.

#121 is attributed per claim wherever the claim names something: a path claim by its own key or the
entry it resolves to, a compatibility claim by the paths in its detail; `only_touches` reads every
path of the PR, so the PR's moved keys are the claim's.

Inputs (G-C0): in differential mode every corpus file is hashed into the payload and a missing one is
a FAILURE, not a note on stderr -- the payload is meant to be the RESULT's receipt, and a receipt that
cannot say which corpus it read is not one. Corpus mode records the shelf's file name, its byte size
and the row counts of its `pr` and `f` tables (NOTE_path2_fourth_pass_2026_09_25, P-1); the shelf is
not hashed, because it is large and opened immutable.

THE FOURTH PASS (NOTE_path2_fourth_pass_2026_09_25). Four rule changes landed after the amendment and
each is attributed here by this file's own test of whether it can have acted on a record:

    F-1  the prefix-shape test undots the prefix    no new attribution: against the baseline it
                                                    restores main's reading, and C-3's accusations
                                                    are the dotted-prefix exception already here
    F-2  a diff splits on \\r\\n, \\r, \\n only       `f2_applies`: str.splitlines() and the git split
                                                    of the diff differ. Any kind, any direction; a new
                                                    accusation it explains is admitted and COUNTED
    F-3  `got` reads a [ \\t]* indent                `f3_applies`: an added line the old `got` pattern
                                                    counts and the new one does not. tests_added only;
                                                    a new accusation it explains is admitted and COUNTED
    F-4  an off-tree prefix beside an on-tree one   only_touches CONTRADICTED -> UNCHECKABLE whose reason
         no longer withdraws a sure accusation      is the off-tree abstention, on a claim with an
                                                    off-tree prefix key: the safe direction, admitted

Every admitted F-2 / F-3 accusation is listed under `fourth_pass` in the payload, so the operator sees
how many there were; none is silent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DIFFERENTIAL = ROOT / "web" / "gate" / "differential"
PREREG = "PREREG_path2_resolution_2026_09_17.md"
AMENDMENT = "AMENDMENT_path2_resolution_2026_09_17.md"
NOTE = ["NOTE_path2_third_pass_2026_09_25.md", "NOTE_path2_fourth_pass_2026_09_25.md"]
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
PROVENANCE_FILES = ("papers/closed-model-frontier/path2_gates.py", "papers/closed-model-frontier/external1_harness.py",
                    "styxx/diffgate.py")
# AMENDMENT C-1, written out: one pattern per kind, no \s, \w or \b, one optional leading U+FEFF.
DEF_TEST = re.compile(r"^\uFEFF?[ \t]*def (test_[^ \t(:]*)")
# NOTE_path2_third_pass R-2: the REMOVED side alone accepts `async`, as the instrument does.
DEF_TEST_REMOVED = re.compile(r"^\uFEFF?[ \t]*(?:async[ \t]+)?def (test_[^ \t(:]*)")
# NOTE_path2_third_pass R-3: a dotfile prefix is one dot then a name character; `..` is not.
_DOTFILE_PREFIX = re.compile(r"^\.[^./\\]")
# NOTE_path2_fourth_pass F-2 / F-3, written out here rather than borrowed from the repair.
GIT_LINE_BREAK = re.compile(r"\r\n|\r|\n")
GOT_BEFORE_F3 = re.compile(r"^\uFEFF?\s*def test_")
GOT_AFTER_F3 = re.compile(r"^\uFEFF?[ \t]*def test_")
OFF_TREE_WHY = "is relative to a directory the diff does not name (#121)"

sys.path.insert(0, str(ROOT))  # the checkout FIRST: the repaired instrument is this tree's
import styxx.diffgate as new  # noqa: E402
if not Path(new.__file__).resolve().is_relative_to(ROOT):
    sys.exit("path2_gates: styxx.diffgate resolved to an installed package, not this checkout")
sys.modules["styxx.claimdetect"] = None  # type: ignore[assignment]  -- observer blocked, see docstring
sys.path.insert(0, str(HERE))
from external1_harness import _fold_statuses, reconstruct  # noqa: E402


def _sha(b: bytes) -> str:
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def _git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, timeout=60)
    if r.returncode != 0:
        sys.exit(f"path2_gates: git {' '.join(args)} failed: {r.stderr.decode('utf-8', 'replace')[:200]}")
    return r.stdout.decode("utf-8", "replace")


def load_base() -> types.ModuleType:
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:styxx/diffgate.py"],
                       capture_output=True, timeout=60)
    if r.returncode != 0:
        sys.exit(f"path2_gates: git show {BASE_COMMIT[:8]}:styxx/diffgate.py failed: "
                 f"{r.stderr.decode('utf-8', 'replace')[:200]}")
    if _sha(r.stdout) != BASE_SHA256:
        sys.exit(f"path2_gates: the baseline hashes to {_sha(r.stdout)[:16]}, not {BASE_SHA256[:16]}")
    mod = types.ModuleType("styxx_diffgate_base")
    mod.__file__ = f"<git show {BASE_COMMIT[:8]}:styxx/diffgate.py>"
    # The baseline file carries `from .declare import declaration_pass` (DECLARE-1, on main before
    # this branch). `styxx.declare` is byte-identical on main and on this branch -- the branch
    # touches one file under styxx/ -- so resolving the relative import against this checkout's
    # package gives the baseline the same reader main has.
    mod.__package__ = "styxx"
    sys.modules[mod.__name__] = mod
    exec(compile(r.stdout.decode("utf-8"), mod.__file__, "exec"), mod.__dict__)  # noqa: S102
    return mod


BASE = load_base()
NEW_SHA256 = _sha(Path(new.__file__).read_bytes())


def provenance() -> dict:
    """G-C0: the bytes that produced a payload, and whether the tree matched HEAD when they ran."""
    dirty = [line for line in _git("status", "--porcelain", "--", *PROVENANCE_FILES).splitlines() if line.strip()]
    return {"scorer_sha256": _sha(Path(__file__).read_bytes()),
            "harness_sha256": _sha((HERE / "external1_harness.py").read_bytes()),
            "repaired_sha256": NEW_SHA256,
            "baseline_commit": BASE_COMMIT,
            "prereg_baseline_commit": PREREG_BASE_COMMIT,
            "baseline_moved_from_prereg": BASE_COMMIT != PREREG_BASE_COMMIT,
            "git_head": _git("rev-parse", "HEAD").strip(),
            "unmodified_against_head": not dirty,
            "modified": dirty}


# ── attribution, written out independently of the repair ─────────────────────────────────────

def git_lines(diff: str) -> list:
    """The diff's lines as git writes them: split on \\r\\n, \\r and \\n, no trailing empty line."""
    lines = GIT_LINE_BREAK.split(diff)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def f2_applies(diff: str) -> bool:
    """F-2 can have moved this record only if str.splitlines() and the git split read it differently."""
    return diff.splitlines() != git_lines(diff)


def f3_applies(diff: str) -> bool:
    """F-3 can have moved a tests_added claim only if some added line is counted by the old `got`
    pattern and not by the new one."""
    added = [ln[1:] for ln in git_lines(diff) if ln.startswith("+") and not ln.startswith("+++")]
    return any(GOT_BEFORE_F3.match(a) and not GOT_AFTER_F3.match(a) for a in added)


def raw_paths(diff: str) -> list:
    out = []
    for line in git_lines(diff):                # NOTE_path2_fourth_pass F-2: the git split
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


def _test_counts(lines) -> Counter:
    return Counter(m.group(1) for m in map(DEF_TEST.match, lines) if m)


def _test_counts_removed(lines) -> Counter:
    return Counter(m.group(1) for m in map(DEF_TEST_REMOVED.match, lines) if m)


def test_def_changed(sides: dict, status: dict) -> bool:
    """#101 for tests_added: a non-`A` file where one test name is defined by an added and a removed line."""
    return any(set(a) & set(r) for a, r in _pairs(sides, status, _test_counts, _test_counts_removed))


def test_def_excess(sides: dict, status: dict) -> bool:
    """G-C5: a non-`A` file with a changed test name defined in more added lines than removed lines."""
    return any(a[n] > r[n] for a, r in _pairs(sides, status, _test_counts, _test_counts_removed)
               for n in set(a) & set(r))


def symbol_def_changed(sides: dict, status: dict, name: str) -> bool:
    rx = re.compile(r"^\uFEFF?[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+" + re.escape(name) + r"(?=[ \t(:]|$)")
    return any(a and r for a, r in _pairs(sides, status, lambda lines: sum(1 for x in lines if rx.match(x))))


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
        # NOTE_path2_fourth_pass: moves and new accusations admitted under F-2 / F-3 / F-4, by kind
        self.fourth_pass = {"f2_records": 0, "f3_records": 0, "moves_by_rule": Counter(),
                            "new_accusations_admitted": Counter(), "f4_withdrawals": 0}

    def violate(self, rule: str, pid) -> None:
        self.violations[rule] += 1
        if len(self.violating) < 50:
            self.violating.append((rule, pid))

    def pair(self, pid, summary: str, diff: str, paths, *, fold_repeats: bool = False) -> None:
        gb = BASE.gate_diff_text(summary, diff, run=None, strict=False)
        gn = new.gate_diff_text(summary, diff, run=None, strict=False)
        self.n["gated_under_both"] += 1
        if [core(c) for c in gb.claims] != [core(c) for c in gn.claims]:
            self.violate("G-C1_claims_differ", pid)
            return
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
        record_moved = False
        # NOTE_path2_fourth_pass: F-2 and F-3 explain a move only where the amended table below does
        # not, and only on a record where they can have acted; what they explain is counted, not hidden.
        f2, f3 = f2_applies(diff), f3_applies(diff)
        self.fourth_pass["f2_records"] += f2
        self.fourth_pass["f3_records"] += f3
        for cb, cn in zip(gb.claims, gn.claims):
            self._claim(pid, cb, cn, paths, fold_repeats, coll, moved_pr, moved_old, moved_new,
                        base_status, status, sides, f2, f3)
            if (cb.verdict, cb.why) != (cn.verdict, cn.why) or (cb.kind == "compat_claim" and cb.detail != cn.detail):
                record_moved = True
        if record_moved:
            self.n["records_moved"] += 1
            if self.name_prs:
                self.moved_records.append(pid)

    def _claim(self, pid, cb, cn, paths, fold_repeats, coll, moved_pr, moved_old, moved_new,
               base_status, status, sides, f2, f3) -> None:
        k = cb.kind
        self.claims_by_verdict["baseline"][cb.verdict] += 1
        self.claims_by_verdict["repaired"][cn.verdict] += 1
        if cb.verdict == "CONTRADICTED":
            self.accusations_by_kind["baseline"][k] += 1
        if cn.verdict == "CONTRADICTED":
            self.accusations_by_kind["repaired"][k] += 1
        if k == "compat_claim":
            a, b = cb.detail.get("compat2_candidate"), cn.detail.get("compat2_candidate")
            if a != b:
                self.compat2_flips[f"{a}->{b}"] += 1
                self.violate("G-C6_compat2_candidate_flipped", pid)
        moved = (cb.verdict, cb.why) != (cn.verdict, cn.why) or (k == "compat_claim" and cb.detail != cn.detail)
        if not moved:
            return
        vb, vn = cb.verdict, cn.verdict
        if vb == vn:
            self.reason_only[k] += 1
        else:
            self.transitions[f"{k}: {vb} -> {vn}"] += 1
        pending: list = []              # what the amended table would call a violation for this claim
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
            self.new_accusations[k] += 1
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
            if f4_withdrawal:
                self.fourth_pass["f4_withdrawals"] += 1
        elif k == "tests_added":
            if not test_def_changed(sides, status):
                pending.append(f"G-C4_unattributed:{k}")
            elif vb != vn and (vb, vn) not in {("VERIFIED", "UNCHECKABLE"), ("CONTRADICTED", "VERIFIED"),
                                               ("CONTRADICTED", "UNCHECKABLE"), ("UNCHECKABLE", "VERIFIED")}:
                pending.append(f"G-C4_direction:{k}")
            if vn == "VERIFIED" and vb != "VERIFIED":
                if fold_repeats:
                    self.fold_exposed_new_verified += 1
                if test_def_excess(sides, status):
                    self.excess_new_verified += 1
        elif k == "symbol_added":
            if not symbol_def_changed(sides, status, cb.detail.get("name", "")):
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
        if not pending:
            return
        # NOTE_path2_fourth_pass: what the amended table cannot explain, F-2 (any kind) or F-3 (tests_added
        # only) may, on a record where it can have acted. Admitted, and counted by rule, kind and move.
        rule = "F-2" if f2 else ("F-3" if f3 and k == "tests_added" else None)
        if rule is None:
            for r in pending:
                self.violate(r, pid)
            return
        self.fourth_pass["moves_by_rule"][f"{rule} {k}: {vb} -> {vn}"] += 1
        if vn == "CONTRADICTED" and vb != "CONTRADICTED":
            self.fourth_pass["new_accusations_admitted"][f"{rule} {k}"] += 1

    def report(self, prov: dict) -> dict:
        if not prov["unmodified_against_head"]:
            self.violate("G-C0_modified_tree", "(provenance)")
        blocking = {r: v for r, v in self.violations.items()}
        return {
            "provenance": prov,
            "counts": dict(self.n),
            "transitions": dict(sorted(self.transitions.items())),
            "reason_only_moves_by_kind": dict(sorted(self.reason_only.items())),
            "new_accusations_by_kind": dict(sorted(self.new_accusations.items())),
            "claims_by_verdict": {k: dict(v) for k, v in self.claims_by_verdict.items()},
            "accusations_by_kind": {k: dict(sorted(v.items())) for k, v in self.accusations_by_kind.items()},
            "compat2_candidate_flips": dict(self.compat2_flips),
            "fourth_pass": {"f2_records": self.fourth_pass["f2_records"],
                            "f3_records": self.fourth_pass["f3_records"],
                            "moves_admitted_by_rule": dict(sorted(self.fourth_pass["moves_by_rule"].items())),
                            "new_accusations_admitted": dict(sorted(self.fourth_pass["new_accusations_admitted"].items())),
                            "f4_withdrawals": self.fourth_pass["f4_withdrawals"]},
            "new_verified_tests_added_on_prs_whose_rows_repeat_a_filename": self.fold_exposed_new_verified,
            "new_verified_tests_added_where_a_file_adds_a_changed_name_more_often_than_it_removes_it":
                self.excess_new_verified,
            "violations": blocking,
            "G-C0_provenance": {"pass": not any(r.startswith("G-C0") for r in blocking)},
            "G-C1_same_claims": {"pass": not any(r.startswith("G-C1") for r in blocking)},
            "G-C3_no_accusation_added": {"pass": not any(r.startswith("G-C3") for r in blocking)},
            "G-C4_every_move_attributed": {"pass": not any(r.startswith("G-C4") for r in blocking)},
            "G-C6_compat2_candidate_does_not_flip": {"pass": not any(r.startswith("G-C6") for r in blocking)},
        }


def run_differential(out: Path) -> int:
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
    rep = t.report(prov)
    pre = [i for n, i in items if n != "path2_pairs.json"]
    payload = {"prereg": PREREG, "amendment": AMENDMENT, "note": NOTE, "mode": "differential",
               "baseline_sha256": BASE_SHA256, "repaired_sha256": NEW_SHA256,
               "inputs": inputs, "missing_inputs": missing,
               "pairs": len(items), "pairs_before_path2": len(pre),
               "moved_record_ids": t.moved_records, **rep}
    payload["all_attribution_gates_pass"] = not t.violations
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "moved_record_ids"}, indent=1))
    print(f"-> {out}")
    for rule, pid in t.violating:
        print(f"VIOLATION {rule} {pid}", file=sys.stderr)
    return 0 if not t.violations else 1


def run_corpus(shelf: Path, limit: int | None, out: Path) -> int:
    if not shelf.exists():
        sys.exit(f"path2_gates: no shelf at {shelf}")
    prov = provenance()
    con = sqlite3.connect(f"{shelf.resolve().as_uri()}?immutable=1", uri=True)
    # NOTE_path2_fourth_pass P-1: which shelf, by size and row counts, not by file name alone.
    shelf_input = {"name": shelf.name, "bytes": shelf.stat().st_size,
                   "pr_rows": con.execute("SELECT COUNT(*) FROM pr").fetchone()[0],
                   "f_rows": con.execute("SELECT COUNT(*) FROM f").fetchone()[0]}
    t = Tally(name_prs=False)
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
        code = {"added": "A", "removed": "D"}
        ok = {}
        for tag, mod in (("baseline", BASE), ("repaired", new)):
            implied = {}
            for fn, st in net.items():
                implied[mod._norm(fn)] = code.get(st, "M")
            ok[tag] = mod.parse_unified_diff(diff)[0] == implied
            if not ok[tag]:
                excl[tag]["reconstruction_mismatch"] += 1
        names = [fn for fn in net]
        if ok["baseline"] != ok["repaired"]:
            elig_moves["baseline_only" if ok["baseline"] else "repaired_only"] += 1
            if not key_moved(names):
                # NOTE_path2_fourth_pass F-2: a parse that splits the diff as git does can make a
                # reconstruction match (or stop matching) with no key moving; admitted and counted.
                if f2_applies(diff):
                    elig_moves["attributed_to_F-2"] += 1
                else:
                    t.violate("G-C2_eligibility_moved_without_a_key", pid)
        if not (ok["baseline"] and ok["repaired"]):
            continue
        rows = Counter(fn for fn, _s, _p in files if fn)
        t.pair(pid, f"{title or ''}\n\n{body}", diff, names, fold_repeats=any(v > 1 for v in rows.values()))
    con.close()
    rep = t.report(prov)
    payload = {"prereg": PREREG, "amendment": AMENDMENT, "note": NOTE, "mode": "corpus",
               "shelf": shelf.name, "shelf_input": shelf_input, "limit": limit,
               "baseline_sha256": BASE_SHA256, "repaired_sha256": NEW_SHA256,
               "claimdetect": "blocked for both instruments (unparsed_claims only; never a verdict)",
               "prs_seen": seen, "excluded": {k: dict(v) for k, v in excl.items()},
               "G-C2_eligibility": {"moves": dict(elig_moves),
                                    "pass": not any(r.startswith("G-C2") for r in t.violations)},
               **rep, "seconds": round(time.time() - t0)}
    payload["all_blocking_gates_pass"] = not t.violations
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))
    print(f"-> {out}")
    for rule, pid in t.violating:
        print(f"VIOLATION {rule} pr_id={pid}", file=sys.stderr)
    return 0 if not t.violations else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("differential")
    d.add_argument("--out", type=Path, default=HERE / "path2_differential_gates.json")
    c = sub.add_parser("corpus")
    c.add_argument("--shelf", type=Path, default=HERE / "external1_shelf.sqlite")
    c.add_argument("--limit", type=int, default=None, help="score only the leading N PRs (a smoke run)")
    c.add_argument("--out", type=Path, default=HERE / "path2_corpus_gates.json")
    a = ap.parse_args()
    return run_differential(a.out) if a.mode == "differential" else run_corpus(a.shelf, a.limit, a.out)


if __name__ == "__main__":
    sys.exit(main())
