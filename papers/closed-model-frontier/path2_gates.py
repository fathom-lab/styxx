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

The attribution rules are the amended table, implemented here independently of the repair (the tiered
and any-tier resolutions, the one-to-one definition pairing, the key comparisons and the dot-miss
exception are written out below, not imported from the repaired module). #121 is attributed per claim
wherever the claim names something: a path claim by its own key or the entry it resolves to, a
compatibility claim by the paths in its detail; `only_touches` reads every path of the PR, so the PR's
moved keys are the claim's.
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
BASE_COMMIT = "87dded26377a2d0cee1d872a7af9584646db5199"
BASE_SHA256 = "473a7dd7c2dce7b1fefd07eaba27291090dd351a0c108b28f4812c7dc77f536d"
DIFFERENTIAL_FILES = ("corpus_real.json", "corpus_fuzz.json", "bc1_pairs.json", "compat_pairs.json",
                      "bin1_pairs.json", "path2_pairs.json")
PATH_KINDS = ("file_created", "file_deleted", "file_touched")
COMPAT_EXTRAS = ("removed", "languages", "surface_removed", "signature_changed", "compat2_candidate")
PROVENANCE_FILES = ("papers/closed-model-frontier/path2_gates.py", "papers/closed-model-frontier/external1_harness.py",
                    "styxx/diffgate.py")
# AMENDMENT C-1, written out: one pattern per kind, no \s, \w or \b, one optional leading U+FEFF.
DEF_TEST = re.compile(r"^\uFEFF?[ \t]*def (test_[^ \t(:]*)")

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
            "git_head": _git("rev-parse", "HEAD").strip(),
            "unmodified_against_head": not dirty,
            "modified": dirty}


# ── attribution, written out independently of the repair ─────────────────────────────────────

def raw_paths(diff: str) -> list:
    out = []
    for line in diff.splitlines():
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


def _pairs(sides: dict, status: dict, rx_for) -> list:
    """Per file whose repaired status is not `A`: (added count, removed count) under `rx_for`."""
    out = []
    for path, (added, removed) in sides.items():
        if status.get(path) == "A":
            continue
        out.append((rx_for(added), rx_for(removed)))
    return out


def _test_counts(lines) -> Counter:
    return Counter(m.group(1) for m in map(DEF_TEST.match, lines) if m)


def test_def_changed(sides: dict, status: dict) -> bool:
    """#101 for tests_added: a non-`A` file where one test name is defined by an added and a removed line."""
    return any(set(a) & set(r) for a, r in _pairs(sides, status, _test_counts))


def test_def_excess(sides: dict, status: dict) -> bool:
    """G-C5: a non-`A` file with a changed test name defined in more added lines than removed lines."""
    return any(a[n] > r[n] for a, r in _pairs(sides, status, _test_counts) for n in set(a) & set(r))


def symbol_def_changed(sides: dict, status: dict, name: str) -> bool:
    rx = re.compile(r"^\uFEFF?[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+" + re.escape(name) + r"(?=[ \t(:]|$)")
    return any(a and r for a, r in _pairs(sides, status, lambda lines: sum(1 for x in lines if rx.match(x))))


def only_touches_new_accusation_allowed(detail: dict, prefixes: list, paths) -> bool:
    """AMENDMENT G-C3 / C-3: a prefix key carrying a leading dot, or a changed path whose repaired key
    begins with `..` -- the two shapes whose old VERIFIED came from the old key dropping the dot."""
    dotted_prefix = any(new._norm(x).rstrip("/.").startswith(".") and BASE._norm(x) != new._norm(x) for x in prefixes)
    dotdot = any(new._norm(p).startswith("..") and BASE._norm(p) != new._norm(p) for p in paths)
    return dotted_prefix or dotdot


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
        coll = collision(paths)
        self.n["key_moved_prs"] += moved_pr
        self.n["collision_prs"] += coll
        base_status = BASE.parse_unified_diff(diff)[0]
        status = new.parse_unified_diff(diff)[0]
        sides = new.parse_unified_diff_sides(diff)
        record_moved = False
        for cb, cn in zip(gb.claims, gn.claims):
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
                continue
            record_moved = True
            vb, vn = cb.verdict, cn.verdict
            if vb == vn:
                self.reason_only[k] += 1
            else:
                self.transitions[f"{k}: {vb} -> {vn}"] += 1
            prefixes = [cb.detail.get("prefix", "")] + ([cb.detail["prefix2"]] if cb.detail.get("prefix2") else [])
            ot_exception = (k == "only_touches" and vb == "VERIFIED" and vn == "CONTRADICTED"
                            and only_touches_new_accusation_allowed(cb.detail, prefixes, paths))
            if vn == "CONTRADICTED" and vb != "CONTRADICTED":
                self.new_accusations[k] += 1
                if not ((k == "files_changed_count" and coll) or ot_exception):
                    self.violate(f"G-C3_new_accusation:{k}", pid)
            if k in PATH_KINDS:
                claimed = cb.detail.get("path", "")
                by121 = path_claim_by121(claimed, base_status, status, moved_old, moved_new)
                by97 = resolutions_differ(status, claimed)
                if not (by97 or by121):
                    self.violate(f"G-C4_unattributed:{k}", pid)
                elif "CONTRADICTED" in (vb, vn):
                    self.violate(f"G-C4_direction:{k}", pid)
                elif k == "file_touched" and vb != vn and not by121:
                    self.violate(f"G-C4_direction:{k}", pid)
            elif k == "files_changed_count":
                if not coll:
                    self.violate(f"G-C4_unattributed:{k}", pid)
            elif k == "only_touches":
                if not (moved_pr or any(BASE._norm(x) != new._norm(x) for x in prefixes)):
                    self.violate(f"G-C4_unattributed:{k}", pid)
                elif vb != vn and not ((vb == "VERIFIED" and vn == "UNCHECKABLE" and cn.why.endswith("(#121)"))
                                       or ot_exception):
                    self.violate(f"G-C4_direction:{k}", pid)
            elif k == "tests_added":
                if not test_def_changed(sides, status):
                    self.violate(f"G-C4_unattributed:{k}", pid)
                elif vb != vn and (vb, vn) not in {("VERIFIED", "UNCHECKABLE"), ("CONTRADICTED", "VERIFIED"),
                                                   ("CONTRADICTED", "UNCHECKABLE"), ("UNCHECKABLE", "VERIFIED")}:
                    self.violate(f"G-C4_direction:{k}", pid)
                if vn == "VERIFIED" and vb != "VERIFIED":
                    if fold_repeats:
                        self.fold_exposed_new_verified += 1
                    if test_def_excess(sides, status):
                        self.excess_new_verified += 1
            elif k == "symbol_added":
                if not symbol_def_changed(sides, status, cb.detail.get("name", "")):
                    self.violate(f"G-C4_unattributed:{k}", pid)
                elif not (vb == "VERIFIED" and vn == "UNCHECKABLE" and cn.why.endswith("(#101)")):
                    self.violate(f"G-C4_direction:{k}", pid)
            elif k == "compat_claim":
                if not (any(p in moved_old for p in compat_detail_paths(cb.detail))
                        or any(p in moved_new for p in compat_detail_paths(cn.detail))):
                    self.violate(f"G-C4_unattributed:{k}", pid)
                elif vb != "UNCHECKABLE" or vn != "UNCHECKABLE":
                    self.violate(f"G-C4_direction:{k}", pid)
            else:                                   # tests_pass, and any kind the table does not name
                self.violate(f"G-C4_unattributed:{k}", pid)
        if record_moved:
            self.n["records_moved"] += 1
            if self.name_prs:
                self.moved_records.append(pid)

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
    items = []
    for name in DIFFERENTIAL_FILES:
        p = DIFFERENTIAL / name
        if p.exists():
            items += [(name, it) for it in json.loads(p.read_text(encoding="utf-8"))]
        else:
            print(f"(missing {name}: run build_corpus.py / fuzz_corpus.py for the full corpus)", file=sys.stderr)
    prov = provenance()
    t = Tally(name_prs=True)
    for name, it in items:
        t.pair(it["id"], it["summary"], it["diff"], raw_paths(it["diff"]))
    rep = t.report(prov)
    pre = [i for n, i in items if n != "path2_pairs.json"]
    payload = {"prereg": PREREG, "amendment": AMENDMENT, "mode": "differential", "baseline_sha256": BASE_SHA256,
               "repaired_sha256": NEW_SHA256, "pairs": len(items), "pairs_before_path2": len(pre),
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
                t.violate("G-C2_eligibility_moved_without_a_key", pid)
        if not (ok["baseline"] and ok["repaired"]):
            continue
        rows = Counter(fn for fn, _s, _p in files if fn)
        t.pair(pid, f"{title or ''}\n\n{body}", diff, names, fold_repeats=any(v > 1 for v in rows.values()))
    con.close()
    rep = t.report(prov)
    payload = {"prereg": PREREG, "amendment": AMENDMENT, "mode": "corpus", "shelf": shelf.name, "limit": limit,
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
