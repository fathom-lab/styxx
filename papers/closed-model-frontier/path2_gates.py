"""PATH-2 gates (PREREG_path2_resolution_2026_09_17): the instrument before the repair against the
instrument after it, claim by claim, with every moved record attributed to #97, #121 or #101.

    python path2_gates.py differential [--out FILE]
        G-P4's attribution half, over web/gate/differential's corpus files (build_corpus.py and
        fuzz_corpus.py run beforehand; the pinned pair files are committed). Record ids are named.

    python path2_gates.py corpus --shelf DIR/external1_shelf.sqlite [--limit N] [--out FILE]
        G-C1 to G-C5 over the EXTERNAL-1 shelf. Counts only: no ledger is written and no PR is named
        in the output file (the ids of any violating PR go to stderr for the operator, nowhere else).

Baseline: `styxx/diffgate.py` at BASE_COMMIT, read from this checkout's object store with `git show`
and REFUSED unless it hashes to BASE_SHA256 (LF). Repaired: this checkout's `styxx/diffgate.py`, whose
sha256 is recorded. Both run in one process on the same bytes. The shelf is opened `immutable=1`
(read only, no lock, no WAL index), so scoring a live checkout's shelf cannot write to it.

`styxx.claimdetect` is blocked for both instruments: it feeds `unparsed_claims` only, `_gate`
swallows its absence by design, and nothing here reads that field.

Reconstruction and eligibility are EXTERNAL-1's (`external1_harness._fold_statuses` / `reconstruct`,
imported unchanged): an empty body, no file records, or a reconstruction whose parse differs from the
implied status map excludes the PR -- evaluated per instrument, with the implied map keyed by that
instrument's own `_norm`.

The attribution rules are the prereg's table, implemented here independently of the repair (the
tiered and any-tier resolutions, the per-file definition scans and the key comparisons are written
out below, not imported from the repaired module).
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
BASE_COMMIT = "87dded26377a2d0cee1d872a7af9584646db5199"
BASE_SHA256 = "473a7dd7c2dce7b1fefd07eaba27291090dd351a0c108b28f4812c7dc77f536d"
DIFFERENTIAL_FILES = ("corpus_real.json", "corpus_fuzz.json", "bc1_pairs.json", "compat_pairs.json",
                      "bin1_pairs.json", "path2_pairs.json")
PATH_KINDS = ("file_created", "file_deleted", "file_touched")
COMPAT_EXTRAS = ("removed", "languages", "surface_removed", "signature_changed", "compat2_candidate")
DEF_TEST = re.compile(r"^\s*def (test_[A-Za-z0-9_]*)")

sys.path.insert(0, str(ROOT))  # the checkout FIRST: the repaired instrument is this tree's
import styxx.diffgate as new  # noqa: E402
if not Path(new.__file__).resolve().is_relative_to(ROOT):
    sys.exit("path2_gates: styxx.diffgate resolved to an installed package, not this checkout")
sys.modules["styxx.claimdetect"] = None  # type: ignore[assignment]  -- observer blocked, see docstring
sys.path.insert(0, str(HERE))
from external1_harness import _fold_statuses, reconstruct  # noqa: E402


def _sha(b: bytes) -> str:
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


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


def collision(paths) -> bool:
    by_old: dict = {}
    for p in paths:
        by_old.setdefault(BASE._norm(p), set()).add(new._norm(p))
    return any(len(v) > 1 for v in by_old.values())


def resolutions_differ(status: dict, claimed: str) -> bool:
    c = new._norm(claimed)
    base_name = c.rstrip("/").rsplit("/", 1)[-1]

    def name(p):
        return p.rstrip("/").rsplit("/", 1)[-1]

    any_tier = next((p for p in status if p == c or p.endswith("/" + c) or name(p) == base_name), None)
    tiered = (next((p for p in status if p == c), None)
              or next((p for p in status if p.endswith("/" + c)), None)
              or next((p for p in status if name(p) == base_name), None))
    return any_tier != tiered


def test_def_changed(sides: dict) -> bool:
    for added, removed in sides.values():
        gone = {m.group(1) for m in map(DEF_TEST.match, removed) if m}
        if gone and any((m := DEF_TEST.match(a)) is not None and m.group(1) in gone for a in added):
            return True
    return False


def symbol_def_changed(sides: dict, name: str) -> bool:
    rx = re.compile(r"^\s*(?:def|class)\s+" + re.escape(name) + r"\b")
    return any(any(rx.match(a) for a in added) and any(rx.match(r) for r in removed)
               for added, removed in sides.values())


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
        coll = collision(paths)
        self.n["key_moved_prs"] += moved_pr
        self.n["collision_prs"] += coll
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
            moved = (cb.verdict, cb.why) != (cn.verdict, cn.why) or (k == "compat_claim" and cb.detail != cn.detail)
            if not moved:
                continue
            record_moved = True
            vb, vn = cb.verdict, cn.verdict
            if vb == vn:
                self.reason_only[k] += 1
            else:
                self.transitions[f"{k}: {vb} -> {vn}"] += 1
            if vn == "CONTRADICTED" and vb != "CONTRADICTED":
                self.new_accusations[k] += 1
                if k != "files_changed_count" or not coll:
                    self.violate(f"G-C3_new_accusation:{k}", pid)
            if k in PATH_KINDS:
                claimed = cb.detail.get("path", "")
                by121 = moved_pr or BASE._norm(claimed) != new._norm(claimed)
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
                prefixes = [cb.detail.get("prefix", "")] + ([cb.detail["prefix2"]] if cb.detail.get("prefix2") else [])
                if not (moved_pr or any(BASE._norm(x) != new._norm(x) for x in prefixes)):
                    self.violate(f"G-C4_unattributed:{k}", pid)
                elif vb != vn and not (vb == "VERIFIED" and vn == "UNCHECKABLE" and cn.why.endswith("(#121)")):
                    self.violate(f"G-C4_direction:{k}", pid)
            elif k == "tests_added":
                if not test_def_changed(sides):
                    self.violate(f"G-C4_unattributed:{k}", pid)
                elif vb != vn and (vb, vn) not in {("VERIFIED", "UNCHECKABLE"), ("CONTRADICTED", "VERIFIED"),
                                                   ("CONTRADICTED", "UNCHECKABLE"), ("UNCHECKABLE", "VERIFIED")}:
                    self.violate(f"G-C4_direction:{k}", pid)
                if vn == "VERIFIED" and vb != "VERIFIED" and fold_repeats:
                    self.fold_exposed_new_verified += 1
            elif k == "symbol_added":
                if not symbol_def_changed(sides, cb.detail.get("name", "")):
                    self.violate(f"G-C4_unattributed:{k}", pid)
                elif not (vb == "VERIFIED" and vn == "UNCHECKABLE" and cn.why.endswith("(#101)")):
                    self.violate(f"G-C4_direction:{k}", pid)
            elif k == "compat_claim":
                if not moved_pr:
                    self.violate(f"G-C4_unattributed:{k}", pid)
                elif vb != "UNCHECKABLE" or vn != "UNCHECKABLE":
                    self.violate(f"G-C4_direction:{k}", pid)
            else:                                   # tests_pass, and any kind the table does not name
                self.violate(f"G-C4_unattributed:{k}", pid)
        if record_moved:
            self.n["records_moved"] += 1
            if self.name_prs:
                self.moved_records.append(pid)

    def report(self) -> dict:
        blocking = {r: v for r, v in self.violations.items()}
        return {
            "counts": dict(self.n),
            "transitions": dict(sorted(self.transitions.items())),
            "reason_only_moves_by_kind": dict(sorted(self.reason_only.items())),
            "new_accusations_by_kind": dict(sorted(self.new_accusations.items())),
            "claims_by_verdict": {k: dict(v) for k, v in self.claims_by_verdict.items()},
            "accusations_by_kind": {k: dict(sorted(v.items())) for k, v in self.accusations_by_kind.items()},
            "compat2_candidate_flips": dict(self.compat2_flips),
            "new_verified_tests_added_on_prs_whose_rows_repeat_a_filename": self.fold_exposed_new_verified,
            "violations": blocking,
            "G-C1_same_claims": {"pass": not any(r.startswith("G-C1") for r in blocking)},
            "G-C3_no_accusation_added": {"pass": not any(r.startswith("G-C3") for r in blocking)},
            "G-C4_every_move_attributed": {"pass": not any(r.startswith("G-C4") for r in blocking)},
        }


def run_differential(out: Path) -> int:
    items = []
    for name in DIFFERENTIAL_FILES:
        p = DIFFERENTIAL / name
        if p.exists():
            items += [(name, it) for it in json.loads(p.read_text(encoding="utf-8"))]
        else:
            print(f"(missing {name}: run build_corpus.py / fuzz_corpus.py for the full corpus)", file=sys.stderr)
    t = Tally(name_prs=True)
    for name, it in items:
        t.pair(it["id"], it["summary"], it["diff"], raw_paths(it["diff"]))
    rep = t.report()
    pre = [i for n, i in items if n != "path2_pairs.json"]
    payload = {"prereg": PREREG, "mode": "differential", "baseline_sha256": BASE_SHA256,
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
    rep = t.report()
    payload = {"prereg": PREREG, "mode": "corpus", "shelf": shelf.name, "limit": limit,
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
    c.add_argument("--limit", type=int, default=None, help="score only the first N PRs (a smoke run)")
    c.add_argument("--out", type=Path, default=HERE / "path2_corpus_gates.json")
    a = ap.parse_args()
    return run_differential(a.out) if a.mode == "differential" else run_corpus(a.shelf, a.limit, a.out)


if __name__ == "__main__":
    sys.exit(main())
