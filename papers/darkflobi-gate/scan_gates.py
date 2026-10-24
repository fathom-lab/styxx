# -*- coding: utf-8 -*-
"""Scan the darkflobi gate PREREG's ```gates block with the lab's own styxx.protocol and print what it reads.

    py -3.12 papers/darkflobi-gate/scan_gates.py [PREREG.md] [--styxx DIR] [--scratch DIR]

Prints the block's sha256 and byte count as `styxx.protocol._select_gates_block` reads them, builds
`Experiment(require_power_basis=True, require_nonvacuous_gates=True)` (on the file in place when it is committed;
otherwise on a copy frozen in a throwaway git repository under --scratch, said so in the output), lists the gates and
the outcome rows, enumerates every boolean assignment of the gates and asserts that the table is total and every
row reachable, scores one synthetic result per realisable assignment and asserts the verdict walks the intended
row, runs `check_metrics` over a sample result, refuses a document with a second gates fence, and prints the Wilson
arithmetic behind G_C and G_N (`styxx.mind.wilson`, z = 1.96).

Standard library plus the styxx package named by --styxx (default: the repository this file sits in). Nothing is
written outside --scratch. What this does not check: that the result dict a scorer writes at close has these
shapes (score.py's own tests do), or anything about the window's data.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PREREG = HERE / "PREREG_darkflobi_gate_2026_10_23.md"

# metric values that fire each gate true / false (section 5's result dict)
TRUE_FALSE = {
    "G_X_exits": ("result.unledgered_deliveries", 0, 3),
    "G_F1_panel": ("result.families_clearing_decoys", 1, 0),
    "G_F2_families": ("result.families_clearing_decoys", 2, 1),
    "G_C_canaries": ("result.canary_refused_wilson_lower", 0.9601, 0.9012),
    "G_N_power": ("result.n_cell1", 57, 12),
    "G_RN_refusals": ("result.n_refusals", 24, 6),
    "G_R_refusals_upheld": ("result.refusal_upheld_share", 0.9583, 0.7917),
    "G_B_bound": ("result.bound_share_cell1", 0.7368, 0.4211),
    "G_S2_padding": ("result.padding_share", 0.1250, 0.4000),
    "G_P_pairing": ("result.wrong_leaf_share", 0.0476, 0.1905),
}


def result_for(fired: dict) -> dict:
    out = {"result": {}}
    for name, (path, tv, fv) in TRUE_FALSE.items():
        key = path.split(".", 1)[1]
        if key == "families_clearing_decoys":
            continue
        out["result"][key] = tv if fired[name] else fv
    f1, f2 = fired["G_F1_panel"], fired["G_F2_families"]
    out["result"]["families_clearing_decoys"] = 2 if f2 else (1 if f1 else 0)
    return out


def walk(rows, fired):
    for i, row in enumerate(rows):
        if all(fired.get(k) == v for k, v in row["when"].items()):
            return i
    return None


def is_committed(doc: Path) -> bool:
    try:
        r = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--follow", "--", doc.name],
                           cwd=str(doc.parent), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and bool(r.stdout.strip())


def freeze_copy(doc: Path, scratch: Path) -> Path:
    repo = scratch / ("repo_%d" % os.getpid())
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir(parents=True)
    dst = repo / doc.name
    dst.write_bytes(doc.read_bytes())
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-qm", "freeze"]):
        subprocess.run(cmd, cwd=str(repo), check=True, capture_output=True, env=env)
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prereg", nargs="?", default=str(DEFAULT_PREREG))
    ap.add_argument("--styxx", default=str(HERE.parent.parent), help="folder holding the styxx package")
    ap.add_argument("--scratch", default=None, help="where a throwaway repository may be made (uncommitted PREREG)")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.styxx)
    from styxx.protocol import Experiment, GateSpecError, _select_gates_block  # noqa: E402
    from styxx.mind import wilson  # noqa: E402

    doc = Path(a.prereg).resolve()
    text = doc.read_text(encoding="utf-8")
    block = _select_gates_block(text)
    sha = hashlib.sha256(block.encode("utf-8")).hexdigest()
    print("prereg:", doc.name)
    print("styxx from:", Path(a.styxx).resolve().as_posix())
    print("gates block sha256:", sha)
    print("gates block bytes :", len(block.encode("utf-8")))
    scratch_made = None
    if is_committed(doc):
        target = doc
        print("freeze: the file is committed in its repository; scanned in place")
    else:
        scratch = Path(a.scratch) if a.scratch else Path(tempfile.mkdtemp(prefix="gate_scan_"))
        scratch_made = scratch if not a.scratch else None
        target = freeze_copy(doc, scratch)
        print("freeze: the file is NOT committed here; scanned from a copy frozen in a throwaway repository")
    try:
        e = Experiment(target, require_power_basis=True, require_nonvacuous_gates=True)
        assert e.gates_sha256 == sha, (e.gates_sha256, sha)
        print("prereg_commit:", e.prereg_commit)
        gates = list(e.spec["gates"].keys())
        print("gates (%d):" % len(gates), gates)
        print("undeclared_power_gates:", e.undeclared_power_gates)
        print("vacuous_gates:", e.vacuous_gates)
        rows = e.spec["outcomes"]
        print("outcome rows:", len(rows))
        reached = {i: 0 for i in range(len(rows))}
        for bits in itertools.product([False, True], repeat=len(gates)):
            fired = dict(zip(gates, bits))
            hit = walk(rows, fired)
            assert hit is not None, "no row for %r" % fired
            reached[hit] += 1
        print("assignments enumerated:", 2 ** len(gates))
        for i, row in enumerate(rows):
            print("  row %2d reached by %4d assignments -> %s" % (i, reached[i], row["verdict"]))
        assert all(v > 0 for v in reached.values()), "unreachable row"
        seen = {}
        for bits in itertools.product([False, True], repeat=len(gates)):
            fired = dict(zip(gates, bits))
            if fired["G_F2_families"] and not fired["G_F1_panel"]:
                continue        # one metric feeds both gates; this corner has no result dict
            v = e.score(result_for(fired))
            for k, b in v.gates.items():
                if k in ("G_F1_panel", "G_F2_families"):
                    continue
                assert b == fired[k], (k, b, fired[k])
            seen[v.verdict] = seen.get(v.verdict, 0) + 1
        print("verdicts realised by scoring synthetic results:")
        for k in sorted(seen):
            print("  %-70s %4d" % (k, seen[k]))
        missing = {r["verdict"] for r in rows} - set(seen)
        print("rows never realised by a result dict:", sorted(missing) or "none")
        sm = e.score({}, smoke=True)
        print("smoke verdict:", sm.verdict, "(gates read: %d)" % len(sm.gates))
        cm = e.check_metrics(result_for(dict(zip(gates, [True] * len(gates)))))
        print("check_metrics usable:", all(v["usable"] for v in cm.values()), "paths:", len(cm))
        bad = text + "\n\n```gates\n{}\n```\n"
        try:
            _select_gates_block(bad)
            print("REFUSAL CHECK FAILED: two gates fences accepted")
            return 1
        except GateSpecError as ex:
            print("two gates fences refused:", str(ex)[:80])
    finally:
        if scratch_made is not None:
            shutil.rmtree(scratch_made, ignore_errors=True)
    print()
    print("wilson(z=1.96):")
    for k, n in ((73, 73), (72, 72), (109, 110), (108, 109), (140, 142), (139, 141), (111, 112), (110, 112),
                 (28, 40), (27, 30), (9, 15), (19, 20), (18, 20)):
        lo, hi = wilson(k, n)
        print("  k=%3d n=%3d  lower=%.4f upper=%.4f" % (k, n, lo, hi))

    def smallest(bar, misses):
        for n in range(misses + 1, 5000):
            if wilson(n - misses, n)[0] >= bar:
                return n
    print("  smallest n with 0 misses clearing 0.95:", smallest(0.95, 0))
    print("  smallest n with 1 miss  clearing 0.95:", smallest(0.95, 1))
    print("  smallest n with 2 misses clearing 0.95:", smallest(0.95, 2))
    print("  se of a share at p=0.70, n=40: %.4f" % ((0.7 * 0.3 / 40) ** 0.5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
