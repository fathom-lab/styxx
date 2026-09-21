"""G-M3-3 of MUTE-3, checked by id rather than by count: every test that passed on run A's
baseline passes on run B's, plus the two controls. The receipts carry only the counts (534, 537);
this records the ids of both baselines so the scorer can hold the gate as the preregistration
words it, and name whatever else run B's baseline gained.

    python papers/harness/mute3_baselines.py --a /path/to/run-A-tree --b /path/to/run-B-tree
    # writes mute3_baselines.json beside this file

The instrument is imported from the run-B tree, so the oracle is the receipts' oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "mute3_baselines.json"


def head(tree: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(tree), capture_output=True, text=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="a clean worktree of the run-A tree")
    ap.add_argument("--b", required=True, help="a clean worktree of the run-B tree")
    args = ap.parse_args()
    trees = {"A": Path(args.a).resolve(), "B": Path(args.b).resolve()}
    sys.path.insert(0, str(trees["B"]))
    from benchmarks.harness_mutation import mute  # the run-B tree's instrument

    out: dict = {"what": "the ids of every oracle test on the run-A and run-B baselines, for G-M3-3",
                 "instrument_sha256": hashlib.sha256((trees["B"] / "benchmarks/harness_mutation/mute.py").read_bytes()).hexdigest()}
    for name, tree in trees.items():
        files = mute.oracle_files(tree)
        t0 = time.time()
        res = mute.run_oracle(tree, files)
        out[name] = {
            "tree": head(tree), "files": files,
            "passing": sorted(k for k, v in res.items() if v == "passed"),
            "not_passing": sorted(k for k, v in res.items() if v != "passed"),
            "seconds": round(time.time() - t0, 1),
        }
    a, b = set(out["A"]["passing"]), set(out["B"]["passing"])
    out["A_passing_not_passing_on_B"] = sorted(a - b)
    out["B_passing_not_passing_on_A"] = sorted(b - a)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"A {out['A']['tree'][:12]} {len(a)} passing; B {out['B']['tree'][:12]} {len(b)} passing")
    print("lost on B:", out["A_passing_not_passing_on_B"])
    print("gained on B:", out["B_passing_not_passing_on_A"])
    return 0 if not (a - b) else 1


if __name__ == "__main__":
    sys.exit(main())
