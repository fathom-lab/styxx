"""The other half of the pair, demonstrated outside the receipts: the manifest test goes red when
the anchor step is cut from test.yml. `mute3_anchor_demo.py` shows the anchor step failing when
the manifest test file is deleted; this shows the manifest test failing when the anchor step is
deleted. Together they are the claim that only the simultaneous cut of both is silent.

    python papers/harness/mute3_anchor_pin_demo.py --tree /path/to/run-B-worktree   # writes mute3_anchor_pin_demo.json

The mutant is the instrument's own level-1 M-STEP of the anchor step, applied with the
instrument's `apply`, and the oracle is the manifest test file alone. It is a demonstration, not a
verdict: MUTE-3's level-1 receipt (MUTE-2r) was taken on the run-A tree, which has no anchor step,
so this mutant is in no receipt of this cycle.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "mute3_anchor_pin_demo.json"
ANCHOR_NAME_PREFIX = "Anchor"
MANIFEST_TEST_FILE = "tests/test_harness_manifest.py"
MANIFEST_TEST = "tests.test_harness_manifest::test_the_harness_matches_the_committed_manifest"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="a git worktree of the run-B tree; the instrument is imported from it; restored afterwards")
    tree = Path(ap.parse_args().tree).resolve()
    sys.path.insert(0, str(tree))
    from benchmarks.harness_mutation import mute

    subprocess.run(["git", "checkout", "-q", "--", "."], cwd=str(tree), check=True)
    _, mutants = mute.inventory(level=1)
    target = [m for m in mutants if m.operator == "M-STEP" and m.check.path == ".github/workflows/test.yml"
              and m.check.job == "test" and str(m.check.name or "").startswith(ANCHOR_NAME_PREFIX)]
    if len(target) != 1:
        raise SystemExit(f"expected exactly one M-STEP mutant of the anchor step, found {len(target)}")
    m = target[0]

    intact = mute.run_oracle(tree, [MANIFEST_TEST_FILE])
    applied = mute.apply(m, tree)
    mutant = mute.run_oracle(tree, [MANIFEST_TEST_FILE]) if applied else {}
    subprocess.run(["git", "checkout", "-q", "--", "."], cwd=str(tree), check=True)

    payload = {
        "what": "the level-1 M-STEP mutant of the anchor step in test.yml, applied to the run-B tree; the manifest test file run on the intact tree and on the mutant",
        "tree": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(tree), capture_output=True, text=True).stdout.strip(),
        "mutant": {"id": m.id, "operator": m.operator, "check": m.check.__dict__, "applied": applied},
        "manifest_test_intact": intact.get(MANIFEST_TEST),
        "manifest_test_on_mutant": mutant.get(MANIFEST_TEST),
        "other_tests_red_on_mutant": sorted(k for k, v in mutant.items() if v == "failed" and k != MANIFEST_TEST),
        "reading": "a demonstration outside the receipts, not a verdict: the anchor step is a level-1 mutant of a tree no level-1 receipt of this cycle was taken on",
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return 0 if (applied and intact.get(MANIFEST_TEST) == "passed" and mutant.get(MANIFEST_TEST) == "failed") else 1


if __name__ == "__main__":
    sys.exit(main())
