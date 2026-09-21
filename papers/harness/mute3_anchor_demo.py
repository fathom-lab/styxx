"""P8 of MUTE-3, demonstrated outside the oracle: the anchor step in test.yml fails when the
manifest test file is deleted, and passes on the intact tree.

    python papers/harness/mute3_anchor_demo.py --tree /path/to/worktree     # writes mute3_anchor_demo.json

The oracle is the suite, and the suite cannot notice its own manifest test going missing (P4/P5
say so and the receipt shows it). The anchor lives in the workflow, so this is a shell run of that
step, not a verdict: it is reported as a demonstration, and it is the only place MUTE-3 leaves the
suite to look at CI.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "mute3_anchor_demo.json"
ANCHOR_NAME_PREFIX = "Anchor"
MANIFEST_TEST = "tests/test_harness_manifest.py"


def anchor_run(tree: Path) -> str:
    import yaml
    doc = yaml.safe_load((tree / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))
    for step in doc["jobs"]["test"]["steps"]:
        if str(step.get("name", "")).startswith(ANCHOR_NAME_PREFIX):
            return step["run"]
    raise SystemExit("no anchor step in test.yml")


def sh(tree: Path, run: str) -> tuple[int, str]:
    r = subprocess.run(["/bin/bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", run],
                       cwd=str(tree), capture_output=True, text=True, timeout=600)
    last = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or [""])[-1]
    return r.returncode, last[:200]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="a git worktree of the run-B tree; restored afterwards")
    tree = Path(ap.parse_args().tree).resolve()
    run = anchor_run(tree)
    subprocess.run(["git", "checkout", "-q", "--", "."], cwd=str(tree), check=True)
    intact, intact_msg = sh(tree, run)
    (tree / MANIFEST_TEST).unlink()
    deleted, deleted_msg = sh(tree, run)
    subprocess.run(["git", "checkout", "-q", "--", "."], cwd=str(tree), check=True)
    payload = {
        "what": "the anchor step of test.yml, run as a shell on the run-B tree, with and without tests/test_harness_manifest.py",
        "tree": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(tree), capture_output=True, text=True).stdout.strip(),
        "anchor_step_run": run,
        "exit_intact": intact, "last_line_intact": intact_msg,
        "exit_with_manifest_test_deleted": deleted, "last_line_with_manifest_test_deleted": deleted_msg,
        "reading": "a demonstration outside the oracle, not a verdict: the anchor is in CI, which the suite cannot see",
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return 0 if (intact == 0 and deleted != 0) else 1


if __name__ == "__main__":
    sys.exit(main())
