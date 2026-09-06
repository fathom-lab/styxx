"""Re-derive the watched-to-fail outcome for the tree-channel repair and write it as a receipt.

A RESULT that says "3 of 10 failed before and 10 of 10 after" is asserting numbers, and this corpus
does not let a number stand in prose with nothing behind it — that rule caught its own author twice
in one night. So the guard is run against the verifier as it was and as it is, from git, and the
counts are written down where a sworn span can bind to them.

Both versions come from git by revision, never from the working tree, so this re-derives in any
checkout carrying the history.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REPAIR = "4017224c"                       # the commit that added the tree-branch refusal
GUARD = "tests/test_sworn_tree_channel_authorship.py"
TARGET = "styxx/sworn.py"

_SUMMARY = re.compile(r"(\d+) passed|(\d+) failed")


def _show(rev: str, rel: str) -> bytes:
    r = subprocess.run(["git", "-C", str(ROOT), "show", "%s:%s" % (rev, rel)], capture_output=True)
    if r.returncode != 0:
        raise SystemExit("REFUSED: cannot read %s at %s" % (rel, rev))
    return r.stdout


def _run_guard(sworn_bytes: bytes) -> dict:
    """Run the guard in a scratch copy of the package with the given verifier bytes installed."""
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        pkg = work / "styxx"
        pkg.mkdir()
        for p in (ROOT / "styxx").rglob("*.py"):
            dest = pkg / p.relative_to(ROOT / "styxx")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
        (pkg / "sworn.py").write_bytes(sworn_bytes)
        data = ROOT / "styxx" / "_data"
        if data.exists():
            for p in data.rglob("*"):
                if p.is_file():
                    d = pkg / "_data" / p.relative_to(data)
                    d.parent.mkdir(parents=True, exist_ok=True)
                    d.write_bytes(p.read_bytes())
        tests = work / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_bytes(b"")
        (tests / Path(GUARD).name).write_bytes((ROOT / GUARD).read_bytes())
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            str(tests / Path(GUARD).name)],
                           cwd=str(work), capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=600,
                           env={**__import__("os").environ, "PYTHONPATH": str(work)})
        passed = failed = 0
        for m in _SUMMARY.finditer(r.stdout or ""):
            if m.group(1):
                passed = int(m.group(1))
            if m.group(2):
                failed = int(m.group(2))
        ids = sorted(set(re.findall(r"^FAILED\s+(\S+)", r.stdout or "", re.M)))
        return {"passed": passed, "failed": failed, "failed_ids": ids,
                "sworn_sha256": hashlib.sha256(sworn_bytes.replace(b"\r\n", b"\n")).hexdigest()}


def main() -> int:
    before = _run_guard(_show(REPAIR + "~1", TARGET))
    after = _run_guard(_show(REPAIR, TARGET))
    out = {
        "schema": "styxx.sworn.repair-watched-to-fail/v1",
        "spec": "papers/sworn/SPEC_tree_channel_authorship_v01_2026_09_06.md",
        "guard": GUARD,
        "repair_commit": REPAIR,
        "before": before,
        "after": after,
        "guard_tests": before["passed"] + before["failed"],
        "reading": ("the guard fails against the verifier as shipped and passes against it "
                    "repaired; the tests that pass in BOTH states are the honest-case controls, "
                    "which a repair that refused every tree receipt would have failed"),
    }
    dest = HERE / "tree_channel_repair.json"
    dest.write_bytes((json.dumps(out, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    print("before: %d passed %d failed  |  after: %d passed %d failed  -> %s"
          % (before["passed"], before["failed"], after["passed"], after["failed"], dest.name))
    for i in before["failed_ids"]:
        print("   failed before:", i.split("::", 1)[-1][:80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
