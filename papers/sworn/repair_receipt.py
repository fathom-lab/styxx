"""Re-derive a repair's watched-to-fail outcome from git and write it as a receipt.

Generalised from tree_channel_repair_receipt.py, which did this for one repair with two constants
baked in. A RESULT that says "2 of 8 failed before and 0 of 8 after" is asserting numbers, and this
corpus does not let a number stand in prose with nothing behind it. So the guard is run against the
verifier as it was and as it is — both read from git by revision, never from the working tree — and
the counts are written where a sworn span can bind to them.

    python papers/sworn/repair_receipt.py --repair <sha> --guard tests/test_x.py --out name.json
                                          [--target styxx/sworn.py] [--also styxx/_data/f.js ...]

`--also` names other files the repair changed (the JavaScript verifier, say) so the scratch package
carries them at the same revision; without it a two-sided repair would be measured with one side
old and one side new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
_SUMMARY = re.compile(r"(\d+) passed|(\d+) failed")


def _show(rev: str, rel: str) -> bytes:
    r = subprocess.run(["git", "-C", str(ROOT), "show", "%s:%s" % (rev, rel)], capture_output=True)
    if r.returncode != 0:
        raise SystemExit("REFUSED: cannot read %s at %s — is the history present?" % (rel, rev))
    return r.stdout


def _run_guard(rev: str, guard: str, files: list) -> dict:
    """Run the guard in a scratch copy of the package with `files` taken from `rev`."""
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        pkg = work / "styxx"
        for p in (ROOT / "styxx").rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                dest = pkg / p.relative_to(ROOT / "styxx")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(p.read_bytes())
        digests = {}
        for rel in files:
            data = _show(rev, rel)
            (work / rel).parent.mkdir(parents=True, exist_ok=True)
            (work / rel).write_bytes(data)
            digests[rel] = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
        tests = work / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_bytes(b"")
        (tests / Path(guard).name).write_bytes((ROOT / guard).read_bytes())
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            str(tests / Path(guard).name)],
                           cwd=str(work), capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=900,
                           env={**os.environ, "PYTHONPATH": str(work)})
        passed = failed = 0
        for m in _SUMMARY.finditer(r.stdout or ""):
            if m.group(1):
                passed = int(m.group(1))
            if m.group(2):
                failed = int(m.group(2))
        ids = sorted(set(re.findall(r"^FAILED\s+(\S+)", r.stdout or "", re.M)))
        if passed + failed == 0:
            raise SystemExit("REFUSED: the guard ran no tests at %s — a child that never ran is "
                             "not a count:\n%s" % (rev, (r.stdout or "")[-600:]))
        return {"rev": rev, "passed": passed, "failed": failed, "failed_ids": ids,
                "files": digests}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", required=True, help="the commit that made the repair")
    ap.add_argument("--guard", required=True, help="the test file, relative to the repo root")
    ap.add_argument("--out", required=True, help="receipt file name, written under papers/sworn")
    ap.add_argument("--target", default="styxx/sworn.py")
    ap.add_argument("--also", nargs="*", default=[])
    ap.add_argument("--spec", default=None)
    a = ap.parse_args(argv)

    dest = HERE / a.out
    if dest.exists():
        r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(dest)],
                           capture_output=True)
        if r.returncode == 0:
            print("REFUSED: %s is tracked; a receipt is history — write a new file" % dest.name,
                  file=sys.stderr)
            return 2

    files = [a.target] + list(a.also)
    before = _run_guard(a.repair + "~1", a.guard, files)
    after = _run_guard(a.repair, a.guard, files)
    out = {
        "schema": "styxx.sworn.repair-watched-to-fail/v1",
        "spec": a.spec,
        "guard": a.guard,
        "repair_commit": a.repair,
        "files_measured": files,
        "before": before,
        "after": after,
        "guard_tests": before["passed"] + before["failed"],
        "reading": ("the guard fails against the verifier as shipped and passes against it "
                    "repaired; the tests that pass in BOTH states are the controls, which a repair "
                    "that over-refused would have failed"),
    }
    dest.write_bytes((json.dumps(out, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    print("before: %d passed %d failed  |  after: %d passed %d failed  -> %s"
          % (before["passed"], before["failed"], after["passed"], after["failed"], dest.name))
    for i in before["failed_ids"]:
        print("   failed before:", i.split("::", 1)[-1][:90])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
