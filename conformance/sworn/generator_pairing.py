"""Measure how many cases survive a generator change at the same seed, and write the receipt.

`NOTE_unpaired_samples_2026_09_06.md` and `RESULT_aperture_closure_2026_09_06.md` both rest on one
number: how many of the first N cases produce the SAME DOCUMENT under the generator before and
after the aperture widening. That number was measured ad hoc and asserted in prose, which is the
failure this corpus exists to refuse — so it gets a receipt like anything else.

Both generators are loaded from git by commit, never from the working tree, so this re-derives in
any checkout that has the history and does not depend on what happens to be checked out.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# The commit that widened the aperture. Its parent is the generator as it was.
WIDENING = "1edbbb24"
REL = "conformance/sworn/differential.py"


def _load(source: bytes, name: str):
    d = Path(tempfile.mkdtemp())
    (d / "differential.py").write_bytes(source)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(name, str(d / "differential.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _show(rev: str) -> bytes:
    r = subprocess.run(["git", "-C", str(ROOT), "show", "%s:%s" % (rev, REL)],
                       capture_output=True)
    if r.returncode != 0:
        raise SystemExit("REFUSED: cannot read %s at %s — is the history present?" % (REL, rev))
    return r.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out", default=str(HERE / "generator_pairing.json"))
    a = ap.parse_args(argv)

    out = Path(a.out).resolve()
    if out.exists():
        r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(out)],
                           capture_output=True)
        if r.returncode == 0:
            print("REFUSED: %s is tracked; a run is history — write a new file" % out.name,
                  file=sys.stderr)
            return 2

    before_src, after_src = _show(WIDENING + "~1"), _show(WIDENING)
    before, after = _load(before_src, "gen_before"), _load(after_src, "gen_after")

    same_doc = same_all = 0
    for i in range(a.cases):
        b, c = before.case(a.seed, i), after.case(a.seed, i)
        if b["document"] == c["document"]:
            same_doc += 1
            if (b["name"], b["commit"], json.dumps(b["manifest"], sort_keys=True, default=str)) == \
               (c["name"], c["commit"], json.dumps(c["manifest"], sort_keys=True, default=str)):
                same_all += 1

    receipt = {
        "schema": "styxx.sworn.generator-pairing/v1",
        "question": ("how many cases survive a generator change at the same seed — i.e. whether "
                     "'same seed, same size' gives paired samples across generator versions"),
        "seed": a.seed,
        "cases": a.cases,
        "generators": {
            "before": {"rev": WIDENING + "~1", "path": REL,
                       "sha256": hashlib.sha256(before_src.replace(b"\r\n", b"\n")).hexdigest()},
            "after": {"rev": WIDENING, "path": REL,
                      "sha256": hashlib.sha256(after_src.replace(b"\r\n", b"\n")).hexdigest()},
        },
        "identical_documents": same_doc,
        "identical_whole_cases": same_all,
        "different_documents": a.cases - same_doc,
        "reading": ("a case is built from a random.Random seeded on (seed, index); every draw "
                    "advances that stream, so a generator change that ADDS a draw re-randomises "
                    "every case after it. Fixing the seed does not fix the sample, and a "
                    "before/after comparison over these two runs is not paired."),
    }
    out.write_bytes((json.dumps(receipt, indent=1, sort_keys=True, ensure_ascii=False)
                     + "\n").encode("utf-8"))
    assert b"\r" not in out.read_bytes(), "conformance/ is -text pinned"
    print("of %d cases at seed %d: %d identical documents, %d different -> %s"
          % (a.cases, a.seed, same_doc, a.cases - same_doc, out.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
