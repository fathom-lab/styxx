"""Run the Python instrument over the corpus: styxx.diffgate.gate_diff_text on every pair.

    python py_side.py                  # this checkout's styxx/diffgate.py, the file the port was made from -> py_out.json
    python py_side.py --installed      # the installed `styxx` package instead (drift measurement)

The port in ../diffgate.js is a transliteration of one specific file: styxx/diffgate.py at the
BC-2 + COMPAT-1 + BIN-1 checkout (pull requests #113, #115 and the #118 repair), sha256 PINNED
below — the file 7.48.0 ships once they merge. By default this script imports the checkout's module and REFUSES to run
unless it hashes to that pin (after CRLF -> LF normalisation, because a wheel built on Windows
carries CRLF and the same file then hashes differently), so "0 disagreements" always means
"against the file the port claims to be", never against whatever happened to be importable.
`--installed` runs the installed package instead, which is how far the release on PyPI sits from
the port; that run is expected to disagree until 7.48.0 ships, and the README says by how much.

`unparsed_claims` is dropped from the records before comparison: that field comes from
styxx.claimdetect, which the port does not carry, and the port says so.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PINNED = "2f79793e46acd3dd4c2e9f2932db881ec4ebfe65846868f2d6fc0956c42793e6"  # styxx/diffgate.py: main's a7debd58 reader (969abb0d + #174's old path and #175's segments in `only_touches`, on reading paths; 969abb0d is 91d45e3a + the one-sided #178, on a reading path; 91d45e3a is e1441f18 + #179's async read and #178's abstention, on reading paths; e1441f18 is f457d2e4 + the front door's docstring, --help and WITHHOLD comment wording, outside any reading path; f457d2e4 is 4eb3d6f0 + #205's line cut and abstention; 4eb3d6f0 is 68873068, which is 9b620e00 + the demo's CONTRADICTED label, + the record naming its instrument) + the PATH-2a block (LF)
# #174 and #175 (fix/diffgate-174-175) moved reading paths in both files at once: `only_touches` reads a rename's old
# path beside the status keys and abstains where only it lies outside the prefixes, and resolves a prefix's `.` and
# `..` segments as git resolves a pathspec, so the differential compares the new reading on both sides.
# #179 and #178 (fix/diffgate-178-179) moved reading paths in both files at once: both read `async def` where they read
# `def`, and both abstain on the `def` counts where a counted line sits in a file not read as .py or .pyi or after an
# odd number of triple quotes in the added lines of its `+++` section, where #205's seams part the two readers (a
# header's edge, before a `def`, after a name), and where two lines of one `+++` section define one test name, one
# after `async`, so the differential compares the new reading on both sides.
# #205 (fix/diffgate-205-206) moved reading paths in both files at once: the Python cuts git output where the port
# always did, and both abstain on the `def` counts over a line holding a character of _READ_APART, so the differential
# compares the new reading on both sides. #206 moved styxx/declare.py, which this pin does not hold.
# The record naming its instrument (identity phase 2) appends `build` and `runtime` to every record and moves no
# content key, so the differential, which compares named fields only, compares what it compared before.
# The pin moved twice in one step and both moves are deliberate. COMPAT-2 (#124) changed the
# compat reading, so the port had to follow it; and `fetch_pr` landed on main after the previous
# pin was written, which is why this script has been REFUSING TO RUN on main ever since -- the
# check that guards the two-implementation claim was itself disabled, which is how the port fell
# a whole cycle behind without anything failing. `fetch_pr` fetches a pull request over the
# network and is not part of the reading the port transliterates; it moves this whole-file hash
# without changing a single verdict, and that is recorded here rather than worked around.
CORPORA = ("corpus_real.json", "corpus_fuzz.json", "bc1_pairs.json", "compat_pairs.json", "bin1_pairs.json", "compat2_pairs.json", "path1_pairs.json", "declare1_pairs.json")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load(installed: bool):
    if installed:
        # keep the checkout off the path so `import styxx` is the installed package, not this tree
        sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != REPO]
    else:
        sys.path.insert(0, str(REPO))
    mod = importlib.import_module("styxx.diffgate")
    digest = _digest(Path(mod.__file__))
    if not installed and digest != PINNED:
        sys.exit(f"{mod.__file__} hashes to {digest[:16]}…, not the file the port was made from "
                 f"({PINNED[:16]}…). Check out the BC-2 + COMPAT-1 instrument, or pass --installed to "
                 "measure drift against the installed package instead.")
    return mod, digest


def main(argv: list[str]) -> int:
    installed = "--installed" in argv
    mod, digest = load(installed)
    items = []
    for name in CORPORA:
        p = HERE / name
        if p.exists():
            items += json.loads(p.read_text(encoding="utf-8"))
    if not items:
        sys.exit("no corpus: run build_corpus.py and/or fuzz_corpus.py first")
    out = []
    for it in items:
        d = mod.gate_diff_text(it["summary"], it["diff"]).to_dict()
        d.pop("unparsed_claims", None)
        out.append({"id": it["id"], **d})
    (HERE / "py_out.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    which = "installed package" if installed else "this checkout"
    print(f"{len(out)} pairs, {sum(len(d['claims']) for d in out)} claims -> py_out.json "
          f"({which}: {mod.__file__}, sha256 {digest[:16]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
