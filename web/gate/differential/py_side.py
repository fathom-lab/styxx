"""Run the Python instrument over the corpus: styxx.diffgate.gate_diff_text on every pair.

    python py_side.py                  # this checkout's styxx/diffgate.py, the file the port was made from -> py_out.json
    python py_side.py --installed      # the installed `styxx` package instead (drift measurement)

The port in ../diffgate.js is a transliteration of one specific file: styxx/diffgate.py as it stands
on main (BC-2 + COMPAT-1 + BIN-2 + COMPAT-2 + PATH-1 + DECLARE-1), re-cut for the PATH-2 repairs
(#97, #121, #101, as amended by AMENDMENT_path2_resolution_2026_09_17, NOTE_path2_third_pass_2026_09_25,
NOTE_path2_fourth_pass_2026_09_25, NOTE_path2_fifth_pass_2026_09_25, NOTE_path2_sixth_pass_2026_09_25 and
NOTE_path2_seventh_pass_2026_09_25) on the file that carries them, sha256 PINNED below, reading names by
the table styxx/_xid.py carries (PINNED_NAME_TABLE). By default this
script imports the checkout's module and REFUSES to run unless it hashes to that pin (after CRLF -> LF
normalisation, because a wheel built on Windows carries CRLF and the same file then hashes
differently), so a disagreement count always means "against the file the port claims to be", never
against whatever happened to be importable.

Over this corpus the count is 0, and any disagreement on it is a port defect. A count of 0 says
nothing about inputs the corpus does not carry: until NOTE_path2_fourth_pass (F-2, F-3) the two
implementations returned opposite `tests_added` verdicts on a re-indent by U+000B, U+000C, U+2028 or
U+2029 while this corpus read 0, because it held no such line. The COMPAT-2 gap this docstring used to
declare is closed: the port carries COMPAT-2 since #126, and the one PATH-2 pair that was pinned for
the Python alone (`path2:121-compat2-a-dotted-scaffold-directory-stays-scaffolding`) is now pinned at
full width on both sides. The two `tests_added` disagreements that origin/main still shows on the
BOM pairs are closed by this branch (NOTE_path2_third_pass, R-1). The fourth pass left one known
disagreement outside this corpus, a symbol name followed by a non-ASCII letter (JavaScript's `\\b` is
ASCII), and a fifth-pass grid found more: the COMPAT patterns, a header path's strip and repr() read a
diff line differently in the two ports. NOTE_path2_fifth_pass (V-1, V-2) closes them, and its 300-input
definition-line grid is a committed test (`test_v1_the_port_reads_the_grid_as_the_python_does`). The
sixth pass (W-1, W-2) closes two more a round-5 review found: two diff parsers that kept different
lines, and a claimed name and a defined name that ended in different places; its name and test-shape
grids are committed tests too. The seventh pass closes the one its own review found: the two ports read
identifiers from their runtimes' Unicode tables (Python 3.12's 15.0, Node 24's 16.0), and now read one
pinned table. What still disagrees is on the SUMMARY side -- the claim templates read
the description with JavaScript's `\\s`, `\\w` and `\\b` -- and web/gate/README.md gives the count.
`--installed` runs the installed package instead, which is how far the release on PyPI sits from
the port. 7.48.0 is on PyPI and ships main's styxx/diffgate.py (sha256 9b620e00..., LF), which this
branch changes, so against 7.48.0 that run disagrees by the PATH-2 repairs; the README says by how much.

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
PINNED = "67fb1b7510b63cccaf6f8e488466fc14748f2c1b0ace6ee940c0c73bc7cf9ced"  # styxx/diffgate.py, main + PATH-2 seventh pass (LF)
# NOTE_path2_seventh_pass: the instrument reads a name by styxx/_xid.py's table, so the table is pinned too
# (the sha256 of the table string both ports carry; tests/test_diffgate_path2.py holds the two copies equal).
PINNED_NAME_TABLE = "8df68f217cca495ab8a38ced9096213aabac4cf23927068d61397d2c9074d4cb"  # Unicode 15.0.0
# The pin moved twice in one step and both moves are deliberate. COMPAT-2 (#124) changed the
# compat reading, so the port had to follow it; and `fetch_pr` landed on main after the previous
# pin was written, which is why this script has been REFUSING TO RUN on main ever since -- the
# check that guards the two-implementation claim was itself disabled, which is how the port fell
# a whole cycle behind without anything failing. `fetch_pr` fetches a pull request over the
# network and is not part of the reading the port transliterates; it moves this whole-file hash
# without changing a single verdict, and that is recorded here rather than worked around.
CORPORA = ("corpus_real.json", "corpus_fuzz.json", "bc1_pairs.json", "compat_pairs.json", "bin1_pairs.json", "compat2_pairs.json", "path1_pairs.json", "declare1_pairs.json", "path2_pairs.json")


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
                 f"({PINNED[:16]}…). Check out the PATH-2 instrument the pin names (its differential is "
                 "expected to show 0 disagreements), or pass --installed to measure drift against the "
                 "installed package instead.")
    if not installed:
        xid = importlib.import_module("styxx._xid")
        table = hashlib.sha256(xid.TABLE.encode("ascii")).hexdigest()
        if table != PINNED_NAME_TABLE or xid.TABLE_SHA256 != PINNED_NAME_TABLE:
            sys.exit(f"styxx/_xid.py's name table hashes to {table[:16]}…, not the one the port carries "
                     f"({PINNED_NAME_TABLE[:16]}…).")
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
