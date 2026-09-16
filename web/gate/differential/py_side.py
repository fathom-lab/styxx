"""Run the Python instrument over the corpus: styxx.diffgate.gate_diff_text on every pair.

    python py_side.py                  # the RELEASED module the port was made from -> py_out.json
    python py_side.py --working-tree   # the checkout's styxx/diffgate.py instead (drift measurement)

The port in ../diffgate.js is a transliteration of one specific file: styxx/diffgate.py as shipped
in the styxx 7.47.0 wheel, sha256 PINNED below. By default this script imports the installed
`styxx` (`pip install styxx==7.47.0`) and REFUSES to run unless the module it imported hashes to
that pin — so "0 disagreements" always means "against the file the port claims to be", never
against whatever happened to be importable. `--working-tree` runs the checkout's module instead,
which is how far main has moved past the release; that run is expected to disagree, and the
README says by how much.

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
PINNED = "fb2d9b3e8426650bc20fc8613c9bcad16c1dcd86b85b0ca8c532fdd2a23e7304"  # styxx 7.47.0 diffgate.py


def load(working_tree: bool):
    if working_tree:
        sys.path.insert(0, str(REPO))
    else:
        # keep the checkout off the path so `import styxx` is the installed package, not this tree
        sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != REPO]
    mod = importlib.import_module("styxx.diffgate")
    digest = hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
    if not working_tree and digest != PINNED:
        sys.exit(f"{mod.__file__} hashes to {digest[:16]}…, not the 7.47.0 file the port was made "
                 f"from ({PINNED[:16]}…). `pip install styxx==7.47.0`, or pass --working-tree to "
                 "measure drift against this checkout instead.")
    return mod, digest


def main(argv: list[str]) -> int:
    working_tree = "--working-tree" in argv
    mod, digest = load(working_tree)
    items = []
    for name in ("corpus_real.json", "corpus_fuzz.json"):
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
    which = "working tree" if working_tree else "released module"
    print(f"{len(out)} pairs, {sum(len(d['claims']) for d in out)} claims -> py_out.json "
          f"({which}: {mod.__file__}, sha256 {digest[:16]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
