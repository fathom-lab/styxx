"""Run the Python instrument over the corpus: styxx.diffgate.gate_diff_text on every pair.

    python py_side.py                  # this checkout's styxx/diffgate.py, the file the port was made from -> py_out.json
    python py_side.py --installed      # the installed `styxx` package instead (drift measurement)

The port in ../diffgate.js is a transliteration of one specific file: styxx/diffgate.py at the
BC-2 + COMPAT-1 + BIN-1 checkout (pull requests #113, #115 and the #118 repair), re-cut for the PATH-2
repairs (#97, #121, #101) on the file that carries them, sha256 PINNED below. That file also carries
COMPAT-2's compatibility reading, which the port does not: the differential reports those
`compat_claim` records as disagreements (9 on this corpus). By default this script imports the checkout's module and REFUSES to run
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
PINNED = "6ccb9b803d64a3166ede0394a595337082377a13741e6f1b5f0af0c581d48091"  # styxx/diffgate.py, BIN-1 + COMPAT-2 + PATH-2 (LF)
CORPORA = ("corpus_real.json", "corpus_fuzz.json", "bc1_pairs.json", "compat_pairs.json", "bin1_pairs.json", "path2_pairs.json")


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
