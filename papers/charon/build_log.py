# -*- coding: utf-8 -*-
"""Build the ferry log over every tracked verdict-bearing artifact in this repository.

The population is this script, not a sentence: ``git ls-files`` over the three suffixes, sorted,
nothing excluded — including the arXiv staging certificates, which enter as UNRESOLVED lines
rather than as absences. A population defined by what survived the walk is the defect this lane
has catalogued nine times, so nothing is skipped and the enumeration is re-runnable:

    python papers/charon/build_log.py            # rebuild log + verify report + page
    python papers/charon/build_log.py --list     # print the population and exit

The log it writes is append-only afterwards; this script REFUSES to overwrite one that exists
(a receipt is history too — a rebuilt log is a new file at a new commit, never an edit).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx import charon  # noqa: E402

SUFFIXES = ("*.sworn.json", "*.capsule.html", "*.certificate.json")
LOG = HERE / "charon.log.jsonl"
REPORT = HERE / "charon_verify_result.json"
PAGE = HERE / "index.html"

# THE ONE EXCLUSION, and its reason. The RESULT that describes this log states the log's own
# counts and head; a line about that document would be a line the document's next sentence
# invalidates, and every rebuild would leave it DRIFTing. The corpus census made exactly this
# mistake once and the rule that came out of it stands: a snapshot cannot contain its own
# description (`build_corpus_census.py` excludes `CORPUS_STATE_*` for the same reason). The
# document is excluded; the SPEC, the battery and every other sworn document are not.
SELF_DESCRIBING = ("papers/charon/RESULT_charon_v01_ships_2026_09_02.sworn.json",)
POPULATION = ("git ls-files " + " ".join(repr(s) for s in SUFFIXES) +
              " at the repository root, sorted; the arXiv staging certificates enter as UNRESOLVED "
              "lines rather than as absences; the only exclusion is the sworn RESULT that describes "
              "this log, because a snapshot cannot contain its own description; rebuild with "
              "papers/charon/build_log.py")


def population() -> list:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", *SUFFIXES],
                         capture_output=True, check=True).stdout
    rels = sorted(p for p in out.decode("utf-8").split("\0") if p and p not in SELF_DESCRIBING)
    return [ROOT / r for r in rels]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_log.py", description=__doc__)
    ap.add_argument("--list", action="store_true", help="print the population and exit")
    a = ap.parse_args(argv)
    paths = population()
    if a.list:
        for p in paths:
            print(p.relative_to(ROOT).as_posix())
        print("%d artifacts" % len(paths))
        return 0
    if LOG.exists():
        raise SystemExit("REFUSED: %s exists — a rebuilt log is a new file at a new commit, "
                         "never an edit. Move the old one aside deliberately." % LOG.name)
    added = charon.ingest(paths, LOG, ROOT, name="styxx", population=POPULATION)
    print("ingested %d lines" % len(added))
    rep = charon.verify_log(LOG, ROOT)
    with open(REPORT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    charon.render_page(LOG, PAGE, rep, repo=ROOT)
    print("head %s" % rep["head"])
    print("by status: %s" % rep["by_status"])
    print("by kind:   %s" % rep["by_kind"])
    print("reproduced at ingest: %s" % rep["reproduced_at_ingest"])
    print("receipts:  %s" % {k: v for k, v in rep["receipts_n"].items() if k != "by_kind"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
