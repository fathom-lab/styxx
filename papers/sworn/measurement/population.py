# -*- coding: utf-8 -*-
"""The population is a script (SPEC §The population is a script).

DESIGN v2 row 7, applied mechanically at a pinned commit: a document is in the in-house arm iff
its sidecar is tracked under ``papers/``, its stem starts with ``RESULT_``, ``FINDING_`` or
``DECLARATION_``, and its path is not under ``papers/sworn/``. Nothing is sampled: the rule
selects, the script counts. Every count written here is a copy of a committed receipt's leaf or a
unit count reconciled against one; none is a measurement of sworn output.

CLI: ``python papers/sworn/measurement/population.py [--commit HEAD] [--out population.json]``.
Refuses to overwrite an existing population.json: a re-pin is a new file at a new commit.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C                                   # noqa: E402
from styxx import sworn                              # noqa: E402

SCHEMA = "styxx-sworn/measurement-population/v1"
RULE = ("a document is in the in-house arm iff its sidecar is tracked under papers/, its stem "
        "starts with RESULT_, FINDING_ or DECLARATION_, and its path is not under papers/sworn/")
PREFIXES = ("RESULT_", "FINDING_", "DECLARATION_")
WHAT_THIS_IS_NOT = ["not a sample: the rule selects", "no number here is a measurement"]


def _sidecars_at(commit: str, root=None) -> List[str]:
    rc, out = C.git("ls-tree", "-r", "--name-only", commit, "--", "papers", root=root)
    if rc != 0:
        raise SystemExit("REFUSED: git ls-tree %s failed" % commit)
    names = out.decode("utf-8", errors="replace").split("\n")
    return sorted(n for n in names if n.endswith(".sworn.json"))


def select(commit: str, root=None):
    """(selected stems, excluded [{stem, reason}]) under RULE at `commit`."""
    selected, excluded = [], []
    for side in _sidecars_at(commit, root):
        stem = side[: -len(".sworn.json")]
        base = stem.rsplit("/", 1)[-1]
        if stem.startswith("papers/sworn/"):
            excluded.append({"stem": stem, "reason": "under papers/sworn/ (the builder's format documents)"})
        elif not base.startswith(PREFIXES):
            excluded.append({"stem": stem, "reason": "stem does not start with RESULT_/FINDING_/DECLARATION_"})
        else:
            selected.append(stem)
    return selected, excluded


def describe(entry: dict, root=None) -> dict:
    """Counts for one entry: the committed receipt's leaves, the unit set, and the reconciliation."""
    side, tree, rec = C.open_document(entry, root=root)
    side = sworn.load_sidecar(side)
    units = C.units_of(side)
    ok, mine, theirs = C.reconcile_units(units, rec or {})
    if not ok:
        raise SystemExit("REFUSED: %s — %d narrative units here, committed receipt printed %r; a unit "
                         "set that disagrees with the receipt is a different splitter" % (entry["stem"], mine, theirs))
    row = {
        "doc_id": entry.get("doc_id"),
        "stem": entry["stem"],
        "role": entry.get("role", "prospective"),
        "sidecar_commit": side["commit"],
        "document_sha256": side["document"]["sha256"],
        "receipt_digest": (rec or {}).get("digest"),
        "document_verdict": (rec or {}).get("document_verdict"),
        "sworn_total": len(side["spans"]),
        "counts": (rec or {}).get("counts"),
        "narrative_sentences": theirs,
        "sentence_share": ((rec or {}).get("coverage") or {}).get("sentence_share"),
        "units": len(units),
        "fragments": sum(1 for u in units if u["fragment"]),
        "windows": len(C.windows_of(side["text"].encode("utf-8"), units)),
        "receipt_moved": False,
    }
    if entry.get("source"):
        row["source"] = entry["source"]
    # a receipt that no longer re-derives at its commit is a finding, not an input to hide
    live = sworn.verify(sidecar=side, tree=tree)
    if rec is not None and (live["document_verdict"] != rec.get("document_verdict")
                            or live["counts"] != rec.get("counts")):
        row["receipt_moved"] = True
        row["live_document_verdict"] = live["document_verdict"]
        row["live_counts"] = live["counts"]
    return row


def build(entries: List[dict], pinned_commit: str, seed: int = C.SEED, prefix: str = "D",
          excluded: Optional[List[dict]] = None, root=None) -> dict:
    """Assign doc_ids by shuffle, describe every entry, and return the population object."""
    order = list(range(len(entries)))
    random.Random(seed).shuffle(order)
    docs = []
    for k, i in enumerate(order, 1):
        e = dict(entries[i])
        e["doc_id"] = "%s%02d" % (prefix, k)
        e["pinned_commit"] = pinned_commit
        docs.append(describe(e, root=root))
    return {
        "schema": SCHEMA,
        "pinned_commit": pinned_commit,
        "rule": RULE,
        "seed": seed,
        "documents": docs,
        "excluded": excluded or [],
        "what_this_is_not": list(WHAT_THIS_IS_NOT),
    }


def iter_documents(pop: dict) -> List[dict]:
    """Population entries with the pinned commit attached, ready for common.open_document."""
    out = []
    for d in pop["documents"]:
        e = dict(d)
        e["pinned_commit"] = pop["pinned_commit"]
        out.append(e)
    return out


def iter_excluded(pop: dict) -> List[dict]:
    """Excluded entries (the decoy sources) with the pinned commit attached."""
    out = []
    for d in pop["excluded"]:
        e = dict(d)
        e["pinned_commit"] = pop["pinned_commit"]
        out.append(e)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--out", default=str(HERE / "population.json"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit("REFUSED: %s exists; a re-pin is a new file at a new commit" % out)
    rc, raw = C.git("rev-parse", a.commit)
    if rc != 0:
        raise SystemExit("REFUSED: cannot resolve %s" % a.commit)
    pinned = raw.decode("ascii").strip()
    selected, excluded = select(pinned)
    entries = []
    for stem in selected:
        role = "design_eight" if C.tracked_at(C.DESIGN_COMMIT, stem + ".sworn.json") else "prospective"
        entries.append({"stem": stem, "role": role})
    pop = build(entries, pinned, excluded=excluded)
    C.write_json_lf(out, pop)
    docs = pop["documents"]
    print("population: %d documents at %s (%d design_eight, %d prospective), %d excluded -> %s"
          % (len(docs), pinned[:12], sum(1 for d in docs if d["role"] == "design_eight"),
             sum(1 for d in docs if d["role"] == "prospective"), len(excluded), out.name))
    print("sworn spans %d, units %d, fragments %d, windows %d; receipts moved: %d"
          % (sum(d["sworn_total"] for d in docs), sum(d["units"] for d in docs),
             sum(d["fragments"] for d in docs), sum(d["windows"] for d in docs),
             sum(1 for d in docs if d["receipt_moved"])))
    print("no number above is a measurement of sworn output; no seat was run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
