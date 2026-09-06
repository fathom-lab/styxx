# -*- coding: utf-8 -*-
"""The trivially-swearing twin for G-G1 (SPEC §The trivially-swearing twin).

For each document, the Claude family is prompted with the canonical text, the tag grammar of
``SPEC_sworn_output_v02_2026_09_02.md`` in short, the receipt names the document already cites, and
the instruction *swear only what is trivially true*. The returned inline document is canonicalised
with ``styxx.sworn.to_sidecar`` (refuses a document that cannot round-trip); a twin whose canonical
text differs from the original is recorded ``twin_text_changed`` and excluded with its count printed.
The floor of each side is the verifier's ``coverage.sentence_share`` — a span count, as the README
says. Twins are written to ``seat_outputs/<family>/trivial/<doc_id>.trivial-twin.json``; the file name
never ends ``.sworn.json``.

CLI: ``python papers/sworn/measurement/twin_trivial.py --doc D01 [--dry-run] [--dir DIR]``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C                                   # noqa: E402
import population as P                               # noqa: E402
import seat_claude as SC                             # noqa: E402
from styxx import sworn                              # noqa: E402

INSTRUCTION = "swear only what is trivially true"
GRAMMAR = (
    "Tag grammar (sworn output v0.2): wrap one sentence in <sworn r=\"RECEIPT\" k=\"KIND\">...</sworn>. "
    "KIND is numeric (the sentence holds exactly one number, equal to the receipt leaf at the printed "
    "precision), quote (the sentence holds exactly one `backtick needle` that occurs in the receipt), "
    "hash (the sentence holds one 64-hex sha256 of the receipt bytes) or absent (the needle does not occur). "
    "RECEIPT is one of the receipts listed below, optionally with #/json/pointer for a leaf. Tags never go "
    "inside code fences, backticks or HTML comments, never nest, and a span is at most 300 characters. "
    "Return the whole document with your tags inserted and NOTHING else changed: every other byte, "
    "including line breaks, must be identical."
)


def cited_receipts(side: dict) -> List[str]:
    out = []
    for s in side["spans"]:
        r = s["receipt"].split("#", 1)[0]
        if r not in out:
            out.append(r)
    return out


def build(doc_id: str, meas_dir=None, dry_run: bool = False, root=None, family: str = "claude") -> dict:
    meas_dir = Path(meas_dir or HERE)
    root = Path(root or C.ROOT)
    pop = json.loads((meas_dir / "population.json").read_text(encoding="utf-8"))
    entry = next((e for e in P.iter_documents(pop) if e["doc_id"] == doc_id), None)
    if entry is None:
        raise SystemExit("REFUSED: %s is not in the population" % doc_id)
    if not dry_run:
        C.refuse_unless_prereg(False, [], root=root)
    side, tree, rec = C.open_document(entry, root=root)
    side = sworn.load_sidecar(side)
    out_dir = meas_dir / "seat_outputs" / family / "trivial"
    out_path = out_dir / ("%s.trivial-twin.json" % doc_id)
    if out_path.exists():
        raise SystemExit("REFUSED: %s exists; a twin is written once" % out_path)
    status = {"doc_id": doc_id, "family": family, "dry_run": dry_run}
    if dry_run:
        import synthetic as S
        twin: Optional[dict] = S.canned_trivial_twin(side)
        status["transport"] = "canned"
    else:
        system = "You re-tag a technical document. " + GRAMMAR + " Receipts you may cite: " + ", ".join(cited_receipts(side))
        prompt = "Instruction: %s.\n\nDocument:\n\n%s" % (INSTRUCTION, side["text"])
        r = SC.cli(prompt, system, None)
        status["raw_sha256"] = r["raw_sha256"]
        status["error"] = r["error"]
        twin = None
        if r["text"]:
            try:
                twin = sworn.to_sidecar(r["text"].encode("utf-8"), side["document"]["name"], commit=side["commit"])
                twin["manifest"] = side["manifest"]
            except SystemExit:
                # The canonicaliser's own sentence is styxx's and moves with styxx; the status file
                # records the class of refusal in this file's words instead.
                status["refused"] = "to_sidecar refused the returned document: it does not round-trip"
    if twin is None:
        status["twin_text_changed"] = None
        status["written"] = None
        C.write_json_lf(out_dir / ("%s.trivial-status.json" % doc_id), status)
        return status
    status["twin_text_changed"] = twin["text"] != side["text"]
    status["spans_original"] = len(side["spans"])
    status["spans_twin"] = len(twin["spans"])
    C.write_json_lf(out_path, twin)
    status["written"] = str(out_path)
    C.write_json_lf(out_dir / ("%s.trivial-status.json" % doc_id), status)
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--doc", required=True)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    s = build(a.doc, a.dir, dry_run=a.dry_run)
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
