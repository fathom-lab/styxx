# -*- coding: utf-8 -*-
"""Re-issue every committed sworn verdict receipt under `styxx.sworn.verdict-receipt/v1`.

A receipt is history too: the v0 receipts stay in git, byte for byte, at the commits that wrote
them. This script writes NEW receipts, in a new commit, from the same sidecars at the same
commits they name — nothing in any sidecar or document changes, and every span verdict must be
the one the v0 receipt recorded (the script refuses otherwise; a verdict that moved would be a
finding, not a re-issue). What changes is the receipt's shape: the digest covers the core without
coverage (SPEC v0.2 R9), coverage/1 replaces the withdrawn estimate, provenance and rungs are
printed. `tests/test_sworn_dogfood.py` re-derives the result.

    python papers/sworn/reissue_receipts_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.sworn import GitTree, issue_receipt, load_sidecar, verify  # noqa: E402


def main() -> int:
    moved = []
    written = 0
    for side_path in sorted(ROOT.glob("papers/**/*.sworn.json")):
        if ".claude" in side_path.parts:
            continue
        stem = side_path.name[: -len(".sworn.json")]
        rec_path = side_path.with_name(stem + ".sworn-receipt.json")
        old = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else None
        side = load_sidecar(json.loads(side_path.read_text(encoding="utf-8")))
        tree = GitTree(ROOT, side["commit"])
        core = verify(sidecar=side, name=stem + ".md", tree=tree)
        if old is not None:
            before = [(s["at"], s["verdict"], s["reason"]) for s in old["spans"]]
            after = [(s["at"], s["verdict"], s["reason"]) for s in core["spans"]]
            if before != after or old["document_verdict"] != core["document_verdict"]:
                moved.append((stem, before, after))
                continue
        rec = issue_receipt(core)
        with open(rec_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        written += 1
        print("%-64s %s  spans=%d  floor=%s  rungs=%s" % (stem[:64], core["document_verdict"],
                                                          core["sworn_total"],
                                                          core["coverage"]["sentence_share"],
                                                          core["rungs"]))
    if moved:
        print("REFUSED to re-issue %d receipt(s): a verdict moved under the new verifier" % len(moved))
        for stem, b, a in moved:
            print("  ", stem, "\n     before", b, "\n     after ", a)
        return 1
    print("re-issued %d receipts under v1" % written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
