# -*- coding: utf-8 -*-
"""CENSUS — what the v0.1 coverage estimate printed beside every committed sworn document.

Reads every `*.sworn-receipt.json` and its sidecar AS COMMITTED at one commit (git plumbing, never
the working tree — the receipts are re-issued under v1 after this census, and a census that read
the working tree would be measuring its own successor). For each document it records what the
v0.1 receipt printed (`coverage.estimate`, `unsworn_claims_estimate`) and what the v0.2 floor
counts on the same canonical text (`narrative_sentences`, `sentence_share`), so the two can be
read side by side.

The finding it receipts: STRUCT-1 (`styxx.claimdetect`) is a diff-claim detector for agent
pull-request prose and never reads a result-shaped sentence as a claim, so the v0.1 estimate was
near-vacuous beside result-shaped documents. This script measures nothing about load-bearingness;
it counts what two instruments printed.

    python papers/sworn/coverage_census_v01.py [COMMIT]     # default 320b30322d804923dc89b8eaf6e63dfa8e3f45f2
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402

DEFAULT_COMMIT = "320b30322d804923dc89b8eaf6e63dfa8e3f45f2"
OUT = HERE / "coverage_census_v01_result.json"
_SPLIT = sworn._SENTENCE_SPLIT


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def narrative_sentences(canonical: bytes, spans, fenced) -> int:
    buf = bytearray(canonical)
    for s in spans:
        for i in range(s["start"], s["end"]):
            buf[i] = 0x20
    for a, b in sworn._fenced_regions(bytes(buf))[0]:
        for i in range(a, b):
            buf[i] = 0x20
    narrative = bytes(buf)
    pos, n = 0, 0
    pieces = []
    for m in _SPLIT.finditer(narrative):
        pieces.append((pos, m.start()))
        pos = m.end()
    pieces.append((pos, len(narrative)))
    for a, b in pieces:
        if narrative[a:b].strip():
            n += 1
    return n


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commit = argv[0] if argv else DEFAULT_COMMIT
    names = git("ls-tree", "-r", "--name-only", commit, "--", "papers").decode("utf-8").split("\n")
    receipts = sorted(n for n in names if n.endswith(".sworn-receipt.json"))
    rows = []
    for rec_path in receipts:
        stem = rec_path[: -len(".sworn-receipt.json")]
        rec = json.loads(git("show", "%s:%s" % (commit, rec_path)).decode("utf-8"))
        side = json.loads(git("show", "%s:%s.sworn.json" % (commit, stem)).decode("utf-8"))
        canonical = side["text"].encode("utf-8")
        sc = sworn.scan(sworn.render(side))
        n_sent = narrative_sentences(canonical, side["spans"], sc["fenced"])
        sworn_total = rec["sworn_total"]
        cov = rec.get("coverage", {})
        rows.append({
            "document": stem.split("/")[-1],
            "receipt_schema": rec.get("schema"),
            "document_verdict": rec.get("document_verdict"),
            "sworn_total": sworn_total,
            "v01_coverage_estimate": cov.get("estimate"),
            "v01_unsworn_claims_estimate": cov.get("unsworn_claims_estimate"),
            "v01_unsworn_claim_texts": [c.get("text", "")[:120] for c in cov.get("unsworn_claims", [])],
            "narrative_sentences": n_sent,
            "sentence_share_floor": (round(sworn_total / (sworn_total + n_sent), 4)
                                     if (sworn_total + n_sent) else None),
        })
    ests = [r["v01_coverage_estimate"] for r in rows if r["v01_coverage_estimate"] is not None]
    floors = [r["sentence_share_floor"] for r in rows if r["sentence_share_floor"] is not None]
    out = {
        "what": "CENSUS of the v0.1 coverage estimate beside every committed sworn document, read at one commit via git plumbing; counts, not a measurement of load-bearingness",
        "commit": commit,
        "documents": len(rows),
        "v01_estimate_min": min(ests) if ests else None,
        "v01_estimate_max": max(ests) if ests else None,
        "v01_unsworn_claims_total": sum(r["v01_unsworn_claims_estimate"] or 0 for r in rows),
        "narrative_sentences_total": sum(r["narrative_sentences"] for r in rows),
        "sworn_total": sum(r["sworn_total"] for r in rows),
        "floor_min": min(floors) if floors else None,
        "floor_max": max(floors) if floors else None,
        "rows": rows,
        "what_this_is_not": [
            "not a measurement of bound recall: nobody judged which sentences were load-bearing",
            "the floor treats every narrative sentence as load-bearing and so understates coverage by construction",
            "the v0.1 estimate used STRUCT-1, a diff-claim detector, on result-shaped prose — the census records what it printed, not what it should have",
        ],
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for r in rows:
        print("%-62s sworn=%-3d v0.1-est=%-7s unsworn=%-2s narrative=%-3d floor=%s"
              % (r["document"][:62], r["sworn_total"], r["v01_coverage_estimate"],
                 r["v01_unsworn_claims_estimate"], r["narrative_sentences"], r["sentence_share_floor"]))
    print("documents %d | v0.1 estimate %s..%s | unsworn-claims counted %d of %d narrative sentences | floor %s..%s -> %s"
          % (out["documents"], out["v01_estimate_min"], out["v01_estimate_max"],
             out["v01_unsworn_claims_total"], out["narrative_sentences_total"],
             out["floor_min"], out["floor_max"], OUT.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
