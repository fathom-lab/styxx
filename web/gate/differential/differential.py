"""Compare the two runs field by field. Exit 1 on the first disagreement.

    python differential.py            # py_out.json vs js_out.json

Compared per pair: verdict, measured, why_unmeasured, sentences_total, uncovered_sentences,
uncovered_texts (the never-read list, in order), and every claim as the tuple
(kind, verdict, why, text, detail) in order. A port that got the verdict right for the wrong
reason, or read one sentence more or less, is a disagreement here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ["verdict", "measured", "why_unmeasured", "sentences_total", "uncovered_sentences", "uncovered_texts"]


def main() -> int:
    py = {d["id"]: d for d in json.loads((HERE / "py_out.json").read_text(encoding="utf-8"))}
    js = {d["id"]: d for d in json.loads((HERE / "js_out.json").read_text(encoding="utf-8"))}
    if py.keys() != js.keys():
        sys.exit(f"corpus mismatch: {len(py)} python records vs {len(js)} javascript records")
    diffs = []
    for k, a in py.items():
        b = js[k]
        for f in FIELDS:
            if a[f] != b[f]:
                diffs.append((k, f, a[f], b[f]))
        ca = [(c["kind"], c["verdict"], c["why"], c["text"], c["detail"]) for c in a["claims"]]
        cb = [(c["kind"], c["verdict"], c["why"], c["text"], c["detail"]) for c in b["claims"]]
        if ca != cb:
            diffs.append((k, "claims", ca, cb))
    n_claims = sum(len(d["claims"]) for d in py.values())
    by_verdict = {v: sum(1 for d in py.values() for c in d["claims"] if c["verdict"] == v)
                  for v in ("VERIFIED", "CONTRADICTED", "UNCHECKABLE")}
    print(f"{len(py)} pairs, {n_claims} claims "
          f"({by_verdict['VERIFIED']} verified, {by_verdict['CONTRADICTED']} contradicted, "
          f"{by_verdict['UNCHECKABLE']} uncheckable) — {len(diffs)} disagreement(s)")
    for d in diffs[:20]:
        print(json.dumps(d, ensure_ascii=False)[:600])
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
