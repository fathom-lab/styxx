"""Descriptive census: STRUCT-1 over the whole pinned corpus, and over diffgate's blind spot.

The baseline measured the never-read band's claim density from a 294-sentence sample
(0.0204, six A's). This runs the structural detector over all 2,824 sentences of the pinned
corpus and, per commit, over exactly the band diffgate never read — turning "the gate read 6
sentences" into "the gate read 6 and left N structurally-checkable claims unparsed".

Descriptive by the prereg. The gate is Stage 2's fresh blind panel.

  python papers/closed-model-frontier/claim_detector_corpus_census.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.claimdetect import detect, null_n1, null_n2                  # noqa: E402
from styxx.diffgate import gate_diff                                    # noqa: E402

PIN_BASE, PIN_HEAD = "origin/main", "a6994ac"
EXPECT_COMMITS, EXPECT_SENTENCES = 57, 2824
OUT = HERE / "claim_detector_corpus_census.json"


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def sentences(msg: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", msg) if s.strip()]


def main() -> int:
    shas = git("rev-list", "--reverse", f"{PIN_BASE}..{PIN_HEAD}").split()
    assert len(shas) == EXPECT_COMMITS, f"pin drifted: {len(shas)} commits"

    all_sents, bands = [], Counter()
    n1 = n2 = 0
    for sha in shas:
        for s in sentences(git("log", "-1", "--format=%B", sha)):
            all_sents.append(s)
            bands[detect(s).band] += 1
            n1 += null_n1(s)
            n2 += null_n2(s)
    assert len(all_sents) == EXPECT_SENTENCES, f"corpus drifted: {len(all_sents)}"

    # per-commit: what the gate read vs what STRUCT-1 says it left unparsed
    per_commit, tot_unparsed, tot_uncovered, tot_claims = [], 0, 0, 0
    worst = []
    for sha in shas:
        subject = git("log", "-1", "--format=%s", sha).strip()
        g = gate_diff(git("log", "-1", "--format=%B", sha), ROOT, f"{sha}^", sha)
        tot_unparsed += len(g.unparsed_claims)
        tot_uncovered += g.uncovered_sentences
        tot_claims += len(g.claims)
        per_commit.append({"sha": sha[:9], "subject": subject[:70],
                           "templates_parsed": len(g.claims),
                           "never_read": g.uncovered_sentences,
                           "unparsed_claims": len(g.unparsed_claims)})
        for u in g.unparsed_claims:
            worst.append({"sha": sha[:9], "text": u[:140]})

    payload = {
        "census": "STRUCT-1 over the pinned corpus and over diffgate's never-read band",
        "status": "DESCRIPTIVE — the gate is Stage 2's fresh blind panel",
        "prereg": "PREREG_claim_detector_2026_08_30.md",
        "pin": {"base": PIN_BASE, "head": PIN_HEAD, "commits": len(shas),
                "sentences": len(all_sents)},
        "struct1_bands": dict(bands),
        "struct1_claim_rate": round(bands["CLAIM"] / len(all_sents), 4),
        "null_flags": {"n1_path_regex": n1, "n2_verb_stems": n2,
                       "n1_rate": round(n1 / len(all_sents), 4),
                       "n2_rate": round(n2 / len(all_sents), 4)},
        "gate_vs_detector": {
            "templates_parsed_total": tot_claims,
            "never_read_total": tot_uncovered,
            "structurally_checkable_but_unparsed": tot_unparsed,
            "reading_multiple": (round(tot_unparsed / tot_claims, 2) if tot_claims else None),
        },
        "unparsed_claims_every_one": worst,
        "per_commit": per_commit,
        "note": ("STRUCT-1 flags are UNADJUDICATED. Its DEV precision is telemetry on n=2. "
                 "These counts size the blind spot; they do not certify that every flagged "
                 "sentence is a true claim — that is exactly what Stage 2 measures."),
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    gv = payload["gate_vs_detector"]
    print(f"corpus: {len(all_sents)} sentences / {len(shas)} commits")
    print(f"STRUCT-1 bands: {dict(bands)}  claim rate {payload['struct1_claim_rate']}")
    print(f"nulls: N1 {n1} ({payload['null_flags']['n1_rate']})  "
          f"N2 {n2} ({payload['null_flags']['n2_rate']})")
    print(f"\ndiffgate parsed          {gv['templates_parsed_total']:>5} claims")
    print(f"diffgate never read      {gv['never_read_total']:>5} sentences")
    print(f"of those, STRUCT-1 says  {gv['structurally_checkable_but_unparsed']:>5} "
          f"are structurally checkable claims the templates could not parse")
    if gv["reading_multiple"]:
        print(f"  -> {gv['reading_multiple']}x what the templates read")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
