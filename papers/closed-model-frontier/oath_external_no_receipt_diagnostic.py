"""POST-HOC DIAGNOSTIC — how much of NO_RECEIPT is "no results" and how much is "not my filename"?

**This is not part of the frozen protocol and changes nothing about the collected corpus.** It ran
after collection, it is labelled post-hoc, and its numbers may not be substituted for the frozen
run's. It exists because the frozen run produced a number that cannot be read the way the
protocol's outcome table proposed to read it.

## The problem it sizes

`PROTOCOL_oath_external_corpus_2026_08_27.md` row 6 says that if `NO_PAIR` dominates, that is
"almost nobody publishes a claim document beside machine-readable results" — the strongest
available evidence for the contract framing.

The frozen run returned `NO_RECEIPT` for 50 of 140 repositories (36%). Reading row 6 off that
would have been wrong, and three repositories checked by hand show why:

    dwzhu-pku/LongEmbed              results/bge-m3/overall_results.json
    G-Taxonomy-Workgroup/GTaxoGym    agg_results/graph_gcn_results.json
    tsinghua-fib-lab/Token_Signature dynamic_cot/All_results.json

All three publish machine-readable results. All three were `NO_RECEIPT`. Two failure modes:

1. **Affixes.** GitHub's `filename:results.json` qualifier tokenises, so it matches
   `overall_results.json` and `graph_gcn_results.json`. The collector's `RECEIPT_NAMES` is an
   exact basename set, so it does not. The selection rule and the inclusion rule disagree about
   what a results file is, and repositories fall into the gap.
2. **Case.** `All_results.json` is not `all_results.json`; the basename test is case-sensitive
   while the README test (after amendment) is not.

`RECEIPT_NAMES` is a marker standing in for the class *machine-readable results file*, which is
the defect `SYNTHESIS_mention_and_use_2026_08_26.md` catalogues, occurring inside the collector
built to measure that defect's reach.

## What this does and does not do

It counts, for every `NO_RECEIPT` and `NO_DOC` repository, how many carry a file matching a looser
pattern. **It does not re-collect, re-certify, or revise any frozen number.** Widening
`RECEIPT_NAMES` after seeing which repositories fell out would be selection after seeing returns.
The frozen corpus stands as collected; what changes is what may be concluded from it.

  python papers/closed-model-frontier/oath_external_no_receipt_diagnostic.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from oath_external_corpus import RECEIPT_NAMES, gh_json  # noqa: E402

MANIFEST = HERE / "oath_external_corpus.json"
OUT = HERE / "oath_external_no_receipt_diagnostic.json"

# Deliberately loose, and used ONLY to size the gap -- never to admit a receipt.
LOOSE = re.compile(r"(results?|metrics|scores|eval|benchmark).*\.json$", re.I)
STEMS = ("results", "metrics", "scores")


def main() -> int:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    targets = [r for r in man["per_repo"] if r["status"] in ("NO_RECEIPT", "NO_DOC")]
    print(f"probing {len(targets)} repositories that yielded no receipt pair\n")

    rows, unreachable = [], 0
    for i, r in enumerate(targets, 1):
        sha = r.get("sha")
        if not sha:
            unreachable += 1
            continue
        t = gh_json([f"repos/{r['repo']}/git/trees/{sha}?recursive=1"], timeout=120)
        if t is None:
            unreachable += 1
            continue
        paths = [e["path"] for e in t.get("tree", []) if e.get("type") == "blob"]
        loose = [p for p in paths if LOOSE.search(p.rsplit("/", 1)[-1])]
        exact_ci = [p for p in paths
                    if p.rsplit("/", 1)[-1].lower() in {n.lower() for n in RECEIPT_NAMES}]
        affix = [p for p in loose
                 if p not in exact_ci
                 and any(s in p.rsplit("/", 1)[-1].lower() for s in STEMS)]
        rows.append({
            "repo": r["repo"], "status": r["status"], "query": r["query"],
            "loose_matches": len(loose), "case_only_misses": len(exact_ci),
            "affix_misses": len(affix), "examples": sorted(loose)[:4],
        })
        print(f"  [{i:>3}/{len(targets)}] {r['repo'][:46]:<47} loose={len(loose):<4} "
              f"case={len(exact_ci):<3} affix={len(affix)}")

    with_any = [r for r in rows if r["loose_matches"] > 0]
    case_only = [r for r in rows if r["case_only_misses"] > 0]
    payload = {
        "diagnostic": "post-hoc sizing of NO_RECEIPT; NOT part of the frozen protocol",
        "changes_no_frozen_number": True,
        "probed": len(rows), "unreachable": unreachable,
        "carry_a_loosely_matching_results_file": len(with_any),
        "share_of_probed": round(len(with_any) / len(rows), 4) if rows else None,
        "missed_on_CASE_alone": len(case_only),
        "missed_on_AFFIX": sum(1 for r in rows if r["affix_misses"] > 0),
        "reading": (
            "Of the repositories the frozen run reported as publishing no machine-readable "
            "results, this many demonstrably do. Outcome-table row 6 -- 'almost nobody publishes "
            "a claim document beside machine-readable results' -- CANNOT be read off this corpus, "
            "because NO_RECEIPT measures agreement with a frozen filename list rather than the "
            "presence of results. The frozen numbers stand; the inference does not."),
        "per_repo": rows,
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print()
    print(f"of {len(rows)} probed, {len(with_any)} carry a results-like JSON "
          f"({payload['share_of_probed']})")
    print(f"  missed on CASE alone: {len(case_only)}   missed on AFFIX: {payload['missed_on_AFFIX']}")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
