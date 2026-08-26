"""AUDIT — how `build_ledger.py` classifies a cycle, and what it gets wrong.

`papers/LEDGER.md` is the document every other claim in this repository is collateralised
against. It says of itself: *Nothing here is typed by hand; re-run it and check.* This is that
check, run against the classifier rather than against the counts.

Two defects, both the same root cause: **the classifier matches substrings against a free-prose
verdict blob, so it cannot tell a cycle that RETURNED an invalid from one that merely MENTIONS
one.**

  D1  `invalids = [c for c in cy if "INVALID__" in c.get("verdict","")]` — substring, anywhere.
      A cycle whose prose discusses an earlier invalid is counted as a machinery refusal.
  D2  the printer emits only the FIRST WORD of the verdict blob, so the rendered list reads
      `cycle 133 — SHIPPED`, `cycle 152 — DO`, `cycle 156 — BUILT` under a heading that says
      these are the runs where a preregistered gate returned `INVALID__*`.

D2 makes D1 visible to any reader in ten seconds, in the one document whose entire purpose is to
prove this lab does not flatter itself. That is why this audit exists.

The sharpest single specimen: **cycle 156 is the cycle that BUILT the ledger**, and it is counted
as a machinery refusal because its verdict prose quotes the ledger's own negatives count. The
ledger counts itself as a loss.

Note the convergence, because it is not a coincidence: this is the same mention-versus-use defect
that `RECON_oath_external_reach_2026_08_26.md` documents in the OATH verifier, in a second,
independently written instrument. Two tools in this repository, built months apart for unrelated
jobs, both read a line and call it a claim.

This audit CHANGES NOTHING. It measures, so a successor can decide with numbers in hand.

  python papers/ledger_classifier_audit.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "ledger_classifier_audit.json"

# The shipped classifier, quoted verbatim from build_ledger.py so this audit cannot drift from it.
NEG_SHIPPED = (r'INVALID__|NOT_|NO_|BLIND|WRONG|REFUS|_FAIL|DO NOT SHIP|RECALL|QUARANTIN'
               r'|OVERTURN|INCOMPLETE')

# A verdict HEAD: the leading verdict token, which is what the section headings actually describe
# ("cycles where a preregistered gate returned INVALID__*"). Everything after the first sentence
# break is commentary about the run, not the run's verdict.
_HEAD = re.compile(r"^[^.;:\n]{0,120}")


def head(verdict: str) -> str:
    return _HEAD.match(verdict.strip()).group(0).strip() if verdict.strip() else ""


def main() -> int:
    cy = [json.loads(x) for x in
          (HERE / "autopilot" / "CYCLE_LOG.jsonl").read_text(encoding="utf-8").splitlines()
          if x.strip()]

    rows = []
    for c in cy:
        v = c.get("verdict", "") or ""
        h = head(v)
        rows.append({
            "cycle": c.get("cycle"),
            "date": c.get("date"),
            "head": h[:120],
            "first_word": v.strip().split()[0] if v.strip() else "",
            # shipped behaviour
            "shipped_negative": bool(re.search(NEG_SHIPPED, v)),
            "shipped_invalid": "INVALID__" in v,
            # head-scoped behaviour: does the VERDICT ITSELF say it, rather than the commentary
            "head_negative": bool(re.search(NEG_SHIPPED, h)),
            "head_invalid": h.startswith("INVALID__") or bool(re.search(r"\bINVALID__", h)),
        })

    ship_neg = [r for r in rows if r["shipped_negative"]]
    ship_inv = [r for r in rows if r["shipped_invalid"]]
    head_neg = [r for r in rows if r["head_negative"]]
    head_inv = [r for r in rows if r["head_invalid"]]

    # The entries a reader sees in the rendered "Every run the machinery refused" list whose
    # printed token is not an INVALID__ verdict at all.
    rendered_nonsense = [r for r in ship_inv if not r["first_word"].startswith("INVALID__")]
    # Counted as a machinery refusal although the verdict itself never returned one.
    invalid_false_positives = [r for r in ship_inv if not r["head_invalid"]]
    # Counted as a negative although the verdict head says nothing negative.
    negative_only_in_commentary = [r for r in ship_neg if not r["head_negative"]]

    payload = {
        "audit": "how build_ledger.py classifies a cycle, and what it gets wrong",
        "status": "AUDIT. Measures; changes nothing. Inputs to a successor's decision.",
        "root_cause": "the classifier matches substrings against a FREE-PROSE verdict blob, so it "
                      "cannot distinguish a cycle that RETURNED a verdict from one that MENTIONS "
                      "one — the same mention-versus-use defect RECON_oath_external_reach_2026_08_26 "
                      "documents in the OATH verifier, in a second independent instrument",
        "cycles_logged": len(cy),
        "shipped": {
            "negatives": len(ship_neg),
            "machinery_refusals": len(ship_inv),
            "note": "these are the numbers papers/LEDGER.md publishes today",
        },
        "head_scoped": {
            "negatives": len(head_neg),
            "machinery_refusals": len(head_inv),
            "note": "same regexes, applied to the verdict HEAD rather than the whole blob",
        },
        "rendered_nonsense_entries": {
            "n": len(rendered_nonsense),
            "detail": [{"cycle": r["cycle"], "printed_as": r["first_word"], "head": r["head"]}
                       for r in rendered_nonsense],
            "why": "the printer emits the verdict's FIRST WORD, so these render under the heading "
                   "'cycles where a preregistered gate returned INVALID__*' as words like SHIPPED, "
                   "PRODUCT, DO, REWRITTEN, BUILT, TWO",
        },
        "machinery_refusal_false_positives": {
            "n": len(invalid_false_positives),
            "detail": [{"cycle": r["cycle"], "head": r["head"]} for r in invalid_false_positives],
            "why": "the verdict never returned INVALID__; its prose discusses one",
        },
        "negative_only_in_commentary": {
            "n": len(negative_only_in_commentary),
            "detail": [{"cycle": r["cycle"], "first_word": r["first_word"], "head": r["head"]}
                       for r in negative_only_in_commentary],
            "why": "counted toward the flagship negatives ratio although the verdict head carries "
                   "no negative token. NOT automatically a miscount — a cycle can end badly and "
                   "say so in its second clause — which is exactly why this is published as an "
                   "audit for adjudication rather than applied as a correction",
        },
        "all_cycles": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"cycles {len(cy)}")
    print(f"  negatives          shipped {len(ship_neg):>3}   head-scoped {len(head_neg):>3}")
    print(f"  machinery refusals shipped {len(ship_inv):>3}   head-scoped {len(head_inv):>3}")
    print(f"  rendered nonsense entries: {len(rendered_nonsense)}")
    for r in rendered_nonsense:
        print(f"     cycle {r['cycle']:>3} prints as {r['first_word']!r}")
    print(f"  refusal false positives:   {len(invalid_false_positives)}")
    print(f"  negatives only in commentary: {len(negative_only_in_commentary)} -> {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
