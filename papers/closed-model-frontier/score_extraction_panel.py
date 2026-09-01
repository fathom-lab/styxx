"""Score the blind extraction panel against PREREG_extraction_ceiling_v2_2026_09_01.md.

WRITTEN AND COMMITTED BEFORE THE PANEL RETURNED. That is the point of this file existing
separately: a scorer authored after seeing the answers is a scorer with a thumb on it. Every
threshold below is copied from the frozen preregistrations, not chosen here.

Inherited from v1 (PREREG_extraction_ceiling_2026_09_01.md, frozen at 38ab585):
    P = E x A
    E = CLAIM items / scored accusations
    A = upheld items / CLAIM items          (upheld read from the committed V14 key)
    G-E2  upheld/scored must re-derive 0.16 or the run is VOID, not adjusted
    G-E3  E <= 0.23 SUPPORTED | E >= 0.40 REFUTED | between INDETERMINATE

Replaced by v2 (frozen at 308490d), because v1's gate could not be built:
    G-E1a >= 27/30 decoys correct overall
    G-E1b >= 9/15 correct on EACH side separately

Any gate failing means no E is reported. A void panel is published as a void panel.

    python papers/closed-model-frontier/score_extraction_panel.py <panel_result.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = HERE / "extraction_panel_key_SEALED.json"
V14_ANSWERS = HERE / "v14_answers.json"
OUT = HERE / "extraction_panel_result.json"

# every number below is transcribed from a frozen document
G_E1A_MIN = 27          # of 30
G_E1B_MIN = 9           # of 15, each side
G_E2_TARGET = 0.16
G_E3_SUPPORTED = 0.23
G_E3_REFUTED = 0.40


def majority(votes):
    """Modal answer, or None when no strict majority exists."""
    votes = [v for v in votes if v]
    if not votes:
        return None
    c = Counter(votes).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None
    return c[0][0]


def main(path: str) -> int:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    per_item = raw.get("per_item") or raw.get("result", {}).get("per_item") or {}
    key = json.loads(KEY.read_text(encoding="utf-8"))
    v14 = json.loads(V14_ANSWERS.read_text(encoding="utf-8"))

    verdicts, split = {}, []
    for iid, votes in per_item.items():
        verdicts[iid] = majority(list(votes.values()) if isinstance(votes, dict) else votes)
        if verdicts[iid] is None:
            split.append(iid)

    # ---------------- G-E1: reliability, two-sided ----------------
    dec = {i: v for i, v in key.items() if v["kind"] == "decoy"}
    sides = {"CLAIM": [], "NOT-A-CLAIM": []}
    for iid, meta in dec.items():
        got, want = verdicts.get(iid), meta["expected"]
        sides[want].append(got == want)
    n_claim_ok, n_claim = sum(sides["CLAIM"]), len(sides["CLAIM"])
    n_not_ok, n_not = sum(sides["NOT-A-CLAIM"]), len(sides["NOT-A-CLAIM"])
    dec_ok, dec_n = n_claim_ok + n_not_ok, n_claim + n_not
    g1a = dec_ok >= G_E1A_MIN
    g1b = n_claim_ok >= G_E1B_MIN and n_not_ok >= G_E1B_MIN

    # ---------------- G-E2: the join must re-derive 0.16 ----------------
    accs = {i: v for i, v in key.items() if v["kind"] == "accusation"}
    upheld = sum(1 for v in accs.values() if v14.get(v["v14_id"]) == "CONTRADICTED")
    p_recon = upheld / len(accs) if accs else float("nan")
    g2 = abs(p_recon - G_E2_TARGET) < 1e-9

    # ---------------- E and A, computed item by item ----------------
    scored = [i for i in accs if verdicts.get(i) in ("CLAIM", "NOT-A-CLAIM")]
    claim_ids = [i for i in scored if verdicts[i] == "CLAIM"]
    E = len(claim_ids) / len(scored) if scored else float("nan")
    upheld_in_claim = sum(1 for i in claim_ids if v14.get(accs[i]["v14_id"]) == "CONTRADICTED")
    A = upheld_in_claim / len(claim_ids) if claim_ids else float("nan")

    if E <= G_E3_SUPPORTED:
        hyp = "SUPPORTED"
    elif E >= G_E3_REFUTED:
        hyp = "REFUTED"
    else:
        hyp = "INDETERMINATE"

    # a split rate above 10% voids per the frozen "what would make us not ship"
    split_rate = len(split) / len(verdicts) if verdicts else 1.0
    g_split = split_rate <= 0.10

    if not g1a or not g1b:
        verdict = "VOID__reliability_gate_failed"
    elif not g2:
        verdict = "VOID__reconciliation_failed"
    elif not g_split:
        verdict = "VOID__panel_split"
    else:
        verdict = hyp

    res = {
        "prereg": "PREREG_extraction_ceiling_v2_2026_09_01.md (G-E1); v1 for everything else",
        "scorer_committed_before_panel_returned": True,
        "n_seats": raw.get("n_seats") or raw.get("result", {}).get("n_seats"),
        "gates": {
            "G_E1a_overall": {"observed": f"{dec_ok}/{dec_n}", "bar": f">={G_E1A_MIN}/30", "pass": bool(g1a)},
            "G_E1b_each_side": {"claim": f"{n_claim_ok}/{n_claim}",
                                "not_a_claim": f"{n_not_ok}/{n_not}",
                                "bar": f">={G_E1B_MIN}/15 each", "pass": bool(g1b)},
            "G_E2_reconciliation": {"observed": round(p_recon, 4), "target": G_E2_TARGET, "pass": bool(g2)},
            "panel_split": {"n_split": len(split), "rate": round(split_rate, 4),
                            "bar": "<=0.10", "pass": bool(g_split)},
        },
        "decomposition": ({"E": round(E, 4), "A": round(A, 4),
                           "n_scored_accusations": len(scored),
                           "n_claim": len(claim_ids),
                           "upheld_among_claim": upheld_in_claim,
                           "P_reconciled": round(p_recon, 4),
                           "G_E3_verdict": hyp}
                          if verdict not in ("VOID__reliability_gate_failed",
                                             "VOID__reconciliation_failed",
                                             "VOID__panel_split")
                          else "WITHHELD — a gate failed and no E may be reported"),
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
