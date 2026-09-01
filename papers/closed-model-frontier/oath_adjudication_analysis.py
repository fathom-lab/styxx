"""Derived analysis over the blind adjudication, and the diagnosis of its failed sanity gate.

The protocol pre-committed a sanity condition on the VERIFIED arm: *"If this is low the panel and
the instrument disagree about what a claim IS, and every other number here is suspect."* It came
back at 0.4933. This script does not argue that condition away — it records that the gate fired,
and then measures WHY, because the reason turned out to be the cycle's actual finding.

  python papers/closed-model-frontier/oath_adjudication_analysis.py
"""
from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from oath_mention_use_census import _coincident  # noqa: E402

RESULT = HERE / "oath_adjudication_result.json"
LEDGER = HERE / "oath_external_corpus_ledger.jsonl"
OUT = HERE / "oath_adjudication_analysis.json"

SANITY_BAR = 0.80          # what a passing verified-arm agreement would have looked like
DOMINANT = "hopit-ai/Moda"


def main() -> int:
    res = json.loads(RESULT.read_text(encoding="utf-8"))
    D = res["per_arm_detail"]
    rows = [json.loads(ln) for ln in LEDGER.read_text(encoding="utf-8").splitlines() if ln.strip()]

    acc = D["UNGROUNDED"]
    fa = [r for r in acc if r["verdict"] == "NOT_A_CLAIM"]
    out_dom = [r for r in acc if r["repo"] != DOMINANT]
    in_dom = [r for r in acc if r["repo"] == DOMINANT]

    # --- the failed gate, diagnosed -------------------------------------------------------------
    ver = D["VERIFIED"]
    ct = collections.Counter()
    for r in ver:
        ct[(_coincident(r["receipt_ref"]), r["verdict"])] += 1
    nominal_n = ct[(False, "CLAIM")] + ct[(False, "NOT_A_CLAIM")]
    coin_n = ct[(True, "CLAIM")] + ct[(True, "NOT_A_CLAIM")]

    # --- concentration ---------------------------------------------------------------------------
    per = collections.defaultdict(collections.Counter)
    for r in rows:
        per[r["repo"]][r["status"]] += 1
    shares = [c["UNGROUNDED"] / sum(c.values()) for c in per.values() if sum(c.values())]
    by_repo_acc = collections.Counter(r["repo"] for r in rows if r["status"] == "UNGROUNDED")

    payload = {
        "analysis": "derived from the blind three-seat adjudication; no new judgement was made",
        "status": "the pre-committed VERIFIED-arm sanity gate FAILED; see sanity_gate below",

        "sanity_gate": {
            "pre_committed_condition": (
                "If the share of VERIFIED tokens the panel agrees are claims is low, the panel and "
                "the instrument disagree about what a claim IS, and every other number here is "
                "suspect."),
            "observed": res["verified_arm_sanity"]["rate"],
            "n": res["verified_arm_sanity"]["n"],
            "a_passing_value_would_have_been_at_least": SANITY_BAR,
            "verdict": "FAILED",
            "consequence_honoured": (
                "The false-accusation and miss rates below are reported, and are NOT promoted to "
                "headline findings, because the frozen protocol said they would be suspect if this "
                "gate failed. Reinterpreting the row after seeing it is the move the protocol's "
                "opening paragraph forbids."),
        },

        "why_the_gate_failed": {
            "coincident_bindings": {"n": coin_n, "panel_calls_claim": ct[(True, "CLAIM")]},
            "nominal_bindings": {"n": nominal_n, "panel_calls_claim": ct[(False, "CLAIM")],
                                 "share": round(ct[(False, "CLAIM")] / nominal_n, 4)
                                 if nominal_n else None},
            "reading": (
                "The v0.8 coincidence channel explains only the coincident row. The bulk of the "
                "rejections are NOMINALLY bound: the token matched a receipt leaf whose path its "
                "own line names, and the panel still says it is not a claim. Inspected, they are "
                "CLI flags (`--episodes 20`), link labels (`[Blog 6]`), hardware specs ('Apple M4 "
                "(10 logical cores'), range notation ('[0, 1]'), configuration ('256 with K=8') "
                "and a numeral inside an <img> tag. The panel is not being strict; the instrument "
                "is swearing oaths to non-claims."),
            "what_this_means": (
                "The mention/use defect is not confined to accusations. It is in the VERIFIED "
                "channel too, and a false verification is worse than a false accusation because "
                "the affirmative attestation is the entire product. No previous cycle measured "
                "this: the pilot looked only at what the verifier accused."),
        },

        "accusations": {
            "n": len(acc),
            "false_accusation_rate": round(len(fa) / len(acc), 4),
            "share_on_genuine_claims": round(1 - len(fa) / len(acc), 4),
            "within_dominant_repo": {
                "repo": DOMINANT, "n": len(in_dom),
                "false_accusation_rate": round(
                    sum(1 for r in in_dom if r["verdict"] == "NOT_A_CLAIM") / len(in_dom), 4)
                if in_dom else None},
            "excluding_dominant_repo": {
                "n": len(out_dom),
                "false_accusation_rate": round(
                    sum(1 for r in out_dom if r["verdict"] == "NOT_A_CLAIM") / len(out_dom), 4)
                if out_dom else None},
            "pilot_comparison": {
                "pilot_n": 13, "pilot_false_accusation_rate": 1.0,
                "reading": ("The pilot reported that not one of its thirteen accusations was a "
                            "catch. At n=366 across seven query families the rate is far below "
                            "1.0, so the pilot's headline does not survive and is withdrawn.")},
        },

        "misses": {
            "n": res["miss_rate"]["n"],
            "miss_rate": res["miss_rate"]["rate"],
            "reading": ("Of tokens the verifier declined to check, this share are checkable "
                        "claims. Abstention on external text is therefore substantially blindness "
                        "rather than calibrated restraint, and the pilot's flattering abstain "
                        "share cannot be read as caution."),
        },

        "concentration": {
            "accusations_total": sum(by_repo_acc.values()),
            "repos_with_any_accusation": len(by_repo_acc),
            "dominant_repo_share": round(by_repo_acc[DOMINANT] / sum(by_repo_acc.values()), 4),
            "top3_share": round(sum(n for _, n in by_repo_acc.most_common(3))
                                / sum(by_repo_acc.values()), 4),
            "median_per_repo_accusation_share": round(st.median(shares), 4),
            "repos_with_zero_accusations": sum(1 for v in shares if v == 0),
            "repos_scored": len(shares),
            "reading": ("The pooled accusation rate is a statement about a handful of "
                        "claim-dense documents. The median external repository draws none."),
        },

        "panel_independence": {
            "unanimity_share": res["agreement"]["unanimity_share"],
            "split_panels": res["agreement"]["split"],
            "unsure_votes": sum(1 for arm in D for r in D[arm] for v in r["votes"]
                                if v == "UNSURE"),
            "total_votes": sum(len(r["votes"]) for arm in D for r in D[arm]),
            "reading": ("Three seats of one model family agreed on 98% of items. That is the "
                        "correlated-error ceiling the protocol disclosed in advance, not evidence "
                        "of correctness. A human re-adjudication of the retained packets is the "
                        "only thing that would lift it."),
        },
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"sanity gate      : {payload['sanity_gate']['observed']} -> FAILED")
    print(f"false accusation : {payload['accusations']['false_accusation_rate']}  "
          f"(pilot claimed 1.0 at n=13)")
    print(f"miss rate        : {payload['misses']['miss_rate']}")
    print(f"median repo acc  : {payload['concentration']['median_per_repo_accusation_share']}")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
