"""Fold OBLIGATE-1's blind panel: precision first, because obligation manufactures accusations.

Implements PREREG_obligate1_2026_08_31.md. Gates: G-O1P (positive-arm CLAIM share >= 0.70),
G-O1NULL (beats obligate-everything on the same adjudications), G-O1R (population-weighted
recall >= 0.20). Decoys gate seats at >= 0.80; UNSURE and NO-MAJORITY are excluded from every
numerator and denominator and their counts are reported.

  python papers/closed-model-frontier/obligate1_fold.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEALED = Path(os.environ.get("STYXX_SEALED_DIR", r"C:\Users\heyzo\clawd\styxx-sealed"))
OUT = HERE / "obligate1_result.json"

P_BAR = 0.70
R_BAR = 0.20


def main() -> int:
    packets = json.loads((HERE / "obligate1_packets.json").read_text(encoding="utf-8"))
    seats = json.loads((HERE / "obligate1_seat_outputs.json").read_text(encoding="utf-8"))

    kb = (SEALED / "obligate1_key.json").read_bytes()
    salt = (SEALED / "agent_claim_key_salt.txt").read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(kb + salt.encode("utf-8")).hexdigest()
    committed = (HERE / "obligate1_key.sha256").read_text(encoding="utf-8").strip()
    assert digest == committed, "obligate1 key hash does not match the build commit"
    key = json.loads(kb.decode("utf-8"))

    ids_by_packet = {p["packet"]: [r["id"] for r in p["rows"]] for p in packets["packets"]}
    rows_ctx = {r["id"]: r for p in packets["packets"] for r in p["rows"]}

    labels = {}
    for name, payload in seats["seats"].items():
        pk = int(name.split("-")[0][-1])
        want = set(ids_by_packet[pk])
        got = [e["id"] for e in payload["labels"]]
        assert set(got) == want and len(got) == len(set(got)), f"{name}: id mismatch"
        labels[name] = {e["id"]: e["label"] for e in payload["labels"]}

    # ── seat validity on the 16 decoys (UNSURE counts as wrong on a decoy) ───────
    validity = {}
    for name, lab in labels.items():
        pk = int(name.split("-")[0][-1])
        ok = n = 0
        for i in ids_by_packet[pk]:
            k = key[i]
            if k.get("arm") != "decoy":
                continue
            n += 1
            ok += lab[i] == k["truth"]
        validity[name] = {"correct": ok, "n": n, "score": round(ok / n, 4),
                         "passes": ok / n >= 0.80}
    all_pass = all(v["passes"] for v in validity.values())

    # ── majority verdicts ────────────────────────────────────────────────────────
    verdicts, unanimous, no_majority, unsure = {}, 0, 0, 0
    for pk, ids in ids_by_packet.items():
        names = [f"p{pk}-seat{s}" for s in (1, 2, 3)]
        for i in ids:
            votes = [labels[n][i] for n in names]
            top, cnt = Counter(votes).most_common(1)[0]
            if cnt == 1:
                verdicts[i] = "NO-MAJORITY"
                no_majority += 1
                continue
            verdicts[i] = top
            unanimous += cnt == 3
            unsure += top == "UNSURE"

    def arm(name):
        ids = [i for i in verdicts if key[i].get("arm") == name]
        valid = [i for i in ids if verdicts[i] in ("CLAIM", "NOT_A_CLAIM")]
        c = sum(1 for i in valid if verdicts[i] == "CLAIM")
        return {"total": len(ids), "valid": len(valid), "claims": c,
                "excluded_unsure_or_nomaj": len(ids) - len(valid),
                "claim_share": round(c / len(valid), 4) if valid else None}

    pos, neg = arm("positive"), arm("negative")

    # population weights for recall (stratified sample -> corpus estimate)
    P = packets["population"]
    w_pos = P["obligate1_positive"] / packets["sample"]["positive"]
    w_neg = P["obligate1_negative"] / packets["sample"]["negative"]
    caught = w_pos * pos["claims"]
    missed = w_neg * neg["claims"]
    recall = round(caught / (caught + missed), 4) if (caught + missed) else None

    # the obligate-everything null, on the same adjudications
    null_prec = (round((pos["claims"] + neg["claims"]) / (pos["valid"] + neg["valid"]), 4)
                 if (pos["valid"] + neg["valid"]) else None)

    gates = {"G-V": {"all_seats_pass": all_pass,
                     "verdict": "PASS" if all_pass else "FAIL"}}
    prec = pos["claim_share"]
    counts = (f"OBLIGATE-1 precision = {pos['claims']}/{pos['valid']} ({prec}) vs bar "
              f"{P_BAR}; obligate-everything null = "
              f"{pos['claims'] + neg['claims']}/{pos['valid'] + neg['valid']} ({null_prec}); "
              f"weighted recall = {recall} vs bar {R_BAR}; no significance is claimed at "
              f"these n")
    gates["G-O1P"] = {"precision": prec, "bar": P_BAR,
                      "verdict": "PASS" if (prec is not None and prec >= P_BAR) else "FAIL",
                      "mandatory_counts_statement": counts}
    if not (prec is not None and prec >= P_BAR):
        gates["G-O1P"]["mandatory_failure_statement"] = \
            "the structural obligation clause does not survive held-out adjudication"
    gates["G-O1NULL"] = {"null_precision": null_prec,
                         "verdict": "PASS" if (prec is not None and null_prec is not None
                                               and prec > null_prec) else "FAIL"}
    gates["G-O1R"] = {"weighted_recall": recall, "bar": R_BAR,
                      "raw_arm_claims": {"positive": pos["claims"], "negative": neg["claims"]},
                      "verdict": "PASS" if (recall is not None and recall >= R_BAR)
                      else "FAIL"}

    # every positive the panel says is NOT a claim — each one a would-be false accusation
    fps = [{"id": i, "token": key[i].get("token"), "doc": key[i].get("doc"),
            "context": rows_ctx[i]["context"][:130]}
           for i in verdicts
           if key[i].get("arm") == "positive" and verdicts[i] == "NOT_A_CLAIM"]

    payload = {
        "prereg": "PREREG_obligate1_2026_08_31.md",
        "key_hash_verified": digest,
        "protocol_notes": ("seats saw each token with its line context capped near 140 "
                           "characters; the packets file stores up to 240"),
        "seat_validity": validity,
        "verdicts": {"total": len(verdicts),
                     "unanimity_rate": round(unanimous / len(verdicts), 4),
                     "no_majority": no_majority, "majority_unsure": unsure},
        "arms": {"positive": pos, "negative": neg},
        "population_weights": {"w_pos": round(w_pos, 4), "w_neg": round(w_neg, 4)},
        "gates": gates,
        "positive_arm_not_claims_every_one": fps,
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    for n, v in sorted(validity.items()):
        print(f"  {n}: {v['correct']}/{v['n']} decoys ({'PASS' if v['passes'] else 'FAIL'})")
    print(f"\nunanimity {payload['verdicts']['unanimity_rate']}  "
          f"no-majority {no_majority}  majority-UNSURE {unsure}")
    print(f"POSITIVE arm: {pos['claims']}/{pos['valid']} CLAIM -> precision {prec}")
    print(f"NEGATIVE arm: {neg['claims']}/{neg['valid']} CLAIM")
    print(f"weighted recall {recall}   null precision {null_prec}")
    for g, d in gates.items():
        print(f"{g}: {d['verdict']}")
    print("\n" + counts)
    print(f"\npositives adjudicated NOT_A_CLAIM (would-be false accusations): {len(fps)}")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
