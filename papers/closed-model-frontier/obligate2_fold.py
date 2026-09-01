"""Fold OBLIGATE-2's blind panel. Four frozen gates; the corpse is the bar.

Per PREREG_obligate2_2026_08_31.md: G-O2P precision >= 0.70; G-O2NULL beats both the
obligate-everything null AND OBLIGATE-1's held-out 0.4483; G-O2R weighted recall >= 0.10;
G-O2BAR the discarded bar-band must adjudicate < 0.30 CLAIM. UNSURE and NO-MAJORITY excluded
everywhere and counted.

  python papers/closed-model-frontier/obligate2_fold.py
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
OUT = HERE / "obligate2_result.json"

P_BAR = 0.70
R_BAR = 0.10
BARBAND_MAX = 0.30
CORPSE = 0.4483          # OBLIGATE-1's held-out precision, the incumbent to beat


def main() -> int:
    packets = json.loads((HERE / "obligate2_packets.json").read_text(encoding="utf-8"))
    seats = json.loads((HERE / "obligate2_seat_outputs.json").read_text(encoding="utf-8"))

    kb = (SEALED / "obligate2_key.json").read_bytes()
    salt = (SEALED / "agent_claim_key_salt.txt").read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(kb + salt.encode("utf-8")).hexdigest()
    assert digest == (HERE / "obligate2_key.sha256").read_text(encoding="utf-8").strip(), \
        "obligate2 key hash mismatch"
    key = json.loads(kb.decode("utf-8"))

    ids_by_packet = {p["packet"]: [r["id"] for r in p["rows"]] for p in packets["packets"]}
    ctxs = {r["id"]: r for p in packets["packets"] for r in p["rows"]}

    labels = {}
    for name, payload in seats["seats"].items():
        pk = int(name.split("-")[0][-1])
        got = [e["id"] for e in payload["labels"]]
        assert set(got) == set(ids_by_packet[pk]) and len(got) == len(set(got)), name
        labels[name] = {e["id"]: e["label"] for e in payload["labels"]}

    validity = {}
    for name, lab in labels.items():
        pk = int(name.split("-")[0][-1])
        ok = n = 0
        for i in ids_by_packet[pk]:
            if key[i].get("arm") != "decoy":
                continue
            n += 1
            ok += lab[i] == key[i]["truth"]
        validity[name] = {"correct": ok, "n": n, "passes": ok / n >= 0.80}
    all_pass = all(v["passes"] for v in validity.values())

    verdicts, unanimous, nomaj = {}, 0, 0
    for pk, ids in ids_by_packet.items():
        names = [f"p{pk}-seat{s}" for s in (1, 2, 3)]
        for i in ids:
            top, cnt = Counter(labels[n][i] for n in names).most_common(1)[0]
            if cnt == 1:
                verdicts[i] = "NO-MAJORITY"
                nomaj += 1
                continue
            verdicts[i] = top
            unanimous += cnt == 3

    def arm(name):
        ids = [i for i in verdicts if key[i].get("arm") == name]
        valid = [i for i in ids if verdicts[i] in ("CLAIM", "NOT_A_CLAIM")]
        c = sum(1 for i in valid if verdicts[i] == "CLAIM")
        return {"total": len(ids), "valid": len(valid), "claims": c,
                "excluded": len(ids) - len(valid),
                "claim_share": round(c / len(valid), 4) if valid else None}

    pos, bar, neg = arm("positive"), arm("barband"), arm("negative")

    P = packets["population"]
    S = packets["sample"]
    w_pos = P["obligate2_positive"] / S["positive"]
    w_bar = P["bar_band"] / S["barband"]
    w_neg = P["other_negative"] / S["negative"]
    caught = w_pos * pos["claims"]
    missed = w_bar * bar["claims"] + w_neg * neg["claims"]
    recall = round(caught / (caught + missed), 4) if (caught + missed) else None

    pooled_valid = pos["valid"] + bar["valid"] + neg["valid"]
    pooled_claims = pos["claims"] + bar["claims"] + neg["claims"]
    null_prec = round(pooled_claims / pooled_valid, 4) if pooled_valid else None

    prec = pos["claim_share"]
    counts = (f"OBLIGATE-2 precision = {pos['claims']}/{pos['valid']} ({prec}) vs bar {P_BAR}"
              f" and corpse {CORPSE}; obligate-everything null = {pooled_claims}/{pooled_valid}"
              f" ({null_prec}); bar-band CLAIM share = {bar['claims']}/{bar['valid']}"
              f" ({bar['claim_share']}) vs max {BARBAND_MAX}; weighted recall = {recall} vs bar"
              f" {R_BAR}; no significance is claimed at these n")

    gates = {"G-V": {"all_seats_pass": all_pass, "verdict": "PASS" if all_pass else "FAIL"}}
    gates["G-O2P"] = {"precision": prec, "bar": P_BAR,
                      "verdict": "PASS" if (prec is not None and prec >= P_BAR) else "FAIL",
                      "mandatory_counts_statement": counts}
    if not (prec is not None and prec >= P_BAR):
        gates["G-O2P"]["mandatory_failure_statement"] = (
            "bar-blindness was not the whole disease — the structural obligation family is "
            "now two for two dead held-out")
    gates["G-O2NULL"] = {"null_precision": null_prec, "corpse": CORPSE,
                         "verdict": "PASS" if (prec is not None and null_prec is not None
                                               and prec > null_prec and prec > CORPSE)
                         else "FAIL"}
    gates["G-O2R"] = {"weighted_recall": recall, "bar": R_BAR,
                      "verdict": "PASS" if (recall is not None and recall >= R_BAR)
                      else "FAIL"}
    gates["G-O2BAR"] = {"barband_claim_share": bar["claim_share"], "max": BARBAND_MAX,
                        "verdict": "PASS" if (bar["claim_share"] is not None
                                              and bar["claim_share"] < BARBAND_MAX)
                        else "FAIL"}

    fps = [{"id": i, "token": key[i].get("token"), "doc": key[i].get("doc"),
            "context": ctxs[i]["context"][:130]}
           for i in verdicts
           if key[i].get("arm") == "positive" and verdicts[i] == "NOT_A_CLAIM"]
    bar_claims = [{"id": i, "token": key[i].get("token"),
                   "context": ctxs[i]["context"][:130]}
                  for i in verdicts
                  if key[i].get("arm") == "barband" and verdicts[i] == "CLAIM"]

    payload = {
        "prereg": "PREREG_obligate2_2026_08_31.md",
        "key_hash_verified": digest,
        "seat_validity": validity,
        "verdicts": {"total": len(verdicts),
                     "unanimity_rate": round(unanimous / len(verdicts), 4),
                     "no_majority": nomaj},
        "arms": {"positive": pos, "barband": bar, "negative": neg},
        "population_weights": {"w_pos": round(w_pos, 4), "w_bar": round(w_bar, 4),
                               "w_neg": round(w_neg, 4)},
        "gates": gates,
        "positive_not_claims_every_one": fps,
        "barband_claims_every_one": bar_claims,
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    for n, v in sorted(validity.items()):
        print(f"  {n}: {v['correct']}/{v['n']} decoys ({'PASS' if v['passes'] else 'FAIL'})")
    print(f"\nunanimity {payload['verdicts']['unanimity_rate']}  no-majority {nomaj}")
    print(f"POSITIVE : {pos['claims']}/{pos['valid']} CLAIM -> {prec}")
    print(f"BAR-BAND : {bar['claims']}/{bar['valid']} CLAIM -> {bar['claim_share']}")
    print(f"NEGATIVE : {neg['claims']}/{neg['valid']} CLAIM -> {neg['claim_share']}")
    print(f"weighted recall {recall}  null {null_prec}  corpse {CORPSE}")
    for g, d in gates.items():
        print(f"{g}: {d['verdict']}")
    print("\n" + counts)
    print(f"\npositive false-flags {len(fps)} | bar-band claims thrown away {len(bar_claims)}")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
