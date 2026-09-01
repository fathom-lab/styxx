"""Fold Stage 2: does STRUCT-1 beat the verb-stem null on blind ground truth?

Implements PREREG_claim_detector_2026_08_30 as amended by
AMENDMENT_claim_detector_stage2_2026_08_31 — nothing else. Seat validity on the 24 gating
decoys (the 6 mention-vs-use decoys REPORT, never gate), majority verdicts with NO-MAJORITY
exclusion, the two gates with their frozen thresholds and mandated sentences, and the floors.

  python papers/closed-model-frontier/stage2_fold.py
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
OUT = HERE / "stage2_result.json"

BAR = 0.2061                 # N2's weighted precision, frozen at the baseline
FLOOR = 30                   # per arm, per the amendment
GATING = {"gating-A", "gating-B", "structural-C"}
FAIL_SENTENCE = ("the structural detector adds no precision over the verb-stem null "
                 "at this sample size")


def main() -> int:
    packets = json.loads((HERE / "stage2_packets.json").read_text(encoding="utf-8"))
    seats = json.loads((HERE / "stage2_seat_outputs.json").read_text(encoding="utf-8"))

    key_bytes = (SEALED / "stage2_key.json").read_bytes()
    salt = (SEALED / "agent_claim_key_salt.txt").read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(key_bytes + salt.encode("utf-8")).hexdigest()
    committed = (HERE / "stage2_key.sha256").read_text(encoding="utf-8").strip()
    assert digest == committed, "stage2 answer-key hash does not match the build commit"
    key = json.loads(key_bytes.decode("utf-8"))

    texts = {s["id"]: s["text"] for p in packets["packets"] for s in p["sentences"]}
    ids_by_packet = {p["packet"]: [s["id"] for s in p["sentences"]] for p in packets["packets"]}

    labels = {}
    for name, payload in seats["seats"].items():
        pk = int(name.split("-")[0][-1])
        want = set(ids_by_packet[pk])
        got = [e["id"] for e in payload["labels"]]
        assert set(got) == want and len(got) == len(set(got)) == len(want), \
            f"{name}: id set mismatch"
        labels[name] = {e["id"]: e for e in payload["labels"]}

    # ── G-V: validity on the 24 gating decoys; mvu decoys report only ────────────
    validity, mvu = {}, {}
    for name, lab in labels.items():
        pk = int(name.split("-")[0][-1])
        gok = gn = mok = mn = 0
        for i in ids_by_packet[pk]:
            k = key[i]
            if k.get("arm") != "decoy":
                continue
            correct = lab[i]["label"] == k["truth"]
            if k["class"] in GATING:
                gn += 1
                gok += correct
            else:
                mn += 1
                mok += correct
        validity[name] = {"gating_correct": gok, "gating_n": gn,
                          "score": round(gok / gn, 4) if gn else None,
                          "passes": bool(gn) and gok / gn >= 0.80}
        mvu[name] = {"correct": mok, "n": mn}
    all_pass = all(v["passes"] for v in validity.values())

    # ── majority verdicts ────────────────────────────────────────────────────────
    verdicts, unanimous, no_majority = {}, 0, []
    for pk, ids in ids_by_packet.items():
        names = [f"p{pk}-seat{s}" for s in (1, 2, 3)]
        for i in ids:
            votes = [labels[n][i]["label"] for n in names]
            top, cnt = Counter(votes).most_common(1)[0]
            if cnt == 1:
                verdicts[i] = "NO-MAJORITY"
                no_majority.append(i)
                continue
            verdicts[i] = top
            unanimous += cnt == 3

    # ── the two arms ─────────────────────────────────────────────────────────────
    def arm(name):
        ids = [i for i in verdicts if key[i].get("arm") == name]
        valid = [i for i in ids if verdicts[i] != "NO-MAJORITY"]
        a = sum(1 for i in valid if verdicts[i] == "A")
        return {"total": len(ids), "valid": len(valid), "A": a,
                "no_majority": len(ids) - len(valid),
                "A_share": round(a / len(valid), 4) if valid else None,
                "label_counts": dict(Counter(verdicts[i] for i in valid))}

    flagged, control = arm("flagged"), arm("control")
    floors_ok = flagged["valid"] >= FLOOR and control["valid"] >= FLOOR

    # ── gates ────────────────────────────────────────────────────────────────────
    gates = {"G-V": {"all_seats_pass": all_pass, "floors_ok": floors_ok,
                     "verdict": "PASS" if (all_pass and floors_ok) else "FAIL"}}

    if not floors_ok:
        gates["G-S2P"] = {"verdict": "NOT-EVALUABLE",
                          "statement": "measurement failed — insufficient valid adjudications"}
        gates["G-S2LIFT"] = dict(gates["G-S2P"])
    else:
        fs, cs = flagged["A_share"], control["A_share"]
        counts = (f"STRUCT-1 = {flagged['A']}/{flagged['valid']} (A-share {fs}) vs the "
                  f"frozen N2 bar {BAR}; control = {control['A']}/{control['valid']} "
                  f"(A-share {cs}); no significance is claimed at these n")
        gates["G-S2P"] = {"A_share": fs, "bar": BAR,
                          "verdict": "PASS" if fs > BAR else "FAIL",
                          "mandatory_counts_statement": counts}
        if fs <= BAR:
            gates["G-S2P"]["mandatory_failure_statement"] = FAIL_SENTENCE
        gates["G-S2LIFT"] = {"flagged_A_share": fs, "control_A_share": cs,
                             "lift": (round(fs - cs, 4) if (fs is not None and cs is not None)
                                      else None),
                             "verdict": "PASS" if (fs is not None and cs is not None
                                                   and fs > cs) else "FAIL",
                             "mandatory_counts_statement": counts}

    # every flagged sentence the panel did NOT call a claim — the false positives, listed
    fps = [{"id": i, "verdict": verdicts[i], "text": texts[i][:130]}
           for i in verdicts
           if key[i].get("arm") == "flagged" and verdicts[i] not in ("A", "NO-MAJORITY")]
    # every control the panel DID call a claim — STRUCT-1's misses, listed
    misses = [{"id": i, "text": texts[i][:130]}
              for i in verdicts
              if key[i].get("arm") == "control" and verdicts[i] == "A"]

    payload = {
        "stage": 2,
        "prereg": "PREREG_claim_detector_2026_08_30.md",
        "amendment": "AMENDMENT_claim_detector_stage2_2026_08_31.md",
        "struct1_version": packets["struct1_version"],
        "key_hash_verified": digest,
        "seat_validity": validity,
        "mvu_decoys_reported_not_gated": mvu,
        "verdicts": {"total": len(verdicts),
                     "unanimity_rate": round(unanimous / len(verdicts), 4),
                     "no_majority": len(no_majority)},
        "arms": {"flagged": flagged, "control": control},
        "gates": gates,
        "flagged_not_adjudicated_A_every_one": fps,
        "control_adjudicated_A_every_one": misses,
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    for n, v in sorted(validity.items()):
        print(f"  {n}: {v['gating_correct']}/{v['gating_n']} gating "
              f"({'PASS' if v['passes'] else 'FAIL'})  mvu {mvu[n]['correct']}/{mvu[n]['n']}")
    print(f"\nunanimity {payload['verdicts']['unanimity_rate']}  "
          f"no-majority {len(no_majority)}")
    print(f"FLAGGED : {flagged['A']}/{flagged['valid']} adjudicated A  "
          f"-> A-share {flagged['A_share']}   {flagged['label_counts']}")
    print(f"CONTROL : {control['A']}/{control['valid']} adjudicated A  "
          f"-> A-share {control['A_share']}   {control['label_counts']}")
    print(f"\nBAR (N2 weighted precision): {BAR}")
    for g, d in gates.items():
        print(f"{g}: {d['verdict']}")
    if gates.get("G-S2P", {}).get("mandatory_counts_statement"):
        print("\n" + gates["G-S2P"]["mandatory_counts_statement"])
    if gates.get("G-S2P", {}).get("mandatory_failure_statement"):
        print("VERBATIM: " + gates["G-S2P"]["mandatory_failure_statement"])
    print(f"\nflagged-but-not-A: {len(fps)}   control-that-were-A: {len(misses)}")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
