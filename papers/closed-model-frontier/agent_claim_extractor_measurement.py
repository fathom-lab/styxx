"""Fold the blind panel into the extractor-baseline measurement.

Implements PREREG_agent_claim_extractor_baseline_2026_08_30.md, nothing else: seat validity
over the 24 gating decoys (the 6 mention-vs-use decoys REPORT, never gate), majority verdicts
with NO-MAJORITY exclusion, E1/E1b/E2/E3/E4 with the frozen inverse-probability weights for
null-rule precision, gates G-V / G1 / G2 with their mandated verbatim sentences, the
distinguishability probe's frozen decision, and the three hand-adjudicated accusations'
individual panel verdicts with the retraction clause armed. DEV labels publish in the clear;
HELD-OUT labels publish only as salted hashes (the salt stays outside the repository until
the repair cycle).

  python papers/closed-model-frontier/agent_claim_extractor_measurement.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.diffgate import _PATH                                        # noqa: E402

SEALED = Path(os.environ.get("STYXX_SEALED_DIR", r"C:\Users\heyzo\clawd\styxx-sealed"))
OUT = HERE / "agent_claim_extractor_baseline.json"

N1_RX = re.compile(_PATH)
# N2: the file_touched verb stems, verbatim from the template
N2_RX = re.compile(r"\b(?:modif\w+|updat\w+|edit\w+|chang\w+|refactor\w+|"
                   r"fix(?:es|ed|ing)?\b|add\w+|extend\w+|hard\w+|wir\w+|patch\w+)", re.I)

GATING = {"gating-A", "gating-B", "structural-C"}
KNOWN_ACCUSATIONS = [
    "FINDING_behavioral_sycophancy_blackbox_2026_06_09.md: committed OATH-HELD",
    "mind_v0_validation.json -- is present in the tree with content that is not",
    "new prereg changed the corpus and LEDGER.md had not been rebuilt.",
]


def main() -> int:
    packets = json.loads((HERE / "agent_claim_packets.json").read_text(encoding="utf-8"))
    key = json.loads((HERE / "agent_claim_key.json").read_text(encoding="utf-8"))
    seats = json.loads((HERE / "agent_claim_seat_outputs.json").read_text(encoding="utf-8"))

    # key-hash re-verification, in the fold itself
    key_bytes = (SEALED / "agent_claim_key.json").read_bytes()
    salt = (SEALED / "agent_claim_key_salt.txt").read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(key_bytes + salt.encode("utf-8")).hexdigest()
    committed = (HERE / "agent_claim_key.sha256").read_text(encoding="utf-8").strip()
    assert digest == committed, "answer-key hash does not match the freeze commit"

    texts = {s["id"]: s["text"] for p in packets["packets"] for s in p["sentences"]}
    ids_by_packet = {p["packet"]: [s["id"] for s in p["sentences"]] for p in packets["packets"]}

    # ---- seat completeness -----------------------------------------------------
    labels: dict[str, dict[str, dict]] = {}          # seat -> id -> entry
    for seat_name, payload in seats["seats"].items():
        pk = int(seat_name[1])
        want = set(ids_by_packet[pk])
        got = [e["id"] for e in payload["labels"]]
        assert len(got) == len(set(got)) == len(want) and set(got) == want, \
            f"{seat_name}: id set mismatch"
        labels[seat_name] = {e["id"]: e for e in payload["labels"]}

    # ---- distinguishability probe (frozen decision) ----------------------------
    probe = {e["id"]: e["origin"] for e in seats["probe"]["calls"]}
    p1 = ids_by_packet[1]
    dec_ids = [i for i in p1 if key[i]["source"] == "decoy"]
    cor_ids = [i for i in p1 if key[i]["source"] == "corpus"]
    tpr = sum(probe.get(i) == "synthetic" for i in dec_ids) / len(dec_ids)
    fpr = sum(probe.get(i) == "synthetic" for i in cor_ids) / len(cor_ids)
    probe_j = round(tpr - fpr, 4)
    gv_void = probe_j > 0.5

    # ---- G-V: gating decoys ----------------------------------------------------
    seat_validity, mvu_report = {}, {}
    for seat_name, lab in labels.items():
        pk = int(seat_name[1])
        gate_ok = gate_n = mvu_ok = mvu_n = 0
        for i in ids_by_packet[pk]:
            k = key[i]
            if k["source"] != "decoy":
                continue
            correct = lab[i]["label"] == k["truth"]
            if k["class"] in GATING:
                gate_n += 1
                gate_ok += correct
            else:
                mvu_n += 1
                mvu_ok += correct
        seat_validity[seat_name] = {"gating_correct": gate_ok, "gating_n": gate_n,
                                    "score": round(gate_ok / gate_n, 4),
                                    "passes": gate_ok / gate_n >= 0.80}
        mvu_report[seat_name] = {"mvu_correct": mvu_ok, "mvu_n": mvu_n}
    all_seats_pass = all(v["passes"] for v in seat_validity.values())

    # ---- majority verdicts -----------------------------------------------------
    verdicts, unanimous, no_majority = {}, 0, []
    arc_flags = {}
    for pk, ids in ids_by_packet.items():
        seat_names = [f"p{pk}-seat{s}" for s in (1, 2, 3)]
        for i in ids:
            votes = [labels[s][i]["label"] for s in seat_names]
            cnt = Counter(votes)
            top, n = cnt.most_common(1)[0]
            if n == 1:
                verdicts[i] = "NO-MAJORITY"
                no_majority.append(i)
                continue
            verdicts[i] = top
            if n == 3:
                unanimous += 1
            arc_flags[i] = sum(labels[s][i]["also_result_clause"] for s in seat_names) >= 2
    unanimity_rate = round(unanimous / len(verdicts), 4)

    # ---- corpus split ----------------------------------------------------------
    corpus_ids = [i for i in verdicts if key[i]["source"] == "corpus"]
    flagged_ids = [i for i in corpus_ids if key[i]["kinds"]]
    sampled_ids = [i for i in corpus_ids if not key[i]["kinds"]]
    valid = {i for i in corpus_ids if verdicts[i] != "NO-MAJORITY"}
    fl_valid = [i for i in flagged_ids if i in valid]
    sm_valid = [i for i in sampled_ids if i in valid]
    nm_flagged = len(flagged_ids) - len(fl_valid)
    nm_sampled = len(sampled_ids) - len(sm_valid)

    # floors
    floors_ok = len(sm_valid) >= 200 and len(fl_valid) >= len(flagged_ids) / 2

    # ---- E1 / E1b --------------------------------------------------------------
    fl_diff = [i for i in fl_valid
               if any(k != "tests_pass" for k in key[i]["kinds"])]
    fl_tp_only = [i for i in fl_valid
                  if key[i]["kinds"] and all(k == "tests_pass" for k in key[i]["kinds"])]
    e1_num = sum(verdicts[i] == "A" for i in fl_diff)
    e1 = {"numerator_A": e1_num, "denominator": len(fl_diff),
          "precision": round(e1_num / len(fl_diff), 4) if fl_diff else None,
          "no_majority_excluded": nm_flagged}
    e1b_num = sum(verdicts[i] == "B" or (verdicts[i] == "A" and arc_flags.get(i))
                  for i in fl_tp_only)
    e1b = {"tests_pass_flags": len(fl_tp_only), "scored_correct": e1b_num,
           "note": "zero tests_pass flags on this corpus" if not fl_tp_only else ""}

    # ---- E4, then E2 -----------------------------------------------------------
    sm_A = sum(verdicts[i] == "A" for i in sm_valid)
    e4 = {"sampled_A": sm_A, "sampled_valid": len(sm_valid),
          "rate": round(sm_A / len(sm_valid), 4) if sm_valid else None}

    all_A = [i for i in (fl_valid + sm_valid) if verdicts[i] == "A"]
    fl_A = [i for i in fl_valid if verdicts[i] == "A"]
    unflagged_remainder = packets["counts"]["unflagged_remainder"]
    if all_A:
        e2_raw = round(len(fl_A) / len(all_A), 4)
        est_unflagged_A = unflagged_remainder * (e4["rate"] or 0)
        e2_est = round(len(fl_A) / (len(fl_A) + est_unflagged_A), 6) \
            if (len(fl_A) + est_unflagged_A) else None
    else:
        e2_raw = e2_est = None
    e2 = {"flagged_A": len(fl_A), "adjudicated_A": len(all_A),
          "recall_raw_within_sample": e2_raw,
          "recall_corpus_level_ESTIMATE_not_measurement": e2_est,
          "tests_pass_only_on_A_with_result_clause": 0}

    # ---- E3: null rules with frozen inverse-probability weights ----------------
    w_sm = unflagged_remainder / packets["counts"]["sampled"]
    nulls = {}
    for name, rx in (("N1_path_regex", N1_RX), ("N2_verb_stems", N2_RX)):
        wf = wfa = 0.0
        cf = cfa = 0
        hits_A = 0
        for i in fl_valid + sm_valid:
            w = 1.0 if key[i]["kinds"] else w_sm
            if rx.search(texts[i]):
                wf += w
                cf += 1
                if verdicts[i] == "A":
                    wfa += w
                    cfa += 1
                    hits_A += 1
        nulls[name] = {
            "flags_raw": cf, "flags_A_raw": cfa,
            "precision_weighted": round(wfa / wf, 4) if wf else None,
            "precision_raw": round(cfa / cf, 4) if cf else None,
            "recall_within_sample": round(hits_A / len(all_A), 4) if all_A else None,
            "weight_sampled": round(w_sm, 4),
        }

    # ---- gates -----------------------------------------------------------------
    best_null_name = max((n for n in nulls if nulls[n]["precision_weighted"] is not None),
                         key=lambda n: nulls[n]["precision_weighted"], default=None)
    best_null = nulls[best_null_name]["precision_weighted"] if best_null_name else None
    gates = {}
    gates["G-V"] = {"all_seats_pass": all_seats_pass, "floors_ok": floors_ok,
                    "probe_void": gv_void,
                    "verdict": "PASS" if (all_seats_pass and floors_ok and not gv_void)
                    else "FAIL"}
    g1_ok = e4["rate"] is not None and e4["rate"] >= 0.02
    gates["G1"] = {"unflagged_A_rate": e4["rate"], "threshold": 0.02,
                   "verdict": "PASS" if g1_ok else "FAIL"}
    if e1["precision"] is None or best_null is None:
        gates["G2"] = {"verdict": "NOT-EVALUABLE",
                       "statement": "G2 not evaluable — the extractor produced zero "
                                    "non-tests_pass flags on this corpus"
                       if e1["precision"] is None else
                       "G2 not evaluable — no null rule produced a defined precision"}
    else:
        g2_ok = e1["precision"] > best_null
        stmt = (f"E1 = {e1['numerator_A']}/{e1['denominator']} vs best null = "
                f"{nulls[best_null_name]['flags_A_raw']}/{nulls[best_null_name]['flags_raw']} "
                f"(weighted {best_null}); no significance is claimed at these n")
        gates["G2"] = {"E1": e1["precision"], "best_null": best_null_name,
                       "best_null_precision_weighted": best_null,
                       "verdict": "PASS" if g2_ok else "FAIL",
                       "mandatory_counts_statement": stmt}
        if not g2_ok:
            gates["G2"]["mandatory_failure_statement"] = \
                "the templates add no precision over the best null rule at this sample size"

    # instability check
    nm_rate_sampled = nm_sampled / len(sampled_ids) if sampled_ids else 0
    nm_rate_flagged = nm_flagged / len(flagged_ids) if flagged_ids else 0
    unstable = nm_rate_sampled > 0.10 or nm_rate_flagged > 0.10

    # ---- the three hand-adjudicated accusations, individually ------------------
    known = []
    for target in KNOWN_ACCUSATIONS:
        found = [i for i, t in texts.items()
                 if key[i]["source"] == "corpus" and t == target]
        for i in found:
            v = verdicts[i]
            known.append({"id": i, "text": target, "panel_verdict": v,
                          "author_said": "mention, not use (C)",
                          "retraction_fires": v == "A"})

    # ---- descriptives ----------------------------------------------------------
    vc = Counter(verdicts[i] for i in corpus_ids)
    trailer = sum(1 for i in corpus_ids if texts[i].startswith("Co-Authored-By:"))

    # ---- DEV in the clear, HELD-OUT sealed -------------------------------------
    dev_set = set(packets["split"]["dev_commits"])
    salt2 = (SEALED / "agent_claim_heldout_salt.txt").read_text(encoding="utf-8").strip()
    dev_labels, heldout_sealed = {}, {}
    for i in corpus_ids:
        sha = key[i]["sha"]
        if sha in dev_set:
            dev_labels[i] = verdicts[i]
        else:
            heldout_sealed[i] = hashlib.sha256(
                (verdicts[i] + salt2 + i).encode("utf-8")).hexdigest()

    payload = {
        "measurement": "agent-report claim extractor baseline vs blind panel ground truth",
        "prereg": "PREREG_agent_claim_extractor_baseline_2026_08_30.md",
        "key_hash_verified": digest,
        "probe": {"tpr_decoys_called_synthetic": round(tpr, 4),
                  "fpr_corpus_called_synthetic": round(fpr, 4),
                  "J": probe_j, "void_threshold": 0.5, "gv_void": gv_void},
        "seat_validity": seat_validity,
        "mvu_decoys_reported_not_gated": mvu_report,
        "verdicts": {"total": len(verdicts), "unanimity_rate": unanimity_rate,
                     "no_majority": len(no_majority),
                     "no_majority_rate_sampled": round(nm_rate_sampled, 4),
                     "no_majority_rate_flagged": round(nm_rate_flagged, 4),
                     "unstable": unstable,
                     "corpus_label_counts": dict(vc),
                     "trailer_sentences": trailer},
        "E1": e1, "E1b": e1b, "E2": e2, "E3_nulls": nulls, "E4": e4,
        "gates": gates,
        "known_accusations_panel_verdicts": known,
        "dev_labels_in_clear": dev_labels,
        "heldout_labels_salted_sha256": heldout_sealed,
        "heldout_salt_location": "outside the repository (styxx-sealed/), "
                                 "released in the repair cycle",
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    print(f"probe J = {probe_j}  (void if > 0.5: {gv_void})")
    for s, v in sorted(seat_validity.items()):
        print(f"  {s}: {v['gating_correct']}/{v['gating_n']} gating "
              f"({'PASS' if v['passes'] else 'FAIL'})  "
              f"mvu {mvu_report[s]['mvu_correct']}/{mvu_report[s]['mvu_n']}")
    print(f"verdicts: {dict(vc)}  unanimity {unanimity_rate}  "
          f"no-majority {len(no_majority)}")
    print(f"E1 precision {e1['precision']} ({e1['numerator_A']}/{e1['denominator']})")
    print(f"E2 recall raw {e2['recall_raw_within_sample']}  "
          f"corpus-level ESTIMATE {e2['recall_corpus_level_ESTIMATE_not_measurement']}")
    print(f"E4 unflagged A-rate {e4['rate']} ({e4['sampled_A']}/{e4['sampled_valid']})")
    for n, d in nulls.items():
        print(f"{n}: weighted precision {d['precision_weighted']} "
              f"(raw {d['flags_A_raw']}/{d['flags_raw']})  recall {d['recall_within_sample']}")
    for g, d in gates.items():
        print(f"{g}: {d['verdict']}")
    for k in known:
        print(f"known accusation {k['id']}: panel says {k['panel_verdict']}"
              f"{'  << RETRACTION FIRES' if k['retraction_fires'] else ''}")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
