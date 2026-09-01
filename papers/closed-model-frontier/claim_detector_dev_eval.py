"""Stage 1 of PREREG_claim_detector_2026_08_30: STRUCT-1 against the DEV split.

DEV telemetry, not results. The 199 in-clear DEV labels (4 A / 33 B / 162 C) are the only
adjudications this cycle may look at while building; HELD-OUT stays sealed as salted hashes
until the frozen detector's outputs are committed.

  python papers/closed-model-frontier/claim_detector_dev_eval.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.claimdetect import detect, null_n1, null_n2                  # noqa: E402

OUT = HERE / "claim_detector_dev_eval.json"


def _scores(flags: dict, labels: dict) -> dict:
    """Precision/recall of a boolean flagger against A-labels, with raw counts."""
    tp = sum(1 for i, f in flags.items() if f and labels[i] == "A")
    fp = sum(1 for i, f in flags.items() if f and labels[i] != "A")
    fn = sum(1 for i, f in flags.items() if not f and labels[i] == "A")
    prec = round(tp / (tp + fp), 4) if (tp + fp) else None
    rec = round(tp / (tp + fn), 4) if (tp + fn) else None
    return {"flagged": tp + fp, "true_positive": tp, "false_positive": fp,
            "missed_A": fn, "precision": prec, "recall": rec}


def main() -> int:
    base = json.loads((HERE / "agent_claim_extractor_baseline.json").read_text(encoding="utf-8"))
    packets = json.loads((HERE / "agent_claim_packets.json").read_text(encoding="utf-8"))
    texts = {s["id"]: s["text"] for p in packets["packets"] for s in p["sentences"]}

    dev = base["dev_labels_in_clear"]
    labels = {i: v for i, v in dev.items() if v != "NO-MAJORITY"}
    print(f"DEV adjudications: {len(labels)}  {dict(Counter(labels.values()))}")

    readings = {i: detect(texts[i]) for i in labels}
    s1 = _scores({i: r.is_claim for i, r in readings.items()}, labels)
    n1 = _scores({i: null_n1(texts[i]) for i in labels}, labels)
    n2 = _scores({i: null_n2(texts[i]) for i in labels}, labels)

    # RESULT band vs the panel's B label — the second, separate frozen rule
    b_ids = [i for i, v in labels.items() if v == "B"]
    b_caught = sum(1 for i in b_ids if readings[i].band == "RESULT")
    b_miscalled_claim = sum(1 for i in b_ids if readings[i].is_claim)

    # Which conjunct killed each missed A — the boundary, per sentence
    misses = []
    for i, v in labels.items():
        if v == "A" and not readings[i].is_claim:
            failed = [k for k, ok in readings[i].conjuncts.items() if not ok]
            misses.append({"id": i, "text": texts[i][:110], "failed_conjuncts": failed})

    # Every false positive, listed — no summarising away
    fps = [{"id": i, "label": labels[i], "text": texts[i][:110],
            "evidence": {k: v for k, v in readings[i].evidence.items() if v}}
           for i, r in readings.items() if r.is_claim and labels[i] != "A"]

    payload = {
        "stage": "1 — DEV telemetry (NOT results; the gate is Stage 2's fresh blind panel)",
        "prereg": "PREREG_claim_detector_2026_08_30.md",
        "struct1_version": readings[next(iter(readings))].version if readings else None,
        "dev_composition": dict(Counter(labels.values())),
        "struct1": s1,
        "null_n1_path_regex": n1,
        "null_n2_verb_stems": n2,
        "bar_from_baseline_weighted": base["E3_nulls"]["N2_verb_stems"]["precision_weighted"],
        "result_band": {"panel_B_total": len(b_ids), "caught_as_RESULT": b_caught,
                        "miscalled_as_CLAIM": b_miscalled_claim},
        "missed_A_with_failing_conjunct": misses,
        "false_positives_every_one": fps,
        "note": ("DEV precision/recall may not be quoted as a result. The prereg's gate is a "
                 "fresh blind panel over STRUCT-1's own flags (Stage 2); these numbers only "
                 "tell the builder whether Stage 2 is worth running."),
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nSTRUCT-1 : flagged {s1['flagged']:>3}  TP {s1['true_positive']}  "
          f"FP {s1['false_positive']}  precision {s1['precision']}  recall {s1['recall']}")
    print(f"N1 path  : flagged {n1['flagged']:>3}  TP {n1['true_positive']}  "
          f"FP {n1['false_positive']}  precision {n1['precision']}  recall {n1['recall']}")
    print(f"N2 verbs : flagged {n2['flagged']:>3}  TP {n2['true_positive']}  "
          f"FP {n2['false_positive']}  precision {n2['precision']}  recall {n2['recall']}")
    print(f"\nRESULT band: {b_caught}/{len(b_ids)} panel-B sentences read as RESULT; "
          f"{b_miscalled_claim} miscalled CLAIM")
    if misses:
        print("\nmissed A sentences (and the conjunct that blocked each):")
        for m in misses:
            print(f"  [{m['id']}] {m['text'][:76]}")
            print(f"        blocked by: {', '.join(m['failed_conjuncts'])}")
    if fps:
        print(f"\nfalse positives ({len(fps)}), every one:")
        for f in fps[:12]:
            print(f"  [{f['id']}:{f['label']}] {f['text'][:76]}")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
