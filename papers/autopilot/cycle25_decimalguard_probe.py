"""cycle-25 decimal+range-guarded trigger-recall probe — battery + G3 artifact classification
under the ON-DISK verifier (this cycle SHIPS the change if all bars pass, so no monkeypatch).

  1. battery: mutate every VERIFIED token of the 13 cycle-18 certs (seed 1, frozen
     mutate_token scheme) -> caught / false_verify / abstain_degrade (G2 fv<=26, G4 catch>=116).
  2. G3: certify the 13 CLEAN docs and classify every clean UNGROUNDED as REAL (a doc number
     with no grounding summary receipt) or ARTIFACT (a summary leaf value-matches yet the token
     is UNGROUNDED -> a non-measurement number / coincidental integer collision). G3 bar = 0 artifacts.

Usage:  python papers/autopilot/cycle25_decimalguard_probe.py
Writes: cycle25_decimalguard_battery_result.json, cycle25_decimalguard_g3_result.json
"""
from __future__ import annotations
import json, random, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.certify import certify_doc, receipt_values, _match  # noqa (on-disk verifier)
sys.path.insert(0, str(ROOT / "papers" / "closed-model-frontier"))
from validate_oath_v0 import mutate_token  # noqa

CERTS = [
    "papers/ai-human-alignment/RESULT_ai_brain_2026_06_03.certificate.json",
    "papers/ai-human-alignment/RESULT_ai_brain_vision_2026_06_03.certificate.json",
    "papers/ai-human-alignment/RESULT_ai_human_2026_06_03.certificate.json",
    "papers/ai-human-alignment/en/RESULT_llm_breadth_2026_06_03.certificate.json",
    "papers/council-reference-free-truth/FINDING_council_2026_05_25.certificate.json",
    "papers/council-reference-free-truth/FINDING_fame_vs_truth_2026_05_25.certificate.json",
    "papers/dogfood-self-audit/FINDING_dogfood_2026_05_25.certificate.json",
    "papers/grounded-honesty-axis/FINDING_adversarial_curve_v3_2026_06_08.certificate.json",
    "papers/grounded-honesty-axis/FINDING_council_grounding_2026_05_28.certificate.json",
    "papers/grounded-honesty-axis/FINDING_detection_locus_gpt_2026_05_30.certificate.json",
    "papers/representational-integrity/RESULT_geometry_integrity_2026_06_03.certificate.json",
    "papers/rhythm-rescue/RESULT_rhythm_rescue_2026_06_03.certificate.json",
    "papers/sycophancy-target-gate/FINDING_promptopinion_2026_05_24.certificate.json",
]


def resolve(cp, cert):
    return [cp.parent / name for name in cert["receipts_sha256"]]


def tstatus(cert, line, token):
    for e in cert["ledger"]:
        if e["line"] == line and e["token"] == token:
            return e["status"]
    return None


def main() -> int:
    rng = random.Random(1)
    per_doc, fv_rows, g3_rows = {}, [], []
    tot = dict(n_mutants=0, caught=0, false_verify=0, abstain_degrade=0, dropped=0)
    g3 = dict(total=0, real_absent=0, real_bulk_only=0, artifact=0)
    clean_ung_total = 0

    for rel in CERTS:
        cp = ROOT / rel
        shipped = json.loads(cp.read_text(encoding="utf-8"))
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        recs = resolve(cp, shipped)

        summ, bulk = [], []
        for rp in recs:
            j = json.loads(rp.read_text(encoding="utf-8"))
            summ += [(rp.name, p, v) for p, v in receipt_values(j, include_bulk=False)]
            bulk += [(rp.name, p, v) for p, v in receipt_values(j, include_bulk=True)]

        clean = certify_doc(doc, recs)
        clean_ung_total += clean["counts"]["UNGROUNDED"]
        for e in [l for l in clean["ledger"] if l["status"] == "UNGROUNDED"]:
            v, dec = e["value"], e["decimals"]
            summ_match = any(_match(v, dec, rv, True) for _, _, rv in summ)
            bulk_match = any(_match(v, dec, rv, True) for _, _, rv in bulk)
            g3["total"] += 1
            if summ_match:
                kind = "artifact"; g3["artifact"] += 1
            elif bulk_match:
                kind = "real_bulk_only"; g3["real_bulk_only"] += 1
            else:
                kind = "real_absent"; g3["real_absent"] += 1
            g3_rows.append({"doc": doc.name, "line": e["line"], "token": e["token"],
                            "kind": kind, "context": e["context"][:120]})

        verified = [l for l in clean["ledger"] if l["status"] == "VERIFIED"]
        lines = doc.read_text(encoding="utf-8").splitlines(keepends=True)
        d = dict(caught=0, false_verify=0, abstain_degrade=0, dropped=0, n_mutants=len(verified),
                 clean_verdict=clean["verdict"], clean_ungrounded=clean["counts"]["UNGROUNDED"])
        for e in verified:
            mut = mutate_token(e["token"], rng)
            li = e["line"] - 1
            if e["token"] not in lines[li]:
                d["dropped"] += 1; tot["dropped"] += 1; continue
            ml = list(lines); ml[li] = lines[li].replace(e["token"], mut, 1)
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tf:
                tf.write("".join(ml)); tmp = Path(tf.name)
            try:
                mc = certify_doc(tmp, recs)
            finally:
                tmp.unlink(missing_ok=True)
            st = tstatus(mc, e["line"], mut)
            tot["n_mutants"] += 1
            if st == "UNGROUNDED":
                d["caught"] += 1; tot["caught"] += 1
            elif st == "VERIFIED":
                d["false_verify"] += 1; tot["false_verify"] += 1
                fv_rows.append({"doc": doc.name, "line": e["line"], "orig": e["token"], "mutant": mut})
            elif st == "ABSTAIN":
                d["abstain_degrade"] += 1; tot["abstain_degrade"] += 1
            else:
                d["dropped"] += 1; tot["dropped"] += 1
        per_doc[doc.name] = d

    n = max(tot["n_mutants"], 1)
    battery = {
        "what": "cycle-25 decimal+range-guarded trigger-recall battery (on-disk verifier)",
        "prereg": "papers/closed-model-frontier/PREREG_oath_v04_recall_rangeguard_2026_07_04.md",
        "seed": 1, "n_docs": len(CERTS), **tot,
        "catch_rate": round(tot["caught"] / n, 3),
        "false_verify_rate": round(tot["false_verify"] / n, 3),
        "abstain_degrade_rate": round(tot["abstain_degrade"] / n, 3),
        "clean_ungrounded_total": clean_ung_total,
        "baseline_v03": {"n_mutants": 269, "caught": 58, "false_verify": 26, "abstain_degrade": 182},
        "cycle23_blunt": {"caught": 128, "false_verify": 26, "abstain_degrade": 112, "artifacts": 6},
        "per_doc": per_doc, "false_verify_rows": fv_rows,
    }
    (HERE / "cycle25_decimalguard_battery_result.json").write_text(json.dumps(battery, indent=2) + "\n",
                                                                encoding="utf-8")
    g3out = {
        "what": "cycle-24 G3: clean UNGROUNDED under the range-guarded recall, REAL vs ARTIFACT",
        "prereg": "papers/closed-model-frontier/PREREG_oath_v04_recall_rangeguard_2026_07_04.md",
        "clean_ungrounded_total": g3["total"], "real_absent": g3["real_absent"],
        "real_bulk_only": g3["real_bulk_only"], "artifact": g3["artifact"], "rows": g3_rows,
    }
    (HERE / "cycle25_decimalguard_g3_result.json").write_text(json.dumps(g3out, indent=2) + "\n",
                                                            encoding="utf-8")
    print(f"BATTERY caught={tot['caught']}/{tot['n_mutants']} false_verify={tot['false_verify']} "
          f"abstain_degrade={tot['abstain_degrade']} dropped={tot['dropped']}")
    print(f"clean UNGROUNDED total={g3['total']} (real_absent={g3['real_absent']} "
          f"real_bulk_only={g3['real_bulk_only']} ARTIFACT={g3['artifact']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
