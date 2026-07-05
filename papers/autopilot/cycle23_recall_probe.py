"""cycle-23 CLOSED_NEGATIVE probe — reproduce the trigger-recall extension's measured
boundary WITHOUT shipping it.

The extension (PREREG_oath_v04_trigger_recall_2026_07_04.md) widened `styxx.certify._TRIGGERS`
with the correlation/similarity register. It MISSED bar G3 (introduced certifier-artifact
UNGROUNDED) and was reverted, so `styxx/certify.py` on disk is the shipped v0.3 verifier.
This script re-applies the exact one-line change IN MEMORY (monkeypatch), so the negative is
reproducible from the committed tree:

  1. battery: mutate every VERIFIED token of the 13 cycle-18 certs (seed 1, the frozen
     mutate_token scheme) under the extended triggers -> caught / false_verify / abstain_degrade
     (bars G4 catch>=116, G2 false_verify<=26).
  2. G3 hand-check: certify the 13 CLEAN docs under the extended triggers and classify every
     clean UNGROUNDED as REAL (a doc number with no grounding summary receipt) or ARTIFACT
     (a non-measurement number obligated only because a register word shares its line).

Usage:  python papers/autopilot/cycle23_recall_probe.py
Writes: cycle23_recall_battery_result.json, cycle23_g3_handcheck_result.json
"""
from __future__ import annotations
import hashlib, json, random, re, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
import styxx.certify as C  # noqa
from styxx.certify import receipt_values, _match  # noqa
sys.path.insert(0, str(ROOT / "papers" / "closed-model-frontier"))
from validate_oath_v0 import mutate_token  # noqa

# --- the EXACT candidate change: shipped v0.3 _TRIGGERS + the recall register (frozen in the prereg)
_EXTENDED_TRIGGERS = re.compile(
    r"\b(aurocs?|aucs?|margins?|cis?\b|boot(strap)?|perm(utation)?(_p\d+)?|p9\d|recall|precision|"
    r"fpr|fnr|accuracy|rate|median|mean|elevation|floor|delta|n\s*=|n_held|n_caved|held|caved|"
    r"gated|dropped|grounded|sycophancy|deception|surface|lens|firewall|collapse|wilson|"
    r"concordance|stability|score[sd]?|"
    r"rsa|rdm|spearman|pearson|correlations?|rho|consistency|reliability|ceiling|agreement|"
    r"convergence|drift|entropy|similarity|variance)\b", re.I)

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
    C._TRIGGERS = _EXTENDED_TRIGGERS   # apply the candidate extension in memory only

    rng = random.Random(1)
    per_doc, fv_rows = {}, []
    tot = dict(n_mutants=0, caught=0, false_verify=0, abstain_degrade=0, dropped=0, verdict_flips=0)
    # G3 classification of clean UNGROUNDED
    g3 = dict(total=0, real_absent=0, real_bulk_only=0, artifact=0)
    g3_rows = []
    clean_ungrounded_total = 0

    for rel in CERTS:
        cp = ROOT / rel
        shipped = json.loads(cp.read_text(encoding="utf-8"))
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        recs = resolve(cp, shipped)

        # summary + bulk leaves for the G3 REAL/ARTIFACT decision
        summ, bulk = [], []
        for rp in recs:
            j = json.loads(rp.read_text(encoding="utf-8"))
            summ += [(rp.name, p, v) for p, v in receipt_values(j, include_bulk=False)]
            bulk += [(rp.name, p, v) for p, v in receipt_values(j, include_bulk=True)]

        clean = C.certify_doc(doc, recs)
        clean_ungrounded_total += clean["counts"]["UNGROUNDED"]
        # --- G3: classify each clean UNGROUNDED ---
        for e in [l for l in clean["ledger"] if l["status"] == "UNGROUNDED"]:
            v, dec = e["value"], e["decimals"]
            summ_match = any(_match(v, dec, rv, True) for _, _, rv in summ)
            bulk_match = any(_match(v, dec, rv, True) for _, _, rv in bulk)
            g3["total"] += 1
            if summ_match:
                # a SUMMARY leaf value-matches yet the token is UNGROUNDED -> count-binding rejected
                # a coincidental integer collision (the value is a DIFFERENT quantity), OR the token
                # is a non-measurement number (spec/ordinal) obligated only by a register word.
                kind = "artifact"; g3["artifact"] += 1
            elif bulk_match:
                kind = "real_bulk_only"; g3["real_bulk_only"] += 1
            else:
                kind = "real_absent"; g3["real_absent"] += 1
            g3_rows.append({"doc": doc.name, "line": e["line"], "token": e["token"],
                            "kind": kind, "context": e["context"][:120]})

        # --- battery (identical to mutant_battery.py, under the extended triggers) ---
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
                mc = C.certify_doc(tmp, recs)
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
        "what": "cycle-23 trigger-recall extension battery (candidate reverted; monkeypatched in memory)",
        "prereg": "papers/closed-model-frontier/PREREG_oath_v04_trigger_recall_2026_07_04.md",
        "seed": 1, "n_docs": len(CERTS),
        **tot,
        "catch_rate": round(tot["caught"] / n, 3),
        "false_verify_rate": round(tot["false_verify"] / n, 3),
        "abstain_degrade_rate": round(tot["abstain_degrade"] / n, 3),
        "clean_ungrounded_total": clean_ungrounded_total,
        "baseline_v03": {"n_mutants": 269, "caught": 58, "false_verify": 26, "abstain_degrade": 182},
        "per_doc": per_doc, "false_verify_rows": fv_rows,
    }
    (HERE / "cycle23_recall_battery_result.json").write_text(json.dumps(battery, indent=2) + "\n",
                                                             encoding="utf-8")
    g3out = {
        "what": "cycle-23 G3 hand-check: clean UNGROUNDED under the trigger-recall extension, REAL vs ARTIFACT",
        "prereg": "papers/closed-model-frontier/PREREG_oath_v04_trigger_recall_2026_07_04.md",
        "clean_ungrounded_total": g3["total"],
        "real_absent": g3["real_absent"], "real_bulk_only": g3["real_bulk_only"],
        "artifact": g3["artifact"], "rows": g3_rows,
    }
    (HERE / "cycle23_g3_handcheck_result.json").write_text(json.dumps(g3out, indent=2) + "\n",
                                                           encoding="utf-8")
    print(f"BATTERY caught={tot['caught']}/{tot['n_mutants']} "
          f"false_verify={tot['false_verify']} abstain_degrade={tot['abstain_degrade']} "
          f"dropped={tot['dropped']}")
    print(f"clean UNGROUNDED total = {g3['total']}  "
          f"(real_absent={g3['real_absent']} real_bulk_only={g3['real_bulk_only']} "
          f"ARTIFACT={g3['artifact']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
