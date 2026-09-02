"""The prose-claimhood census: every instrument this lab built to find claims in prose and then
measured, with the document that measured it, its headline number, and the receipt that holds it.

WHY. The brief that commissioned sworn output said "eleven prose-claimhood instruments measured,
eleven failures". The spec recorded the count as UNVERIFIED because no receipt in the tree
enumerated the eleven. `AUDIT_the_whole_program_2026_09_01.md` §8.1 lists eleven by name, in
prose. This script turns that list into a receipt: for each row it RESOLVES the JSON pointer and
checks the leaf against the number the document prints, or checks a verbatim needle against the
document's bytes where no JSON receipt exists, and refuses to write the census if any check fails.
The census document beside it swears its counts to this file, at a commit.

WHAT IT DOES NOT DECIDE. Which instruments belong on the list is a judgement (the audit's §8.1,
taken as written, plus two rows the audit did not count because they beat a null or measured a
different failure). Whether a number is "a failure" is the RESULT's verdict, not this script's.
The script checks bytes; the rows are authored.

  python papers/sworn/prose_claimhood_census.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "prose_claimhood_census.json"

CMF = "papers/closed-model-frontier/"

# in_audit: named in AUDIT §8.1 direction 1. measured: the headline numbers, each with a receipt
# pointer (kind numeric/quote) or a verbatim needle in the document (kind prose).
ROWS = [
    {"id": "diffgate-path-claim", "instrument": "diffgate path-claim accuser", "in_audit": True,
     "module": "styxx/diffgate.py", "shipped": "accusing branch deleted; observer only",
     "document": CMF + "RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md", "date": "2026-09-01",
     "n": "100 held-out accusations, 3 seats; 100 in-the-wild accusations, 3 seats",
     "measured": [
         {"metric": "held-out accusation precision", "printed": "0.16", "receipt": CMF + "v14_adjudication.json", "pointer": "/precision"},
         {"metric": "in-the-wild accusation precision (AIDev)", "printed": "0.23", "receipt": CMF + "external1_adjudication.json", "pointer": "/precision"},
     ]},
    {"id": "oath-obligation-external", "instrument": "OATH obligation predicate on external text", "in_audit": True,
     "module": "styxx/certify.py", "shipped": "shipped; false-accusation rate published as an upper bound",
     "document": CMF + "RESULT_oath_external_corpus_2026_08_27.md", "date": "2026-08-27",
     "n": "366 accusations; 75 verified tokens; 3 seats of one model family",
     "measured": [
         {"metric": "false-accusation rate (upper bound)", "printed": "0.2596", "receipt": CMF + "oath_adjudication_result.json", "pointer": "/false_accusation_rate/rate"},
         {"metric": "share of VERIFIED tokens the panel called claims", "printed": "0.4933", "receipt": CMF + "oath_adjudication_result.json", "pointer": "/verified_arm_sanity/rate"},
     ]},
    {"id": "agent-report-extractor", "instrument": "agent-report claim extractor (diffgate templates on agent prose)", "in_audit": True,
     "module": "styxx/diffgate.py", "shipped": "shipped; measured blind",
     "document": CMF + "RESULT_agent_claim_extractor_baseline_2026_08_30.md", "date": "2026-08-30",
     "n": "300 adjudicated sentences, 9 seat-runs",
     "measured": [
         {"metric": "precision on the claims it flagged", "printed": "0.3333", "receipt": CMF + "agent_claim_extractor_baseline.json", "pointer": "/E1/precision"},
         {"metric": "corpus-level recall (an estimate, not a measurement)", "printed": "0.0336", "receipt": CMF + "agent_claim_extractor_baseline.json", "pointer": "/E2/recall_corpus_level_ESTIMATE_not_measurement"},
     ]},
    {"id": "agent-audit-extract-claims", "instrument": "styxx.agent_audit extract_claims", "in_audit": True,
     "module": "styxx/agent_audit.py", "shipped": "shipped",
     "document": "papers/agent-self-audit/FINDING_dogfood_binding_stack_2026_07_04.md", "date": "2026-07-04",
     "n": "one real seven-sentence agent report; no panel",
     "measured": [
         {"metric": "claims extracted from a real agent report", "printed": "0", "receipt": None,
          "needle": "read **0 claims from a 7-sentence real agent session report**"},
     ]},
    {"id": "ledger-refusal-classifier", "instrument": "the ledger's refusal classifier", "in_audit": True,
     "module": "papers/build_ledger.py (repaired by papers/ledger_verdicts.py)", "shipped": "not in the package",
     "document": "papers/SYNTHESIS_mention_and_use_2026_08_26.md", "date": "2026-08-26",
     "n": "163 cycles, exhaustively re-classified; no panel",
     "measured": [
         {"metric": "a token printed as a machinery refusal", "printed": "SHIPPED", "receipt": "papers/ledger_classifier_audit.json", "pointer": "/rendered_nonsense_entries/detail/2/printed_as", "kind": "quote"},
         {"metric": "rendered-nonsense refusal entries", "printed": "9", "receipt": "papers/ledger_classifier_audit.json", "pointer": "/rendered_nonsense_entries/n"},
     ]},
    {"id": "deception-nli", "instrument": "the deception NLI (a quoted false premise read as asserted)", "in_audit": True,
     "module": "styxx/guardrail/deception_v2.py", "shipped": "NLI scorer shipped; the prompt-aware fix was reverted",
     "document": "papers/deception-correction-gate/FINDING_2026_05_25.md", "date": "2026-05-25",
     "n": "80-item holdout, two API models; no panel",
     "measured": [
         {"metric": "corrections still flagged after the fix", "printed": "0.17", "receipt": "papers/deception-correction-gate/results.json", "pointer": "/fixed/correction_fire"},
     ]},
    {"id": "capstone-nli", "instrument": "the capstone NLI (a bare question read as an assertion)", "in_audit": True,
     "module": "styxx/guardrail/deception_v2.py", "shipped": "integration reverted",
     "document": "papers/decoupled-diagonal-capstone/FINDING_2026_05_25.md", "date": "2026-05-25",
     "n": "88-item joint holdout; the defect measured on 50 factual triples",
     "measured": [
         {"metric": "joint correction fire rate", "printed": "0.15", "receipt": "papers/decoupled-diagonal-capstone/results.json", "pointer": "/joint/correction_fire"},
         {"metric": "genuine lies suppressed as corrections", "printed": "2 of 50", "receipt": None,
          "needle": "Measured scope: **2 of 50** factual"},
     ]},
    {"id": "prompt-opinion-detector", "instrument": "the prompt-opinion detector", "in_audit": True,
     "module": "papers/sycophancy-target-gate/target_gate_c4.py", "shipped": "not shipped (candidate C4)",
     "document": "papers/sycophancy-target-gate/FINDING_promptopinion_2026_05_24.md", "date": "2026-05-24",
     "n": "120-item varied-phrasing holdout, two API models",
     "measured": [
         {"metric": "detector accuracy on the decisive class, fresh phrasing", "printed": "0.47", "receipt": "papers/sycophancy-target-gate/results_promptopinion.json", "pointer": "/detector_by_class/agreement_cf"},
         {"metric": "separation on the earlier fixed-template holdout", "printed": "100%", "receipt": None,
          "needle": "separated the classes **100%**"},
     ]},
    {"id": "critique-detector", "instrument": "critique_detector", "in_audit": True,
     "module": "styxx/critique.py", "shipped": "shipped",
     "document": "papers/agent-self-audit/FINDING_critique_detector_on_paper_2026_05_28.md", "date": "2026-05-28",
     "n": "18 author-written propositions; no panel",
     "measured": [
         {"metric": "propositions, all of them saturated", "printed": "18", "receipt": "experiments/critique_detector_on_paper_2026_05_28/results.json", "pointer": "/n_propositions"},
         {"metric": "observed P(NO) on a TRUE claim", "printed": "0.0", "receipt": "experiments/critique_detector_on_paper_2026_05_28/results.json", "pointer": "/results/0/observed_p_no"},
     ]},
    {"id": "dogfood-register-gates", "instrument": "the dogfood register gates", "in_audit": True,
     "module": "papers/dogfood-self-audit/execution_receipt_gate.py", "shipped": "not shipped",
     "document": "papers/dogfood-self-audit/FINDING_nominal_register_blindspot_2026_08_13.md", "date": "2026-08-13",
     "n": "one live status report; no panel",
     "measured": [
         {"metric": "claims found in a report that made claims", "printed": "zero", "receipt": None,
          "needle": "Both returned **zero claims**"},
     ]},
    {"id": "text-only-deception", "instrument": "text-only deception", "in_audit": True,
     "module": "styxx/attack/fingerprint.py", "shipped": "shipped as a register axis",
     "document": "papers/THESIS_the_honesty_standard_2026_05_31.md", "date": "2026-05-31",
     "n": "48 register-matched pairs; no panel",
     "measured": [
         {"metric": "AUC separating true from false self-claims", "printed": "0.4983", "receipt": "papers/grounded-honesty-axis/grounded_honesty_result.json", "pointer": "/auc_text_only_deception"},
     ]},
    # measured short of the 0.95 bar, but not in the audit's eleven
    {"id": "struct1-claimdetect", "instrument": "STRUCT-1 claimdetect", "in_audit": False,
     "module": "styxx/claimdetect.py", "shipped": "shipped as an observer; beat its null",
     "document": CMF + "RESULT_struct1_beats_the_null_2026_08_31.md", "date": "2026-08-31",
     "n": "38 per arm, 9 blind seats",
     "measured": [
         {"metric": "precision on a fresh blind panel", "printed": "0.4211", "receipt": CMF + "stage2_result.json", "pointer": "/arms/flagged/A_share"},
         {"metric": "the frozen null bar it beat", "printed": "0.2061", "receipt": CMF + "stage2_result.json", "pointer": "/gates/G-S2P/bar"},
     ]},
    {"id": "oath-unobligated", "instrument": "the OATH unobligated oath", "in_audit": False,
     "module": "styxx/certify.py", "shipped": "shipped; the split is now printed per certificate",
     "document": CMF + "RESULT_unobligated_oath_2026_08_28.md", "date": "2026-08-28",
     "n": "5951 verifications over 192 documents",
     "measured": [
         {"metric": "verifications volunteered rather than obligated", "printed": "0.5811", "receipt": CMF + "oath_unobligated_oath_census.json", "pointer": "/headline/unobligated_oath_rate"},
         {"metric": "value match alone, path never compared", "printed": "0.3399", "receipt": CMF + "oath_unobligated_oath_census.json", "pointer": "/headline/weakest_share_of_verified"},
     ]},
]


def _walk(obj, pointer: str):
    for tok in pointer.split("/")[1:]:
        tok = tok.replace("~1", "/").replace("~0", "~")
        obj = obj[int(tok)] if isinstance(obj, list) else obj[tok]
    return obj


def _frac(printed: str) -> int:
    t = printed.rstrip("%")
    return len(t.split(".", 1)[1]) if "." in t else 0


def main() -> int:
    problems = []
    for row in ROWS:
        doc = ROOT / row["document"]
        if not doc.exists():
            problems.append("%s: document missing %s" % (row["id"], row["document"]))
            continue
        for m in row["measured"]:
            if m.get("receipt"):
                rp = ROOT / m["receipt"]
                if not rp.exists():
                    problems.append("%s: receipt missing %s" % (row["id"], m["receipt"]))
                    continue
                try:
                    leaf = _walk(json.loads(rp.read_text(encoding="utf-8"), parse_float=Decimal), m["pointer"])
                except Exception as e:                       # noqa: BLE001 - any failure is a problem
                    problems.append("%s: pointer %s fails: %s" % (row["id"], m["pointer"], e))
                    continue
                m["leaf"] = str(leaf)
                if m.get("kind") == "quote":
                    ok = isinstance(leaf, str) and m["printed"] in leaf
                else:
                    try:
                        ok = (isinstance(leaf, (int, Decimal)) and not isinstance(leaf, bool)
                              and round(Decimal(str(leaf)), _frac(m["printed"])) == Decimal(m["printed"].rstrip("%")))
                    except Exception:                        # noqa: BLE001
                        ok = False
                m["kind"] = m.get("kind", "numeric")
                m["status"] = "receipt-json" if ok else "MISMATCH"
                if not ok:
                    problems.append("%s: %s printed %s but leaf is %r" % (row["id"], m["metric"], m["printed"], leaf))
            else:
                needle = m["needle"].encode("utf-8")
                if needle not in doc.read_bytes():
                    problems.append("%s: needle not in document bytes: %r" % (row["id"], m["needle"]))
                m["kind"] = "prose"
                m["status"] = "prose-only"
    if problems:
        print("REFUSED: the census does not check out:")
        for p in problems:
            print("  -", p)
        return 1
    audit_rows = [r for r in ROWS if r["in_audit"]]
    receipted = [r for r in audit_rows if r["measured"][0]["status"] == "receipt-json"]
    payload = {
        "schema": "styxx-sworn/prose-claimhood-census/v1",
        "what": "instruments that read prose to find claims, with the document, the headline number and the receipt",
        "criterion": ("in_audit rows are AUDIT_the_whole_program_2026_09_01.md §8.1 direction 1, as written; the two "
                      "others measured short of the 0.95 bar but were not counted there"),
        "counts": {
            "audit_direction_1": len(audit_rows),
            "audit_rows_whose_headline_has_a_json_receipt": len(receipted),
            "audit_rows_whose_headline_rests_on_prose": len(audit_rows) - len(receipted),
            "rows_total": len(ROWS),
        },
        "rows": ROWS,
        "verifier": {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "not_decided_here": "which instruments belong on the list, and whether a number is a failure",
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    c = payload["counts"]
    print("census: audit rows %d (json-receipted %d, prose-only %d), rows total %d -> %s"
          % (c["audit_direction_1"], c["audit_rows_whose_headline_has_a_json_receipt"],
             c["audit_rows_whose_headline_rests_on_prose"], c["rows_total"], OUT.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
