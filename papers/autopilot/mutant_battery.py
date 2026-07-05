"""Mutant battery over shipped certificates — the cycle-19 diligence audit as a
committed, re-runnable script (cycle 19 ran it in memory; that gap closes here).

Method (identical to cycle 19, seed 1):
  For each certificate: re-certify the CLEAN doc against its recorded receipt set
  (filenames from receipts_sha256, resolved next to the doc, SHA-verified), then
  mutate every VERIFIED token once (single significant digit, the mutate_token
  scheme of validate_oath_v0.py), re-certify the mutant in a temp copy, and
  classify the corrupted token's fate:
    caught          -> the token is UNGROUNDED in the mutant cert (verdict flips)
    false_verify    -> the token is VERIFIED (the oath swears to the corruption)
    abstain_degrade -> the token is ABSTAIN (oath stops swearing; verdict stays)
    dropped         -> the token is no longer extracted at all
  Nothing on disk is modified; mutants live in temp files.

Usage:  python papers/autopilot/mutant_battery.py [--out RESULT.json]
Baseline for comparison: cycle18_mutant_battery_result.json (v0.3 verifier:
269 mutants, caught 58, FALSE-VERIFY 26, abstain-degrade 182).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # papers/autopilot
ROOT = HERE.parent.parent                       # repo root
sys.path.insert(0, str(ROOT))
from styxx.certify import certify_doc  # noqa: E402
sys.path.insert(0, str(ROOT / "papers" / "closed-model-frontier"))
from validate_oath_v0 import mutate_token  # noqa: E402  (the exact frozen scheme)

SEED = 1

# The 13 cycle-18 certificates (CYCLE_LOG.jsonl cycle 18 receipts, verbatim).
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


def resolve_receipts(cert_path: Path, cert: dict) -> tuple[list[Path], list[str]]:
    """Receipt filenames recorded in the cert, resolved next to the doc. SHA drift
    is reported loudly (the battery still runs — the receipts are what they are)."""
    paths, drift = [], []
    for name, sha in cert["receipts_sha256"].items():
        rp = cert_path.parent / name
        if not rp.exists():
            raise FileNotFoundError(f"{cert_path.name}: receipt {name} not found next to doc")
        actual = hashlib.sha256(rp.read_bytes()).hexdigest()
        if actual != sha:
            drift.append(name)
        paths.append(rp)
    return paths, drift


def token_status(cert: dict, line: int, token: str) -> str | None:
    for entry in cert["ledger"]:
        if entry["line"] == line and entry["token"] == token:
            return entry["status"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "cycle22_v04_battery_result.json"))
    a = ap.parse_args()

    rng = random.Random(SEED)
    per_doc, fv_rows = {}, []
    totals = dict(n_mutants=0, caught=0, false_verify=0, abstain_degrade=0,
                  dropped=0, verdict_flips=0)
    sha_drift_all = {}

    for rel in CERTS:
        cert_path = ROOT / rel
        shipped = json.loads(cert_path.read_text(encoding="utf-8"))
        doc_path = cert_path.with_name(cert_path.name.replace(".certificate.json", ".md"))
        receipts, drift = resolve_receipts(cert_path, shipped)
        if drift:
            sha_drift_all[doc_path.name] = drift

        clean = certify_doc(doc_path, receipts)
        verified = [e for e in clean["ledger"] if e["status"] == "VERIFIED"]
        text = doc_path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)

        d = dict(caught=0, false_verify=0, abstain_degrade=0, dropped=0,
                 verdict_flips=0, n_mutants=len(verified),
                 clean_counts=clean["counts"], clean_verdict=clean["verdict"])
        for e in verified:
            mut_tok = mutate_token(e["token"], rng)
            li = e["line"] - 1
            # replace the FIRST occurrence of the token on its line (matches the
            # ledger's line/token anchor; duplicate tokens on a line share fate)
            if e["token"] not in lines[li]:
                d["dropped"] += 1
                totals["dropped"] += 1
                continue
            mut_lines = list(lines)
            mut_lines[li] = lines[li].replace(e["token"], mut_tok, 1)
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                             encoding="utf-8") as tf:
                tf.write("".join(mut_lines))
                tmp = Path(tf.name)
            try:
                mcert = certify_doc(tmp, receipts)
            finally:
                tmp.unlink(missing_ok=True)
            st = token_status(mcert, e["line"], mut_tok)
            totals["n_mutants"] += 1
            if st == "UNGROUNDED":
                d["caught"] += 1; totals["caught"] += 1
                if mcert["verdict"] != clean["verdict"]:
                    d["verdict_flips"] += 1; totals["verdict_flips"] += 1
            elif st == "VERIFIED":
                d["false_verify"] += 1; totals["false_verify"] += 1
                fv_rows.append({"doc": doc_path.name, "line": e["line"],
                                "orig": e["token"], "mutant": mut_tok,
                                "context": e["context"]})
            elif st == "ABSTAIN":
                d["abstain_degrade"] += 1; totals["abstain_degrade"] += 1
            else:
                d["dropped"] += 1; totals["dropped"] += 1
        per_doc[doc_path.name] = d
        print(f"[{doc_path.name}] clean {clean['verdict']} "
              f"V={clean['counts']['VERIFIED']} A={clean['counts']['ABSTAIN']} "
              f"U={clean['counts']['UNGROUNDED']} | mutants={d['n_mutants']} "
              f"caught={d['caught']} fv={d['false_verify']} ad={d['abstain_degrade']}")

    n = max(totals["n_mutants"], 1)
    result = {
        "what": "mutant battery over the 13 cycle-18 certificates (committed script)",
        "prereg": "papers/closed-model-frontier/PREREG_oath_v04_float_binding_2026_07_03.md",
        "seed": SEED,
        "n_docs": len(CERTS),
        "receipt_sha_drift": sha_drift_all,
        **totals,
        "catch_rate": round(totals["caught"] / n, 3),
        "false_verify_rate": round(totals["false_verify"] / n, 3),
        "abstain_degrade_rate": round(totals["abstain_degrade"] / n, 3),
        "clean_ungrounded_total": sum(per_doc[k]["clean_counts"]["UNGROUNDED"] for k in per_doc),
        "per_doc": per_doc,
        "false_verify_rows": fv_rows,
    }
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nTOTAL mutants={totals['n_mutants']} caught={totals['caught']} "
          f"FALSE-VERIFY={totals['false_verify']} abstain-degrade={totals['abstain_degrade']} "
          f"dropped={totals['dropped']}")
    print(f"clean UNGROUNDED across 13 docs = {result['clean_ungrounded_total']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
