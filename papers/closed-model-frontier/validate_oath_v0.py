"""OATH v0 validation — the pre-registered kill-gate run (D1/D2/D3).

PREREG_oath_v0_certify_doc_2026_06_09.md, frozen before this run:
  D1: 20 seeded single-claim mutations across the corpus docs -> verifier must flag >=16 as UNGROUNDED.
  D2: zero false UNGROUNDED on unmutated docs (a flagged claim that hand-verifies as a REAL doc<->receipt
      discrepancy is a CATCH, reported, not a false alarm).
  D3: every extracted number classified; report the VERIFIED share per doc.
FAIL any bar -> OATH v0 is a linter prototype, not an oath; no certificate language ships.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))   # repo root -> import styxx.certify
from styxx.certify import certify_doc, extract_numbers  # noqa: E402

SEED = 1   # v0.1 re-validation: FRESH mutation seed (v0 used SEED=0 and FAILED D1 8/20, D2 13;
           # mechanism fixes are pre-registered in the PREREG amendment, bars unchanged)
N_MUT = 20

CORPUS = [
    ("FINDING_b24_whitebox_vs_behavioral_2026_06_09.md",
     ["b24_headtohead_result.json", "b24_controls_addendum.json"]),
    ("FINDING_b22_nonacknowledged_caving_2026_06_09.md",
     ["behavioral_sycophancy_b22_result.json", "b22_findings_addendum.json"]),
    ("FINDING_behavioral_sycophancy_blackbox_2026_06_09.md",
     ["behavioral_sycophancy_result.json", "b22_findings_addendum.json",
      "../grounded-honesty-axis/intent_metapc_3.json",
      # v0.3 receipt-set repair (hand-check): the R2 intent AUROC the doc cites lives in the
      # rung-2 ladder receipt, not the metapc file
      "../grounded-honesty-axis/intent_ladder_result.json"]),
]


def mutate_token(tok: str, rng: random.Random) -> str:
    """Perturb one significant digit, keeping format (decimal count, sign, commas)."""
    digits = [i for i, ch in enumerate(tok) if ch.isdigit()]
    # prefer a significant digit (skip leading zeros like the 0 in 0.94)
    sig = [i for i in digits if not (tok[i] == "0" and (i == 0 or not tok[:i].strip("+-0.")))]
    pos = rng.choice(sig or digits)
    old = int(tok[pos])
    new = rng.choice([d for d in range(10) if d != old])
    return tok[:pos] + str(new) + tok[pos + 1:]


def main() -> int:
    rng = random.Random(SEED)
    report = {"prereg": "PREREG_oath_v0_certify_doc_2026_06_09.md", "seed": SEED}

    # ---------- D2 + D3 on the clean corpus ----------
    d2_false, d3 = [], {}
    clean_verified = {}
    for doc, recs in CORPUS:
        cert = certify_doc(HERE / doc, [HERE / r for r in recs])
        (HERE / doc).with_suffix(".certificate.json").write_text(
            json.dumps(cert, indent=2) + "\n", encoding="utf-8")
        c = cert["counts"]
        d3[doc] = {"numbers": sum(c.values()), "verified": c["VERIFIED"],
                   "abstained": c["ABSTAIN"], "ungrounded": c["UNGROUNDED"]}
        clean_verified[doc] = [l for l in cert["ledger"] if l["status"] == "VERIFIED"]
        for bad in cert["ungrounded"]:
            d2_false.append({"doc": doc, "line": bad["line"], "token": bad["token"],
                             "context": bad["context"]})
        print(f"[clean] {doc}: {cert['verdict']} v={c['VERIFIED']} a={c['ABSTAIN']} c={c['UNGROUNDED']}")
    report["D2_contradicted_on_clean"] = d2_false
    report["D3_coverage"] = d3

    # ---------- D1: seeded mutations of VERIFIED claims ----------
    targets = []
    docs_cycle = [d for d, _ in CORPUS]
    di = 0
    while len(targets) < N_MUT:
        doc = docs_cycle[di % len(docs_cycle)]; di += 1
        cands = clean_verified[doc]
        if not cands:
            continue
        targets.append((doc, rng.choice(cands)))
    caught, missed = 0, []
    mdir = HERE / "_oath_mutants"; mdir.mkdir(exist_ok=True)
    for k, (doc, claim) in enumerate(targets):
        text = (HERE / doc).read_text(encoding="utf-8")
        lines = text.splitlines()
        ln = lines[claim["line"] - 1]
        mut_tok = mutate_token(claim["token"], rng)
        # replace only the first occurrence of the exact token on that line
        lines[claim["line"] - 1] = ln.replace(claim["token"], mut_tok, 1)
        mp = mdir / f"mut{k:02d}_{doc}"
        mp.write_text("\n".join(lines), encoding="utf-8")
        recs = dict(CORPUS)[doc]
        cert = certify_doc(mp, [HERE / r for r in recs])
        flagged = any(b["line"] == claim["line"] and b["token"] == mut_tok
                      for b in cert["ungrounded"])
        if flagged:
            caught += 1
        else:
            missed.append({"k": k, "doc": doc, "line": claim["line"],
                           "orig": claim["token"], "mut": mut_tok,
                           "context": claim["context"][:100]})
    report["D1"] = {"n_mutations": N_MUT, "caught": caught, "bar": 16,
                    "pass": caught >= 16, "missed": missed}
    print(f"\nD1: caught {caught}/{N_MUT} (bar 16) -> {'PASS' if caught >= 16 else 'FAIL'}")
    print(f"D2: {len(d2_false)} UNGROUNDED on clean docs (0 required unless hand-verified real)")
    for f in d2_false:
        print(f"  [D2?] {f['doc']} L{f['line']}: {f['token']} | {f['context'][:90]}")

    verdict = "OATH-V0-VALID" if (report["D1"]["pass"] and not d2_false) else (
        "OATH-V0-PENDING-HAND-CHECK" if report["D1"]["pass"] else "OATH-V0-FAILED (linter prototype)")
    report["validation_verdict"] = verdict
    (HERE / "oath_v0_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nVALIDATION:", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
