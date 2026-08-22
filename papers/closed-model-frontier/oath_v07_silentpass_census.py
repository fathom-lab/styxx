"""OATH v0.7 residual census — how much of the claim surface is silently mutable?

The trigger-recall debt is usually quoted as a share of the full-precision pool. This measures
the denominator that share should be read against: the WHOLE certified claim surface.

For every claim the shipped verifier currently certifies VERIFIED, perturb one significant digit
(the `validate_oath_v0.py` mutation operator, seeded) and re-certify the document. Three outcomes:

  UNGROUNDED  the verifier accuses the mutant       -> the instrument works
  ABSTAIN     the verifier says nothing             -> SILENT PASS, the document stays OATH-HELD
  VERIFIED    the mutant matches some other leaf    -> FALSE ATTESTATION, worse than silence

Reported under G2c of PREREG_oath_v07_precision_obligation_2026_08_22. Any release that publishes
the v0.7 catch without this residual overstates its own coverage.

Non-destructive: mutants live in temp files; only this script's result JSON is written.
Runtime ~3 min.

  python papers/closed-model-frontier/oath_v07_silentpass_census.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import certify_doc                # noqa: E402
from styxx.corpus_audit import _resolve_receipts     # noqa: E402

OUT = HERE / "oath_v07_silentpass_census.json"
SEED = 1


def mutate_token(tok: str, rng: random.Random) -> str:
    """The repo's own battery operator (validate_oath_v0.mutate_token)."""
    digits = [i for i, ch in enumerate(tok) if ch.isdigit()]
    sig = [i for i in digits if not (tok[i] == "0" and (i == 0 or not tok[:i].strip("+-0.")))]
    pos = rng.choice(sig or digits)
    old = int(tok[pos])
    new = rng.choice([d for d in range(10) if d != old])
    return tok[:pos] + str(new) + tok[pos + 1:]


def main() -> int:
    t0 = time.time()
    rng = random.Random(SEED)
    docs = []
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        if "anc" in cp.parts:
            continue
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        if not doc.exists():
            continue
        try:
            rec = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        receipts, missing, _ = _resolve_receipts(cp, rec)
        if receipts and not missing:
            docs.append((doc, receipts))
    print(f"docs with fully-resolvable receipts: {len(docs)}", flush=True)

    rows, docs_silent, docs_total, skipped = [], set(), set(), 0
    for i, (doc, receipts) in enumerate(docs, 1):
        try:
            base = certify_doc(doc, receipts)
        except Exception:
            continue
        verified = [e for e in base["ledger"] if e["status"] == "VERIFIED"]
        if not verified:
            continue
        docs_total.add(doc.name)
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        for claim in verified:
            tok, ln = claim["token"], claim["line"] - 1
            mut = mutate_token(tok, rng)
            if mut == tok or ln >= len(lines) or tok not in lines[ln]:
                skipped += 1
                continue
            ml = list(lines)
            ml[ln] = ml[ln].replace(tok, mut, 1)
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                             encoding="utf-8") as tf:
                tf.write("\n".join(ml))
                tmp = Path(tf.name)
            try:
                mc = certify_doc(tmp, receipts)
            except Exception:
                skipped += 1
                continue
            finally:
                tmp.unlink(missing_ok=True)
            status = next((e["status"] for e in mc["ledger"]
                           if e["line"] == claim["line"] and e["token"] == mut), "NOT_EXTRACTED")
            rows.append({"doc": doc.name, "line": claim["line"], "token": tok, "mutant": mut,
                         "decimals": claim["decimals"], "status": status,
                         "doc_verdict_held": mc["verdict"] == "OATH-HELD"})
            if status != "UNGROUNDED":
                docs_silent.add(doc.name)
        if i % 20 == 0:
            print(f"  [{i}/{len(docs)}] claims so far: {len(rows)} ({time.time()-t0:.0f}s)",
                  flush=True)

    n = len(rows)
    silent = [r for r in rows if r["status"] != "UNGROUNDED"]
    silent_by_dec = Counter(r["decimals"] for r in silent)
    closure = {str(T): sum(c for d, c in silent_by_dec.items() if d >= T)
               for T in (2, 3, 4, 5, 7, 13, 15, 16, 17)}

    report = {
        "note": "G2c residual for PREREG_oath_v07_precision_obligation_2026_08_22",
        "verifier_sha256": hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "seed": SEED,
        "docs": len(docs), "docs_with_verified_claims": len(docs_total),
        "verified_claims_mutated": n, "skipped": skipped,
        "mutant_status": dict(Counter(r["status"] for r in rows)),
        "silent_total": len(silent),
        "silent_share": round(len(silent) / n, 4) if n else 0.0,
        "silent_abstain": sum(1 for r in silent if r["status"] == "ABSTAIN"),
        "silent_false_verified": sum(1 for r in silent if r["status"] == "VERIFIED"),
        "silent_not_extracted": sum(1 for r in silent if r["status"] == "NOT_EXTRACTED"),
        "docs_with_at_least_one_silent": len(docs_silent),
        "silent_by_decimal_width": {str(k): v for k, v in sorted(silent_by_dec.items())},
        "closure_if_obligate_decimals_ge": closure,
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\nVERIFIED claims mutated: {n} (skipped {skipped})")
    print(f"  mutant status: {report['mutant_status']}")
    print(f"  SILENT (not accused): {len(silent)} = {report['silent_share']}")
    print(f"    ABSTAIN {report['silent_abstain']} | FALSE-VERIFIED "
          f"{report['silent_false_verified']} | NOT_EXTRACTED {report['silent_not_extracted']}")
    print(f"  docs with >=1 silent claim: {len(docs_silent)}/{len(docs_total)}")
    print(f"  silent by decimal width: {report['silent_by_decimal_width']}")
    for T, c in closure.items():
        print(f"    a 'decimals >= {T}' rule reaches {c}/{len(silent)} "
              f"({c/max(len(silent),1):.4f}) of the silent surface")
    print(f"\nelapsed {time.time()-t0:.1f}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
