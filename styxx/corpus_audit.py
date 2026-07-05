"""styxx.corpus_audit — re-certify a whole corpus of OATH certificates under the CURRENT verifier.

The mutant battery and the cycle-18/26 sweeps, productized into a standing, anyone-can-run check:
for every ``*.certificate.json`` under a root, resolve the receipts it recorded (next to the doc),
SHA-verify them (flag drift), re-run :func:`styxx.certify.certify_doc` under the *current* verifier,
and report each document's live verdict. Answers, on demand: *is every number we ever shipped still
grounded at the receipts it cited?*

Open by design (see ``OPEN_CORE.md``): this is a measurement primitive, never gated.

Two modes:
  * default — re-certification only (fast, deterministic): HELD / FAILED / receipt-drift / verdict-drift.
  * ``--tamper`` — additionally mutate every VERIFIED token once (single significant digit, seeded)
    and report the corpus tamper-catch rate (caught / false-verify / abstain-degrade). This is the
    ``papers/autopilot/mutant_battery.py`` scheme lifted into the package.

CLI::

    python -m styxx.corpus_audit [ROOT] [--tamper] [--seed N] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

from styxx.certify import certify_doc

__all__ = ["discover_certificates", "audit_document", "audit_corpus", "mutate_token"]


def discover_certificates(root: Path) -> list[Path]:
    """Every ``*.certificate.json`` under *root*, sorted."""
    return sorted(root.rglob("*.certificate.json"))


def _resolve_receipts(cert_path: Path, cert: dict) -> tuple[list[Path], list[str], list[str]]:
    """Receipt filenames recorded in the cert, resolved next to the doc. Returns
    (existing_paths, missing_names, sha_drifted_names)."""
    paths, missing, drift = [], [], []
    for name, sha in cert.get("receipts_sha256", {}).items():
        rp = cert_path.parent / name
        if not rp.exists():
            missing.append(name)
            continue
        if hashlib.sha256(rp.read_bytes()).hexdigest() != sha:
            drift.append(name)
        paths.append(rp)
    return paths, missing, drift


def mutate_token(tok: str, rng: random.Random) -> str:
    """Perturb one significant digit, keeping format (the frozen validate_oath_v0 scheme)."""
    digits = [i for i, ch in enumerate(tok) if ch.isdigit()]
    sig = [i for i in digits if not (tok[i] == "0" and (i == 0 or not tok[:i].strip("+-0.")))]
    pos = rng.choice(sig or digits)
    old = int(tok[pos])
    new = rng.choice([d for d in range(10) if d != old])
    return tok[:pos] + str(new) + tok[pos + 1:]


def _doc_for(cert_path: Path) -> Path:
    return cert_path.with_name(cert_path.name.replace(".certificate.json", ".md"))


def audit_document(cert_path: Path, tamper: bool = False, seed: int = 1) -> dict:
    """Re-certify one document under the current verifier. Optionally run the tamper battery."""
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    doc = _doc_for(cert_path)
    rec = {"certificate": cert_path.name, "document": doc.name,
           "recorded_verdict": cert.get("verdict")}
    if not doc.exists():
        rec.update(status="MISSING_DOC", live_verdict=None)
        return rec
    receipts, missing, drift = _resolve_receipts(cert_path, cert)
    rec["receipt_drift"] = drift
    rec["missing_receipts"] = missing
    if not receipts:
        rec.update(status="NO_RECEIPTS", live_verdict=None)
        return rec
    live = certify_doc(doc, receipts)
    rec["live_verdict"] = live["verdict"]
    rec["counts"] = live["counts"]
    rec["verdict_changed"] = (live["verdict"] != cert.get("verdict"))
    rec["status"] = "OK"
    if tamper:
        rng = random.Random(seed)
        text = doc.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        verified = [e for e in live["ledger"] if e["status"] == "VERIFIED"]
        caught = fv = ad = dropped = 0
        for e in verified:
            li = e["line"] - 1
            if li >= len(lines) or e["token"] not in lines[li]:
                dropped += 1
                continue
            mut = mutate_token(e["token"], rng)
            ml = list(lines)
            ml[li] = lines[li].replace(e["token"], mut, 1)
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                             encoding="utf-8") as tf:
                tf.write("".join(ml))
                tmp = Path(tf.name)
            try:
                mc = certify_doc(tmp, receipts)
            finally:
                tmp.unlink(missing_ok=True)
            st = next((x["status"] for x in mc["ledger"]
                       if x["line"] == e["line"] and x["token"] == mut), None)
            if st == "UNGROUNDED":
                caught += 1
            elif st == "VERIFIED":
                fv += 1
            elif st == "ABSTAIN":
                ad += 1
            else:
                dropped += 1
        rec["tamper"] = {"n_mutants": len(verified) - dropped, "caught": caught,
                         "false_verify": fv, "abstain_degrade": ad, "dropped": dropped}
    return rec


def audit_corpus(root: Path, tamper: bool = False, seed: int = 1) -> dict:
    """Audit every certificate under *root*; return per-doc records + a corpus summary."""
    docs = [audit_document(cp, tamper, seed) for cp in discover_certificates(root)]
    held = sum(1 for d in docs if d.get("live_verdict") == "OATH-HELD")
    failed = sum(1 for d in docs if d.get("live_verdict") == "OATH-FAILED")
    unresolved = sum(1 for d in docs if d.get("status") in ("MISSING_DOC", "NO_RECEIPTS"))
    changed = sum(1 for d in docs if d.get("verdict_changed"))
    drifted = sum(1 for d in docs if d.get("receipt_drift"))
    summary = {"root": str(root), "n_certificates": len(docs), "held": held, "failed": failed,
               "unresolved": unresolved, "verdict_changed": changed, "receipt_drift": drifted}
    if tamper:
        tot = {"n_mutants": 0, "caught": 0, "false_verify": 0, "abstain_degrade": 0}
        for d in docs:
            for k in tot:
                tot[k] += d.get("tamper", {}).get(k, 0)
        n = max(tot["n_mutants"], 1)
        summary["tamper"] = {**tot, "catch_rate": round(tot["caught"] / n, 3),
                             "false_verify_rate": round(tot["false_verify"] / n, 3)}
    return {"summary": summary, "documents": docs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.corpus_audit",
                                 description="Re-certify a corpus of OATH certificates under the current verifier.")
    ap.add_argument("root", nargs="?", default=".", help="directory to scan (default: cwd)")
    ap.add_argument("--tamper", action="store_true", help="also run the single-digit tamper battery")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", default=None, help="write the full report to this path")
    a = ap.parse_args(argv)
    report = audit_corpus(Path(a.root), tamper=a.tamper, seed=a.seed)
    s = report["summary"]
    print(f"corpus {s['root']}: {s['n_certificates']} certificates | "
          f"HELD {s['held']}  FAILED {s['failed']}  unresolved {s['unresolved']}  "
          f"verdict-drift {s['verdict_changed']}  receipt-drift {s['receipt_drift']}")
    for d in report["documents"]:
        if d.get("live_verdict") == "OATH-FAILED" or d.get("verdict_changed") or d.get("receipt_drift"):
            tag = d.get("live_verdict") or d.get("status")
            extra = " receipt-drift" if d.get("receipt_drift") else ""
            extra += " verdict-CHANGED" if d.get("verdict_changed") else ""
            print(f"  [{tag}]{extra}  {d['document']}")
    if a.tamper and "tamper" in s:
        t = s["tamper"]
        print(f"tamper-catch: {t['caught']}/{t['n_mutants']} = {t['catch_rate']}  "
              f"(false-verify {t['false_verify']} = {t['false_verify_rate']})")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"-> {a.json}")
    return 1 if s["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
