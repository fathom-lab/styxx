"""styxx.ladder -- the probe-robustness ladder as a first-class object.

The four-rung adversarial ladder a substrate probe's robustness claim should survive, assembled
from this program's frozen, pre-registered attack arcs (2026-07). Each rung is a distinct attacker
CLASS, run against the same audit surface with the same admissibility discipline (bite gates,
knowledge floors, pre-committed VOID branches):

  1. POISONING      -- calibration poisoning: mis-point the auditor's probe without removing the
                       signal (the stage-2/E1 arc; defense: private-calibration re-lock).
  2. PARITY         -- probe-capacity attribution: does a capacity-matched probe family fit on the
                       ATTACKER'S OWN poisoned split recover the read? The rung nobody runs on
                       their own work: it demoted THIS program's flagship attribution. Its output
                       -- the capacity share of the recovery -- is the MANDATORY line item of any
                       ladder report.
  3. STATIC ERASURE -- removal: drive the gold subspace to verifiable zero (attacker loss
                       converges) and ask whether the reads survive with knowledge held.
  4. ADAPTIVE ERASURE -- chase: re-fit the erased subspace on the current model as the signal
                       relocates; measures whether relocation can be chased to convergence.

v1 scope (honest): the ladder over THIS repository's canonical honesty-construct receipts --
registry, report assembly, parity-attribution extraction, receipt verification. It does NOT yet
execute the ladder against arbitrary (model, probe_family) pairs; that execution API is the
documented v2 (backlog G1), and the frozen per-rung scripts listed in the registry are its
backends. Receipts live under papers/ and ship with the repository, not the wheel -- every
function takes a repo root and fails loudly, never silently, when receipts are absent.

CLI: python -m styxx.ladder [--root PATH] [--json]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Rung", "RUNGS", "load_receipt", "parity_attribution", "admissibility_line_item",
           "report", "verify"]


@dataclass(frozen=True)
class Rung:
    """One rung of the ladder: an attacker class with its frozen harness and canonical receipt."""
    key: str
    question: str
    attacker_class: str
    script: str                     # repo-relative frozen harness (the v2 execution backend)
    prereg: str                     # repo-relative frozen prereg
    receipt: str                    # repo-relative canonical result JSON
    result_doc: str                 # repo-relative certified RESULT/backlog doc
    canonical_verdict: str          # the frozen verdict string this receipt carries
    decisive_fields: tuple = field(default_factory=tuple)  # per-cell reads a report surfaces


RUNGS: tuple = (
    Rung(
        key="poisoning",
        question="Can the auditor's probe be mis-pointed without removing the signal, and does a private-calibration re-lock recover the read?",
        attacker_class="calibration poisoning (redirection)",
        script="papers/read-neq-write/e1_clean_auditor.py",
        prereg="papers/read-neq-write/PREREG_E1_clean_calibrated_auditor_2026_07_04.md",
        receipt="papers/read-neq-write/e1_result.json",
        result_doc="papers/read-neq-write/RESULT_E1_read_neq_write_STANDS_2026_07_04.md",
        canonical_verdict="POISONING_CONFIRMED__read_neq_write_STANDS",
        decisive_fields=(),
    ),
    Rung(
        key="parity",
        question="How much of the recovery is probe CAPACITY rather than calibration privacy -- does a capacity-matched family fit on the attacker's own poisoned split recover the read?",
        attacker_class="attribution control (the mandatory line item)",
        script="papers/calib-poison-general/honesty_parity_confirm.py",
        prereg="papers/calib-poison-general/PREREG_honesty_parity_confirm_2026_07_11.md",
        receipt="papers/calib-poison-general/honesty_parity_confirm_result.json",
        result_doc="papers/calib-poison-general/RESULT_honesty_parity_confirm_2026_07_11.md",
        canonical_verdict="PARTIAL_CONSOLIDATED__residual_real_but_not_robust",
        decisive_fields=("private13_auroc", "naive_matched13_auroc", "parity_gap", "baseline_gap"),
    ),
    Rung(
        key="static-erasure",
        question="Does the read survive verifiable REMOVAL of the gold subspace with knowledge held?",
        attacker_class="subspace erasure (removal, frozen target)",
        script="papers/calib-poison-general/b2_subspace_erasure.py",
        prereg="papers/calib-poison-general/PREREG_B2_subspace_erasure_2026_07_12.md",
        receipt="papers/calib-poison-general/b2_subspace_erasure_result.json",
        result_doc="papers/calib-poison-general/RESULT_B2_subspace_erasure_SURVIVES_2026_07_12.md",
        canonical_verdict="SURVIVES__vs_subspace_erasure_1p5B",
        decisive_fields=("private13_auroc", "naive_dom6_auroc", "eval_knowledge", "frozen18_read"),
    ),
    Rung(
        key="adaptive-erasure",
        question="Can the relocating signal be CHASED to convergence by re-fitting the erased subspace on the current model?",
        attacker_class="subspace erasure (removal, moving target)",
        script="papers/calib-poison-general/b2_adaptive_erasure.py",
        prereg="papers/calib-poison-general/PREREG_B2_adaptive_erasure_2026_07_13.md",
        receipt="papers/calib-poison-general/b2_adaptive_erasure_result.json",
        result_doc="papers/calib-poison-general/RESULT_B2_adaptive_erasure_SURVIVES_2026_07_13.md",
        canonical_verdict="SURVIVES__vs_adaptive_erasure_1p5B",
        decisive_fields=("private13_auroc", "naive_dom6_auroc", "eval_knowledge", "frozen18_read"),
    ),
)


def load_receipt(rung: Rung, root: Path) -> dict:
    """Load a rung's canonical receipt. Raises FileNotFoundError with the expected path -- receipts
    are repo artifacts; silence is never an answer."""
    p = Path(root) / rung.receipt
    if not p.exists():
        raise FileNotFoundError(
            f"ladder rung '{rung.key}' receipt missing: {p} -- run from the styxx repository root")
    return json.loads(p.read_text(encoding="utf-8"))


def parity_attribution(root: Path) -> dict:
    """The mandatory line item: what share of the private>naive recovery is probe capacity?

    Computed from the parity rung's receipt (never hardcoded): per admissible cell,
    capacity_share = 1 - parity_gap / baseline_gap (guarded for tiny denominators), plus the
    receipt's own medians when present. Returns per-cell shares and the median."""
    rung = next(r for r in RUNGS if r.key == "parity")
    rec = load_receipt(rung, root)
    cells = []
    for c in rec.get("points", []):
        bg, pg = c.get("baseline_gap"), c.get("parity_gap")
        if bg is None or pg is None or abs(bg) < 0.05:
            continue
        cells.append({
            "seed": c.get("seed"), "lambda": c.get("lambda"),
            "baseline_gap": bg, "parity_gap": pg,
            "capacity_share": round(1.0 - pg / bg, 4),
        })
    shares = sorted(x["capacity_share"] for x in cells)
    median = shares[len(shares) // 2] if len(shares) % 2 else round(
        (shares[len(shares) // 2 - 1] + shares[len(shares) // 2]) / 2, 4) if shares else None
    return {"verdict": rec.get("verdict"), "n_cells": len(cells),
            "median_capacity_share": median, "points": cells,
            "reading": "share of the private-auditor recovery explained by probe capacity alone "
                       "(fit on the attacker's own poisoned split); the residual is the privacy term"}


# the canonical on-disk home for the flagship probe's two-sided admissibility certificate
# (issued via ConscienceMount.certify_admissibility -> report.certificate(out_path=...))
ADMISSIBILITY_RECEIPT = "papers/conscience-mount/mount_admissibility_certificate.json"


def admissibility_line_item(root: Path | str = ".") -> dict:
    """The two-sided admissibility line item: is the flagship probe itself SENSITIVE and SPECIFIC
    on its own score (styxx.admissibility)? Surfaces the canonical certificate when it exists on
    disk, and says so honestly when it has not been issued yet — mirroring parity_attribution's
    receipts-in/dict-out discipline (never hardcoded, never silently absent)."""
    p = Path(root) / ADMISSIBILITY_RECEIPT
    reading = ("two-sided instrument admissibility of the flagship mount probe on its OWN deployed "
               "score (divergence margin): sensitive on target-present episodes AND specific on "
               "target-absent ones; styxx.admissibility")
    if not p.exists():
        return {"status": "not yet issued",
                "expected_receipt": ADMISSIBILITY_RECEIPT,
                "how": ("ConscienceMount.certify_admissibility(positive_states, null_states, "
                        "fire_threshold=<calibrated tau>) then report.certificate(out_path=...)"),
                "reading": reading}
    cert = json.loads(p.read_text(encoding="utf-8"))
    return {"status": "issued",
            "receipt": ADMISSIBILITY_RECEIPT,
            "verdict": cert.get("admissibility_verdict"),
            "admissible": cert.get("admissible"),
            "threshold_derived": cert.get("specificity", {}).get("threshold_derived"),
            "discrim": cert.get("sensitivity", {}).get("discrim"),
            "fire_rate": cert.get("specificity", {}).get("fire_rate"),
            "verify": "python -m styxx.admissibility --verify " + ADMISSIBILITY_RECEIPT,
            "reading": reading}


def report(root: Path | str = ".") -> dict:
    """Assemble the full ladder report from the canonical receipts: per-rung verdict + decisive
    reads + the mandatory parity-attribution line item. Pure receipts in, dict out."""
    root = Path(root)
    rungs_out = []
    for rung in RUNGS:
        rec = load_receipt(rung, root)
        entry: dict[str, Any] = {
            "rung": rung.key,
            "attacker_class": rung.attacker_class,
            "question": rung.question,
            "verdict": rec.get("verdict"),
            "verdict_matches_canonical": rec.get("verdict") == rung.canonical_verdict,
            "receipt": rung.receipt,
            "prereg": rung.prereg,
        }
        if rung.decisive_fields and "points" in rec:
            entry["points"] = [
                {f: c.get(f) for f in ("seed", "alpha", "lambda", "admissible") + rung.decisive_fields
                 if f in c}
                for c in rec["points"]
            ]
        rungs_out.append(entry)
    return {
        "what": "styxx probe-robustness ladder report -- four attacker classes, one audit surface, "
                "pre-registered gates; assembled verbatim from the canonical receipts",
        "construct": "honesty (Qwen2.5-1.5B decisive family); scale and further attacker classes "
                     "tracked in papers/PROGRAM_BACKLOG.md (B6, B7, accumulating eraser)",
        "rungs": rungs_out,
        "parity_attribution": parity_attribution(root),
        "instrument_admissibility": admissibility_line_item(root),
        "all_verdicts_canonical": all(r["verdict_matches_canonical"] for r in rungs_out),
    }


def verify(root: Path | str = ".") -> list:
    """Ladder-wide receipt check: every rung's script/prereg/receipt/result-doc exists and the
    receipt carries its frozen canonical verdict. Returns a list of problem strings (empty = OK)."""
    root = Path(root)
    problems = []
    for rung in RUNGS:
        for kind in ("script", "prereg", "receipt", "result_doc"):
            p = root / getattr(rung, kind)
            if not p.exists():
                problems.append(f"{rung.key}: missing {kind} {p}")
        try:
            rec = load_receipt(rung, root)
            if rec.get("verdict") != rung.canonical_verdict:
                problems.append(f"{rung.key}: receipt verdict {rec.get('verdict')!r} != "
                                f"canonical {rung.canonical_verdict!r}")
        except FileNotFoundError:
            pass  # already recorded above
    return problems


# --- the erasure-resistance certificate -------------------------------------------------------
#
# PRE-COMMITTED BEHAVIOR (written 2026-07-13 while the B7 3B run was STILL TRAINING, blind to its
# verdict): the certificate composes whatever removal-class receipts exist. A SURVIVES__* verdict
# enters `claim_scope`; an ERASED__* verdict enters `measured_breaks` WITH EQUAL PROMINENCE — a
# certificate that hides a break is not a certificate; VOID/PARTIAL verdicts enter `unadjudicated`
# (neither certified nor broken); an absent receipt enters `pending`. The certificate REFUSES to
# exist without a non-empty `unbounded_dimensions` section — an adversarial bound with no named
# unmeasured flank is dishonest by construction.

# scale replications of the removal-class attack (receipt may not exist yet -> pending)
SCALE_RECEIPTS: tuple = (
    ("Qwen2.5-3B-Instruct", "papers/calib-poison-general/b7_erasure_3b_result.json",
     "papers/calib-poison-general/PREREG_B7_erasure_3b_2026_07_13.md"),
)

# the flanks the current receipts do NOT bound — frozen from the preregs' own owed lists
UNBOUNDED_DIMENSIONS: tuple = (
    "accumulating eraser (union of all past subspaces -- closes the flow-back channel)",
    "chased whole-stack subspace (the 13-family's full span, not its rank-2 projection)",
    "full fine-tuning attacker (B6; all receipts are LoRA r=16)",
    "training budgets beyond 300 steps / cadences beyond refit-50",
    "constructs beyond honesty; families beyond Qwen at the receipted scales",
    "knowledge invariant is behavioral EVAL accuracy on disjoint stems, not a capability battery (B4)",
)


def _sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _classify_removal_receipt(rec: dict, label: str, path: str) -> tuple:
    """(bucket, entry) for one removal-class receipt. Buckets: claim_scope / measured_breaks /
    unadjudicated. Decisive minima are computed over ADMISSIBLE cells only."""
    verdict = rec.get("verdict", "")
    adm = [c for c in rec.get("points", []) if c.get("admissible")]
    reads = [c.get("private13_auroc") for c in adm if c.get("private13_auroc") is not None]
    know = [c.get("eval_knowledge") for c in adm if c.get("eval_knowledge") is not None]
    entry = {
        "attacker": label,
        "model": rec.get("model"),
        "verdict": verdict,
        "receipt": path,
        "budget": {"adapter": "LoRA r=16", "steps": rec.get("steps"),
                   "lambda": rec.get("lambda"), "alphas": rec.get("alphas"),
                   "refit_every": rec.get("refit_every")},
        "n_admissible": rec.get("n_admissible"),
        "private13_min_admissible": min(reads) if reads else None,
        "knowledge_min_admissible": min(know) if know else None,
    }
    if verdict.startswith("SURVIVES__"):
        return "claim_scope", entry
    if verdict.startswith("ERASED"):
        return "measured_breaks", entry
    return "unadjudicated", entry


def erasure_resistance_certificate(root: Path | str = ".", out_path: Path | str | None = None) -> dict:
    """Compose the removal-class receipts into the erasure-resistance certificate: what survived,
    what broke, what is unadjudicated, what is pending, and -- mandatorily -- what is unbounded.
    Writes JSON to out_path when given. Never states anything not in a receipt."""
    root = Path(root)
    buckets: dict = {"claim_scope": [], "measured_breaks": [], "unadjudicated": [], "pending": []}
    receipt_hashes = {}
    removal = [("static subspace erasure", "static-erasure"),
               ("adaptive re-fit erasure", "adaptive-erasure")]
    for label, key in removal:
        rung = next(r for r in RUNGS if r.key == key)
        rec = load_receipt(rung, root)   # loud on absence: the base rungs are required
        receipt_hashes[rung.receipt] = _sha256(root / rung.receipt)
        bucket, entry = _classify_removal_receipt(rec, label, rung.receipt)
        buckets[bucket].append(entry)
    for model, rel, prereg in SCALE_RECEIPTS:
        p = root / rel
        if not p.exists():
            buckets["pending"].append({"attacker": "static subspace erasure", "model": model,
                                       "receipt": rel, "prereg": prereg,
                                       "status": "receipt not yet on disk (run pre-registered)"})
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        receipt_hashes[rel] = _sha256(p)
        bucket, entry = _classify_removal_receipt(rec, "static subspace erasure (scale)", rel)
        buckets[bucket].append(entry)
    if not UNBOUNDED_DIMENSIONS:
        raise ValueError("refusing to issue a certificate with no unbounded dimensions")
    survived_models = sorted({e["model"] for e in buckets["claim_scope"] if e["model"]})
    broken_models = sorted({e["model"] for e in buckets["measured_breaks"] if e["model"]})
    cert = {
        "what": "styxx erasure-resistance certificate -- the private-13 substrate honesty read vs "
                "removal-class attackers, composed verbatim from pre-registered receipts",
        "claim": (f"the read survived every admissible removal-class cell on: {', '.join(survived_models)}"
                  if survived_models else "no surviving scope -- see measured_breaks/unadjudicated"),
        "measured_breaks_summary": (f"the read was REMOVED with knowledge held on: {', '.join(broken_models)}"
                                    if broken_models else "none in the receipts composed here"),
        **buckets,
        "unbounded_dimensions": list(UNBOUNDED_DIMENSIONS),
        "receipts_sha256": receipt_hashes,
        "reverify": "re-run the scripts named in styxx.ladder.RUNGS at the commits recorded in each "
                    "RESULT doc; python -m styxx.ladder verifies verdict-string integrity",
        "issued_by": "styxx.ladder.erasure_resistance_certificate (behavior pre-committed blind to "
                     "the 3B verdict; see module comment)",
    }
    if out_path is not None:
        Path(out_path).write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
    return cert


def verify_erasure_certificate(cert: dict | str | Path, root: Path | str = ".") -> dict:
    """Tamper-check an issued erasure-resistance certificate against the live receipts: re-hash every
    receipt the certificate recorded and confirm it still matches. This is what makes the certificate
    an ASSURANCE rather than a snapshot -- an auditor runs it to prove the evidence has not drifted
    since issuance. Returns {ok, checked, mismatches, missing}; ok iff every recorded receipt exists
    and re-hashes to its recorded value. Does NOT re-run the experiments (that is `reverify`); it
    verifies the certificate is a faithful, un-tampered index of receipts that are still on disk."""
    if isinstance(cert, (str, Path)):
        cert = json.loads(Path(cert).read_text(encoding="utf-8"))
    root = Path(root)
    recorded = cert.get("receipts_sha256", {})
    mismatches, missing, checked = [], [], 0
    for rel, sha in recorded.items():
        p = root / rel
        if not p.exists():
            missing.append(rel)
            continue
        checked += 1
        live = _sha256(p)
        if live != sha:
            mismatches.append({"receipt": rel, "recorded": sha, "live": live})
    return {"ok": not mismatches and not missing, "checked": checked,
            "n_recorded": len(recorded), "mismatches": mismatches, "missing": missing}


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m styxx.ladder",
                                 description="the styxx probe-robustness ladder over the repo's canonical receipts")
    ap.add_argument("--root", default=".", help="styxx repository root (default: cwd)")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument("--certificate", metavar="OUT",
                    help="issue the erasure-resistance certificate to OUT (JSON) and print its claim")
    ap.add_argument("--verify", metavar="CERT",
                    help="tamper-check an issued certificate: re-hash its receipts against the live repo")
    a = ap.parse_args()
    if a.verify:
        v = verify_erasure_certificate(a.verify, a.root)
        print(f"certificate receipts: checked {v['checked']}/{v['n_recorded']} -> "
              f"{'OK (un-tampered)' if v['ok'] else 'DRIFT DETECTED'}")
        for m in v["mismatches"]:
            print(f"  MISMATCH {m['receipt']}: recorded {m['recorded'][:12]} != live {m['live'][:12]}")
        for miss in v["missing"]:
            print(f"  MISSING  {miss}")
        return 0 if v["ok"] else 1
    if a.certificate:
        cert = erasure_resistance_certificate(a.root, a.certificate)
        print(f"claim          : {cert['claim']}")
        print(f"measured breaks: {cert['measured_breaks_summary']}")
        print(f"pending        : {len(cert['pending'])} | unadjudicated: {len(cert['unadjudicated'])}")
        print(f"unbounded      : {len(cert['unbounded_dimensions'])} named dimensions")
        print(f"-> {a.certificate}")
        return 0
    problems = verify(a.root)
    if problems:
        print("LADDER: receipts incomplete --")
        for p in problems:
            print(f"  {p}")
        return 1
    rep = report(a.root)
    if a.json:
        print(json.dumps(rep, indent=2))
        return 0
    print("styxx probe-robustness ladder -- canonical receipts\n")
    for r in rep["rungs"]:
        mark = "OK " if r["verdict_matches_canonical"] else "!! "
        print(f"  {mark}{r['rung']:<18} [{r['attacker_class']}]")
        print(f"     verdict: {r['verdict']}")
    pa = rep["parity_attribution"]
    print(f"\n  mandatory line item -- parity attribution: median capacity share "
          f"{pa['median_capacity_share']} over {pa['n_cells']} admissible cells")
    print(f"  ({pa['reading']})")
    ia = rep["instrument_admissibility"]
    if ia["status"] == "issued":
        print(f"\n  line item -- instrument admissibility: {ia['verdict']} "
              f"(discrim {ia['discrim']}, fire_rate {ia['fire_rate']})")
    else:
        print("\n  line item -- instrument admissibility: not yet issued "
              f"(expected at {ia['expected_receipt']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
