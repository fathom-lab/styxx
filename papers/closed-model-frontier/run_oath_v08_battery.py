"""OATH v0.8 battery — G1/G2/G3/G4/G5 + invariant I1 per
PREREG_oath_v08_float_field_binding_2026_08_23.

Under test: one severable clause in ``styxx.certify``.

  V08_FLOAT_FIELD_BINDING   a float claim at 1..V08_FIELD_BIND_MAX_DECIMALS fractional digits whose
                            value-matches all sit on receipt paths unrelated to its binding context
                            (widened by V08_FIELD_BIND_PREV_LINES) is DEMOTED from VERIFIED to
                            ABSTAIN with reason ``unbound-field:<receipt>:<path>`` -- but only where
                            the cited receipts carry SOME path the context names (the NAMEABLE
                            test), so an unbindable claim is never demoted for nothing.

The decisive measurement is the silent-pass census run in BOTH arms: mutate one significant digit
of every claim the SHIPPED verifier certifies VERIFIED, re-certify, and count how many mutants come
back VERIFIED -- affirmative false attestation. **The OFF arm is the positive control: if the ON arm
does not fall strictly below it, the run is VOID** (G1), whatever else it reads.

The mutation frame is the OFF arm's VERIFIED set in both arms, so the two are scored on identical
claims. Note honestly that part of the ON arm's gain comes from the clause refusing to swear to the
CLEAN claim rather than to its mutant; that is precisely why G3 bounds the cost ratio and G4
hand-adjudicates whether those refusals are right.

G4 cannot be scored by machine. This harness emits the seeded sample as a dossier; adjudication is
done by hand against the frozen definition in the prereg and recorded in the RESULT note.

Non-destructive: mutants live in temp files, corpus passes are in-memory, and the only files written
are this battery's own result JSON and its G4 dossier.

  python papers/closed-model-frontier/run_oath_v08_battery.py
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

import styxx.certify as C                                                  # noqa: E402
from styxx.certify import certify_doc                                      # noqa: E402
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

GATING_SEED = 2      # the operating point was chosen against seed 1; G2/G3 score a FRESH seed
REPORT_SEED = 1      # reported for comparability with the published 604 baseline; does NOT gate
ADJ_SEED = 11
ADJ_N = 40
BASELINE = HERE / "oath_v08_baseline_ledger.json"
OUT = HERE / "oath_v08_battery_result.json"
DOSSIER = HERE / "oath_v08_g4_adjudication_sample.json"


def mutate_token(tok: str, rng: random.Random) -> str:
    """The repo's own battery operator (validate_oath_v0.mutate_token), unchanged."""
    digits = [i for i, ch in enumerate(tok) if ch.isdigit()]
    sig = [i for i in digits if not (tok[i] == "0" and (i == 0 or not tok[:i].strip("+-0.")))]
    pos = rng.choice(sig or digits)
    old = int(tok[pos])
    new = rng.choice([d for d in range(10) if d != old])
    return tok[:pos] + str(new) + tok[pos + 1:]


def substitute(line: str, tok: str, mut: str) -> tuple[str, bool]:
    """Land *mut* in place of *tok*, honouring the typographic minus.

    Copied from `run_oath_v07_battery.py`. Extraction normalizes U+2212 to ASCII '-', so a negative
    token is reported in ASCII while the document holds U+2212 and a bare `line.replace` silently
    no-ops on every signed claim -- scoring a harness miss as a verifier miss."""
    if tok in line:
        return line.replace(tok, mut, 1), True
    if tok.startswith("-"):
        alt, alt_mut = tok.replace("-", "−", 1), mut.replace("-", "−", 1)
        if alt in line:
            return line.replace(alt, alt_mut, 1), True
    return line, False


def resolvable_docs():
    out = []
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
            out.append((doc, receipts))
    return out


def corpus_pass(docs, flag: bool) -> dict:
    C.V08_FLOAT_FIELD_BINDING = flag
    ledger, verdicts, roster = {}, {}, []
    for doc, receipts in docs:
        try:
            cert = certify_doc(doc, receipts)
        except Exception:
            continue
        rel = doc.relative_to(ROOT).as_posix()
        verdicts[rel] = cert["verdict"]
        for e in cert["ledger"]:
            ledger[f"{rel}|L{e['line']}|{e['token']}"] = e["status"]
            ref = e.get("receipt_ref")
            if isinstance(ref, str) and ref.startswith("unbound-field:"):
                roster.append({"doc": doc.name, "line": e["line"], "token": e["token"],
                               "decimals": e["decimals"], "ref": ref,
                               "context": e["context"][:170]})
    return {"ledger": ledger, "verdicts": verdicts, "roster": roster}


def census(docs, frame, seed: int, flag: bool) -> dict:
    """Mutate every claim in *frame* and re-certify under *flag*. Frame is fixed across arms."""
    C.V08_FLOAT_FIELD_BINDING = flag
    rng = random.Random(seed)
    rows, skipped = [], 0
    for doc, receipts in docs:
        claims = frame.get(doc.name)
        if not claims:
            continue
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        for ln_no, tok, dec in claims:
            mut = mutate_token(tok, rng)
            if mut == tok or ln_no - 1 >= len(lines):
                skipped += 1
                continue
            ml = list(lines)
            ml[ln_no - 1], landed = substitute(ml[ln_no - 1], tok, mut)
            if not landed:
                skipped += 1
                continue
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
                           if e["line"] == ln_no and e["token"] == mut), "NOT_EXTRACTED")
            rows.append({"doc": doc.name, "line": ln_no, "token": tok, "mutant": mut,
                         "decimals": dec, "status": status})
    n = len(rows)
    silent = [r for r in rows if r["status"] != "UNGROUNDED"]
    return {"flag": flag, "seed": seed, "mutated": n, "skipped": skipped,
            "status": dict(Counter(r["status"] for r in rows)),
            "false_verified": sum(1 for r in rows if r["status"] == "VERIFIED"),
            "silent_total": len(silent),
            "silent_share": round(len(silent) / n, 4) if n else 0.0,
            "false_verified_by_decimals": {str(k): v for k, v in sorted(
                Counter(r["decimals"] for r in rows if r["status"] == "VERIFIED").items())},
            "rows": rows}


def main() -> int:
    t0 = time.time()
    if not hasattr(C, "V08_FLOAT_FIELD_BINDING"):
        print("FATAL: styxx.certify has no V08_FLOAT_FIELD_BINDING — nothing to test.")
        return 2
    if not BASELINE.exists():
        print(f"FATAL: baseline {BASELINE.name} missing — G5 cannot be scored.")
        return 2
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    original = C.V08_FLOAT_FIELD_BINDING

    docs = resolvable_docs()
    print(f"docs with resolvable receipts: {len(docs)}")
    print(f"V08_FIELD_BIND_MAX_DECIMALS={C.V08_FIELD_BIND_MAX_DECIMALS}  "
          f"V08_FIELD_BIND_PREV_LINES={C.V08_FIELD_BIND_PREV_LINES}\n", flush=True)

    # ---- corpus passes: the clean-side cost, the severability reference, and invariant I1
    c_off = corpus_pass(docs, False)
    c_on = corpus_pass(docs, True)

    diffs = [k for k in set(c_off["ledger"]) | set(base["ledger"])
             if c_off["ledger"].get(k) != base["ledger"].get(k)]
    print(f"G5 severability: {len(diffs)} ledger differences with the flag OFF (bar 0)")
    for k in diffs[:8]:
        print(f"  [DIFF] {k}: baseline={base['ledger'].get(k)} off={c_off['ledger'].get(k)}")

    changed = [k for k in c_on["ledger"] if c_on["ledger"][k] != c_off["ledger"].get(k)]
    kinds = Counter((c_off["ledger"].get(k), c_on["ledger"][k]) for k in changed)
    ung_off = sum(1 for v in c_off["ledger"].values() if v == "UNGROUNDED")
    ung_on = sum(1 for v in c_on["ledger"].values() if v == "UNGROUNDED")
    held_off = sum(1 for v in c_off["verdicts"].values() if v == "OATH-HELD")
    held_on = sum(1 for v in c_on["verdicts"].values() if v == "OATH-HELD")
    i1 = (ung_off == ung_on and held_off == held_on
          and set(kinds) <= {("VERIFIED", "ABSTAIN")})
    print(f"\nI1 invariant (asserted, NOT gated): UNGROUNDED {ung_off}->{ung_on}  "
          f"HELD {held_off}->{held_on}  transitions {dict(kinds)}  -> {'HOLDS' if i1 else 'VIOLATED'}")
    print(f"clean-corpus demotions VERIFIED->ABSTAIN: {len(changed)}")

    # ---- mutation frame: the SHIPPED verifier's attestation surface, fixed across both arms
    C.V08_FLOAT_FIELD_BINDING = False
    frame, total_claims = {}, 0
    for doc, receipts in docs:
        try:
            cert = certify_doc(doc, receipts)
        except Exception:
            continue
        cl = [(e["line"], e["token"], e["decimals"]) for e in cert["ledger"]
              if e["status"] == "VERIFIED"]
        if cl:
            frame[doc.name] = cl
            total_claims += len(cl)
    print(f"\nmutation frame (OFF-arm VERIFIED claims): {total_claims}\n", flush=True)

    arms = {}
    for label, seed in (("gating", GATING_SEED), ("reported", REPORT_SEED)):
        print(f"census seed {seed} ({label}) ...", flush=True)
        off = census(docs, frame, seed, False)
        on = census(docs, frame, seed, True)
        arms[label] = {"seed": seed, "off": off, "on": on,
                       "false_verified_removed": off["false_verified"] - on["false_verified"]}
        print(f"  false-VERIFIED  OFF {off['false_verified']}  ON {on['false_verified']}  "
              f"removed {arms[label]['false_verified_removed']}")
        print(f"  silent total    OFF {off['silent_total']} ({off['silent_share']})  "
              f"ON {on['silent_total']} ({on['silent_share']})", flush=True)

    g = arms["gating"]
    removed = g["false_verified_removed"]
    positive_control = g["on"]["false_verified"] < g["off"]["false_verified"]
    cost_ratio = len(changed) / max(removed, 1)

    # ---- G4 dossier: seeded sample for hand-adjudication
    rng = random.Random(ADJ_SEED)
    roster = sorted(c_on["roster"], key=lambda r: (r["doc"], r["line"], r["token"]))
    sample = rng.sample(roster, min(ADJ_N, len(roster)))
    DOSSIER.write_text(json.dumps({
        "note": "G4 hand-adjudication sample for PREREG_oath_v08_float_field_binding_2026_08_23; "
                "classify each row GENUINE-BINDING-DESTROYED / COINCIDENCE-CORRECTED / "
                "SPEC-CORRECTED per the frozen definition. Ties resolve to GENUINE.",
        "seed": ADJ_SEED, "n": len(sample), "roster_total": len(roster), "rows": sample,
    }, indent=2) + "\n", encoding="utf-8")

    C.V08_FLOAT_FIELD_BINDING = original

    report = {
        "prereg": "PREREG_oath_v08_float_field_binding_2026_08_23.md",
        "verifier_sha256": hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "baseline_verifier_sha256": base["verifier_sha256"],
        "V08_FIELD_BIND_MAX_DECIMALS": C.V08_FIELD_BIND_MAX_DECIMALS,
        "V08_FIELD_BIND_PREV_LINES": C.V08_FIELD_BIND_PREV_LINES,
        "docs": len(docs), "mutation_frame_claims": total_claims,
        "I1_invariant_asserted_not_gated": {
            "ungrounded_off": ung_off, "ungrounded_on": ung_on,
            "held_off": held_off, "held_on": held_on,
            "transition_kinds": {f"{a}->{b}": n for (a, b), n in kinds.items()},
            "holds": i1},
        "clean_demotions": len(changed),
        "unbound_field_roster_size": len(roster),
        "G1_positive_control": {"role": "VOID condition", "on": g["on"]["false_verified"],
                                "off": g["off"]["false_verified"], "pass": positive_control},
        "G2_benefit": {"role": "decisive", "seed": GATING_SEED, "removed": removed,
                       "bar": 60, "pass": removed >= 60},
        "G3_cost_ratio": {"role": "regression guard", "demotions": len(changed),
                          "removed": removed, "ratio": round(cost_ratio, 3),
                          "bar": 1.5, "pass": cost_ratio <= 1.5},
        "G4_adjudication": {"role": "decisive", "seed": ADJ_SEED, "n": len(sample),
                            "bar": "GENUINE-BINDING-DESTROYED <= 12 of 40",
                            "dossier": DOSSIER.name, "pass": None,
                            "note": "hand-scored in the RESULT note; not machine-gradable"},
        "G5_severability": {"role": "gated", "ledger_differences": len(diffs),
                            "bar": 0, "examples": diffs[:20], "pass": len(diffs) == 0},
        "census": arms,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'gate':22s} {'bar':>10s} {'measured':>12s}  verdict")
    print("-" * 60)
    print(f"{'G1 positive control':22s} {'ON < OFF':>10s} "
          f"{str(g['on']['false_verified']) + ' < ' + str(g['off']['false_verified']):>12s}  "
          f"{'OK' if positive_control else 'VOID'}")
    print(f"{'G2 benefit':22s} {'>= 60':>10s} {removed:>12d}  "
          f"{'PASS' if removed >= 60 else 'FAIL'}")
    print(f"{'G3 cost ratio':22s} {'<= 1.5':>10s} {cost_ratio:>12.3f}  "
          f"{'PASS' if cost_ratio <= 1.5 else 'FAIL'}")
    print(f"{'G4 adjudication':22s} {'<= 12/40':>10s} {'by hand':>12s}  -> {DOSSIER.name}")
    print(f"{'G5 severability':22s} {'0':>10s} {len(diffs):>12d}  "
          f"{'PASS' if not diffs else 'FAIL'}")
    print(f"{'I1 (not a gate)':22s} {'invariant':>10s} {'':>12s}  "
          f"{'HOLDS' if i1 else 'VIOLATED'}")
    print(f"\nelapsed {report['elapsed_s']}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
