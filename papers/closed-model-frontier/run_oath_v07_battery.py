"""OATH v0.7 battery — G1/G2/G2b/G3/G4/G5 per PREREG_oath_v07_precision_obligation_2026_08_22.

Under test: two severable clauses in ``styxx.certify``.

  V07_PRECISION_OBLIGATION  a token printed at >= V07_PRECISION_DIGITS fractional digits is
                            OBLIGATED regardless of line vocabulary.
  V07_ULP_ESCAPE            an obligation created by that clause ALONE, with no match but a
                            receipt leaf within V07_ULP_N ULP, degrades to ABSTAIN with a named
                            ``ulp-neighbour`` reason — never to VERIFIED.

Every mutation gate runs in BOTH arms, flags OFF (the shipped v0.6.2 verifier) and flags ON, on
the identical sample with the identical mutation seed. **The OFF arm is the positive control: if
the ON arm does not exceed it on G2, the run is VOID** — a battery that reports the same number in
both arms is not measuring the clause, whatever else it clears.

Non-destructive: mutants live in temp files, the corpus pass is in-memory, and the only file
written is this battery's own result JSON.

  python papers/closed-model-frontier/run_oath_v07_battery.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import styxx.certify as C                                                  # noqa: E402
from styxx.certify import (certify_doc, extract_numbers,                   # noqa: E402
                           _TRIGGERS, _TRIGGERS_CORR)
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

SEED = 1
N = 20
FULLPREC_MIN = 7
BASELINE = HERE / "oath_v07_baseline_ledger.json"


def is_bound_shipped(bctx: str, value: float, decimals: int) -> bool:
    """The v0.6.2 obligation predicate — the split that defines the two pools."""
    if _TRIGGERS.search(bctx):
        return True
    return bool(decimals > 0 and -1.0 <= value <= 1.0 and _TRIGGERS_CORR.search(bctx))


def mutate_sig16(tok: str, rng: random.Random) -> str:
    """Perturb one significant fractional digit among positions 1-6 (the v0.6.1 scheme)."""
    frac_at = tok.index(".") + 1
    frac = tok[frac_at:]
    sig = [i for i in range(min(6, len(frac))) if not (frac[i] == "0" and set(frac[:i]) <= {"0"})]
    pos = frac_at + rng.choice(sig or [0])
    old = int(tok[pos])
    new = rng.choice([d for d in range(10) if d != old])
    return tok[:pos] + str(new) + tok[pos + 1:]


def substitute(line: str, tok: str, mut: str) -> tuple[str, bool]:
    """Land *mut* in place of *tok*, honouring the typographic minus.

    HARNESS DEFECT, OWNED AND FIXED HERE: extraction normalizes U+2212 to ASCII '-' (v0.6.2), so a
    negative token is reported in ASCII while the document holds U+2212. A bare
    ``line.replace(tok, mut, 1)`` therefore silently no-ops on every signed claim, the mutant never
    lands, and the gate reads NOT_EXTRACTED — scoring a harness miss as a verifier miss. The
    inherited `run_oath_v061_battery.py` carries the same bare replace; it is not modified here
    (the instrument never moves), the defect is simply not repeated in this cycle's harness."""
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


def build_pools(docs):
    unbound, bound = [], []
    for doc, receipts in docs:
        text = doc.read_text(encoding="utf-8", errors="replace")
        for e in extract_numbers(text):
            if e["decimals"] < FULLPREC_MIN:
                continue
            bctx = e.get("binding_context", e["context"])
            item = (doc, receipts, e["line"], e["token"])
            (bound if is_bound_shipped(bctx, e["value"], e["decimals"])
             else unbound).append(item)
    return unbound, bound


def draw(pool, rng, n):
    seen, out = set(), []
    for item in rng.sample(pool, len(pool)):
        key = (item[0].name, item[3])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) == n:
            break
    return out


def set_flags(precision: bool, ulp: bool) -> None:
    C.V07_PRECISION_OBLIGATION = precision
    C.V07_ULP_ESCAPE = ulp


def run_arm(items, mut_seed: int, precision: bool, ulp: bool) -> dict:
    set_flags(precision, ulp)
    rng = random.Random(mut_seed)
    caught = verified = abstained = other = unlanded = 0
    rows = []
    for doc, receipts, ln_no, tok in items:
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        mut = mutate_sig16(tok, rng)
        ml = list(lines)
        ml[ln_no - 1], landed = substitute(ml[ln_no - 1], tok, mut)
        unlanded += not landed
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                         encoding="utf-8") as tf:
            tf.write("\n".join(ml))
            tmp = Path(tf.name)
        try:
            cert = certify_doc(tmp, receipts)
        finally:
            tmp.unlink(missing_ok=True)
        entry = next((e for e in cert["ledger"]
                      if e["line"] == ln_no and e["token"] == mut), None)
        status = entry["status"] if entry else "NOT_EXTRACTED"
        caught += status == "UNGROUNDED"
        verified += status == "VERIFIED"
        abstained += status == "ABSTAIN"
        other += status not in ("UNGROUNDED", "VERIFIED", "ABSTAIN")
        rows.append({"doc": doc.name, "line": ln_no, "token": tok, "mutant": mut,
                     "status": status, "landed": landed,
                     "ref": (entry or {}).get("receipt_ref")})
    return {"precision": precision, "ulp_escape": ulp, "n": len(items),
            "caught_ungrounded": caught, "false_verified": verified,
            "abstained": abstained, "other": other,
            "mutants_that_did_not_land": unlanded, "rows": rows}


def corpus_pass(docs, precision: bool, ulp: bool) -> dict:
    set_flags(precision, ulp)
    ledger, verdicts, ungrounded, ulp_roster = {}, {}, {}, []
    for doc, receipts in docs:
        try:
            cert = certify_doc(doc, receipts)
        except Exception:
            continue
        rel = doc.relative_to(ROOT).as_posix()
        verdicts[rel] = cert["verdict"]
        for e in cert["ledger"]:
            key = f"{rel}|L{e['line']}|{e['token']}"
            ledger[key] = e["status"]
            if e["status"] == "UNGROUNDED":
                ungrounded[key] = e["context"][:170]
            if isinstance(e.get("receipt_ref"), str) and e["receipt_ref"].startswith("ulp-neighbour:"):
                ulp_roster.append({"key": key, "ref": e["receipt_ref"],
                                   "context": e["context"][:150]})
    return {"ledger": ledger, "verdicts": verdicts,
            "ungrounded": ungrounded, "ulp_roster": ulp_roster}


def main() -> int:
    t0 = time.time()
    for flag in ("V07_PRECISION_OBLIGATION", "V07_ULP_ESCAPE",
                 "V07_PRECISION_DIGITS", "V07_ULP_N"):
        if not hasattr(C, flag):
            print(f"FATAL: styxx.certify has no {flag} — nothing to test.")
            return 2
    if not BASELINE.exists():
        print(f"FATAL: baseline {BASELINE.name} missing — G5 cannot be scored.")
        return 2
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    digits, ulp_n = C.V07_PRECISION_DIGITS, C.V07_ULP_N
    original = (C.V07_PRECISION_OBLIGATION, C.V07_ULP_ESCAPE)

    docs = resolvable_docs()
    unbound_pool, bound_pool = build_pools(docs)
    total = len(unbound_pool) + len(bound_pool)
    print(f"docs with resolvable receipts: {len(docs)}")
    print(f"full-precision pool: {total}  unbound: {len(unbound_pool)}  bound: {len(bound_pool)}")
    print(f"V07_PRECISION_DIGITS={digits}  V07_ULP_N={ulp_n}\n")

    rng = random.Random(SEED)
    g2_items = draw(unbound_pool, rng, N)
    g4_items = draw(bound_pool, rng, N)

    # ---- G1 (recall): the sampled unbound tokens must be visible to extraction.
    extracted = 0
    for doc, _r, ln_no, tok in g2_items:
        line = doc.read_text(encoding="utf-8", errors="replace").splitlines()[ln_no - 1]
        extracted += any(e["token"] == tok for e in extract_numbers(line))
    print(f"G1 recall (extracted)   : {extracted}/{N} (bar 18)")

    # ---- G2 / G2b: unbound-line catch, ON vs the OFF positive control.
    g2_off = run_arm(g2_items, SEED + 100, False, False)
    g2_on = run_arm(g2_items, SEED + 100, True, True)
    print(f"G2 unbound catch  ON    : {g2_on['caught_ungrounded']}/{N} (bar 16)")
    print(f"G2b positive control OFF: {g2_off['caught_ungrounded']}/{N} (must be <=2 and < ON)")

    # ---- G4: no regression on the bound half.
    g4_off = run_arm(g4_items, SEED + 200, False, False)
    g4_on = run_arm(g4_items, SEED + 200, True, True)
    print(f"G4 bound catch    ON    : {g4_on['caught_ungrounded']}/{N} "
          f"(bar 16, OFF {g4_off['caught_ungrounded']})")
    false_verified = g2_on["false_verified"] + g4_on["false_verified"]
    print(f"G4 false-VERIFIED total : {false_verified} (bar 0)")

    # ---- G3: false accusation on the CLEAN corpus, ON vs baseline.
    print(f"\nG3 clean corpus: {len(docs)} documents", flush=True)
    c_on = corpus_pass(docs, True, True)
    base_ung = {k for k, s in base["ledger"].items() if s == "UNGROUNDED"}
    new_flags = sorted(set(c_on["ungrounded"]) - base_ung)
    lost_flags = sorted(base_ung - set(c_on["ungrounded"]))
    held_before = {d for d, v in base["verdicts"].items() if v == "OATH-HELD"}
    held_after = {d for d, v in c_on["verdicts"].items() if v == "OATH-HELD"}
    flipped = sorted(held_before - held_after)
    print(f"G3 UNGROUNDED: baseline {len(base_ung)} -> ON {len(c_on['ungrounded'])} "
          f"| NEW {len(new_flags)} | LOST {len(lost_flags)}")
    for k in new_flags:
        print(f"  [NEW] {k}\n        {c_on['ungrounded'][k][:130]}")
    print(f"G3 certificates flipped HELD->FAILED: {len(flipped)}")
    for d in flipped:
        print(f"  [FLIP] {d}")
    print(f"G3b ulp-neighbour roster: {len(c_on['ulp_roster'])} token(s)")
    for u in c_on["ulp_roster"]:
        print(f"  [ULP] {u['key']}  <- {u['ref'][:90]}")

    # ---- G5: severability. Flags OFF must reproduce the pre-fix ledger exactly.
    c_off = corpus_pass(docs, False, False)
    diffs = [k for k in set(c_off["ledger"]) | set(base["ledger"])
             if c_off["ledger"].get(k) != base["ledger"].get(k)]
    print(f"\nG5 severability: {len(diffs)} ledger differences with both flags OFF (bar 0)")
    for k in diffs[:10]:
        print(f"  [DIFF] {k}: baseline={base['ledger'].get(k)} off={c_off['ledger'].get(k)}")

    C.V07_PRECISION_OBLIGATION, C.V07_ULP_ESCAPE = original

    positive_control = g2_on["caught_ungrounded"] > g2_off["caught_ungrounded"]
    report = {
        "prereg": "PREREG_oath_v07_precision_obligation_2026_08_22.md",
        "seed": SEED, "n": N,
        "V07_PRECISION_DIGITS": digits, "V07_ULP_N": ulp_n,
        "verifier_sha256": hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "baseline_verifier_sha256": base["verifier_sha256"],
        "pool": {"total": total, "unbound": len(unbound_pool), "bound": len(bound_pool),
                 "unbound_share": round(len(unbound_pool) / max(total, 1), 4),
                 "docs": len(docs)},
        "G1_recall": {"extracted": extracted, "bar": 18, "pass": extracted >= 18},
        "G2_unbound_catch": {"on": g2_on, "bar": 16,
                             "pass": g2_on["caught_ungrounded"] >= 16},
        "G2b_positive_control": {"off": g2_off, "bar_max": 2,
                                 "pass": g2_off["caught_ungrounded"] <= 2 and positive_control,
                                 "on_exceeds_off": positive_control},
        "G3_no_false_accusation": {
            "baseline_ungrounded": len(base_ung),
            "on_ungrounded": len(c_on["ungrounded"]),
            "new_flags": [{"key": k, "context": c_on["ungrounded"][k]} for k in new_flags],
            "lost_flags": lost_flags,
            "certificates_flipped": flipped,
            "ulp_neighbour_roster": c_on["ulp_roster"],
            "pass_mechanical": len(lost_flags) == 0,
            "bar": "every NEW flag must hand-adjudicate GENUINE per the frozen artifact "
                   "definition; zero lost flags",
        },
        "G4_no_tamper_regression": {
            "on": g4_on, "off": g4_off, "bar": 16,
            "false_verified_total": false_verified,
            "pass": (g4_on["caught_ungrounded"] >= 16
                     and g4_on["caught_ungrounded"] >= g4_off["caught_ungrounded"]
                     and false_verified == 0)},
        "G5_severability": {"ledger_differences": len(diffs), "bar": 0,
                            "examples": diffs[:20], "pass": len(diffs) == 0},
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = HERE / "oath_v07_battery_result.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    gated = {"G1": report["G1_recall"]["pass"],
             "G2": report["G2_unbound_catch"]["pass"],
             "G2b": report["G2b_positive_control"]["pass"],
             "G3": report["G3_no_false_accusation"]["pass_mechanical"],
             "G4": report["G4_no_tamper_regression"]["pass"],
             "G5": report["G5_severability"]["pass"]}
    print(f"\npositive control (ON must exceed OFF on G2): "
          f"{'OK' if positive_control else 'VOID — the battery is not measuring the clause'}")
    print("gates: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in gated.items()))
    print("G3 new flags still require hand-adjudication against the frozen artifact definition.")
    print(f"elapsed {report['elapsed_s']}s -> {out.name}")
    return 0 if all(gated.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
