"""OATH v0.12 battery — scored under PREREG_oath_v12_formula_constant_2026_08_26.

Two-armed (flag OFF / flag ON) at the ship-candidate verifier, on the frame as frozen in the
prereg. Non-destructive: nothing is written but this battery's own result JSON.

**This battery reports a KILL, and it stops where the outcome table tells it to.** G2 under-fires,
which the pre-committed table maps to `V12_UNDERREACH` — revert and publish. The gates after it
are NOT scored, and the result JSON says so by name rather than leaving a reader to assume they
passed. Running a warrant panel to adjudicate a clause that does not reach its own class would be
theatre, and scoring gates a dead clause cannot benefit from is how a battery starts flattering.

The one thing worth reading twice: the clause misses **the prereg's own motivating specimen**. The
prereg quotes a formula, pre-commits that its own certificate must flip to OATH-HELD when the
clause lands, and the formula is written as an indented code block — so there is no inline-code
span, no `$` delimiter, and conjunct 1 never fires. The prereg froze G2 against a LINE-level
census and then specified a SPAN-level clause; the gap between those two populations is the whole
defect.

  python papers/closed-model-frontier/run_oath_v12_battery.py
"""
from __future__ import annotations

import collections
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

C = importlib.import_module("styxx.certify")
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

OUT = HERE / "oath_v12_battery_result.json"
CENSUS = HERE / "oath_mention_use_census.json"

# ---- frozen expectations, quoted from the prereg ------------------------------------------
G1_FRAME = {"documents": 181, "tokens": 7665,
            "VERIFIED": 5646, "ABSTAIN": 2011, "UNGROUNDED": 8}
G2_ROSTER_SIZE = 11          # "the 11-token roster the census records for latex_on_line"
G2_ROSTER_SPLIT = {"UNGROUNDED": 3, "VERIFIED": 8}
PROOF_OF_REPAIR = ("papers/SYNTHESIS_mention_and_use_2026_08_26.md",
                   "papers/closed-model-frontier/PREREG_oath_v12_formula_constant_2026_08_26.md")

# Certificates created AFTER the census froze the frame. Excluded from the frame pass so G1's
# pin is reconstructible without moving files off disk — which mattered, because the first
# version of this harness required holding the v0.12 prereg's certificate aside, and that is
# also the document whose proof-of-repair had to be scored. A reconstruction that hides its
# own subject answers a question nobody asked.
#
# Declared here rather than applied silently: this is a NAMED reconstruction of the frozen
# frame, and any reader can check the list against the census's date.
FRAME_EXCLUDE = ("papers/closed-model-frontier/"
                 "PREREG_oath_v12_formula_constant_2026_08_26.certificate.json",)


def set_v12(on: bool) -> None:
    C.V12_FORMULA_CONSTANT = on


def resolvable_docs():
    out = []
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        if "anc" in cp.parts:
            continue
        if cp.relative_to(ROOT).as_posix() in FRAME_EXCLUDE:
            continue
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        if not doc.exists():
            continue
        try:
            rec = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        receipts, missing, _ = _resolve_receipts(cp, rec, ROOT / "papers")
        if receipts and not missing:
            out.append((doc, receipts))
    return out


def frame_pass(docs, on: bool) -> dict:
    set_v12(on)
    return {d.relative_to(ROOT).as_posix(): C.certify_doc(d, r) for d, r in docs}


def firings(cert) -> set:
    return {(e["line"], e.get("col"), e["token"])
            for e in cert["ledger"] if e["receipt_ref"] == "formula_constant"}


def main() -> int:
    t0 = time.time()
    docs = resolvable_docs()
    off = frame_pass(docs, False)
    on = frame_pass(docs, True)
    set_v12(False)

    # ---- G1 -------------------------------------------------------------------------------
    counts = collections.Counter()
    for c in off.values():
        counts.update(c["counts"])
    frame = {"documents": len(off), "tokens": sum(len(c["ledger"]) for c in off.values()),
             "VERIFIED": counts["VERIFIED"], "ABSTAIN": counts["ABSTAIN"],
             "UNGROUNDED": counts["UNGROUNDED"]}
    g1 = {"gate": "G1", "name": "INSTRUMENT VALIDITY (VOID-producing)",
          "verdict": "PASS" if frame == G1_FRAME else "VOID:V12_BATTERY_VOID",
          "frame_at_freeze": G1_FRAME, "frame_at_run": frame}

    # ---- G2 -------------------------------------------------------------------------------
    fired = {rel: firings(c) for rel, c in on.items()}
    total = sum(len(f) for f in fired.values())
    split = collections.Counter()
    for rel, f in fired.items():
        by_coord = {(e["line"], e.get("col"), e["token"]): e["status"]
                    for e in off[rel]["ledger"]}
        for coord in f:
            split[by_coord.get(coord, "?")] += 1
    g2_ok = total == G2_ROSTER_SIZE and dict(split) == G2_ROSTER_SPLIT
    g2 = {"gate": "G2", "name": "FIRING-SURFACE EXACTNESS",
          "verdict": "PASS" if g2_ok else
                     ("FAIL:V12_OVERREACH" if total > G2_ROSTER_SIZE else "FAIL:V12_UNDERREACH"),
          "roster_expected": G2_ROSTER_SIZE, "roster_split_expected": G2_ROSTER_SPLIT,
          "firings_observed": total, "firings_split_observed": dict(split),
          "firing_documents": {rel: sorted(f) for rel, f in fired.items() if f},
          "note": "The prereg froze this roster from the census's LINE-level marker "
                  "(`latex_on_line`) and then specified a SPAN-level clause. Those are different "
                  "populations; the clause reaches the smaller one."}

    # ---- proof of repair, pre-committed in the prereg --------------------------------------
    # Scored DIRECTLY from each document's own certificate rather than read out of the frame
    # pass. Reconstructing the frozen frame means holding a certificate aside, and a
    # proof-of-repair that silently reports `None` because its subject was the thing held aside
    # is a gate answering a question nobody asked.
    por = []
    for rel in PROOF_OF_REPAIR:
        doc = ROOT / rel
        cp = doc.with_name(doc.name.replace(".md", ".certificate.json"))
        if not (doc.exists() and cp.exists()):
            por.append({"document": rel, "flips": None, "note": "document or certificate absent"})
            continue
        rec = json.loads(cp.read_text(encoding="utf-8"))
        receipts, missing, _ = _resolve_receipts(cp, rec, ROOT / "papers")
        if missing:
            por.append({"document": rel, "flips": None, "note": f"receipts missing: {missing}"})
            continue
        set_v12(False)
        o = C.certify_doc(doc, receipts)
        set_v12(True)
        n = C.certify_doc(doc, receipts)
        set_v12(False)
        por.append({"document": rel, "off": o["verdict"], "on": n["verdict"],
                    "flips": o["verdict"] != "OATH-HELD" and n["verdict"] == "OATH-HELD",
                    "still_accused": [{"line": u["line"], "token": u["token"]}
                                      for u in n["ungrounded"]]})
    por_ok = all(p["flips"] for p in por)

    # ---- severability: OFF must be inert ---------------------------------------------------
    inert = all(e["receipt_ref"] != "formula_constant"
                for c in off.values() for e in c["ledger"])

    outcome = ("V12_BATTERY_VOID" if g1["verdict"].startswith("VOID")
               else g2["verdict"].split(":", 1)[1] if g2["verdict"].startswith("FAIL")
               else "V12_UNDERREACH" if not por_ok
               else "V12_GATES_G1_G2_PASS__REMAINDER_NOT_SCORED")

    payload = {
        "battery": "OATH v0.12 — the formula constant",
        "prereg": "papers/closed-model-frontier/PREREG_oath_v12_formula_constant_2026_08_26.md",
        "verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "clause": "V12_FORMULA_CONSTANT",
        "shipped_value": C.V12_FORMULA_CONSTANT,
        "outcome": outcome,
        "gates_scored": [g1, g2],
        "proof_of_repair": {
            "requirement": "the prereg pre-commits that BOTH its own certificate and the "
                           "SYNTHESIS's must flip OATH-FAILED -> OATH-HELD when the clause "
                           "lands; if either fails to flip the cycle under-reached regardless "
                           "of the other gates",
            "documents": por, "all_flip": por_ok},
        "severability_off_arm_inert": inert,
        "frame_reconstruction": {
            "excluded_from_frame": list(FRAME_EXCLUDE),
            "why": "certificates created after the census froze the frame; excluded so G1's pin "
                   "is reconstructible in place. Named, not silent.",
            "note": "the excluded document is still scored for proof-of-repair, directly from "
                    "its own certificate — the frame is reconstructed, the subject is not hidden"},
        "gates_not_scored": {
            "G3 conversion ledger": "not reached — G2 is terminal under the outcome table",
            "G4 cost is zero genuine verifications": "not reached",
            "G5 value-blindness": "not reached",
            "G6 collateral": "not reached",
            "G7 blind adjudication": "NOT RUN. A warrant panel adjudicates whether a retraction "
                                     "is DESERVED; it cannot rescue a clause that does not reach "
                                     "its class. Running one here would be theatre, and its "
                                     "population was in any case unsatisfiable as frozen — the "
                                     "prereg asks for 10 non-roster tokens from lines carrying a "
                                     "LaTeX span and the frame contains fewer.",
            "G8 suite closure": "not reached",
            "G9 mechanism proof": "not reached"},
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"G1 {g1['verdict']}   G2 {g2['verdict']}")
    print(f"  firings {total} (expected {G2_ROSTER_SIZE})  split {dict(split)}")
    for p in por:
        print(f"  proof-of-repair {p['document'].split('/')[-1][:50]:<52} flips={p.get('flips')}")
    print(f"\nOUTCOME: {outcome}  -> {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
