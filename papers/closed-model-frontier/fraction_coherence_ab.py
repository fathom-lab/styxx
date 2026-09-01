"""G-F3: the absolute corpus A/B for V11 FRACTION-COHERENCE.

Re-certifies every certified document with the clause ON vs OFF at the same verifier.
The frozen condition: zero tokens move in any direction other than UNGROUNDED -> VERIFIED
via `derived-fraction`, and zero documents move HELD -> FAILED. Every moved token is listed.
One wrong movement fails the gate.

  python papers/closed-model-frontier/fraction_coherence_ab.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import styxx.certify as C                                                # noqa: E402
from styxx.corpus_audit import (_doc_for, _resolve_receipts,             # noqa: E402
                                discover_certificates)

OUT = HERE / "fraction_coherence_ab.json"


def run(flag: bool):
    C.V11_FRACTION_COHERENCE = flag
    out = {}
    for cp in discover_certificates(ROOT / "papers"):
        stored = json.loads(cp.read_text(encoding="utf-8"))
        doc = _doc_for(cp)
        if doc is None or not doc.exists():
            continue
        try:
            paths, _m, _d = _resolve_receipts(cp, stored, ROOT / "papers")
            cert = C.certify_doc(doc, paths)
        except Exception:
            continue
        out[doc.name] = {
            "verdict": cert["verdict"],
            "tokens": {(e["line"], e["col"], str(e["token"])):
                       (e["status"], str(e.get("receipt_ref"))) for e in cert["ledger"]},
        }
    return out


def main() -> int:
    off = run(False)
    on = run(True)
    C.V11_FRACTION_COHERENCE = True          # restore the shipped default

    docs = sorted(set(off) & set(on))
    assert len(docs) >= 150, f"denominator guard: {len(docs)}"
    moved, wrong, verdict_flips, held_to_failed = [], [], [], []
    for d in docs:
        vo, vn = off[d]["verdict"], on[d]["verdict"]
        if vo != vn:
            verdict_flips.append({"doc": d, "off": vo, "on": vn})
            if vo == "OATH-HELD" and vn == "OATH-FAILED":
                held_to_failed.append(d)
        keys = set(off[d]["tokens"]) | set(on[d]["tokens"])
        for k in keys:
            so = off[d]["tokens"].get(k, ("<absent>", ""))
            sn = on[d]["tokens"].get(k, ("<absent>", ""))
            if so == sn:
                continue
            rec = {"doc": d, "line": k[0], "token": k[2],
                   "off": so[0], "on": sn[0], "on_ref": sn[1][:80]}
            moved.append(rec)
            ok = (so[0] == "UNGROUNDED" and sn[0] == "VERIFIED"
                  and sn[1].startswith("derived-fraction"))
            if not ok:
                wrong.append(rec)

    gate = "PASS" if (not wrong and not held_to_failed) else "FAIL"
    payload = {
        "gate": "G-F3 (absolute)",
        "prereg": "PREREG_fraction_coherence_2026_08_31.md",
        "documents_compared": len(docs),
        "tokens_moved": len(moved),
        "moves_every_one": moved,
        "wrong_movements": wrong,
        "verdict_flips": verdict_flips,
        "held_to_failed": held_to_failed,
        "verdict": gate,
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"documents compared: {len(docs)}")
    print(f"tokens moved: {len(moved)}  (wrong: {len(wrong)})")
    for m in moved:
        print(f"  {m['doc'][:52]:52s} L{m['line']} {m['token']:>6} "
              f"{m['off']} -> {m['on']}  {m['on_ref'][:48]}")
    print(f"verdict flips: {verdict_flips}")
    print(f"HELD->FAILED: {held_to_failed}")
    print(f"G-F3: {gate}")
    print(f"-> {OUT.name}")
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
