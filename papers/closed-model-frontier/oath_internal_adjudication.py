"""Build the internal arm's ledger and blind packets.

Prereg: `PREREG_oath_verified_channel_internal_2026_08_27.md`, frozen and committed first.

The external cycle found that only about half of external `OATH-VERIFIED` tokens are claims at
all. The defence is that those authors never signed the contract. This builds the same measurement
over the documents that did — this laboratory's own certified corpus — using the same question,
the same tie direction and the same seed, so the two are comparable.

Tokens are presented to adjudicators identically to the external run. Nothing in a packet says
which corpus it came from.

  python papers/closed-model-frontier/oath_internal_adjudication.py build
  python papers/closed-model-frontier/oath_internal_adjudication.py score judgements.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from oath_adjudication import PACKET_SIZE, PANEL, QUESTION, SEED, _verdict  # noqa: E402
from styxx.certify import _TRIGGERS, certify_doc                            # noqa: E402
from styxx.corpus_audit import (_doc_for, _resolve_receipts,                # noqa: E402
                                discover_certificates)

LEDGER = HERE / "oath_internal_ledger.jsonl"
PACKETS = HERE / "oath_internal_packets.json"
KEY = HERE / "oath_internal_key.json"
RESULT = HERE / "oath_internal_result.json"

# --- frozen by the prereg ----------------------------------------------------------------------
N_VERIFIED = 150          # the arm under test
N_ABSTAIN_DECOYS = 75
CONTEXT_CHARS = 200       # identical presentation to the external run


def build_ledger() -> list[dict]:
    rows, docs, failed = [], 0, 0
    for cp in discover_certificates(ROOT / "papers"):
        try:
            cert = json.loads(cp.read_text(encoding="utf-8"))
            doc = _doc_for(cp)
            paths, _missing, _drift = _resolve_receipts(cp, cert, ROOT / "papers")
            if not doc.exists() or not paths:
                failed += 1
                continue
            live = certify_doc(doc, paths)
        except Exception:
            failed += 1
            continue
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        for e in live["ledger"]:
            ln = lines[e["line"] - 1] if 0 < e["line"] <= len(lines) else ""
            rows.append({
                "repo": doc.name,          # same field name as the external ledger, on purpose
                "sha": cert.get("document_sha256", "")[:40],
                "line": e["line"], "col": e.get("col"), "token": e["token"],
                "value": e["value"], "status": e["status"], "receipt_ref": e["receipt_ref"],
                "obligating_words": sorted({m.group(0).lower()
                                            for m in _TRIGGERS.finditer(ln)}),
                "context": ln.strip()[:CONTEXT_CHARS],
            })
        docs += 1
    LEDGER.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                      encoding="utf-8")
    print(f"internal ledger: {docs} documents, {len(rows)} tokens, {failed} unresolvable")
    return rows


def build() -> dict:
    rows = build_ledger()
    rng = random.Random(SEED)
    ver = [r for r in rows if r["status"] == "VERIFIED"]
    ab = [r for r in rows if r["status"] == "ABSTAIN"]
    ung = [r for r in rows if r["status"] == "UNGROUNDED"]

    pick_v = rng.sample(ver, min(N_VERIFIED, len(ver)))
    pick_a = rng.sample(ab, min(N_ABSTAIN_DECOYS, len(ab)))
    pick_u = list(ung)                       # all of them; there are few

    items, key = [], {}
    for src in (pick_v, pick_a, pick_u):
        for r in src:
            iid = f"I{len(items):04d}"
            items.append({"id": iid, "repo": r["repo"], "line": r["line"], "token": r["token"],
                          "context": r["context"],
                          "obligating_words": r.get("obligating_words", [])})
            key[iid] = {"status": r["status"], "repo": r["repo"], "line": r["line"],
                        "token": r["token"], "receipt_ref": r.get("receipt_ref")}
    rng.shuffle(items)
    packets = [items[i:i + PACKET_SIZE] for i in range(0, len(items), PACKET_SIZE)]

    payload = {
        "prereg": "papers/closed-model-frontier/PREREG_oath_verified_channel_internal_2026_08_27.md",
        "arm": "this laboratory's own certified corpus, re-certified live at the pinned verifier",
        "seed": SEED, "packet_size": PACKET_SIZE, "panel": PANEL, "question": QUESTION,
        "composition": {"verified": len(pick_v), "abstain_decoys": len(pick_a),
                        "ungrounded_decoys": len(pick_u), "total_items": len(items),
                        "packets": len(packets)},
        "available": {"verified": len(ver), "abstained": len(ab), "ungrounded": len(ung)},
        "blinding": ("Identical presentation to the external run; nothing in a packet indicates "
                     "which corpus an item came from, or what the verifier decided about it."),
        "packets": packets,
    }
    PACKETS.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    KEY.write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"verified {len(pick_v)}  abstain-decoys {len(pick_a)}  ungrounded-decoys {len(pick_u)}")
    print(f"-> {PACKETS.name} ({len(packets)} packets)  /  {KEY.name} (withheld)")
    return payload


def score(judgements_path: Path) -> dict:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    judged = json.loads(Path(judgements_path).read_text(encoding="utf-8"))
    by_item: dict[str, list[str]] = {}
    for j in judged:
        by_item.setdefault(j["id"], []).append(j["verdict"])

    arms: dict[str, list] = {"VERIFIED": [], "ABSTAIN": [], "UNGROUNDED": []}
    unan, split = 0, 0
    for iid, votes in by_item.items():
        if iid not in key:
            continue
        arms[key[iid]["status"]].append({"id": iid, "verdict": _verdict(votes),
                                         "votes": votes, **key[iid]})
        unan += len(set(votes)) == 1
        split += len(set(votes)) != 1

    def share(rows, want):
        return round(sum(1 for r in rows if r["verdict"] == want) / len(rows), 4) if rows else None

    ext = json.loads((HERE / "oath_adjudication_result.json").read_text(encoding="utf-8"))
    v_int = share(arms["VERIFIED"], "CLAIM")
    v_ext = ext["verified_arm_sanity"]["rate"]
    out = {
        "prereg": "papers/closed-model-frontier/PREREG_oath_verified_channel_internal_2026_08_27.md",
        "items_scored": sum(len(v) for v in arms.values()),
        "verified_claim_share_internal": {"n": len(arms["VERIFIED"]), "rate": v_int},
        "verified_claim_share_external": {"n": ext["verified_arm_sanity"]["n"], "rate": v_ext},
        "gap": round(v_int - v_ext, 4) if v_int is not None else None,
        "miss_rate_internal": {"n": len(arms["ABSTAIN"]),
                               "rate": share(arms["ABSTAIN"], "CLAIM")},
        "false_accusation_rate_internal": {"n": len(arms["UNGROUNDED"]),
                                           "rate": share(arms["UNGROUNDED"], "NOT_A_CLAIM")},
        "agreement": {"unanimous": unan, "split": split,
                      "unanimity_share": round(unan / (unan + split), 4) if unan + split else None,
                      "external_unanimity": ext["agreement"]["unanimity_share"]},
        "per_arm_detail": arms,
    }
    RESULT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_arm_detail"}, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "score"))
    ap.add_argument("judgements", nargs="?", default=None)
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    else:
        if not a.judgements:
            raise SystemExit("score needs a judgements JSON path")
        score(Path(a.judgements))
    return 0


if __name__ == "__main__":
    sys.exit(main())
