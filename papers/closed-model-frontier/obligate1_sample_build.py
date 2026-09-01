"""OBLIGATE-1 ground-truth build: fresh abstentions, stratified by the frozen predicate.

Per PREREG_obligate1_2026_08_31.md: numeric tokens the CURRENT verifier abstains on, drawn
from live re-certification of the certified corpus, EXCLUDING every (source, line, token)
triple the 2026-08-28 RECON adjudicated and every sentence used in either claim-detector
packet set. Stratified: all OBLIGATE-1-positive abstentions up to 60, plus an equal count of
negatives, seed 20260901. Token decoys (authored at build time, truths sealed) ride in every
packet. The predicate is lifted VERBATIM from the census's best rule.

  python papers/closed-model-frontier/obligate1_sample_build.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import certify_doc                                     # noqa: E402
from styxx.corpus_audit import (_doc_for, _resolve_receipts,              # noqa: E402
                                discover_certificates)

SEED = 20260901
CAP = 60
N_PACKETS = 3
SEALED = Path(os.environ.get("STYXX_SEALED_DIR", r"C:\Users\heyzo\clawd\styxx-sealed"))
PACKETS_OUT = HERE / "obligate1_packets.json"
KEYHASH_OUT = HERE / "obligate1_key.sha256"


# ── the frozen candidate, lifted verbatim from oath_structural_obligation_census.py ──────
def _find(ctx: str, tok: str) -> int:
    for m in re.finditer(re.escape(tok), ctx):
        a, b = m.start(), m.end()
        if (a == 0 or not ctx[a - 1].isdigit()) and (b == len(ctx) or not ctx[b].isdigit()):
            return a
    return -1


def _code_spans(ctx: str):
    spans, open_at = [], None
    for m in re.finditer("`", ctx):
        if open_at is None:
            open_at = m.end()
        else:
            spans.append((open_at, m.start()))
            open_at = None
    return spans


def obligate1(ctx: str, tok: str) -> bool:
    if not ("." in tok and len(tok.split(".")[-1]) >= 2):
        return False
    i = _find(ctx, tok)
    if i < 0:
        return False
    return not any(a <= i < b for a, b in _code_spans(ctx))


# ── token decoys: authored here, truths sealed, gated at >= 0.80 ─────────────────────────
DECOYS = [
    (1, "Held-out recall landed at 0.8213 after the split.", "0.8213", "CLAIM"),
    (2, "The margin widened to 0.4471 on the second run.", "0.4471", "CLAIM"),
    (3, "Mean latency settled at 2.35 seconds per call.", "2.35", "CLAIM"),
    (4, "Coverage reached 0.9127 of the addressable corpus.", "0.9127", "CLAIM"),
    (5, "The floor held at 0.6644 across every fold.", "0.6644", "CLAIM"),
    (6, "Agreement came out at 0.7789 between the two seats.", "0.7789", "CLAIM"),
    (7, "The pooled share fell to 0.0912 after exclusions.", "0.0912", "CLAIM"),
    (8, "Throughput averaged 41.27 documents per minute.", "41.27", "CLAIM"),
    (9, "See section 3.2.1 for the derivation.", "3.2.1", "NOT_A_CLAIM"),
    (10, "Reproduce with seed 343 and the pinned wheel.", "343", "NOT_A_CLAIM"),
    (11, "Shipped in v0.6.2 alongside the loader fix.", "0.6.2", "NOT_A_CLAIM"),
    (12, "Row 14 of the table carries the exception.", "14", "NOT_A_CLAIM"),
    (13, "The run finished on 2026-07-12 without incident.", "2026", "NOT_A_CLAIM"),
    (14, "Issue #38 tracks the replication effort.", "38", "NOT_A_CLAIM"),
    (15, "Set `timeout=30.00` in the client config.", "30.00", "NOT_A_CLAIM"),
    (16, "The third experiment of five is still queued.", "five", "NOT_A_CLAIM"),
]


def main() -> int:
    # exclusions: the RECON's adjudicated triples + both claim-detector packet sentence sets
    excl_triples = set()
    pairs = [("oath_adjudication_result.json", "external"),
             ("oath_internal_result.json", "internal")]
    for res_name, _arm in pairs:
        res = json.loads((HERE / res_name).read_text(encoding="utf-8"))
        for band, items in res.get("per_arm_detail", {}).items():
            for a in items:
                excl_triples.add((a.get("repo"), a.get("line"), a.get("token")))
    excl_lines = set()
    for pk_name in ("agent_claim_packets.json", "stage2_packets.json"):
        pk = json.loads((HERE / pk_name).read_text(encoding="utf-8"))
        for p in pk["packets"]:
            for s in p["sentences"]:
                excl_lines.add(s["text"].strip())

    # live re-certification of the corpus; collect ABSTAIN tokens with their line context
    rows, docs = [], 0
    for cp in discover_certificates(ROOT / "papers"):
        stored = json.loads(cp.read_text(encoding="utf-8"))
        doc = _doc_for(cp)
        if doc is None or not doc.exists():
            continue
        try:
            paths, _m, _d = _resolve_receipts(cp, stored, ROOT / "papers")
            cert = certify_doc(doc, paths)
        except Exception:
            continue
        docs += 1
        text_lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        for e in cert.get("ledger", []):
            if e.get("status") != "ABSTAIN":
                continue
            ln = e.get("line")
            ctx = (e.get("context") or "").strip()[:240]
            if not ctx and isinstance(ln, int) and 1 <= ln <= len(text_lines):
                ctx = text_lines[ln - 1].strip()[:240]
            if not ctx:
                continue
            tok = str(e.get("token"))
            if (str(doc.name), ln, tok) in excl_triples or ctx in excl_lines:
                continue
            rows.append({"doc": doc.name, "line": ln, "token": tok, "context": ctx})
    assert docs >= 150, f"denominator guard: only {docs} documents re-certified"

    # dedupe on (doc, line, token)
    seen, uniq = set(), []
    for r in rows:
        k = (r["doc"], r["line"], r["token"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)

    pos = [r for r in uniq if obligate1(r["context"], r["token"])]
    neg = [r for r in uniq if not obligate1(r["context"], r["token"])]
    rng = random.Random(SEED)
    pos_s = pos if len(pos) <= CAP else rng.sample(pos, CAP)
    assert len(pos_s) >= 30, f"positive arm below floor before adjudication: {len(pos_s)}"
    neg_s = rng.sample(neg, len(pos_s))

    items = ([{"arm": "positive", **r} for r in pos_s]
             + [{"arm": "negative", **r} for r in neg_s])
    rng2 = random.Random(SEED)
    rng2.shuffle(items)
    parts = [items[i::N_PACKETS] for i in range(N_PACKETS)]

    packets, key = [], {}
    for k, part in enumerate(parts, 1):
        blob = part + [{"arm": "decoy", "decoy_id": i, "truth": t,
                        "token": tok, "context": ctx}
                       for i, ctx, tok, t in DECOYS]
        random.Random(SEED + k).shuffle(blob)
        rows_out = []
        for i, it in enumerate(blob, 1):
            sid = f"o1p{k}-{i:03d}"
            rows_out.append({"id": sid, "token": it["token"], "context": it["context"]})
            key[sid] = {kk: vv for kk, vv in it.items() if kk not in ("context",)}
        packets.append({"packet": k, "n": len(rows_out), "rows": rows_out})

    payload = {
        "prereg": "PREREG_obligate1_2026_08_31.md",
        "population": {"documents_recertified": docs,
                       "abstained_tokens_after_exclusions": len(uniq),
                       "obligate1_positive": len(pos), "obligate1_negative": len(neg)},
        "sample": {"positive": len(pos_s), "negative": len(neg_s),
                   "decoys_per_packet": len(DECOYS), "packets": N_PACKETS},
        "packets": packets,
    }
    PACKETS_OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    kb = json.dumps(key, sort_keys=True, ensure_ascii=False).encode("utf-8")
    (SEALED / "obligate1_key.json").write_bytes(kb)
    salt = (SEALED / "agent_claim_key_salt.txt").read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(kb + salt.encode("utf-8")).hexdigest()
    KEYHASH_OUT.write_text(digest + "\n", encoding="utf-8")

    print(f"docs {docs}  abstained(after excl) {len(uniq)}  "
          f"OBLIGATE-1 pos {len(pos)} / neg {len(neg)}")
    print(f"sample: {len(pos_s)} positive + {len(neg_s)} negative + "
          f"{len(DECOYS)} decoys x {N_PACKETS} packets -> {[p['n'] for p in packets]}")
    print(f"key sealed; sha256 -> {KEYHASH_OUT.name}: {digest[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
