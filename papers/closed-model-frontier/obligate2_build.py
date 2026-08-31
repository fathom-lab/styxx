"""OBLIGATE-2: implement the frozen predicate, DEV-check it, and build the fresh sample.

Per PREREG_obligate2_2026_08_31.md. The four bar-markers are implemented exactly as frozen
(windows 12 and 40 are frozen numbers). DEV telemetry runs against OBLIGATE-1's 115 spent
adjudications and may not be quoted as a result. The feasibility precondition aborts the
cycle pre-panel if fewer than 30 fresh positives exist.

  python papers/closed-model-frontier/obligate2_build.py
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

SEED = 20260902
CAP = 60
N_PACKETS = 3
SEALED = Path(os.environ.get("STYXX_SEALED_DIR", r"C:\Users\heyzo\clawd\styxx-sealed"))
PACKETS_OUT = HERE / "obligate2_packets.json"
KEYHASH_OUT = HERE / "obligate2_key.sha256"
DEV_OUT = HERE / "obligate2_dev_eval.json"


# ── OBLIGATE-1 base, verbatim from the census rule ───────────────────────────────────────
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


# ── the four frozen bar-markers ──────────────────────────────────────────────────────────
_CMP = re.compile(r"(?:≥|≤|<=|>=|<|>|=|\bat least\b|\bat most\b|\babove\b|\bbelow\b|"
                  r"\bunder\b|\bover\b|\bagainst\b|\bvs\.?\b|\bversus\b)", re.I)
_BARVOCAB = re.compile(r"\b(?:bar|threshold|floor|ceiling|cap|cutoff|criterion|alpha|"
                       r"margin of|chance|frozen|pre-?registered|prereg'?d)\b", re.I)


def bar_adjacent(ctx: str, tok: str) -> tuple[bool, str]:
    """Any of the four frozen bar-markers. Returns (blocked, which)."""
    i = _find(ctx, tok)
    if i < 0:
        return False, ""
    j = i + len(tok)
    # 1. comparator within 12 chars either side
    if _CMP.search(ctx[max(0, i - 12):i]) or _CMP.search(ctx[j:j + 12]):
        return True, "comparator-adjacency"
    # 2. bar vocabulary within 40 chars either side
    if _BARVOCAB.search(ctx[max(0, i - 40):i]) or _BARVOCAB.search(ctx[j:j + 40]):
        return True, "bar-vocabulary"
    # 3. interval position: inside [a, b] or {a, b, ...}
    for op, cl in (("[", "]"), ("{", "}")):
        a = ctx.rfind(op, 0, i)
        b = ctx.find(cl, j)
        if a != -1 and b != -1 and "," in ctx[a:b] and b - a < 60:
            return True, "interval-position"
    # 4. gate-table criterion cell: pipe row, and a comparator in this token's own cell
    if ctx.lstrip().startswith("|"):
        cell_a = ctx.rfind("|", 0, i)
        cell_b = ctx.find("|", j)
        cell = ctx[cell_a + 1:cell_b if cell_b != -1 else len(ctx)]
        if _CMP.search(cell.replace(tok, "", 1)):
            return True, "criterion-cell"
    return False, ""


def obligate2(ctx: str, tok: str) -> bool:
    if not obligate1(ctx, tok):
        return False
    blocked, _ = bar_adjacent(ctx, tok)
    return not blocked


def main() -> int:
    # ── DEV telemetry on OBLIGATE-1's 115 spent adjudications ────────────────────
    o1_pk = json.loads((HERE / "obligate1_packets.json").read_text(encoding="utf-8"))
    o1_key = json.loads((HERE / "obligate1_key.json").read_text(encoding="utf-8"))
    o1_res = json.loads((HERE / "obligate1_result.json").read_text(encoding="utf-8"))
    o1_seats = json.loads((HERE / "obligate1_seat_outputs.json").read_text(encoding="utf-8"))
    rows_ctx = {r["id"]: r for p in o1_pk["packets"] for r in p["rows"]}
    from collections import Counter
    verdicts = {}
    for pk in (1, 2, 3):
        names = [f"p{pk}-seat{s}" for s in (1, 2, 3)]
        labs = {n: {e["id"]: e["label"] for e in o1_seats["seats"][n]["labels"]} for n in names}
        for r in o1_pk["packets"][pk - 1]["rows"]:
            i = r["id"]
            top, cnt = Counter(labs[n][i] for n in names).most_common(1)[0]
            verdicts[i] = top if cnt >= 2 else "NO-MAJORITY"
    dev = {"note": "DEV TELEMETRY on OBLIGATE-1's spent adjudications; may not be quoted as a result"}
    for arm in ("positive", "negative"):
        ids = [i for i, k in o1_key.items()
               if k.get("arm") == arm and verdicts.get(i) in ("CLAIM", "NOT_A_CLAIM")]
        o2 = [i for i in ids if obligate2(rows_ctx[i]["context"], o1_key[i].get("token") or rows_ctx[i]["token"])]
        c = sum(1 for i in o2 if verdicts[i] == "CLAIM")
        dev[arm] = {"adjudicated": len(ids), "obligate2_fires": len(o2), "of_those_CLAIM": c,
                    "precision_dev": round(c / len(o2), 4) if o2 else None}
    DEV_OUT.write_text(json.dumps(dev, indent=1) + "\n", encoding="utf-8")
    print("DEV (spent, telemetry only):", json.dumps({k: v for k, v in dev.items() if k != 'note'}))

    # ── exclusions ───────────────────────────────────────────────────────────────
    excl = set()
    for res_name in ("oath_adjudication_result.json", "oath_internal_result.json"):
        res = json.loads((HERE / res_name).read_text(encoding="utf-8"))
        for band, items in res.get("per_arm_detail", {}).items():
            for a in items:
                excl.add((a.get("repo"), a.get("line"), a.get("token")))
    for i, k in o1_key.items():
        if k.get("arm") in ("positive", "negative"):
            excl.add((k.get("doc"), k.get("line"), k.get("token")))
    excl_lines = set()
    for pk_name in ("agent_claim_packets.json", "stage2_packets.json"):
        pk = json.loads((HERE / pk_name).read_text(encoding="utf-8"))
        for p in pk["packets"]:
            for s in p["sentences"]:
                excl_lines.add(s["text"].strip())

    # ── fresh abstentions ────────────────────────────────────────────────────────
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
        for e in cert.get("ledger", []):
            if e.get("status") != "ABSTAIN":
                continue
            ctx = (e.get("context") or "").strip()[:240]
            tok = str(e.get("token"))
            if not ctx or (str(doc.name), e.get("line"), tok) in excl or ctx in excl_lines:
                continue
            rows.append({"doc": doc.name, "line": e.get("line"), "token": tok, "context": ctx})
    assert docs >= 150, f"denominator guard: {docs}"
    seen, uniq = set(), []
    for r in rows:
        k = (r["doc"], r["line"], r["token"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)

    pos = [r for r in uniq if obligate2(r["context"], r["token"])]
    barband = [r for r in uniq
               if obligate1(r["context"], r["token"]) and not obligate2(r["context"], r["token"])]
    other_neg = [r for r in uniq if not obligate1(r["context"], r["token"])]
    print(f"fresh population: {len(uniq)} abstained | O2-positive {len(pos)} | "
          f"bar-band {len(barband)} | other {len(other_neg)}")

    # ── feasibility precondition (frozen) ────────────────────────────────────────
    assert len(pos) >= 30, ("measurement failed — population insufficient: "
                            f"{len(pos)} fresh OBLIGATE-2 positives < 30")

    rng = random.Random(SEED)
    pos_s = pos if len(pos) <= CAP else rng.sample(pos, CAP)
    n = len(pos_s)
    n_bar = min(len(barband), n // 2)
    bar_s = barband if len(barband) <= n_bar else rng.sample(barband, n_bar)
    rest_s = rng.sample(other_neg, n - len(bar_s))
    print(f"sample: {n} positive + {len(bar_s)} bar-band + {len(rest_s)} other-negative")

    DECOYS = json.loads((SEALED / "obligate2_decoys.json").read_text(encoding="utf-8")) \
        if (SEALED / "obligate2_decoys.json").exists() else None
    if DECOYS is None:
        # reuse the 16 revealed token decoys, per the prereg's disclosed decision
        o1_decoys = [(k["decoy_id"], rows_ctx[i]["context"], rows_ctx[i]["token"], k["truth"])
                     for i, k in o1_key.items() if k.get("arm") == "decoy"]
        seen_d, DECOYS = set(), []
        for did, ctx, tok, tr in sorted(o1_decoys):
            if did not in seen_d:
                seen_d.add(did)
                DECOYS.append({"id": did, "context": ctx, "token": tok, "truth": tr})
    assert len(DECOYS) == 16

    items = ([{"arm": "positive", **r} for r in pos_s]
             + [{"arm": "barband", **r} for r in bar_s]
             + [{"arm": "negative", **r} for r in rest_s])
    random.Random(SEED).shuffle(items)
    parts = [items[i::N_PACKETS] for i in range(N_PACKETS)]

    packets, key = [], {}
    for k, part in enumerate(parts, 1):
        blob = part + [{"arm": "decoy", "decoy_id": d["id"], "truth": d["truth"],
                        "token": d["token"], "context": d["context"]} for d in DECOYS]
        random.Random(SEED + k).shuffle(blob)
        out_rows = []
        for i, it in enumerate(blob, 1):
            sid = f"o2p{k}-{i:03d}"
            out_rows.append({"id": sid, "token": it["token"], "context": it["context"]})
            key[sid] = {kk: vv for kk, vv in it.items() if kk != "context"}
        packets.append({"packet": k, "n": len(out_rows), "rows": out_rows})

    payload = {
        "prereg": "PREREG_obligate2_2026_08_31.md",
        "population": {"documents_recertified": docs, "fresh_abstained": len(uniq),
                       "obligate2_positive": len(pos), "bar_band": len(barband),
                       "other_negative": len(other_neg)},
        "sample": {"positive": n, "barband": len(bar_s), "negative": len(rest_s),
                   "decoys_per_packet": len(DECOYS), "packets": N_PACKETS},
        "packets": packets,
    }
    PACKETS_OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    kb = json.dumps(key, sort_keys=True, ensure_ascii=False).encode("utf-8")
    (SEALED / "obligate2_key.json").write_bytes(kb)
    salt = (SEALED / "agent_claim_key_salt.txt").read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(kb + salt.encode("utf-8")).hexdigest()
    KEYHASH_OUT.write_text(digest + "\n", encoding="utf-8")
    print(f"packets {[p['n'] for p in packets]}  key sealed  sha256 {digest[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
