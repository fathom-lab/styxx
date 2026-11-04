# -*- coding: utf-8 -*-
"""vectorlib.py — the transforms a conformance vector applies to base.jsonl (Python side; run_vectors.cjs is the
JavaScript side, and the two must produce the same bytes). `canon` is the writer's one byte form."""
from __future__ import annotations

import json


def canon(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def bridge(chain_problems, entries, withheld):
    """ledger.py's chain problems read the way verify.js reads a withheld list: at the published line after a
    withheld run the seq gap is the bridge (recorded in `bridged`), and `prev` must name the last withheld
    digest — a mismatch stays a problem. Returns (problems, bridged)."""
    wmap = {w["seq"]: w["entry_id"] for w in (withheld or [])}
    if not wmap:
        return list(chain_problems), []
    bridged, kept = [], []
    for pr in chain_problems:
        s = pr.get("seq")
        if s is None or (s - 1) not in wmap:
            kept.append(pr)
            continue
        k, run = s - 1, []
        while k in wmap:
            run.append(k)
            k -= 1
        bridged.extend(sorted(run))
        e = next(x for x in entries if x.get("seq") == s)
        left = [p for p in pr["problems"] if not p.startswith("seq is ")]
        if e.get("prev") == wmap[s - 1]:
            left = [p for p in left if p != "prev does not name the line before"]
        if left:
            kept.append({"line": pr.get("line"), "seq": s, "problems": left})
    return kept, sorted(bridged)


def anchors_ok(L, entries, withheld, texts):
    """verify.js's checkAnchor, answered from ledger.py's reading of the entries."""
    wmap = {w["seq"]: w["entry_id"] for w in (withheld or [])}
    out = []
    for t in texts:
        a = L.parse_anchor_text(t)
        if not a:
            out.append(False)
            continue
        by = next((e for e in entries if e.get("seq") == a["seq"]), None)
        after = sum(1 for e in entries if e.get("day") == a["date"] and isinstance(e.get("seq"), int) and e["seq"] > a["seq"])
        own = next((e for e in entries if e.get("kind") == "anchor" and e.get("anchor_date") == a["date"]), None)
        in_chain = (by is not None and by.get("entry_id") == a["head"]) or (by is None and wmap.get(a["seq"]) == a["head"])
        out.append(bool(in_chain and after == 0 and (own is None or (own.get("anchor_head") == a["head"] and own.get("anchor_head_seq") == a["seq"]))))
    return out


def answer(L, text: bytes, withheld, anchors):
    """What a Python verifier answers for a vector, in the vector's terms."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "v.jsonl"
        p.write_bytes(text)
        header, entries, problems = L.read_log(p)
    chain, bridged = bridge(L.check_chain(header, entries), entries, withheld)
    allp = problems + chain
    out = {"ok": not allp, "problem_seqs": sorted({p["seq"] for p in allp if p.get("seq") is not None}),
           "problems_without_seq": sum(1 for p in allp if p.get("seq") is None), "head": L.head_of(header, entries),
           "bridged": bridged}
    if anchors:
        out["anchors_ok"] = anchors_ok(L, entries, withheld, anchors)
    return out


def apply(base: bytes, t: dict, L=None) -> bytes:
    """`L` is the gate's ledger module (for entry_id_of / header_digest in the rechain op); the other ops need none."""
    op = t["op"]
    lines = base.split(b"\n")
    if op == "identity":
        return base
    if op == "crlf_first_line":
        return base.replace(b"\n", b"\r\n", 1)
    if op == "truncate_last":
        kept = base.rstrip(b"\n").split(b"\n")[:-1]
        return b"\n".join(kept) + b"\n"
    out = []
    for ln in lines:
        if not ln.strip():
            out.append(ln)
            continue
        o = json.loads(ln)
        s = o.get("seq")
        if op == "set_field" and s == t["seq"]:
            o[t["field"]] = t["value"]
            out.append(canon(o))
        elif op == "drop_seq" and s == t["seq"]:
            continue
        elif op == "dup_key" and s == t["seq"]:
            out.append(ln[:-1] + b',"%s":%s}' % (t["key"].encode("utf-8"), json.dumps(t["value"]).encode("utf-8")))
        else:
            out.append(ln)
    if op == "set_field_rechain":
        assert L is not None
        prev = None
        res = []
        for ln in out:
            if not ln.strip():
                res.append(ln)
                continue
            o = json.loads(ln)
            if o.get("schema") == L.LOG_SCHEMA:
                prev = L.header_digest(o)
                res.append(ln)
                continue
            if o["seq"] == t["seq"]:
                o[t["field"]] = t["value"]
            if o["seq"] >= t["seq"]:
                o["prev"] = prev
                o["entry_id"] = L.entry_id_of({k: v for k, v in o.items() if k != "entry_id"})
                res.append(canon(o))
            else:
                res.append(ln)
            prev = o["entry_id"]
        out = res
    return b"\n".join(out)
