# -*- coding: utf-8 -*-
"""A transport smoke of the seat runners on P10.4's six invented items, with a hard cap on model calls.

    py -3.12 checks/seat_smoke.py <output folder> [--claude] [--local] [--cap 10]

For each seat asked: step 1 of the six items, then step 2 of at most two of them (the leading "-" and the final line
feed), each send counted with its retry; a seat stops at `--cap` sends. The local seat here is the development server
(127.0.0.1:18790), which P10.3 keeps off every real packet; it answers the transport items only. Nothing here is a
packet item, and no answer here is scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import tpc_seats  # noqa: E402


def smoke(seat, out: Path, cap: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    pk = tpc_seats.transport_packet()
    one = json.loads(json.dumps(pk))
    for it in one["items"]:
        it["steps"] = 1
    p1 = out / "packet_TRANSPORT_step1.json"
    p1.write_bytes((json.dumps(one, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    r1 = tpc_seats.run_seat(p1, seat, out, limit=min(6, cap))
    p2 = out / "packet_TRANSPORT.json"
    p2.write_bytes((json.dumps(pk, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    left = cap - r1["calls"] - 1  # one more send may be a retry
    r2 = tpc_seats.run_seat(p2, seat, out, limit=max(0, min(2, left))) if left > 0 else {"calls": 0, "stopped": "cap"}
    rows = tpc_seats.read_ledger(tpc_seats.ledger_path(out, "TRANSPORT", seat.name))
    sent = [r for r in rows if r.get("input_sha256")]
    expect = {}
    for it in pk["items"]:
        expect[(it["id"], 1)] = hashlib.sha256(tpc_seats.step1_input(it["step1_text"])).hexdigest()
    ok_bytes = all(r["input_sha256"] == expect[(r["id"], 1)] for r in sent if r["step"] == 1)
    return {"seat": seat.describe(), "sends": r1["calls"] + r2["calls"], "step1": r1, "step2": r2,
            "answered": [(r["id"], r["step"]) for r in rows if r.get("answer")],
            "no_answer": [(r["id"], r["step"], r.get("no_answer")) for r in rows if not r.get("answer")],
            "step1_inputs_are_the_packet_bytes": ok_bytes,
            "replies": [{"id": r["id"], "step": r["step"], "answer": r.get("answer"),
                         "transport": {k: v for k, v in (r.get("transport") or {}).items() if k != "session_id"}}
                        for r in rows]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--claude", action="store_true")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--cap", type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out)
    report = {}
    if a.claude:
        report["C"] = smoke(tpc_seats.ClaudeSeat(timeout=300.0), out / "C", a.cap)
    if a.local:
        seat = tpc_seats.ChatSeat(tpc_seats.DEV_LOCAL_URL, "qwen2.5-3b-instruct", dev=True, timeout=120.0)
        report["dev_local"] = smoke(seat, out / "dev_local", a.cap)
    data = (json.dumps(report, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    (out / "seat_smoke.json").write_bytes(data)
    sys.stdout.buffer.write(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
