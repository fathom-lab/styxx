#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""anchor_route_scan.py: did each delivered anchor leave a send line, and did the hook runner run for it (F-1).

    py -3.12 -B anchor_route_scan.py --gate C:/Users/heyzo/clawd/gate --log <gateway log> [--log ...]
                                     [--clawdbot-json ~/.clawdbot/clawdbot.json]

Reads, never writes: the ledger (every `kind: anchor` line, every `kind: send` line with `payload_kind: anchor`),
`anchor_state.json` (the days the plugin marked delivered, and when), `plugin_failures.jsonl` (present or not), the
gate's `sources/` folder (gate.py's check keeps a source file for every text in the anchor's form it judges, so an
anchor text there means gate.py saw it), and the gateway's own log lines `[hooks] running message_sending (N
handlers` / `message_sent`, with their times. For each delivered day it prints whether a send line, a source file
and a hook run exist within two minutes of the delivery. With --clawdbot-json it prints the KEY NAMES of the
config's `plugins.entries` and `channels` and nothing else of that file (it holds credentials). Every file input is
named with its sha256, and the `sources/` folder, read live, by a digest over its file names and their sha256, so the
reading can be checked against the bytes it was taken from. It also prints, for each delivery, the log lines of any
kind within two minutes, and for each E2 `outbound` send line the nearest `message_sending` hook line before it. Prints one JSON object; standard
library only. Names no destination, prints no text but the anchor's own fixed form, which the ledger already holds.
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

HOOK_RE = re.compile(r"\[hooks\] running (message_sending|message_sent) \((\d+) handlers")
ANCHOR_RE = re.compile(r"gate anchor (\d{4}-\d{2}-\d{2}) head ([0-9a-f]{64}) seq (\d+)")


def _t(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _file(p: Path) -> dict:
    b = p.read_bytes()
    return {"name": p.name, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--log", action="append", default=[])
    ap.add_argument("--clawdbot-json", default=None)
    ap.add_argument("--ledger", default=None, help="a frozen copy of the ledger in place of <gate>/ledger.jsonl")
    ap.add_argument("--anchor-state", default=None, help="a frozen copy in place of <gate>/anchor_state.json")
    ap.add_argument("--until", default=None, help="count only hook lines stamped at or before this UTC time; the log "
                    "file is any clawdbot process's, so lines a test process wrote later are left out by time")
    a = ap.parse_args(argv)
    gate = Path(a.gate)
    ledger_path = Path(a.ledger) if a.ledger else gate / "ledger.jsonl"
    state_path = Path(a.anchor_state) if a.anchor_state else gate / "anchor_state.json"
    until = _t(a.until) if a.until else None
    lines = [json.loads(x) for x in ledger_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    lines = [x for x in lines if "seq" in x]
    anchors = [x for x in lines if x.get("kind") == "anchor"]
    anchor_sends = [x for x in lines if x.get("kind") == "send" and x.get("payload_kind") == "anchor"]
    sends = [x for x in lines if x.get("kind") == "send"]
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    delivered = state.get("delivered") or {}
    src_texts = set()
    src_digest = hashlib.sha256()
    src_files = 0
    for p in sorted((gate / "sources").glob("*")) if (gate / "sources").is_dir() else []:
        try:
            b = p.read_bytes()
        except OSError:
            continue
        src_files += 1
        src_digest.update(p.name.encode("utf-8") + b"\n" + hashlib.sha256(b).hexdigest().encode() + b"\n")
        m = ANCHOR_RE.search(b.decode("utf-8", errors="replace"))
        if m:
            src_texts.add(m.group(1))
    hooks = []
    any_lines = []
    for lp in a.log:
        for raw in Path(lp).read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(raw)
            except ValueError:
                continue
            when = (r.get("_meta") or {}).get("date")
            if not when or (until is not None and _t(when) > until):
                continue
            any_lines.append(_t(when))
            m = HOOK_RE.search(str(r.get("1", "")))
            if m:
                hooks.append((_t(when), m.group(1), int(m.group(2))))
    sending = sorted(h[0] for h in hooks if h[1] == "message_sending")
    days = []
    for day, rec in sorted(delivered.items()):
        at = _t(rec["at"])
        near = [h for h in hooks if abs((h[0] - at).total_seconds()) <= 120]
        days.append({
            "day": day,
            "delivered_at": rec["at"],
            "anchor_line_seq": rec.get("anchor_seq"),
            "send_lines_payload_kind_anchor_that_day": [x["seq"] for x in anchor_sends
                                                        if x.get("timestamp", "")[:10] == rec["at"][:10]],
            "source_file_with_this_day_anchor_text": day in src_texts,
            "hook_runs_within_120s": [{"hook": h[1], "handlers": h[2], "at": h[0].isoformat()} for h in near],
            "log_lines_of_any_kind_within_120s": sum(1 for t in any_lines if abs((t - at).total_seconds()) <= 120),
        })
    # every E2 outbound send line, and the message_sending hook line nearest before it (the hook runs before gate.py
    # writes the line)
    e2 = []
    for x in sends:
        if x.get("exit") != "E2" or not x.get("timestamp"):
            continue
        ts = _t(x["timestamp"])
        before = [s for s in sending if s <= ts + dt.timedelta(seconds=1)]
        gap = (ts - before[-1]).total_seconds() if before else None
        e2.append({"seq": x["seq"], "timestamp": x["timestamp"], "nearest_message_sending_before_s": gap})
    out = {
        "inputs": {
            "ledger": {**_file(ledger_path), "lines": len(lines), "head_seq": lines[-1]["seq"] if lines else None},
            "anchor_state": _file(state_path) if state_path.exists() else None,
            "logs": [_file(Path(lp)) for lp in a.log],
            "sources_dir": {"files": src_files, "digest_over_names_and_sha256": src_digest.hexdigest(),
                            "read_live": True},
            "plugin_failures_jsonl": _file(gate / "plugin_failures.jsonl") if (gate / "plugin_failures.jsonl").exists() else None,
        },
        "e2_outbound_sends_and_their_hook_lines": e2,
        "anchor_lines": [{"seq": x["seq"], "anchor_date": x.get("anchor_date"), "timestamp": x.get("timestamp"),
                          "process": x.get("process")} for x in anchors],
        "send_lines": len(sends),
        "send_lines_payload_kind_anchor": len(anchor_sends),
        "send_lines_by_payload_kind": {k: sum(1 for x in sends if x.get("payload_kind") == k)
                                       for k in sorted({x.get("payload_kind") for x in sends}, key=str)},
        "plugin_failures_jsonl_present": (gate / "plugin_failures.jsonl").exists(),
        "hook_runs_in_logs": {k: sum(1 for h in hooks if h[1] == k) for k in ("message_sending", "message_sent")},
        "hook_handler_counts_seen": sorted({h[2] for h in hooks}),
        "hook_line_span": [min(h[0] for h in hooks).isoformat(), max(h[0] for h in hooks).isoformat()] if hooks else None,
        "hook_lines_until": a.until,
        "delivered_days": days,
    }
    if a.clawdbot_json:
        cfg = json.loads(Path(a.clawdbot_json).read_text(encoding="utf-8"))
        plugins = cfg.get("plugins") if isinstance(cfg, dict) else None
        entries = plugins.get("entries") if isinstance(plugins, dict) else None
        channels = cfg.get("channels") if isinstance(cfg, dict) else None
        out["config_key_names"] = {
            "plugins_entries": sorted(entries.keys()) if isinstance(entries, dict) else None,
            "channels": sorted(channels.keys()) if isinstance(channels, dict) else None,
        }
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
