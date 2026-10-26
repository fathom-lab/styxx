#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gate_words_scan.py: the lint on this checkout, run over the stored texts of the gate's public send lines.

A count the RESULT prints beside the gate's own lint refusals, nothing more. The gate runs
``scripts/surface_lint.py`` at its pin, so a rule added on ``main`` after the pin (the ``withdrawn/gate-*`` rules of
``papers/darkflobi-gate/AMENDMENT_darkflobi_gate_lint_overclaims_2026_10_25.md``) reaches the gate only at the next
pin; this script says how many delivered public lines the checkout's lint would have hit, per rule id, by seq and by
day, so the window's record carries that number without moving the pin.

    py -3.12 papers/darkflobi-gate/gate_words_scan.py [--w-root C:/Users/heyzo/clawd] [--ledger PATH]
        [--sources DIR] [--lint PATH] [--classes public[,operator_dm,...]] [--json]

What it reads: ``W:gate/ledger.jsonl`` (one JSON object per line; a line is a send when ``kind`` is ``send``) and,
for every send line whose ``dest_class`` is listed (default ``public``, the PREREG's population), the stored checked
text ``W:gate/sources/<checked_sha256>.md``. The gate wrote that file at check time; it is the checked text, which
for a prepared (minted) post is the text above the capsule footer, where the gate's own lint ran over the whole
payload as handed, so the two can differ on a footer, never on a sentence the agent wrote. A send line with no stored
text (no ``checked_sha256``, or the file is gone) is counted as ``no_source`` and listed by seq.

What it prints: per rule id, the number of lines hit and their seqs; the same for matches the lint counts as
allowed (a withdrawal form within its window); the ledger's own ``lint_hit:<rule id>`` reason codes per seq, so the
pin's lint at send time stands beside this checkout's; and the hit lines by day. Never a byte of any text: no
excerpt, no token, no destination hash. Exit 0 when every listed send line was read, 1 when some had no stored text,
2 on a usage or read error. Standard library only; read-only on every path it is given.

What it is not: a check that any sentence is true (the lint matches strings and passes paraphrase), a count the gates
block reads (the gates block reads the ledger's own codes), or a replay of the gate (``gate_verify.py`` is that).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_LINT = REPO / "scripts" / "surface_lint.py"
DEFAULT_W_ROOT = "C:/Users/heyzo/clawd"
LINT_CODE = "lint_hit:"


def load_lint(path: Path):
    """The lint module loaded from its file, so the rules counted are the ones at this path and no other."""
    spec = importlib.util.spec_from_file_location("surface_lint_for_scan", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the lint at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "lint_text") or not hasattr(mod, "RULES"):
        raise RuntimeError(f"{path} has no lint_text/RULES")
    return mod


def ledger_lines(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """(line number, object) for every JSON object in the ledger; a line that is not an object is an error."""
    with path.open("r", encoding="utf-8") as f:
        for n, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{n}: not JSON ({e.msg})")
            if not isinstance(obj, dict):
                raise RuntimeError(f"{path}:{n}: not an object")
            yield n, obj


def scan(ledger: Path, sources: Path, lint, classes: Sequence[str]) -> Dict[str, Any]:
    seqs_seen: List[int] = []
    head: Optional[str] = None
    lines_total = 0
    selected: List[Dict[str, Any]] = []
    for _, obj in ledger_lines(ledger):
        lines_total += 1
        if isinstance(obj.get("entry_id"), str):
            head = obj["entry_id"]
        if obj.get("kind") != "send":
            continue
        seq = obj.get("seq")
        if isinstance(seq, int):
            seqs_seen.append(seq)
        if obj.get("dest_class") not in classes:
            continue
        selected.append(obj)

    hits: Dict[str, List[int]] = {}
    allowed: Dict[str, List[int]] = {}
    ledger_codes: Dict[str, List[int]] = {}
    by_day: Dict[str, Dict[str, int]] = {}
    no_source: List[Dict[str, Any]] = []
    with_source = 0
    for obj in selected:
        seq = obj.get("seq")
        day = obj.get("day") or "?"
        d = by_day.setdefault(day, {"lines": 0, "hit": 0, "no_source": 0})
        d["lines"] += 1
        for code in obj.get("reasons") or []:
            if isinstance(code, str) and code.startswith(LINT_CODE):
                ledger_codes.setdefault(code[len(LINT_CODE):], []).append(seq)
        sha = obj.get("checked_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            no_source.append({"seq": seq, "why": "no_checked_sha256"})
            d["no_source"] += 1
            continue
        src = sources / (sha + ".md")
        if not src.is_file():
            no_source.append({"seq": seq, "why": "file_missing"})
            d["no_source"] += 1
            continue
        text = src.read_bytes().decode("utf-8", errors="replace")
        with_source += 1
        h, a = lint.lint_text(text, f"seq:{seq}")
        rids = sorted({x.rid for x in h})
        for rid in rids:
            hits.setdefault(rid, []).append(seq)
        for rid in sorted({x.rid for x in a}):
            allowed.setdefault(rid, []).append(seq)
        if rids:
            d["hit"] += 1

    lint_path = Path(lint.__file__)
    return {
        "ledger": ledger.as_posix(), "ledger_lines": lines_total,
        "seq_range": [min(seqs_seen), max(seqs_seen)] if seqs_seen else None,
        "head_entry_id_prefix": head[:16] if head else None,
        "sources": sources.as_posix(),
        "lint": lint_path.as_posix(), "lint_sha256": hashlib.sha256(lint_path.read_bytes()).hexdigest(),
        "lint_rules": len(lint.RULES) + 1,
        "classes": list(classes), "send_lines": len(selected), "with_source": with_source,
        "no_source": no_source,
        "hits": {k: v for k, v in sorted(hits.items())},
        "allowed": {k: v for k, v in sorted(allowed.items())},
        "ledger_lint_codes": {k: v for k, v in sorted(ledger_codes.items())},
        "by_day": {k: by_day[k] for k in sorted(by_day)},
    }


def render(r: Dict[str, Any]) -> str:
    out = [f"gate_words_scan: ledger {r['ledger']} ({r['ledger_lines']} lines, send seq "
           f"{'-'.join(str(x) for x in r['seq_range']) if r['seq_range'] else 'none'}, head {r['head_entry_id_prefix']})",
           f"lint {r['lint']} (sha256 {r['lint_sha256'][:16]}..., {r['lint_rules']} rules); sources {r['sources']}",
           f"classes {','.join(r['classes'])}: send lines {r['send_lines']}, with a stored text {r['with_source']}, "
           f"without {len(r['no_source'])}"]
    out.append("by rule (this checkout's lint over the stored texts):")
    out += [f"  {rid:38s} lines {len(seqs):3d}  seqs {', '.join(str(s) for s in seqs)}"
            for rid, seqs in r["hits"].items()] or ["  none"]
    out.append("allowed (a withdrawal form within the lint's window):")
    out += [f"  {rid:38s} lines {len(seqs):3d}  seqs {', '.join(str(s) for s in seqs)}"
            for rid, seqs in r["allowed"].items()] or ["  none"]
    out.append("the ledger's own lint codes (the pin's lint at send time):")
    out += [f"  {rid:38s} lines {len(seqs):3d}  seqs {', '.join(str(s) for s in seqs)}"
            for rid, seqs in r["ledger_lint_codes"].items()] or ["  none"]
    out.append("by day: " + ("; ".join(f"{day} lines {d['lines']} hit {d['hit']} no_source {d['no_source']}"
                                        for day, d in r["by_day"].items()) or "none"))
    if r["no_source"]:
        out.append("no stored text: " + ", ".join(f"seq {x['seq']} ({x['why']})" for x in r["no_source"]))
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="gate_words_scan", description=(
        "Run this checkout's surface lint over the stored texts of the gate's public send lines and count the "
        "hits per rule id, by seq and by day. Prints counts and seqs, never text."))
    ap.add_argument("--w-root", default=DEFAULT_W_ROOT, help="the box's W: folder (default: C:/Users/heyzo/clawd)")
    ap.add_argument("--ledger", help="the ledger (default: <w-root>/gate/ledger.jsonl)")
    ap.add_argument("--sources", help="the stored texts (default: <w-root>/gate/sources)")
    ap.add_argument("--lint", default=str(DEFAULT_LINT), help="the lint to run (default: this checkout's)")
    ap.add_argument("--classes", default="public", help="comma-separated dest_class values to scan (default: public)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="backslashreplace")
        except (ValueError, OSError):
            pass
    w_root = Path(a.w_root)
    ledger = Path(a.ledger) if a.ledger else w_root / "gate" / "ledger.jsonl"
    sources = Path(a.sources) if a.sources else w_root / "gate" / "sources"
    classes = tuple(c.strip() for c in a.classes.split(",") if c.strip())
    try:
        if not ledger.is_file():
            raise RuntimeError(f"no ledger at {ledger}")
        if not sources.is_dir():
            raise RuntimeError(f"no sources folder at {sources}")
        if not classes:
            raise RuntimeError("--classes names no class")
        lint = load_lint(Path(a.lint))
        r = scan(ledger, sources, lint, classes)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"gate_words_scan: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r, indent=1, ensure_ascii=True))
    else:
        print(render(r))
    return 1 if r["no_source"] else 0


if __name__ == "__main__":
    sys.exit(main())
