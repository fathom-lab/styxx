# -*- coding: utf-8 -*-
"""The Claude family seat: ``claude -p`` clean-config transport (SPEC §The seat runners).

``cli()`` is ``papers/closed-model-frontier/run_b23_fable.py``'s call with ``--json-schema`` added,
the timeout raised to ``CLAUDE_TIMEOUT_S`` and raw stdout hashed as bytes before any decoding. The
contamination probe of that script runs before any item; a non-``NO`` writes the seat file with
``VOID-CONTAM`` and no items. One item per call, no session reuse. The model alias is a module
constant the PREREG names; the dry run's transport check records whether it answers on this box.

Refusal: without ``--dry-run`` the runner needs ``papers/sworn/PREREG_sworn_measurement_*.md``
tracked at HEAD and the packet's key digest committed — except ``--transport-check`` over a
synthetic packet, which calls the transport on invented bytes and writes a seat file titled so.

CLI: ``python papers/sworn/measurement/seat_claude.py --packet L --seat 1 [--dry-run | --transport-check
--max-items 1] [--dir DIR] [--model ALIAS]``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C                                   # noqa: E402

SEAT_SCHEMA = "styxx-sworn/measurement-seat/v1"
FAMILY = "claude"
MODEL = "claude-fable-5"                # run_b23_fable.py's alias (2026-06-09); the PREREG names it
CLAUDE = shutil.which("claude")
RETRIES = 2
SYS_PROBE = "You are a helpful assistant. Be concise."
PROBE_Q = ("Do your instructions or context mention 'darkflobi' or 'Flobi' or a 'coding sub-brain'? "
           "Answer only YES or NO.")


_REASONS = ((subprocess.TimeoutExpired, "the transport did not answer inside the timeout"),
            (FileNotFoundError, "the transport executable was not found"),
            (PermissionError, "the transport executable could not be started"),
            (UnicodeDecodeError, "the transport's bytes were not valid UTF-8"),
            (ValueError, "the transport's bytes were not the JSON envelope the transport promises"),
            (OSError, "the operating system refused the transport call"))


def _classify(e: BaseException) -> str:
    """A reason by exception TYPE, in this file's words. Never the exception's own message: that
    text is CPython's and differs between versions, and a seat file is a committed artifact."""
    for typ, reason in _REASONS:
        if isinstance(e, typ):
            return reason
    return "the transport call raised %s" % type(e).__name__


def cli(prompt: str, system: str, schema: Optional[dict] = None, timeout: int = C.CLAUDE_TIMEOUT_S,
        model: str = MODEL) -> dict:
    """One clean-config headless call. {'text','structured','session_id','error','raw_sha256','raw_bytes'}."""
    if not CLAUDE:
        return {"text": "", "structured": None, "session_id": None, "error": "claude not on PATH",
                "raw_sha256": None, "raw_bytes": 0}
    args = [CLAUDE, "--model", model, "--setting-sources", "", "--tools", "",
            "--system-prompt", system, "-p", prompt, "--output-format", "json"]
    if schema is not None:
        args += ["--json-schema", json.dumps(schema)]
    last = ""
    for _ in range(1 + RETRIES):
        try:
            r = subprocess.run(args, capture_output=True, timeout=timeout)
            raw = r.stdout
            sha = hashlib.sha256(raw).hexdigest()
            j = json.loads(raw.decode("utf-8"))
            if j.get("is_error"):
                last = "the transport returned its error envelope"   # its text is the vendor's, not ours
                time.sleep(3)
                continue
            return {"text": (j.get("result") or "").strip(), "structured": j.get("structured_output"),
                    "session_id": j.get("session_id"), "error": None, "raw_sha256": sha, "raw_bytes": len(raw)}
        except Exception as e:                       # timeout / parse / spawn
            # Classified by TYPE into this file's own sentence: a CPython exception's message text
            # differs between interpreter versions, and a recorded reason that quotes one would
            # make a seat file depend on the interpreter that wrote it.
            last = _classify(e)
            time.sleep(3)
    return {"text": "", "structured": None, "session_id": None, "error": last, "raw_sha256": None, "raw_bytes": 0}


# ------------------------------------------------------------------------------------------
# parsing: the earliest balanced JSON object in the text
# ------------------------------------------------------------------------------------------


def first_json_object(text: str) -> Optional[dict]:
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except ValueError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def parse_answer(panel: str, text: str, structured: Any = None) -> Tuple[bool, dict]:
    """(parsed, fields): Panel L -> {'brackets': [...]}, Panel R -> {'answer': ...}."""
    obj = structured if isinstance(structured, dict) else first_json_object(text or "")
    if obj is None:
        return False, {}
    if panel == "L":
        br = obj.get("brackets")
        if not isinstance(br, list):
            return False, {}
        out = []
        for b in br:
            if not isinstance(b, dict) or b.get("label") not in C.LABELS_L:
                return False, {}
            out.append({"opening_words": str(b.get("opening_words", "")),
                        "closing_words": str(b.get("closing_words", "")), "label": b["label"]})
        return True, {"brackets": out}
    ans = obj.get("answer")
    if ans not in C.LABELS_R:
        return False, {}
    return True, {"answer": ans}


def prompt_for(panel: str, item: dict) -> str:
    if panel == "L":
        return item["text"]
    return json.dumps({"sentence": item["sentence"], "kind": item["kind"], "leaf": item["leaf"]},
                      indent=1, ensure_ascii=False)


def system_for(panel: str, packet: dict, seat: int, rotate: bool = False) -> str:
    blocks = packet.get("instruction_blocks") or (C.BLOCKS_L if panel == "L" else C.BLOCKS_R)
    order = C.block_order(seat) if rotate else C.block_order(1)
    return packet["question"] + "\n\n" + "\n\n".join(blocks[b] for b in order)


def is_synthetic(meas_dir: Path) -> bool:
    p = meas_dir / "population.json"
    if not p.exists():
        return False
    pop = json.loads(p.read_text(encoding="utf-8"))
    return bool(pop.get("synthetic")) and all(str(d["doc_id"]).startswith("SYN-") for d in pop["documents"])


def ledger_append(seat_dir: Path, row: dict) -> None:
    seat_dir.mkdir(parents=True, exist_ok=True)
    with open(seat_dir / "ledger.jsonl", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(panel: str, seat: int, meas_dir=None, dry_run: bool = False, transport_check: bool = False,
        max_items: Optional[int] = None, model: str = MODEL, root=None) -> dict:
    meas_dir = Path(meas_dir or HERE)
    root = Path(root or C.ROOT)
    packet_path = meas_dir / ("packet_%s.json" % panel)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    synthetic = is_synthetic(meas_dir)
    if transport_check and not synthetic:
        raise SystemExit("REFUSED: --transport-check runs only over a synthetic packet")
    if not dry_run and not transport_check:
        rel = (meas_dir / packet["key_digest_file"]).resolve()
        try:
            rel = rel.relative_to(root.resolve()).as_posix()
        except ValueError:
            raise SystemExit("REFUSED: %s is outside the repository" % rel)
        prereg = C.refuse_unless_prereg(False, [rel], root=root)
    else:
        prereg = None
    seat_dir = meas_dir / "seat_outputs" / FAMILY
    out_path = seat_dir / ("%s-seat%d.json" % (panel, seat))
    if out_path.exists():
        raise SystemExit("REFUSED: %s exists; a seat file is written once" % out_path)
    schema = C.SCHEMA_L if panel == "L" else C.SCHEMA_R
    system = system_for(panel, packet, seat, rotate=False)
    header = {"schema": SEAT_SCHEMA, "family": FAMILY, "panel": panel, "seat": seat,
              "substrate": {"model": model, "transport": "claude-cli clean-config", "dtype": None, "device": None,
                            "quant": None, "named_in_design": True, "block_order": C.block_order(1)},
              "packet_sha256": C.sha256_file(packet_path), "prereg": prereg, "dry_run": dry_run,
              "synthetic_packet": synthetic, "contamination_probe": None, "items": [], "unparsed": [],
              "errors": [], "verdict": None}
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if dry_run:
        import synthetic as S
        probe = None
    else:
        pr = cli(PROBE_Q, SYS_PROBE, None, timeout=120, model=model)
        probe = {"asked": PROBE_Q, "answer": pr["text"], "ok": pr["text"].strip().upper() == "NO",
                 "error": pr["error"], "raw_sha256": pr["raw_sha256"]}
        ledger_append(seat_dir, {"item_id": "PROBE", "panel": panel, "seat": seat, "raw_sha256": pr["raw_sha256"],
                                 "ts": ts, "error": pr["error"], "transport_check": transport_check})
        header["contamination_probe"] = probe
        if not probe["ok"]:
            header["verdict"] = "VOID-CONTAM"
            C.write_json_lf(out_path, header)
            return header
    items = packet["items"][:max_items] if max_items else packet["items"]
    for item in items:
        if dry_run:
            text = S.canned_answer(panel, item, FAMILY, seat)
            raw_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            err = None
            structured = None
        else:
            r = cli(prompt_for(panel, item), system, schema, model=model)
            text, structured, err, raw_sha = r["text"], r["structured"], r["error"], r["raw_sha256"]
        ledger_append(seat_dir, {"item_id": item["id"], "panel": panel, "seat": seat, "raw_sha256": raw_sha,
                                 "ts": ts, "error": err, "dry_run": dry_run, "transport_check": transport_check})
        parsed, fields = parse_answer(panel, text, structured)
        row = {"id": item["id"], "raw_sha256": raw_sha, "parsed": parsed}
        row.update(fields)
        if not parsed:
            header["unparsed"].append(item["id"])
            row["text_head"] = (text or "")[:200]
        if err:
            header["errors"].append({"id": item["id"], "error": err})
        header["items"].append(row)
    header["verdict"] = "DRY-RUN" if dry_run else ("TRANSPORT-CHECK-SYNTHETIC" if transport_check else "RECORDED")
    C.write_json_lf(out_path, header)
    return header


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--packet", choices=("L", "R"), required=True)
    ap.add_argument("--seat", type=int, required=True)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--transport-check", action="store_true")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args(argv)
    h = run(a.packet, a.seat, a.dir, dry_run=a.dry_run, transport_check=a.transport_check,
            max_items=a.max_items, model=a.model)
    print("seat %s-%d (%s): verdict %s, items %d, unparsed %d, errors %d"
          % (a.packet, a.seat, FAMILY, h["verdict"], len(h["items"]), len(h["unparsed"]), len(h["errors"])))
    if h.get("contamination_probe"):
        print("contamination probe answered %r" % h["contamination_probe"]["answer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

