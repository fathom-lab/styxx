"""Tranche A's seat facts: headers, counts and sha256 read from the seat outputs, never an answer.

    python papers/sworn/measurement/tranche_A_seat_facts.py SEAT_OUTPUTS_DIR (--out PATH | --check PATH)

SEAT_OUTPUTS_DIR is tranche A's seat_outputs/ directory (claude/ and local/). Until tranche A's key-release commit
the twelve seat files and the local family's ledger are not in the tree, so the directory is the run worktree's
untracked one; after that commit it is papers/sworn/measurement/seat_outputs, and the same command with --check
must reproduce the committed receipt byte for byte. The output names no absolute path and carries no timestamp of
its own, so where it runs does not change its bytes.

What it reads: each of the twelve seat files (header fields, and counts over the items, unparsed, errors and timing
arrays), the two ledgers (row counts, the row each seat starts at and its timestamp, the rows that carry an error, and how
many distinct call hashes each panel's rows hold), the nine twin status files and twin files, and the committed
population and sidecars (through common.open_document, at the pinned commit) to compare each twin's text with its
document's. What it prints: no item's answer or label, no per-item token count, no per-item call hash; an item id
appears only where a seat file lists it as unparsed or as an error, or where a ledger row carries an error.
No sealed file, no key, no model, no network; it writes only --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MEAS = Path(__file__).resolve().parent
ROOT = MEAS.parents[2]
for _p in (str(MEAS), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common as C                                   # noqa: E402
import population as P                               # noqa: E402
from styxx import sworn                              # noqa: E402

SCHEMA = "styxx-sworn/measurement-tranche-A-seat-facts/v1"
FAMILIES = ("claude", "local")
PANELS = ("L", "R")
SEATS = (1, 2, 3)
DOCS = ("D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09")
HEADER = ("schema", "family", "panel", "seat", "substrate", "packet_sha256", "prereg", "dry_run",
          "synthetic_packet", "verdict", "note")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read(path: Path) -> bytes:
    if not path.is_file():
        raise SystemExit("REFUSED: %s is missing" % path.as_posix())
    return path.read_bytes()


def seat_file_facts(seat_dir: Path, fam: str, panel: str, seat: int) -> dict:
    rel = "%s/%s-seat%d.json" % (fam, panel, seat)
    raw = _read(seat_dir / rel)
    d = json.loads(raw.decode("utf-8"))
    out = {"path": rel, "sha256": _sha(raw), "bytes": len(raw)}
    for k in HEADER:
        if k in d:
            out[k] = d[k]
    probe = d.get("contamination_probe")
    out["contamination_probe"] = (None if probe is None else
                                  {"answer": probe.get("answer"), "ok": probe.get("ok"),
                                   "error": probe.get("error")})
    items = d.get("items", [])
    out["items_count"] = len(items)
    out["items_parsed_count"] = sum(1 for r in items if r.get("parsed"))
    out["unparsed"] = list(d.get("unparsed", []))
    out["unparsed_count"] = len(out["unparsed"])
    out["errors"] = [{"id": e.get("id"), "error": e.get("error")} for e in d.get("errors", [])]
    out["errors_count"] = len(out["errors"])
    timing = d.get("timing")
    if timing is None:
        out["timing_count"] = None
    else:
        capped = [t.get("id") for t in timing if t.get("new_tokens") == C.LOCAL_MAX_NEW_TOKENS]
        out["timing_count"] = len(timing)
        out["new_tokens_cap"] = C.LOCAL_MAX_NEW_TOKENS
        out["timing_rows_at_cap_count"] = len(capped)
        out["timing_rows_at_cap_and_unparsed_count"] = len(set(capped) & set(out["unparsed"]))
    return out


def ledger_facts(seat_dir: Path, fam: str) -> dict:
    rel = "%s/ledger.jsonl" % fam
    raw = _read(seat_dir / rel)
    rows = [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    out = {"path": rel, "sha256": _sha(raw), "bytes": len(raw), "rows": len(rows),
           "rows_probe": sum(1 for r in rows if r.get("item_id") == "PROBE"),
           "rows_item": sum(1 for r in rows if r.get("item_id") != "PROBE"),
           "keys": sorted({k for r in rows for k in r})}
    seats = {}
    for n, r in enumerate(rows, 1):
        key = "%s-seat%d" % (r.get("panel"), r.get("seat"))
        s = seats.setdefault(key, {"start_line": n, "start_ts": r.get("ts"), "end_line": n,
                                   "rows": 0, "rows_probe": 0, "rows_item": 0})
        s["rows"] += 1
        s["rows_probe" if r.get("item_id") == "PROBE" else "rows_item"] += 1
        s["end_line"] = n
    out["seats"] = {k: seats[k] for k in sorted(seats)}
    out["error_rows"] = [{"line": n, "item_id": r.get("item_id"), "panel": r.get("panel"),
                          "seat": r.get("seat"), "error": r.get("error"),
                          "raw_sha256_is_null": r.get("raw_sha256") is None}
                         for n, r in enumerate(rows, 1) if r.get("error") is not None]
    out["error_rows_count"] = len(out["error_rows"])
    per_panel = {}
    for panel in PANELS:
        hs = [r.get("raw_sha256") for r in rows if r.get("panel") == panel and r.get("item_id") != "PROBE"]
        per_panel[panel] = {"item_rows": len(hs), "null": sum(1 for h in hs if h is None),
                            "distinct_non_null": len({h for h in hs if h is not None})}
    out["call_hashes_per_panel"] = per_panel
    return out


def twin_facts(seat_dir: Path, documents: dict) -> dict:
    out = {}
    for doc in DOCS:
        st_rel = "claude/trivial/%s.trivial-status.json" % doc
        tw_rel = "claude/trivial/%s.trivial-twin.json" % doc
        st_raw, tw_raw = _read(seat_dir / st_rel), _read(seat_dir / tw_rel)
        st, tw = json.loads(st_raw.decode("utf-8")), json.loads(tw_raw.decode("utf-8"))
        a = documents[doc]
        b = tw.get("text") or ""
        out[doc] = {"status_path": st_rel, "status_sha256": _sha(st_raw),
                    "twin_path": tw_rel, "twin_sha256": _sha(tw_raw),
                    "status_error": st.get("error"), "status_has_refused_key": "refused" in st,
                    "twin_text_changed": st.get("twin_text_changed"),
                    "spans_original": st.get("spans_original"), "spans_twin": st.get("spans_twin"),
                    "document_text_chars": len(a), "twin_text_chars": len(b),
                    "document_text_ends_with_lf": a.endswith("\n"),
                    "twin_text_equals_document_text": b == a,
                    "twin_text_plus_lf_equals_document_text": b + "\n" == a}
    return out


def build(seat_dir: Path) -> dict:
    pop_raw = _read(MEAS / "population.json")
    pop = json.loads(pop_raw.decode("utf-8"))
    documents = {}
    for e in P.iter_documents(pop):
        side, _tree, _rec = C.open_document(e)
        documents[e["doc_id"]] = sworn.load_sidecar(side)["text"]
    seat_files = {f: {"%s-seat%d" % (p, s): seat_file_facts(seat_dir, f, p, s) for p in PANELS for s in SEATS}
                  for f in FAMILIES}
    totals = {}
    for f in FAMILIES:
        for p in PANELS:
            rows = [seat_files[f]["%s-seat%d" % (p, s)] for s in SEATS]
            t = {"items": sum(r["items_count"] for r in rows),
                 "unparsed": sum(r["unparsed_count"] for r in rows),
                 "errors": sum(r["errors_count"] for r in rows)}
            if all(r.get("timing_count") is not None for r in rows):
                t["timing_rows_at_cap"] = sum(r["timing_rows_at_cap_count"] for r in rows)
                t["timing_rows_at_cap_and_unparsed"] = sum(r["timing_rows_at_cap_and_unparsed_count"] for r in rows)
            totals["%s_%s" % (f, p)] = t
    return {
        "schema": SCHEMA,
        "script": "papers/sworn/measurement/tranche_A_seat_facts.py",
        "script_sha256": _sha(Path(__file__).resolve().read_bytes()),
        "population_sha256": _sha(pop_raw),
        "what_this_is": ("headers, counts and sha256 read from tranche A's twelve seat files, two ledgers and nine "
                         "trivial twins, so a RESULT can swear to counts over arrays the seat files hold"),
        "what_this_is_not": ("not a re-score, not a key read, and no item's answer or label: an item id appears "
                             "only where a seat file lists it as unparsed or as an error, or a ledger row carries "
                             "an error"),
        "seat_files": seat_files,
        "totals": totals,
        "ledgers": {f: ledger_facts(seat_dir, f) for f in FAMILIES},
        "twins": twin_facts(seat_dir, documents),
    }


def render(obj: dict) -> bytes:
    return (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seat_outputs", help="tranche A's seat_outputs/ directory (claude/ and local/)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", help="write the receipt here (LF, UTF-8)")
    g.add_argument("--check", help="re-derive and compare with this committed receipt, byte for byte")
    a = ap.parse_args(argv)
    seat_dir = Path(a.seat_outputs)
    if not seat_dir.is_dir():
        raise SystemExit("REFUSED: %s is not a directory" % seat_dir.as_posix())
    data = render(build(seat_dir))
    if a.check:
        same = Path(a.check).read_bytes() == data
        print("tranche A seat facts: %s (sha256 %s)" % ("REPRODUCED" if same else "DIFFERS", _sha(data)))
        return 0 if same else 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    if out.read_bytes() != data:
        raise SystemExit("REFUSED: the write did not land byte for byte")
    print("wrote %s (%d bytes, sha256 %s)" % (out.as_posix(), len(data), _sha(data)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
