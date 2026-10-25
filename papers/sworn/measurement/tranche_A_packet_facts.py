"""Tranche A's packet facts, read only from committed files: two receipts, one of counts alone.

    python papers/sworn/measurement/tranche_A_packet_facts.py (--out-dir DIR | --check-dir DIR)

Writes two files into DIR (papers/sworn/measurement/receipts in the tree):

- tranche_A_decoy_text_count.json: how many packet_L.json item texts equal a committed decoys_L.json decoy text.
  Counts only: no item id, no decoy id, no side. It does not say which items are decoys.
- tranche_A_packet_facts.json: the Panel L items whose text begins with "-" (their ids and character counts), the
  character counts of the items a RESULT compares with them by id, and the keys a Panel R item carries (a Panel R
  item has no text field; seat_claude.prompt_for sends it as a JSON object). Nothing about a decoy, a side or a seat.

Reads packet_L.json, packet_R.json and decoys_L.json beside this file, and nothing else. No seat output, no sealed
file, no key, no model. --check-dir re-derives both and compares them with the files in DIR byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MEAS = Path(__file__).resolve().parent
COMPARED = ("L-0008",)          # the item the L-0033 finding compares by length; named by id, nothing else


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _render(obj: dict) -> bytes:
    return (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def build() -> dict:
    pk_b = (MEAS / "packet_L.json").read_bytes()
    pr_b = (MEAS / "packet_R.json").read_bytes()
    dl_b = (MEAS / "decoys_L.json").read_bytes()
    pk, pr, dl = (json.loads(b.decode("utf-8")) for b in (pk_b, pr_b, dl_b))
    script = {"script": "papers/sworn/measurement/tranche_A_packet_facts.py",
              "script_sha256": _sha(Path(__file__).resolve().read_bytes())}
    texts = [d["text"] for d in dl["decoys"]]
    decoy_texts = set(texts)
    count = dict(script)
    count.update({
        "schema": "styxx-sworn/measurement-tranche-A-decoy-text-count/v1",
        "what_this_is_not": "counts only: no item id, no decoy id and no side; it does not say which items are decoys",
        "packet_L_sha256": _sha(pk_b), "decoys_L_sha256": _sha(dl_b),
        "packet_L_items": len(pk["items"]),
        "committed_decoys": len(texts), "committed_decoy_distinct_texts": len(decoy_texts),
        "packet_L_item_texts_equal_to_a_committed_decoy_text": sum(1 for it in pk["items"]
                                                                   if it.get("text") in decoy_texts),
    })
    items_l = pk["items"]
    dash = [{"index": i, "id": it["id"], "text_chars": len(it["text"])}
            for i, it in enumerate(items_l) if it.get("text", "").startswith("-")]
    facts = dict(script)
    facts.update({
        "schema": "styxx-sworn/measurement-tranche-A-packet-facts/v1",
        "what_this_is_not": "no decoy, no side, no seat and no answer: packet bytes only",
        "packet_L_sha256": _sha(pk_b), "packet_R_sha256": _sha(pr_b),
        "packet_L_items": len(items_l), "packet_R_items": len(pr["items"]),
        "packet_L_items_beginning_with_dash": dash,
        "packet_L_items_beginning_with_dash_count": len(dash),
        "packet_R_item_keys": sorted({k for it in pr["items"] for k in it}),
        "compared_by_length": [{"index": i, "id": it["id"], "text_chars": len(it["text"])}
                               for i, it in enumerate(items_l) if it["id"] in COMPARED],
    })
    return {"tranche_A_decoy_text_count.json": count, "tranche_A_packet_facts.json": facts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out-dir")
    g.add_argument("--check-dir")
    a = ap.parse_args(argv)
    outs = {name: _render(obj) for name, obj in build().items()}
    if a.check_dir:
        bad = [n for n, data in outs.items() if (Path(a.check_dir) / n).read_bytes() != data]
        for n, data in outs.items():
            print("%s: %s (sha256 %s)" % (n, "DIFFERS" if n in bad else "REPRODUCED", _sha(data)))
        return 1 if bad else 0
    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for n, data in outs.items():
        (d / n).write_bytes(data)
        if (d / n).read_bytes() != data:
            raise SystemExit("REFUSED: %s did not land byte for byte" % n)
        print("wrote %s (%d bytes, sha256 %s)" % ((d / n).as_posix(), len(data), _sha(data)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
