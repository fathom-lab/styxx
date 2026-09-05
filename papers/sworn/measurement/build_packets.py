# -*- coding: utf-8 -*-
"""Canonical text -> the two blind packets, the decoys, and the sealed keys (SPEC §The packets).

Panel L items are windows of canonical text (tags already stripped) cut only at blank lines;
Panel R items are one sworn span each with the leaf view of the receipt its author named, the
verdict never included. Decoys are shuffled into the item order and a seat cannot tell one from a
document item by id, position or shape: what an id is lives in the sealed key.

Panel L decoys are authored: ``DECOY_PICKS_L`` names, for each of the thirty, a document the
population rule excludes, a unit index in its unit set, and the builder's side. Panel R decoys are
built: the YES side is a HELD span from an excluded document with the leaf its author named, the
NO side the same span retargeted by construction A of ``canaries.py``.

CLI: ``python papers/sworn/measurement/build_packets.py [--population population.json]
[--seed 20260905] [--sealed DIR] [--out DIR] [--keys-dir DIR]``. Refuses to overwrite a packet.
Prints ``NO seat was run and no number was computed by this file.``
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import canaries as CAN                               # noqa: E402
import common as C                                   # noqa: E402
import population as P                               # noqa: E402
import seal_key as K                                 # noqa: E402
from styxx import sworn                              # noqa: E402

PACKET_SCHEMA = "styxx-sworn/measurement-packet/v1"
DECOYS_SCHEMA = "styxx-sworn/measurement-decoys/v1"
KEY_L = "sworn_measurement_key_L"
KEY_R = "sworn_measurement_key_R"
SEED_SHUFFLE_L = 1        # random.Random(SEED + 1): Panel L item order
SEED_SHUFFLE_R = 2        # random.Random(SEED + 2): Panel R item order
SEED_DECOY_R = 3          # random.Random(SEED + 3): the retargeting of the Panel R NO side

SELECTION_RULE_L = (
    "Each keyed sentence is a unit of an excluded document's unit set at the pinned commit, chosen "
    "by the builder: LOAD-BEARING when it states a count or rate the excluded document's conclusion "
    "depends on; NOT when it is a title, a header line, a date, a version, a file name, or a "
    "sentence about method. The passage is the whole lines from the unit before to the unit after."
)
AUTHORSHIP_DISCLOSURE_L = (
    "Authorship, disclosed rather than buried: the Panel L decoys were selected and sided by the "
    "builder of this machinery from documents the population rule excludes. A family's agreement "
    "with them measures agreement with the builder's reading, not truth, and any result quoting the "
    "decoy gate carries this sentence."
)
CONSTRUCTION_RULE_R = (
    "The YES side is the first %d spans, in excluded-document order then span order, whose committed "
    "receipt verdict is HELD and whose receipt names a JSON pointer leaf that construction A can "
    "retarget; the NO side is the same spans retargeted. Expected answers live in the sealed key."
)

# (stem, unit index in common.units_of at the pinned commit, side) — thirty, authored.
DECOY_PICKS_L: List[Tuple[str, int, str]] = [
    ("papers/sworn/RESULT_sworn_v01_ships_2026_09_01", 20, "LOAD-BEARING"),
    ("papers/sworn/RESULT_sworn_v01_ships_2026_09_01", 31, "LOAD-BEARING"),
    ("papers/sworn/RESULT_sworn_v02_ships_2026_09_02", 31, "LOAD-BEARING"),
    ("papers/sworn/RESULT_sworn_v02_ships_2026_09_02", 51, "LOAD-BEARING"),
    ("papers/sworn/CENSUS_prose_claimhood_instruments_2026_09_01", 32, "LOAD-BEARING"),
    ("papers/sworn/CENSUS_prose_claimhood_instruments_2026_09_01", 45, "LOAD-BEARING"),
    ("papers/sworn/CENSUS_prose_claimhood_instruments_2026_09_01", 69, "LOAD-BEARING"),
    ("papers/sworn/CENSUS_prose_claimhood_instruments_2026_09_01", 84, "LOAD-BEARING"),
    ("papers/sworn/CENSUS_prose_claimhood_instruments_2026_09_01", 97, "LOAD-BEARING"),
    ("papers/PLAN_the_next_level_2026_09_02", 16, "LOAD-BEARING"),
    ("papers/PLAN_the_next_level_2026_09_02", 21, "LOAD-BEARING"),
    ("papers/PLAN_the_next_level_2026_09_02", 36, "LOAD-BEARING"),
    ("papers/PLAN_the_next_level_2026_09_02", 50, "LOAD-BEARING"),
    ("papers/sworn/ATTACKS_sworn_v01_battery_2026_09_02", 54, "LOAD-BEARING"),
    ("papers/sworn/ATTACKS_sworn_v01_battery_2026_09_02", 58, "LOAD-BEARING"),
    ("papers/sworn/RESULT_sworn_v01_ships_2026_09_01", 1, "NOT"),
    ("papers/sworn/RESULT_sworn_v01_ships_2026_09_01", 3, "NOT"),
    ("papers/sworn/RESULT_sworn_v01_ships_2026_09_01", 25, "NOT"),
    ("papers/sworn/RESULT_sworn_v02_ships_2026_09_02", 3, "NOT"),
    ("papers/sworn/RESULT_sworn_v02_ships_2026_09_02", 5, "NOT"),
    ("papers/sworn/RESULT_sworn_v02_ships_2026_09_02", 35, "NOT"),
    ("papers/sworn/CENSUS_prose_claimhood_instruments_2026_09_01", 0, "NOT"),
    ("papers/sworn/CENSUS_prose_claimhood_instruments_2026_09_01", 6, "NOT"),
    ("papers/sworn/CENSUS_prose_claimhood_instruments_2026_09_01", 11, "NOT"),
    ("papers/sworn/CENSUS_prose_claimhood_instruments_2026_09_01", 14, "NOT"),
    ("papers/PLAN_the_next_level_2026_09_02", 0, "NOT"),
    ("papers/PLAN_the_next_level_2026_09_02", 7, "NOT"),
    ("papers/PLAN_the_next_level_2026_09_02", 67, "NOT"),
    ("papers/sworn/ATTACKS_sworn_v01_battery_2026_09_02", 0, "NOT"),
    ("papers/sworn/ATTACKS_sworn_v01_battery_2026_09_02", 20, "NOT"),
]


def _open(entries: List[dict], root) -> Dict[str, Tuple[dict, object, Optional[dict]]]:
    out = {}
    for e in entries:
        side, tree, rec = C.open_document(e, root=root)
        out[e["stem"]] = (sworn.load_sidecar(side), tree, rec)
    return out


def _r_item(sentence: bytes, kind: str, receipt: str, side: dict, tree) -> dict:
    data, why = C.receipt_bytes(side, tree, receipt)
    if data is None:
        parsed, _ = sworn._parse_receipt(receipt)
        name = (parsed["target"].rsplit("/", 1)[-1] if parsed and parsed["form"] == "path"
                else (parsed["target"] if parsed else receipt))
        leaf = {"receipt_name": name, "pointer": None, "lines": None, "value": "",
                "value_kind": "unresolvable_receipt", "truncated": False, "note": why}
    else:
        leaf = C.leaf_view(receipt, data, kind)
    return {"sentence": sentence.decode("utf-8"), "kind": kind, "leaf": leaf}


def decoys_L(pop: dict, picks: List[Tuple[str, int, str]], root=None) -> List[dict]:
    """The authored Panel L decoys materialised from the excluded documents at the pinned commit."""
    wanted = {stem for stem, _, _ in picks}
    entries = [e for e in P.iter_excluded(pop) if e["stem"] in wanted]
    missing = wanted - {e["stem"] for e in entries}
    if missing:
        raise SystemExit("REFUSED: decoy source not in the population's excluded list: %s" % sorted(missing))
    docs = _open(entries, root)
    rows = []
    for n, (stem, idx, side_) in enumerate(picks, 1):
        side, _, _ = docs[stem]
        canonical = side["text"].encode("utf-8")
        units = C.units_of(side)
        if not (0 <= idx < len(units)):
            raise SystemExit("REFUSED: %s has no unit %d" % (stem, idx))
        p = C.passage_around(canonical, units, idx)
        rows.append({"decoy_id": "LD-%02d" % n, "side": side_, "source_stem": stem, "unit_index": idx,
                     "passage": {"start": p["start"], "end": p["end"]}, "keyed": p["keyed"],
                     "text": canonical[p["start"]:p["end"]].decode("utf-8")})
    per_side = {s: sum(1 for r in rows if r["side"] == s) for s in ("LOAD-BEARING", "NOT")}
    if any(v != C.N_DECOYS_PER_SIDE for v in per_side.values()):
        raise SystemExit("REFUSED: Panel L decoys per side %s, N_DECOYS_PER_SIDE is %d"
                         % (per_side, C.N_DECOYS_PER_SIDE))
    return rows


def decoys_R(pop: dict, seed: int, root=None) -> List[dict]:
    """The built Panel R decoys: YES = a HELD pointer span as authored, NO = the same retargeted."""
    rng = random.Random(seed + SEED_DECOY_R)
    docs = _open(P.iter_excluded(pop), root)
    rows = []
    for e in P.iter_excluded(pop):
        side, tree, rec = docs[e["stem"]]
        if rec is None:
            continue
        verdicts = [s.get("verdict") for s in rec.get("spans", [])]
        canonical = side["text"].encode("utf-8")
        for i, s in enumerate(side["spans"]):
            if len(rows) >= 2 * C.N_DECOYS_PER_SIDE:
                break
            if i >= len(verdicts) or verdicts[i] != "HELD":
                continue
            r = CAN.retarget(side, tree, i, rng)
            if r is None:
                continue
            sent = canonical[s["start"]:s["end"]]
            n = len(rows) // 2 + 1
            yes = _r_item(sent, s["kind"], s["receipt"], side, tree)
            yes.update(decoy_id="RD-%02dY" % n, side="YES", source_stem=e["stem"], span_index=i,
                       construction="as_authored", receipt=s["receipt"])
            no = _r_item(sent, s["kind"], r["receipt"], side, tree)
            no.update(decoy_id="RD-%02dN" % n, side="NO", source_stem=e["stem"], span_index=i,
                      construction="A", receipt=r["receipt"], falsehood_check=r["falsehood_check"])
            rows.extend([yes, no])
    if len(rows) != 2 * C.N_DECOYS_PER_SIDE:
        raise SystemExit("REFUSED: %d Panel R decoys built, %d needed" % (len(rows), 2 * C.N_DECOYS_PER_SIDE))
    return rows


def build(pop: dict, pop_path, picks: Optional[List[Tuple[str, int, str]]] = None, root=None,
          sealed=None, out_dir=None, keys_dir=None, seed: int = C.SEED) -> dict:
    """Write packet_L.json, packet_R.json, decoys_L.json, packets_digest.txt and seal both keys."""
    picks = DECOY_PICKS_L if picks is None else picks
    out_dir = Path(out_dir or HERE)
    keys_dir = Path(keys_dir or out_dir / "keys")
    for name in ("packet_L.json", "packet_R.json", "decoys_L.json", "packets_digest.txt"):
        if (out_dir / name).exists():
            raise SystemExit("REFUSED: %s exists; a rebuild is a new file at a new commit" % (out_dir / name))
    pop_sha = C.sha256_file(pop_path)
    items_L: List[Tuple[dict, dict]] = []      # (item without id, key meta)
    items_R: List[Tuple[dict, dict]] = []
    windows_total = 0
    for e in P.iter_documents(pop):
        side, tree, _ = C.open_document(e, root=root)
        side = sworn.load_sidecar(side)
        canonical = side["text"].encode("utf-8")
        units = C.units_of(side)
        for w in C.windows_of(canonical, units):
            windows_total += 1
            items_L.append(({"text": canonical[w["start"]:w["end"]].decode("utf-8")},
                            {"kind": "document", "doc_id": e["doc_id"],
                             "window": {"start": w["start"], "end": w["end"]}, "oversize": w["oversize"]}))
        for i, s in enumerate(side["spans"]):
            item = _r_item(canonical[s["start"]:s["end"]], s["kind"], s["receipt"], side, tree)
            items_R.append((item, {"kind": "document", "doc_id": e["doc_id"], "span_index": i}))
    dl = decoys_L(pop, picks, root=root)
    for d in dl:
        items_L.append(({"text": d["text"]}, {"kind": "decoy", "decoy_id": d["decoy_id"], "side": d["side"],
                                              "keyed": dict(d["keyed"])}))
    dr = decoys_R(pop, seed, root=root)
    for d in dr:
        items_R.append(({"sentence": d["sentence"], "kind": d["kind"], "leaf": d["leaf"]},
                        {"kind": "decoy", "decoy_id": d["decoy_id"], "side": d["side"],
                         "source_stem": d["source_stem"], "construction": d["construction"]}))
    random.Random(seed + SEED_SHUFFLE_L).shuffle(items_L)
    random.Random(seed + SEED_SHUFFLE_R).shuffle(items_R)
    key_L, key_R = {}, {}
    packet_items_L, packet_items_R = [], []
    for n, (item, meta) in enumerate(items_L, 1):
        iid = "L-%04d" % n
        packet_items_L.append(dict(id=iid, **item))
        key_L[iid] = meta
    for n, (item, meta) in enumerate(items_R, 1):
        iid = "R-%04d" % n
        packet_items_R.append(dict(id=iid, **item))
        key_R[iid] = meta
    decoys_doc = {"schema": DECOYS_SCHEMA, "panel": "L", "seed": seed, "selection_rule": SELECTION_RULE_L,
                  "authorship_disclosure": AUTHORSHIP_DISCLOSURE_L, "per_side": C.N_DECOYS_PER_SIDE,
                  "decoys": dl}
    dpath = C.write_json_lf(out_dir / "decoys_L.json", decoys_doc)
    decoys_sha = C.sha256_file(dpath)

    def packet(panel, question, blocks, schema, items, key_name, extra):
        p = {"schema": PACKET_SCHEMA, "panel": panel, "question": question,
             "instructions": C.instructions(panel, 1), "instruction_blocks": dict(blocks),
             "output_schema": schema,
             "built_from": {"population_sha256": pop_sha, "seed": seed, "decoys_sha256": decoys_sha},
             "key_digest_file": "keys/%s.sha256" % key_name, "items": items}
        p["built_from"].update(extra)
        return p

    pl = packet("L", C.QUESTION_L, C.BLOCKS_L, C.SCHEMA_L, packet_items_L, KEY_L,
                {"window_max_units": C.WINDOW_MAX_UNITS, "edge_words": C.EDGE_WORDS})
    pr = packet("R", C.QUESTION_R, C.BLOCKS_R, C.SCHEMA_R, packet_items_R, KEY_R,
                {"leaf_view_max_chars": C.LEAF_VIEW_MAX_CHARS,
                 "decoy_construction_rule": CONSTRUCTION_RULE_R % C.N_DECOYS_PER_SIDE})
    pL = C.write_json_lf(out_dir / "packet_L.json", pl)
    pR = C.write_json_lf(out_dir / "packet_R.json", pr)
    with open(out_dir / "packets_digest.txt", "w", encoding="utf-8", newline="\n") as fh:
        for p in (pL, pR, dpath):
            fh.write("%s  %s\n" % (C.sha256_file(p), p.name))
    sL = K.seal(KEY_L, key_L, sealed=sealed, keys_dir=keys_dir)
    sR = K.seal(KEY_R, key_R, sealed=sealed, keys_dir=keys_dir)
    return {"items_L": len(packet_items_L), "items_R": len(packet_items_R), "windows": windows_total,
            "decoys_L": len(dl), "decoys_R": len(dr), "oversize_windows": sum(1 for _, m in items_L
                                                                              if m.get("oversize")),
            "key_L_digest": sL["digest"], "key_R_digest": sR["digest"],
            "packet_L_sha256": C.sha256_file(pL), "packet_R_sha256": C.sha256_file(pR),
            "decoys_L_sha256": decoys_sha}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--population", default=str(HERE / "population.json"))
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--sealed", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keys-dir", default=None)
    a = ap.parse_args(argv)
    pop = json.loads(Path(a.population).read_text(encoding="utf-8"))
    r = build(pop, a.population, sealed=a.sealed, out_dir=a.out, keys_dir=a.keys_dir, seed=a.seed)
    print("packet_L: %d items (%d windows, %d oversize, %d decoys); packet_R: %d items (%d decoys)"
          % (r["items_L"], r["windows"], r["oversize_windows"], r["decoys_L"], r["items_R"], r["decoys_R"]))
    print("key digests: L %s  R %s" % (r["key_L_digest"][:12], r["key_R_digest"][:12]))
    print("NO seat was run and no number was computed by this file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
