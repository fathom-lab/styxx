# -*- coding: utf-8 -*-
"""The scorer, committed before any seat runs (SPEC §The gates, each as a function of committed
inputs). Every threshold below is DESIGN v2's bar, transcribed, and **proposed, unsigned** until a
PREREG names it; a change after the PREREG commit is a moved bar.

The scorer obtains every verdict it uses by calling ``styxx.sworn.verify`` on a twin at the commit
the twin names, or by reading a committed receipt; it adjudicates nothing itself. It refuses to fold
when a sealed key's salted digest differs from the committed one, when a twin's sha256 differs from
``twins/canary_digest.txt``, or — outside ``--dry-run`` — when no PREREG is committed at HEAD or the
PREREG's lock does not cover this file's committed blob. A gate that fails prints its title; nothing
is re-run. Under ``--dry-run`` only ``SYN-`` items are accepted and every share, interval, kappa and
Q3 value is the literal ``DRYRUN-NO-RATE``.

CLI: ``python papers/sworn/measurement/score.py [--dry-run --dir dryrun --sealed DIR --out FILE]``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import canaries as CAN                               # noqa: E402
import common as C                                   # noqa: E402
import population as P                               # noqa: E402
import seal_key as K                                 # noqa: E402
from styxx import sworn                              # noqa: E402

RESULT_SCHEMA = "styxx-sworn/measurement-result/v1"
FAMILIES = ("claude", "local")
WITHHELD = "WITHHELD"
NORATE = "DRYRUN-NO-RATE"

# DESIGN v2 lines 50-61, transcribed — proposed, unsigned
G_D_OVERALL = 27
G_D_SIDE = 9
G_S1 = 0.70
G_S1X = 0.50
G_S2 = 0.25
G_P = 0.10
G_C_LOWER = 0.95
G_G1 = 0.80

TITLES = {
    "G_D": "void panel; the void is the result",
    "G_F": "one-family: counts only, no precision",
    "G_S1": "authors leave bindable sentences unbound",
    "G_S2": "the numerator is padding",
    "G_P": "the author named the wrong leaf",
    "G_C": "the verifier misses planted falsehoods",
    "G_G1": "the floor cannot price gaming; the floor leaves the headline",
}

DISCLOSURE = [
    "The authors of the in-house documents knew the bars before writing them; the in-house arm is "
    "bar-aware by construction.",
    "Both families ran on one machine, and one of them is the family that wrote the documents; "
    "correlated error is the ceiling, and kappa between families is a description, never validity.",
    "The Panel L decoys were authored by the builder from documents the population rule excludes; "
    "the Panel R decoys were constructed by the same code that plants the canaries.",
    "Under v0.2, sentence_share = sworn_total / (sworn_total + narrative_sentences) does not know "
    "which sentences were sworn; a G-G1 pass says the trivial twin swore fewer sentences, and "
    "nothing about whether the floor prices gaming.",
    "The canaries for the nine existing documents were planted after authorship; a prospective "
    "rate for documents not yet written is owed and enforced by a harness that runs before the "
    "author sees a verdict.",
    "Three seats of one deterministic local model under three instruction-block orders are not "
    "three judgements; the Claude family's three seats are three fresh sessions under the "
    "transport's default sampling.",
]


def _rate(k: int, n: int, dry: bool):
    if dry:
        return NORATE
    return None if n == 0 else round(k / n, 4)


# ------------------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------------------


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def seat_file(seat_dir: Path, family: str, panel: str, seat: int) -> Optional[dict]:
    p = seat_dir / family / ("%s-seat%d.json" % (panel, seat))
    return _load(p) if p.exists() else None


def _documents(pop: dict, root) -> Dict[str, dict]:
    docs = {}
    for e in P.iter_documents(pop):
        side, tree, rec = C.open_document(e, root=root)
        side = sworn.load_sidecar(side)
        docs[e["doc_id"]] = {"entry": e, "side": side, "tree": tree, "receipt": rec,
                             "units": C.units_of(side), "canonical": side["text"].encode("utf-8")}
    return docs


# ------------------------------------------------------------------------------------------
# Panel L
# ------------------------------------------------------------------------------------------


def _locate_brackets(item_text: bytes, brackets: List[dict], counts: dict) -> List[dict]:
    out = []
    for b in brackets:
        hit, how = C.locate(item_text, str(b.get("opening_words", "")), str(b.get("closing_words", "")))
        if hit is None:
            counts["unlocated_brackets"] += 1
            continue
        if how == "collapsed":
            counts["located_by_second_pass"] += 1
        counts["located_brackets"] += 1
        out.append({"start": hit[0], "end": hit[1], "label": b.get("label")})
    return out


def panel_L(packet: dict, key: dict, docs: Dict[str, dict], seat_dir: Path, families=FAMILIES) -> dict:
    items = {i["id"]: i for i in packet["items"]}
    seats = {f: [seat_file(seat_dir, f, "L", s) for s in range(1, C.SEATS_PER_FAMILY + 1)] for f in families}
    present = {f: all(s is not None and s.get("verdict") in ("RECORDED", "DRY-RUN") for s in seats[f])
               for f in families}
    # per family: item id -> unit ranges (item coordinates) and their identities
    unit_ranges: Dict[str, List[Tuple[int, int]]] = {}
    unit_ids: Dict[str, List[Tuple[str, Any]]] = {}
    for iid, meta in key.items():
        if meta["kind"] == "document":
            d = docs[meta["doc_id"]]
            w = meta["window"]
            idx = [k for k, u in enumerate(d["units"]) if w["start"] <= u["start"] < w["end"]]
            unit_ranges[iid] = [(d["units"][k]["start"] - w["start"], d["units"][k]["end"] - w["start"]) for k in idx]
            unit_ids[iid] = [("document", (meta["doc_id"], k)) for k in idx]
        else:
            unit_ranges[iid] = [(meta["keyed"]["start"], meta["keyed"]["end"])]
            unit_ids[iid] = [("decoy", (meta["decoy_id"], meta["side"]))]
    counts = {f: {"unlocated_brackets": 0, "located_by_second_pass": 0, "located_brackets": 0,
                  "unparsed_items": 0} for f in families}
    family_labels: Dict[str, Dict[str, List[str]]] = {f: {} for f in families}   # f -> iid -> labels per unit
    seat1_brackets: Dict[str, Dict[str, List[dict]]] = {f: {} for f in families}
    for f in families:
        if not present[f]:
            continue
        for iid, item in items.items():
            text = item["text"].encode("utf-8")
            per_seat = []
            for s, sf in enumerate(seats[f], 1):
                row = next((r for r in sf.get("items", []) if r.get("id") == iid), None)
                if row is None or not row.get("parsed"):
                    counts[f]["unparsed_items"] += 1
                    per_seat.append(["NO-LABEL"] * len(unit_ranges[iid]))
                    continue
                located = _locate_brackets(text, row.get("brackets") or [], counts[f])
                if s == 1:
                    seat1_brackets[f][iid] = located
                per_seat.append(C.project_labels(located, unit_ranges[iid]))
            family_labels[f][iid] = [C.family_label([per_seat[s][u] for s in range(len(per_seat))])
                                     for u in range(len(unit_ranges[iid]))]
    # G-D per family
    gd = {}
    for f in families:
        if not present[f]:
            gd[f] = {"present": False, "pass": False, "title": TITLES["G_D"], "proposed_unsigned": True}
            continue
        per_side = {}
        for iid, meta in key.items():
            if meta["kind"] != "decoy":
                continue
            side = meta["side"]
            per_side.setdefault(side, {"n": 0, "correct": 0})
            per_side[side]["n"] += 1
            if family_labels[f][iid][0] == side:
                per_side[side]["correct"] += 1
        overall_n = sum(v["n"] for v in per_side.values())
        overall_k = sum(v["correct"] for v in per_side.values())
        ok = overall_k >= G_D_OVERALL and all(v["correct"] >= G_D_SIDE for v in per_side.values())
        gd[f] = {"present": True, "correct_overall": overall_k, "n_overall": overall_n, "per_side": per_side,
                 "bar": {"overall": G_D_OVERALL, "per_side": G_D_SIDE}, "pass": ok,
                 "title": None if ok else TITLES["G_D"], "proposed_unsigned": True}
    clearing = [f for f in families if gd[f]["pass"]]
    # labels per document unit
    labels: Dict[str, Dict[int, dict]] = {}
    for iid, ids in unit_ids.items():
        for u, (kind, ident) in enumerate(ids):
            if kind != "document":
                continue
            doc_id, k = ident
            row = {f: (family_labels[f][iid][u] if present[f] else None) for f in families}
            if len(clearing) == 2:
                row["final"] = C.final_label(row[clearing[0]], row[clearing[1]])
            elif len(clearing) == 1:
                row["final"] = None
                row["one_family"] = row[clearing[0]]
            else:
                row["final"] = None
            labels.setdefault(doc_id, {})[k] = row
    return {"G_D": gd, "clearing": clearing, "counts": counts, "labels": labels,
            "family_labels": family_labels, "seat1_brackets": seat1_brackets, "unit_ranges": unit_ranges,
            "unit_ids": unit_ids, "present": present, "items": items}


def cells_and_gates_L(res: dict, docs: Dict[str, dict], dry: bool) -> dict:
    fam = FAMILIES
    clearing = res["clearing"]
    out: Dict[str, Any] = {}
    label_key = "final" if len(clearing) == 2 else ("one_family" if len(clearing) == 1 else None)
    cells = {"LOAD-BEARING": 0, "NOT": 0, "UNSURE": 0, "NO-MAJORITY": 0, "FAMILY-SPLIT": 0, "NO-LABEL": 0}
    s1_num = s1_den = 0
    gu = 0
    s2_num = s2_den = 0
    q3: Dict[str, Any] = {}
    for doc_id, d in docs.items():
        lb_narrative = 0
        for k, u in enumerate(d["units"]):
            row = res["labels"].get(doc_id, {}).get(k)
            lab = row.get(label_key) if (row and label_key) else None
            if lab is None:
                continue
            cells[lab] = cells.get(lab, 0) + 1
            if u["fragment"] or lab not in ("LOAD-BEARING", "NOT"):
                continue
            bind = True if u["sworn"] else C.bindable(u["text"].encode("utf-8"))["any"]
            if lab == "LOAD-BEARING":
                if bind:
                    s1_den += 1
                    if u["sworn"]:
                        s1_num += 1
                else:
                    gu += 1
                if not u["sworn"]:
                    lb_narrative += 1
            if u["sworn"]:
                s2_den += 1
                if lab == "NOT":
                    s2_num += 1
        sworn_total = sum(1 for u in d["units"] if u["sworn"])
        rec_share = ((d["receipt"] or {}).get("coverage") or {}).get("sentence_share")
        denom = sworn_total + lb_narrative
        if dry:
            q3[doc_id] = {"panel_coverage": NORATE, "sentence_share": NORATE, "difference": NORATE,
                          "lb_narrative_units": lb_narrative}
        elif label_key == "final" and denom and rec_share is not None:
            pc = round(sworn_total / denom, 4)
            q3[doc_id] = {"panel_coverage": pc, "sentence_share": rec_share, "difference": round(pc - rec_share, 4),
                          "lb_narrative_units": lb_narrative}
        else:
            q3[doc_id] = {"panel_coverage": WITHHELD, "sentence_share": rec_share, "difference": WITHHELD,
                          "lb_narrative_units": lb_narrative}
    two = len(clearing) == 2
    label = "final" if two else ("one-family" if len(clearing) == 1 else "none")
    out["G_S1"] = {"numerator": s1_num, "denominator": s1_den, "labels": label,
                   "share": _rate(s1_num, s1_den, dry) if two else WITHHELD, "bar": G_S1, "proposed_unsigned": True}
    out["G_S1"]["pass"] = (None if not two or dry or out["G_S1"]["share"] is None
                           else out["G_S1"]["share"] >= G_S1)
    out["G_S1"]["title"] = TITLES["G_S1"] if out["G_S1"]["pass"] is False else None
    out["G_S1X"] = {"share": WITHHELD, "note": "external arm not built in v0.1 (owed)", "bar": G_S1X,
                    "proposed_unsigned": True}
    out["G_U"] = {"count": gu, "labels": label, "bar": None, "proposed_unsigned": True}
    out["G_S2"] = {"numerator": s2_num, "denominator": s2_den, "labels": label,
                   "share": _rate(s2_num, s2_den, dry) if two else WITHHELD, "bar": G_S2, "proposed_unsigned": True}
    out["G_S2"]["pass"] = (None if not two or dry or out["G_S2"]["share"] is None
                           else out["G_S2"]["share"] <= G_S2)
    out["G_S2"]["title"] = TITLES["G_S2"] if out["G_S2"]["pass"] is False else None
    out["cells"] = {"final_labels": cells, "labels": label,
                    "unlocated_brackets": {f: res["counts"][f]["unlocated_brackets"] for f in fam},
                    "located_brackets": {f: res["counts"][f]["located_brackets"] for f in fam},
                    "located_by_second_pass": {f: res["counts"][f]["located_by_second_pass"] for f in fam},
                    "unparsed_items": {f: res["counts"][f]["unparsed_items"] for f in fam}}
    out["q3"] = q3
    # kappa over the splitter's unit set (document units only), both families' labels
    a, b = [], []
    for doc_id, rows in res["labels"].items():
        for k, row in rows.items():
            a.append(row.get(fam[0]) or "NO-LABEL")
            b.append(row.get(fam[1]) or "NO-LABEL")
    # panel-boundary variant: claude seat-1 brackets as units, local seat-1 projected by overlap
    pa, pb = [], []
    for iid, ids in res["unit_ids"].items():
        if not ids or ids[0][0] != "document":
            continue
        ca = res["seat1_brackets"][fam[0]].get(iid)
        cb = res["seat1_brackets"][fam[1]].get(iid)
        if ca is None or cb is None:
            continue
        pa.extend(br["label"] for br in ca)
        pb.extend(C.project_labels(cb, [(br["start"], br["end"]) for br in ca]))
    if two and not dry:
        out["kappa"] = {"splitter": C.cohen_kappa(a, b), "panel_boundary": C.cohen_kappa(pa, pb),
                        "note": "a description of two families on one machine, never validity"}
    elif two:
        ks, kp = C.cohen_kappa(a, b), C.cohen_kappa(pa, pb)
        out["kappa"] = {"splitter": {"kappa": NORATE, "n": ks["n"], "excluded": ks["excluded"]},
                        "panel_boundary": {"kappa": NORATE, "n": kp["n"], "excluded": kp["excluded"]},
                        "note": "dry run: no rate"}
    else:
        out["kappa"] = {"splitter": WITHHELD, "panel_boundary": WITHHELD, "note": "fewer than two families clear Panel L"}
    return out


# ------------------------------------------------------------------------------------------
# Panel R
# ------------------------------------------------------------------------------------------


def panel_R(packet: dict, key: dict, docs: Dict[str, dict], seat_dir: Path, dry: bool, families=FAMILIES) -> dict:
    seats = {f: [seat_file(seat_dir, f, "R", s) for s in range(1, C.SEATS_PER_FAMILY + 1)] for f in families}
    present = {f: all(s is not None and s.get("verdict") in ("RECORDED", "DRY-RUN") for s in seats[f])
               for f in families}
    answers: Dict[str, Dict[str, str]] = {f: {} for f in families}
    unparsed = {f: 0 for f in families}
    for f in families:
        if not present[f]:
            continue
        for iid in key:
            votes = []
            for sf in seats[f]:
                row = next((r for r in sf.get("items", []) if r.get("id") == iid), None)
                if row is None or not row.get("parsed"):
                    unparsed[f] += 1
                    votes.append("NO-LABEL")
                else:
                    votes.append(row.get("answer"))
            answers[f][iid] = C.family_label(votes, C.LABELS_R)
    gd = {}
    for f in families:
        if not present[f]:
            gd[f] = {"present": False, "pass": False, "title": TITLES["G_D"], "proposed_unsigned": True}
            continue
        per_side = {}
        for iid, meta in key.items():
            if meta["kind"] != "decoy":
                continue
            per_side.setdefault(meta["side"], {"n": 0, "correct": 0})
            per_side[meta["side"]]["n"] += 1
            if answers[f][iid] == meta["side"]:
                per_side[meta["side"]]["correct"] += 1
        k_all = sum(v["correct"] for v in per_side.values())
        ok = k_all >= G_D_OVERALL and all(v["correct"] >= G_D_SIDE for v in per_side.values())
        gd[f] = {"present": True, "correct_overall": k_all, "n_overall": sum(v["n"] for v in per_side.values()),
                 "per_side": per_side, "bar": {"overall": G_D_OVERALL, "per_side": G_D_SIDE}, "pass": ok,
                 "title": None if ok else TITLES["G_D"], "proposed_unsigned": True}
    clearing = [f for f in families if gd[f]["pass"]]
    two = len(clearing) == 2
    finals = {"YES": 0, "NO": 0, "UNSURE": 0, "NO-MAJORITY": 0, "FAMILY-SPLIT": 0}
    gr = 0
    for iid, meta in key.items():
        if meta["kind"] != "document":
            continue
        if two:
            fin = C.final_label(answers[clearing[0]][iid], answers[clearing[1]][iid])
        elif len(clearing) == 1:
            fin = answers[clearing[0]][iid]
        else:
            continue
        finals[fin] = finals.get(fin, 0) + 1
        rec = docs[meta["doc_id"]]["receipt"] or {}
        spans = rec.get("spans") or []
        verdict = spans[meta["span_index"]].get("verdict") if meta["span_index"] < len(spans) else None
        if fin == "NO" and verdict == "HELD":
            gr += 1
    label = "final" if two else ("one-family" if len(clearing) == 1 else "none")
    den = finals["YES"] + finals["NO"]
    gp = {"numerator": finals["NO"], "denominator": den, "labels": label,
          "share": _rate(finals["NO"], den, dry) if two else WITHHELD, "bar": G_P, "proposed_unsigned": True}
    gp["pass"] = None if not two or dry or gp["share"] is None else gp["share"] <= G_P
    gp["title"] = TITLES["G_P"] if gp["pass"] is False else None
    return {"G_D": gd, "clearing": clearing, "present": present, "G_P": gp,
            "G_R": {"count": gr, "labels": label, "note": "HELD by the committed receipt and NO by the panel",
                    "proposed_unsigned": True},
            "finals": finals, "unparsed_items": unparsed}


# ------------------------------------------------------------------------------------------
# G-C and G-G1
# ------------------------------------------------------------------------------------------


def gate_GC(docs: Dict[str, dict], sealed: Path, digest_path: Path, keys_dir: Path, dry: bool) -> dict:
    key = K.load_key(CAN.CANARY_KEY, sealed, keys_dir)
    lines = digest_path.read_text(encoding="utf-8").splitlines()
    want = {ln.split()[1]: ln.split()[0] for ln in lines if ln.strip()}
    twins_dir = sealed / CAN.TWINS_DIRNAME
    k = n = 0
    per_twin = {}
    for doc_id, meta in key.items():
        name = "%s.canary-twin.json" % doc_id
        p = twins_dir / name
        if name not in want or not p.exists():
            raise SystemExit("REFUSED: twin %s is not listed in %s or is missing" % (name, digest_path))
        if C.sha256_file(p) != want[name]:
            raise SystemExit("REFUSED: twin %s does not match its committed digest" % name)
        twin = _load(p)
        if docs.get(doc_id) is None:
            raise SystemExit("REFUSED: twin %s names a document outside the population" % name)
        if twin["text"] != docs[doc_id]["side"]["text"]:
            raise SystemExit("REFUSED: twin %s does not share its document's text" % name)
        core = sworn.verify(sidecar=twin, tree=docs[doc_id]["tree"])
        row = {"k": 0, "n": 0, "malformed": 0, "unresolved": 0, "held": 0}
        for rec in meta["canaries"]:
            v = core["spans"][rec["twin_span_index"]]["verdict"]
            row["n"] += 1
            if v == "FAILED":
                row["k"] += 1
            elif v == "MALFORMED":
                row["malformed"] += 1
            elif v == "UNRESOLVED":
                row["unresolved"] += 1
            elif v == "HELD":
                row["held"] += 1
        per_twin[doc_id] = row
        k += row["k"]
        n += row["n"]
    lo, hi = C.wilson(k, n) if n else (float("nan"), float("nan"))
    out = {"k": k, "n": n, "per_twin": per_twin, "bar": G_C_LOWER, "proposed_unsigned": True,
           "smallest_n_clearing_bar_at_k_eq_n": C.smallest_n_clearing(G_C_LOWER, 0),
           "note": "MALFORMED and UNRESOLVED canaries count in n and not in k"}
    if dry:
        out.update(wilson95=[NORATE, NORATE], **{"pass": None, "title": None})
    else:
        ok = bool(n) and lo >= G_C_LOWER
        out.update(wilson95=[round(lo, 4), round(hi, 4)], **{"pass": ok, "title": None if ok else TITLES["G_C"]})
    return out


def gate_GG1(docs: Dict[str, dict], seat_dir: Path, dry: bool, families=FAMILIES) -> dict:
    pairs = lower = changed = 0
    rows = {}
    for f in families:
        tdir = seat_dir / f / "trivial"
        if not tdir.exists():
            continue
        for p in sorted(tdir.glob("*.trivial-twin.json")):
            doc_id = p.name[:-len(".trivial-twin.json")]
            d = docs.get(doc_id)
            if d is None:
                continue
            twin = _load(p)
            if twin.get("text") != d["side"]["text"]:
                changed += 1
                rows["%s/%s" % (f, doc_id)] = "twin_text_changed"
                continue
            floor_twin = sworn.verify(sidecar=twin, tree=d["tree"])["coverage"]["sentence_share"]
            floor_orig = ((d["receipt"] or {}).get("coverage") or {}).get("sentence_share")
            pairs += 1
            is_lower = floor_orig is not None and floor_twin is not None and floor_twin < floor_orig
            lower += 1 if is_lower else 0
            rows["%s/%s" % (f, doc_id)] = ({"twin": NORATE, "original": NORATE, "lower": is_lower} if dry
                                            else {"twin": floor_twin, "original": floor_orig, "lower": is_lower})
    out = {"pairs": pairs, "lower": lower, "twin_text_changed": changed, "per_pair": rows, "bar": G_G1,
           "proposed_unsigned": True,
           "note": "a span-count comparison under the v0.2 floor: 'lower' means the twin swore fewer "
                   "sentences or masked fewer of the narrative, nothing about gaming"}
    out["share"] = _rate(lower, pairs, dry)
    out["pass"] = None if dry or pairs == 0 else out["share"] >= G_G1
    out["title"] = TITLES["G_G1"] if out["pass"] is False else None
    return out


# ------------------------------------------------------------------------------------------
# the fold
# ------------------------------------------------------------------------------------------


def _prereg_lock_check(root, prereg: str):
    """Outside a dry run: the PREREG at HEAD must cover this file's committed blob."""
    head = C.head_commit(root)
    rc, out = C.git("ls-tree", head, "--", Path(__file__).resolve().relative_to(Path(root).resolve()).as_posix(),
                    root=root)
    blob = out.split()[2].decode("ascii") if rc == 0 and out.split() else None
    body = C.show_at(head, prereg, root=root) or b""
    mine = C.sha256_file(__file__)
    if blob is None or (blob.encode("ascii") not in body and mine.encode("ascii") not in body):
        raise SystemExit("REFUSED: the PREREG's lock does not cover score.py's committed blob %s" % blob)


def fold(meas_dir=None, sealed=None, out_path=None, dry_run: bool = False, root=None,
         families=FAMILIES) -> dict:
    meas_dir = Path(meas_dir or HERE)
    root = Path(root or C.ROOT)
    sealed = Path(sealed or C.SEALED)
    keys_dir = meas_dir / "keys"
    seat_dir = meas_dir / "seat_outputs"
    pop = _load(meas_dir / "population.json")
    if dry_run:
        if sealed.resolve() == Path(C.SEALED).resolve():
            raise SystemExit("REFUSED: a dry run never reads the real sealed directory")
        bad = [d["doc_id"] for d in pop["documents"] if not str(d["doc_id"]).startswith("SYN-")]
        if bad or not pop.get("synthetic"):
            raise SystemExit("REFUSED: a dry run accepts only SYN- items from a synthetic population, got %s" % bad)
        if meas_dir.name != "dryrun":
            raise SystemExit("REFUSED: a dry run folds only under a directory named dryrun/")
        prereg = None
    else:
        if pop.get("synthetic") or any(str(d["doc_id"]).startswith("SYN-") for d in pop["documents"]):
            raise SystemExit("REFUSED: a synthetic population folds only under --dry-run")
        prereg = C.refuse_unless_prereg(False, [(meas_dir / "keys" / (n + ".sha256")).resolve().relative_to(root.resolve()).as_posix()
                                                 for n in ("sworn_measurement_key_L", "sworn_measurement_key_R",
                                                           CAN.CANARY_KEY)], root=root)
        _prereg_lock_check(root, prereg)
    packet_L = _load(meas_dir / "packet_L.json")
    packet_R = _load(meas_dir / "packet_R.json")
    key_L = K.load_key("sworn_measurement_key_L", sealed, keys_dir)
    key_R = K.load_key("sworn_measurement_key_R", sealed, keys_dir)
    docs = _documents(pop, root)
    resL = panel_L(packet_L, key_L, docs, seat_dir, families)
    gatesL = cells_and_gates_L(resL, docs, dry_run)
    resR = panel_R(packet_R, key_R, docs, seat_dir, dry_run, families)
    gc = gate_GC(docs, sealed, meas_dir / "twins" / "canary_digest.txt", keys_dir, dry_run)
    gg1 = gate_GG1(docs, seat_dir, dry_run, families)
    substrates = {}
    for f in families:
        sf = seat_file(seat_dir, f, "L", 1) or seat_file(seat_dir, f, "R", 1)
        substrates[f] = (sf or {}).get("substrate")
    gf = {"families_clearing": {"L": len(resL["clearing"]), "R": len(resR["clearing"])},
          "bar": "two families, each clearing its decoys", "proposed_unsigned": True,
          "pass": len(resL["clearing"]) == 2 and len(resR["clearing"]) == 2}
    gf["title"] = None if gf["pass"] else TITLES["G_F"]
    withheld = []
    for name, g in (("G_S1", gatesL["G_S1"]), ("G_S2", gatesL["G_S2"]), ("G_P", resR["G_P"])):
        if g["share"] == WITHHELD:
            withheld.append(name)
    if gatesL["kappa"]["splitter"] == WITHHELD:
        withheld.append("kappa")
    if any(v.get("difference") == WITHHELD for v in gatesL["q3"].values()):
        withheld.append("q3")
    lock_paths = [Path(__file__), meas_dir / "packet_L.json", meas_dir / "packet_R.json",
                  meas_dir / "twins" / "canary_digest.txt"] + sorted(keys_dir.glob("*.sha256"))
    result = {
        "schema": RESULT_SCHEMA, "dry_run": dry_run, "quotable": (not dry_run) and gf["pass"],
        "prereg": prereg, "lock": C.bind_inputs(lock_paths, root=root),
        "population": {"documents": len(docs), "sworn_spans": sum(1 for d in docs.values() for u in d["units"] if u["sworn"]),
                       "units": sum(len(d["units"]) for d in docs.values()),
                       "fragments": sum(1 for d in docs.values() for u in d["units"] if u["fragment"]),
                       "synthetic": bool(pop.get("synthetic"))},
        "families": substrates,
        "gates": {"G_D": {"L": resL["G_D"], "R": resR["G_D"]}, "G_F": gf, "G_S1": gatesL["G_S1"],
                  "G_S1X": gatesL["G_S1X"], "G_U": gatesL["G_U"], "G_S2": gatesL["G_S2"], "G_P": resR["G_P"],
                  "G_C": gc, "G_R": resR["G_R"], "G_G1": gg1},
        "cells": dict(gatesL["cells"], panel_R_finals=resR["finals"], panel_R_unparsed_items=resR["unparsed_items"]),
        "kappa": gatesL["kappa"],
        "q3": gatesL["q3"],
        "disclosure": list(DISCLOSURE),
        "withheld": withheld,
        "what_this_is_not": (["DRY RUN over synthetic bytes: no quotable number; every rate is DRYRUN-NO-RATE"]
                             if dry_run else ["a fold of committed inputs under proposed, unsigned bars"]),
    }
    if out_path is None:
        out_path = meas_dir / ("dry_run_result.json" if dry_run else "measurement_result.json")
    out_path = Path(out_path)
    if dry_run and out_path.resolve().parent.name != "dryrun":
        raise SystemExit("REFUSED: a dry run writes only under dryrun/")
    C.write_json_lf(out_path, result)
    result["_written"] = str(out_path)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dir", default=None, help="the measurement directory (dryrun/ for a dry run)")
    ap.add_argument("--sealed", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = fold(a.dir, a.sealed, a.out, dry_run=a.dry_run)
    g = r["gates"]
    print("families clearing: L %d, R %d; withheld: %s" % (g["G_F"]["families_clearing"]["L"],
                                                           g["G_F"]["families_clearing"]["R"], r["withheld"]))
    print("G_C: k=%d n=%d wilson95=%s" % (g["G_C"]["k"], g["G_C"]["n"], g["G_C"]["wilson95"]))
    print("written: %s" % r["_written"])
    if a.dry_run:
        print("DRY RUN - no quotable number")
    return 0


if __name__ == "__main__":
    sys.exit(main())
