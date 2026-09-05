# -*- coding: utf-8 -*-
"""The canary inserter (SPEC §The canary rule).

A canary is a well-formed sworn span whose receipt is known — by construction and by a
standard-library check — not to hold what the sentence says. Three constructions, all leaving the
canonical text byte-identical: A retargets an existing ``path:<file>#/pointer`` span to a sibling
scalar leaf; B turns a narrative unit holding one backtick needle into a ``quote`` span over a
cited file that lacks the needle; C turns a narrative unit holding one number into a ``numeric``
span over a cited leaf whose value differs by the rounding rule's own quantum.

Falsehood is established with ``Decimal`` and ``bytes.find`` and never with ``styxx.sworn.verify``
(a canary whose falsehood the instrument under test established would make the canary gate a
tautology). Form — not truth — is checked with the lexer and the loader. Twins go to the sealed
directory under ``sworn_measurement_twins/<doc_id>.canary-twin.json``; the tree carries only
``twins/canary_digest.txt`` and the salted digest of the canary key.

CLI: ``python papers/sworn/measurement/canaries.py build [--population population.json]
[--seed 20260905] [--rate all] [--sealed DIR] [--keys-dir keys] [--digest twins/canary_digest.txt]``.
Prints capacity per document and the smallest-n table; runs no seat and no verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C                                   # noqa: E402
import population as P                               # noqa: E402
import seal_key as K                                 # noqa: E402
from styxx import sworn                              # noqa: E402

TWINS_DIRNAME = "sworn_measurement_twins"
CANARY_KEY = "sworn_measurement_canary_key"
G_C_BAR = 0.95            # DESIGN v2 row 4 — proposed, unsigned; printed beside the capacity table
SEED_CANARY = 100         # random.Random(SEED + SEED_CANARY + doc_index): the canary choice per document
MINUS = chr(0x2212)


# ------------------------------------------------------------------------------------------
# the standard-library falsehood checks
# ------------------------------------------------------------------------------------------


def printed_decimal(tok: str) -> Tuple[Decimal, int]:
    """The printed token as a Decimal and the count of fractional digits it prints."""
    t = tok.replace(",", "").replace(MINUS, "-").rstrip("%")
    d = Decimal(t)
    frac = len(t.split(".", 1)[1]) if "." in t else 0
    return d, frac


def margin_ok(leaf: Decimal, printed: Decimal, frac: int) -> Tuple[bool, str]:
    """|leaf - printed| >= 10 ** -frac: the quantum ROUND_HALF_EVEN at printed precision cannot cross."""
    q = Decimal(1).scaleb(-frac)
    return abs(leaf - printed) >= q, str(q)


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (Decimal, str)) and not isinstance(v, bool)


def scalar_siblings(obj: Any, tokens: List[str]) -> List[Tuple[str, Any]]:
    """Scalar leaves sharing the parent container of the leaf at `tokens`, as (pointer, value)."""
    if not tokens:
        return []
    parent = obj
    for t in tokens[:-1]:
        parent = parent[int(t)] if isinstance(parent, list) else parent[t]
    prefix = "".join("/" + C.pointer_escape(t) for t in tokens[:-1])
    out: List[Tuple[str, Any]] = []
    if isinstance(parent, dict):
        for k, v in parent.items():
            if k != tokens[-1] and _is_scalar(v):
                out.append((prefix + "/" + C.pointer_escape(k), v))
    elif isinstance(parent, list):
        for i, v in enumerate(parent):
            if str(i) != tokens[-1] and _is_scalar(v):
                out.append((prefix + "/" + str(i), v))
    return out


def _pointer_receipt(side: dict, receipt: str) -> Optional[dict]:
    parsed, _ = sworn._parse_receipt(receipt)
    if parsed is None or parsed["form"] != "path":
        return None
    frag = parsed.get("fragment")
    if not frag or frag["type"] != "pointer":
        return None
    return parsed


def _numeric_ok(v: Any) -> bool:
    return isinstance(v, Decimal) and not isinstance(v, bool) and v.is_finite() and v.adjusted() <= 300


# ------------------------------------------------------------------------------------------
# construction A — retarget an existing pointer span to a sibling leaf
# ------------------------------------------------------------------------------------------


def retarget(side: dict, tree, span_index: int, rng: random.Random,
             want_string: bool = False) -> Optional[dict]:
    """A canary record for span `span_index`, or None when the span hosts none. With
    `want_string` (the dry run only) a numeric span is pointed at a STRING sibling so the verifier
    reports MALFORMED rather than FAILED — a planted falsehood the verifier does not FAIL."""
    span = side["spans"][span_index]
    kind = span["kind"]
    if kind not in ("numeric", "quote"):
        return None
    parsed = _pointer_receipt(side, span["receipt"])
    if parsed is None:
        return None
    data, _ = tree.blob(parsed["target"])
    if data is None:
        return None
    try:
        obj = C.load_json_decimal(data)
        sibs = scalar_siblings(obj, parsed["fragment"]["tokens"])
    except (ValueError, KeyError, IndexError, TypeError, UnicodeDecodeError):
        return None
    text = side["text"].encode("utf-8")
    inner = text[span["start"]:span["end"]]
    base = {"construction": "A", "original_span_index": span_index, "kind": kind,
            "start": span["start"], "end": span["end"]}
    if kind == "numeric":
        why, tok, _ = sworn._number_token(inner.decode("utf-8"))
        if why:
            return None
        printed, frac = printed_decimal(tok)
        if want_string:
            cands = [(p, v) for p, v in sibs if isinstance(v, str)]
            if not cands:
                return None
            ptr, v = rng.choice(cands)
            base.update(receipt="path:%s#%s" % (parsed["target"], ptr), printed=tok, needle_sha256=None,
                        leaf_value=v[:80], margin=None, falsehood_check="forced_malformed_for_dry_run")
            return base
        cands = []
        for p, v in sibs:
            if _numeric_ok(v):
                ok, q = margin_ok(v, printed, frac)
                if ok:
                    cands.append((p, v, q))
        if not cands:
            return None
        ptr, v, q = rng.choice(cands)
        base.update(receipt="path:%s#%s" % (parsed["target"], ptr), printed=tok, needle_sha256=None,
                    leaf_value=str(v), margin=q, falsehood_check="decimal_margin")
        return base
    needle, why = sworn._needle_in(inner)
    if needle is None:
        return None
    cands = [(p, v) for p, v in sibs if isinstance(v, str) and v.encode("utf-8").find(needle) == -1]
    if not cands:
        return None
    ptr, v = rng.choice(cands)
    base.update(receipt="path:%s#%s" % (parsed["target"], ptr), printed=None,
                needle_sha256=hashlib.sha256(needle).hexdigest(), leaf_value=v[:80], margin=None,
                falsehood_check="bytes_find")
    return base


# ------------------------------------------------------------------------------------------
# constructions B and C — a narrative unit becomes a span over a cited receipt
# ------------------------------------------------------------------------------------------


def cited(side: dict, tree) -> Tuple[List[str], List[Tuple[str, Any]]]:
    """(paths the document cites, [(pointer receipt, leaf value)] it cites), resolved at its commit."""
    paths: List[str] = []
    leaves: List[Tuple[str, Any]] = []
    for s in side["spans"]:
        parsed, _ = sworn._parse_receipt(s["receipt"])
        if parsed is None or parsed["form"] != "path":
            continue
        if parsed["target"] not in paths:
            paths.append(parsed["target"])
        frag = parsed.get("fragment")
        if frag and frag["type"] == "pointer":
            data, _ = tree.blob(parsed["target"])
            if data is None:
                continue
            try:
                leaf = C.pointer_walk(C.load_json_decimal(data), frag["raw"])
            except (ValueError, KeyError, IndexError, TypeError, UnicodeDecodeError):
                continue
            if _is_scalar(leaf):
                leaves.append((s["receipt"], leaf))
    return paths, leaves


def host_quote(unit: dict, tree, paths: List[str], rng: random.Random) -> Optional[dict]:
    inner = unit["text"].encode("utf-8")
    needle, why = sworn._needle_in(inner)
    if needle is None or len(needle) < sworn.SHORT_NEEDLE_BYTES:
        return None
    cands = []
    for p in paths:
        data, _ = tree.blob(p)
        if data is not None and data.find(needle) == -1:
            cands.append(p)
    if not cands:
        return None
    p = rng.choice(cands)
    return {"construction": "B", "original_span_index": None, "kind": "quote", "receipt": "path:" + p,
            "printed": None, "needle_sha256": hashlib.sha256(needle).hexdigest(), "leaf_value": None,
            "margin": None, "falsehood_check": "bytes_find", "start": unit["start"], "end": unit["end"]}


def host_number(unit: dict, leaves: List[Tuple[str, Any]], rng: random.Random) -> Optional[dict]:
    why, tok, _ = sworn._number_token(unit["text"])
    if why:
        return None
    printed, frac = printed_decimal(tok)
    cands = []
    for receipt, v in leaves:
        if _numeric_ok(v):
            ok, q = margin_ok(v, printed, frac)
            if ok:
                cands.append((receipt, v, q))
    if not cands:
        return None
    receipt, v, q = rng.choice(cands)
    return {"construction": "C", "original_span_index": None, "kind": "numeric", "receipt": receipt,
            "printed": tok, "needle_sha256": None, "leaf_value": str(v), "margin": q,
            "falsehood_check": "decimal_margin", "start": unit["start"], "end": unit["end"]}


# ------------------------------------------------------------------------------------------
# form: the lexer and the loader, never the verifier
# ------------------------------------------------------------------------------------------


def _twin_of(side: dict, spans: List[dict]) -> dict:
    return {"spec": side["spec"], "commit": side["commit"], "document": dict(side["document"]),
            "text": side["text"], "spans": sorted(spans, key=lambda s: (s["start"], s["end"])),
            "manifest": side["manifest"]}


def reproduces(twin: dict) -> bool:
    """load_sidecar accepts it and scan(render(twin)) reproduces its spans exactly."""
    try:
        sworn.load_sidecar(twin)
    except SystemExit:
        return False
    sc = sworn.scan(sworn.render(twin))
    if sc["document_malformed"] or not sc["lexical_ok"]:
        return False
    if any(d["malformed"] for d in sc["declarations"]):
        return False
    seen = sorted((d["start"], d["end"], d["receipt"], d["kind"]) for d in sc["declarations"])
    want = [(s["start"], s["end"], s["receipt"], s["kind"]) for s in twin["spans"]]
    return seen == want


def _overlaps(a: int, b: int, spans: List[dict]) -> bool:
    return any(s["start"] < b and a < s["end"] for s in spans)


# ------------------------------------------------------------------------------------------
# the twin
# ------------------------------------------------------------------------------------------


def build_twin(side: dict, tree, doc_index: int, seed: int = C.SEED, rate: str = C.CANARY_RATE,
               force_malformed: bool = False) -> Tuple[dict, List[dict], dict]:
    """(twin sidecar, canary records with twin_span_index, capacity {A, B, C, total, rejected})."""
    if rate != "all":
        raise SystemExit("REFUSED: CANARY_RATE is %r; only 'all' is implemented (the rate is capacity)" % rate)
    side = sworn.load_sidecar(side)
    rng = random.Random(seed + SEED_CANARY + doc_index)
    spans = [dict(s) for s in side["spans"]]
    records: List[dict] = []
    rejected = 0
    forced = False
    for i in range(len(spans)):
        rec = None
        if force_malformed and not forced and spans[i]["kind"] == "numeric":
            rec = retarget(side, tree, i, rng, want_string=True)
            forced = rec is not None
        if rec is None:
            rec = retarget(side, tree, i, rng)
        if rec is None:
            continue
        trial = [dict(s) for s in spans]
        trial[i]["receipt"] = rec["receipt"]
        if not reproduces(_twin_of(side, trial)):
            rejected += 1
            continue
        spans = trial
        records.append(rec)
    paths, leaves = cited(side, tree)
    canonical = side["text"].encode("utf-8")
    comments = sworn.scan(canonical)["comments"]
    for u in C.units_of(side):
        if u["sworn"] or u["fragment"] or len(u["text"]) > sworn.SPAN_CAP_CODEPOINTS:
            continue
        if _overlaps(u["start"], u["end"], spans):
            continue
        if any(a <= u["start"] < b for a, b in comments):
            continue
        rec = host_quote(u, tree, paths, rng) or host_number(u, leaves, rng)
        if rec is None:
            continue
        cand = {"start": u["start"], "end": u["end"], "receipt": rec["receipt"], "kind": rec["kind"]}
        if not reproduces(_twin_of(side, spans + [cand])):
            rejected += 1
            continue
        spans.append(cand)
        records.append(rec)
    twin = _twin_of(side, spans)
    if not reproduces(twin):
        raise SystemExit("REFUSED: the twin does not reproduce under the lexer")
    index = {(s["start"], s["end"]): i for i, s in enumerate(twin["spans"])}
    for r in records:
        r["twin_span_index"] = index[(r["start"], r["end"])]
    records.sort(key=lambda r: r["twin_span_index"])
    cap = {"A": sum(1 for r in records if r["construction"] == "A"),
           "B": sum(1 for r in records if r["construction"] == "B"),
           "C": sum(1 for r in records if r["construction"] == "C"),
           "total": len(records), "rejected_by_form": rejected}
    return twin, records, cap


def smallest_n_table(bar: float = G_C_BAR, misses=(0, 1, 2, 3, 5)) -> Dict[str, Optional[int]]:
    return {"misses_%d" % m: C.smallest_n_clearing(bar, m) for m in misses}


def build(pop: dict, root=None, sealed=None, keys_dir=None, digest_path=None, seed: int = C.SEED,
          rate: str = C.CANARY_RATE, force_malformed_in: Optional[str] = None) -> dict:
    """Twins for every population document into the sealed directory; the canary key sealed;
    twins/canary_digest.txt written. Returns the capacity summary (counts only)."""
    sealed = Path(sealed or C.SEALED)
    twins_dir = sealed / TWINS_DIRNAME
    twins_dir.mkdir(parents=True, exist_ok=True)
    digest_path = Path(digest_path or HERE / "twins" / "canary_digest.txt")
    if digest_path.exists():
        raise SystemExit("REFUSED: %s exists; a rebuild is a new file at a new commit" % digest_path)
    key: Dict[str, dict] = {}
    lines: List[str] = []
    capacity: Dict[str, dict] = {}
    for k, e in enumerate(P.iter_documents(pop)):
        side, tree, _ = C.open_document(e, root=root)
        twin, records, cap = build_twin(side, tree, k, seed=seed, rate=rate,
                                        force_malformed=(e["doc_id"] == force_malformed_in))
        tb = (json.dumps(twin, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
        name = "%s.canary-twin.json" % e["doc_id"]
        with open(twins_dir / name, "wb") as fh:
            fh.write(tb)
        sha = C.sha256_bytes(tb)
        key[e["doc_id"]] = {"stem": e["stem"], "commit": side["commit"], "twin_sha256": sha,
                            "canaries": records}
        lines.append("%s  %s  canaries=%d" % (sha, name, cap["total"]))
        capacity[e["doc_id"]] = cap
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(digest_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    sealed_info = K.seal(CANARY_KEY, key, sealed=sealed, keys_dir=keys_dir)
    pooled = sum(c["total"] for c in capacity.values())
    return {"capacity": capacity, "pooled_n": pooled, "smallest_n_clearing_bar_at_k_eq_n": smallest_n_table(),
            "bar_proposed_unsigned": G_C_BAR, "canary_key_digest": sealed_info["digest"],
            "twins_dir": str(twins_dir), "digest_file": str(digest_path)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    b = sub.add_parser("build")
    b.add_argument("--population", default=str(HERE / "population.json"))
    b.add_argument("--seed", type=int, default=C.SEED)
    b.add_argument("--rate", default=C.CANARY_RATE)
    b.add_argument("--sealed", default=None)
    b.add_argument("--keys-dir", default=None)
    b.add_argument("--digest", default=None)
    a = ap.parse_args(argv)
    pop = json.loads(Path(a.population).read_text(encoding="utf-8"))
    r = build(pop, sealed=a.sealed, keys_dir=a.keys_dir, digest_path=a.digest, seed=a.seed, rate=a.rate)
    for doc_id, cap in r["capacity"].items():
        print("%s  A=%d B=%d C=%d total=%d rejected_by_form=%d"
              % (doc_id, cap["A"], cap["B"], cap["C"], cap["total"], cap["rejected_by_form"]))
    print("pooled n = %d canaries (capacity, not a measurement); bar %.2f proposed, unsigned"
          % (r["pooled_n"], r["bar_proposed_unsigned"]))
    print("smallest n clearing the bar at k = n: %s" % json.dumps(r["smallest_n_clearing_bar_at_k_eq_n"]))
    print("canary key sealed, digest %s; twins under %s; %s written"
          % (r["canary_key_digest"][:12], r["twins_dir"], r["digest_file"]))
    print("no verifier was run on any twin by this file; no seat was run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
