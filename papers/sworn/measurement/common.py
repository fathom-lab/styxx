# -*- coding: utf-8 -*-
"""Shared helpers for the sworn measurement machinery.

Spec: ``papers/sworn/SPEC_sworn_measurement_machinery_2026_09_05.md``, frozen before this file
existed. Everything here is an adapter-side helper: the parameters, the question texts, the unit
set and its windows, bracket location and projection, majority and the cross-family rule, Cohen's
kappa, Wilson (imported from ``styxx.mind``), LF-only JSON writes, salted key digests, the
git-plumbing readers, and the PREREG refusal every seat runner shares.

Nothing in this module adjudicates a span. ``styxx/sworn.py`` imports nothing from here.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import sworn                      # noqa: E402
from styxx.mind import wilson                # noqa: E402  (stdlib-only module)

__all__ = [
    "HERE", "ROOT", "SEALED", "SEED", "WINDOW_MAX_UNITS", "N_DECOYS_PER_SIDE", "SEATS_PER_FAMILY",
    "LEAF_VIEW_MAX_CHARS", "EDGE_WORDS", "CANARY_RATE", "WILSON_Z", "LOCAL_MAX_NEW_TOKENS",
    "CLAUDE_TIMEOUT_S", "QUESTION_L", "QUESTION_R", "BLOCKS_L", "BLOCKS_R", "instructions",
    "block_order", "SCHEMA_L", "SCHEMA_R", "LABELS_L", "LABELS_R", "EXCLUDED_LABELS",
    "write_json_lf", "sha256_bytes", "sha256_file", "key_bytes", "salted_digest", "read_digest_file",
    "git", "show_at", "tracked_at", "head_commit", "ls_files",
    "load_json_decimal", "pointer_walk", "pointer_escape", "leaf_view",
    "units_of", "reconcile_units", "windows_of", "bindable", "locate", "project_labels",
    "majority", "family_label", "final_label", "cohen_kappa", "wilson", "smallest_n_clearing",
    "bind_inputs", "prereg_at_head", "refuse_unless_prereg", "open_document",
]

# ------------------------------------------------------------------------------------------
# parameters (SPEC §Parameters) — module constants; a change after the PREREG commit is a moved bar
# ------------------------------------------------------------------------------------------
SEALED = Path(os.environ.get("STYXX_SEALED_DIR", r"C:\Users\heyzo\clawd\styxx-sealed"))
SEED = 20260905
WINDOW_MAX_UNITS = 40
N_DECOYS_PER_SIDE = 15
SEATS_PER_FAMILY = 3
LEAF_VIEW_MAX_CHARS = 2000
EDGE_WORDS = 3
CANARY_RATE = "all"
WILSON_Z = 1.96
LOCAL_MAX_NEW_TOKENS = 1200
CLAUDE_TIMEOUT_S = 600

DESIGN_COMMIT = "d35ed81d6a6bb78f52933bfe92382e13f1bc2be4"     # DESIGN v2 landed here
LABELS_L = ("LOAD-BEARING", "NOT", "UNSURE")
LABELS_R = ("YES", "NO", "UNSURE")
EXCLUDED_LABELS = frozenset({"UNSURE", "NO-LABEL", "NO-MAJORITY", "FAMILY-SPLIT"})

# ------------------------------------------------------------------------------------------
# the questions, verbatim from the SPEC
# ------------------------------------------------------------------------------------------
QUESTION_L = (
    "For each sentence in the passage below, decide whether it is LOAD-BEARING — a claim the "
    "document's conclusion depends on, which could turn out to be right or wrong — or NOT "
    "load-bearing — context, method, a date, a version, a file name, a restatement, a hedge, a "
    "sentence about what the document does not say — or UNSURE. Draw the sentence boundaries "
    "yourself."
)
BLOCKS_L = {
    "TASK": "TASK. You are reading one passage from a technical report. Label every sentence.",
    "FORMAT": (
        "FORMAT. Answer with one JSON object and nothing else: "
        '{"brackets": [{"opening_words": "...", "closing_words": "...", "label": "LOAD-BEARING"}, ...]}. '
        "opening_words is the opening three words of the sentence and closing_words is its closing "
        "three words, copied exactly as they appear in the passage including punctuation and "
        "capitalisation; a sentence shorter than six words may repeat words in both. label is "
        "exactly one of LOAD-BEARING, NOT, UNSURE. Cover every sentence once; do not skip, merge "
        "or reorder."
    ),
    "CAUTIONS": (
        "CAUTIONS. A sentence that reports a number, a quoted string, a hash, or the absence of "
        "something can still be NOT load-bearing if the conclusion would stand without it. A "
        "sentence with no number can be LOAD-BEARING. Use UNSURE honestly; it is counted against "
        "the panel, not against you. Do not reproduce the passage; quote only the opening and "
        "closing three words."
    ),
}
QUESTION_R = (
    "Here is one sentence from a technical report, the kind of check its author declared for it, "
    "and the receipt leaf the author bound it to. Does the leaf evidence the sentence — is what "
    "the leaf holds what the sentence says? Answer YES, NO or UNSURE."
)
BLOCKS_R = {
    "TASK": (
        "TASK. Compare one sentence with one leaf. numeric: the sentence's one number should be "
        "the leaf's value at the precision the sentence prints. quote: the text between backticks "
        "in the sentence should occur in the leaf. absent: it should not. hash: the sentence's "
        "64-hex digest should equal the leaf's digest."
    ),
    "FORMAT": 'FORMAT. Answer with one JSON object and nothing else: {"answer": "YES"}, "NO" or "UNSURE".',
    "CAUTIONS": (
        "CAUTIONS. Judge the pairing, not the sentence's truth in the world. If the leaf view is "
        "marked truncated and the answer would depend on the cut part, answer UNSURE. Do not explain."
    ),
}
_ORDER = ("TASK", "FORMAT", "CAUTIONS")


def block_order(seat: int) -> List[str]:
    """Seat 1 as written, seat 2 starting from FORMAT, seat 3 starting from CAUTIONS."""
    k = (int(seat) - 1) % len(_ORDER)
    return list(_ORDER[k:] + _ORDER[:k])


def instructions(panel: str, seat: int) -> str:
    blocks = BLOCKS_L if panel == "L" else BLOCKS_R
    return "\n\n".join(blocks[b] for b in block_order(seat))


SCHEMA_L = {
    "type": "object",
    "properties": {"brackets": {"type": "array", "items": {
        "type": "object",
        "properties": {"opening_words": {"type": "string"}, "closing_words": {"type": "string"},
                       "label": {"type": "string", "enum": list(LABELS_L)}},
        "required": ["opening_words", "closing_words", "label"], "additionalProperties": False}}},
    "required": ["brackets"], "additionalProperties": False,
}
SCHEMA_R = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": list(LABELS_R)}},
    "required": ["answer"], "additionalProperties": False,
}

# ------------------------------------------------------------------------------------------
# bytes, digests, LF writes
# ------------------------------------------------------------------------------------------


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def write_json_lf(path, obj) -> Path:
    """Every JSON the lab writes is LF-only (styxx.sworn._write_json_lf's form)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    return p


def key_bytes(key: dict) -> bytes:
    """The one serialisation a sealed key has (SPEC §The sealed keys)."""
    return (json.dumps(key, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def salted_digest(data: bytes, salt: str) -> str:
    return hashlib.sha256(data + salt.encode("utf-8")).hexdigest()


def read_digest_file(path) -> Tuple[str, str]:
    """`<hex>  <name>` → (hex, name)."""
    line = Path(path).read_text(encoding="utf-8").strip().splitlines()[0]
    parts = line.split()
    return parts[0], (parts[1] if len(parts) > 1 else "")

# ------------------------------------------------------------------------------------------
# git plumbing — the ambient world enters here and nowhere below this section
# ------------------------------------------------------------------------------------------


def git(*args: str, root=None, stdin: Optional[bytes] = None) -> Tuple[int, bytes]:
    r = subprocess.run(["git", "-C", str(root or ROOT), *args], input=stdin,
                       capture_output=True, check=False)
    return r.returncode, r.stdout


def head_commit(root=None) -> str:
    rc, out = git("rev-parse", "HEAD", root=root)
    if rc != 0:
        raise SystemExit("REFUSED: git rev-parse HEAD failed")
    return out.decode("ascii").strip()


def show_at(commit: str, path: str, root=None) -> Optional[bytes]:
    """The blob at <commit>:<path>, never the working tree."""
    rc, out = git("show", "%s:%s" % (commit, path), root=root)
    return out if rc == 0 else None


def tracked_at(commit: str, path: str, root=None) -> bool:
    rc, _ = git("cat-file", "-e", "%s:%s" % (commit, path), root=root)
    return rc == 0


def ls_files(pattern: str, root=None) -> List[str]:
    rc, out = git("ls-files", "--", pattern, root=root)
    if rc != 0:
        return []
    return [ln for ln in out.decode("utf-8", errors="replace").split("\n") if ln.strip()]

# ------------------------------------------------------------------------------------------
# JSON receipts, read the standard-library way (never through the verifier)
# ------------------------------------------------------------------------------------------


def load_json_decimal(data: bytes):
    return json.loads(data.decode("utf-8"), parse_float=Decimal, parse_int=Decimal)


def pointer_escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def pointer_walk(obj, pointer: str):
    """RFC 6901 walk with ~0/~1 unescaping; raises KeyError/IndexError/TypeError on a miss."""
    if pointer == "":
        return obj
    for tok in pointer.split("/")[1:]:
        tok = tok.replace("~1", "/").replace("~0", "~")
        if isinstance(obj, list):
            obj = obj[int(tok)]
        elif isinstance(obj, dict):
            obj = obj[tok]
        else:
            raise TypeError("pointer into a scalar")
    return obj


def _leaf_text(leaf) -> str:
    if isinstance(leaf, bool) or leaf is None:
        return json.dumps(leaf)
    if isinstance(leaf, Decimal):
        return str(leaf)
    if isinstance(leaf, str):
        return json.dumps(leaf, ensure_ascii=False)
    return json.dumps(leaf, ensure_ascii=False, default=str)


def leaf_view(receipt: str, data: bytes, kind: str) -> dict:
    """The Panel R leaf view (SPEC §The packets), by receipt form."""
    parsed, why = sworn._parse_receipt(receipt)
    if parsed is None:
        return {"receipt_name": receipt, "pointer": None, "lines": None, "value": "",
                "value_kind": "malformed_receipt", "truncated": False, "note": why}
    target = parsed["target"]
    name = target.rsplit("/", 1)[-1] if parsed["form"] == "path" else target
    frag = parsed.get("fragment")
    view = {"receipt_name": name, "pointer": None, "lines": None, "value": "",
            "value_kind": "receipt_text", "truncated": False}
    if kind == "hash":
        view.update(value=sha256_bytes(data), value_kind="sha256")
        return view
    if frag is not None and frag["type"] == "pointer":
        view["pointer"] = frag["raw"]
        try:
            leaf = pointer_walk(load_json_decimal(data), frag["raw"])
            text = _leaf_text(leaf)
            view["value_kind"] = "leaf"
        except (KeyError, IndexError, TypeError, ValueError, UnicodeDecodeError) as e:
            text = ""
            view["value_kind"] = "unresolvable_leaf"
            view["note"] = type(e).__name__
    elif frag is not None and frag["type"] == "lines":
        view["lines"] = frag["raw"]
        sl = sworn._line_slice(data, frag["first"], frag["last"])
        text = (sl or b"").decode("utf-8", errors="replace")
        view["value_kind"] = "slice"
    else:
        text = data.decode("utf-8", errors="replace")
    if len(text) > LEAF_VIEW_MAX_CHARS:
        text = text[:LEAF_VIEW_MAX_CHARS]
        view["truncated"] = True
    view["value"] = text
    return view

# ------------------------------------------------------------------------------------------
# units, fragments, windows (SPEC §Units, fragments and windows)
# ------------------------------------------------------------------------------------------
_ALNUM = re.compile(rb"[A-Za-z0-9]")


def units_of(sidecar: dict) -> List[dict]:
    """The unit set: every sworn span, then every narrative sentence of the masked canonical text
    under the diffgate splitter, byte for byte the loop of styxx.sworn._coverage."""
    canonical = sidecar["text"].encode("utf-8")
    spans = sorted(sidecar["spans"], key=lambda s: s["start"])
    out: List[dict] = []
    for i, s in enumerate(spans):
        out.append({"start": s["start"], "end": s["end"], "sworn": True, "span_index": i,
                    "fragment": False, "text": canonical[s["start"]:s["end"]].decode("utf-8")})
    buf = bytearray(canonical)
    for s in spans:
        for i in range(s["start"], s["end"]):
            buf[i] = 0x20
    for a, b in sworn._fenced_regions(bytes(buf))[0]:
        for i in range(a, b):
            buf[i] = 0x20
    narrative = bytes(buf)
    pos = 0
    pieces = []
    for m in sworn._SENTENCE_SPLIT.finditer(narrative):
        pieces.append((pos, m.start()))
        pos = m.end()
    pieces.append((pos, len(narrative)))
    for a, b in pieces:
        seg = narrative[a:b]
        if not seg.strip():
            continue
        lead = len(seg) - len(seg.lstrip())
        stripped = seg.strip()
        start, end = a + lead, a + lead + len(stripped)
        out.append({"start": start, "end": end, "sworn": False, "span_index": None,
                    "fragment": _ALNUM.search(stripped) is None,
                    "text": canonical[start:end].decode("utf-8")})
    out.sort(key=lambda u: (u["start"], u["end"]))
    return out


def reconcile_units(units: List[dict], receipt: dict) -> Tuple[bool, int, int]:
    """(ok, narrative units counted here, narrative_sentences the committed receipt printed)."""
    mine = sum(1 for u in units if not u["sworn"])
    theirs = receipt.get("coverage", {}).get("narrative_sentences")
    return mine == theirs, mine, theirs


_BLANK = re.compile(rb"\n[ \t]*\n(?:[ \t]*\n)*")


def windows_of(canonical: bytes, units: List[dict], max_units: int = WINDOW_MAX_UNITS) -> List[dict]:
    """Blank-line-delimited paragraphs packed greedily to <= max_units units each."""
    bounds = [0] + [m.end() for m in _BLANK.finditer(canonical)] + [len(canonical)]
    paras = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i] < bounds[i + 1]]
    out: List[dict] = []
    cur: Optional[dict] = None
    for a, b in paras:
        idx = [i for i, u in enumerate(units) if a <= u["start"] < b]
        if cur is not None and (len(cur["units"]) + len(idx) <= max_units or not idx):
            cur["end"] = b
            cur["units"].extend(idx)
            continue
        if cur is not None:
            out.append(cur)
        cur = {"start": a, "end": b, "units": list(idx), "oversize": len(idx) > max_units}
    if cur is not None:
        out.append(cur)
    return out

# ------------------------------------------------------------------------------------------
# bindability (SPEC §Projection …) — a byte predicate over three kinds
# ------------------------------------------------------------------------------------------


def bindable(text: bytes) -> Dict[str, Optional[bool]]:
    s = text.decode("utf-8", errors="replace")
    numeric = sworn._number_token(s)[0] is None
    needle, why = sworn._needle_in(text)
    quote = why == "ok" and needle is not None and len(needle) >= sworn.SHORT_NEEDLE_BYTES
    runs = [m.group(0) for m in sworn._HEXRUN.finditer(s)]
    hsh = len([r for r in runs if len(r) == 64]) == 1
    return {"numeric": numeric, "quote": quote, "hash": hsh, "absent": None,
            "any": bool(numeric or quote or hsh)}

# ------------------------------------------------------------------------------------------
# bracket location and projection
# ------------------------------------------------------------------------------------------
_WS = re.compile(rb"\s+")


def _collapse(text: bytes) -> Tuple[bytes, List[int]]:
    """Whitespace runs → one space; returns the collapsed bytes and a map to original offsets."""
    out = bytearray()
    idx: List[int] = []
    i = 0
    n = len(text)
    while i < n:
        m = _WS.match(text, i)
        if m:
            out.append(0x20)
            idx.append(m.start())
            i = m.end()
        else:
            out.append(text[i])
            idx.append(i)
            i += 1
    idx.append(n)
    return bytes(out), idx


def _find_once(hay: bytes, opening: bytes, closing: bytes) -> Optional[Tuple[int, int]]:
    if not opening or not closing:
        return None
    if hay.count(opening) != 1:
        return None
    start = hay.find(opening)
    end_at = hay.find(closing, start)
    if end_at < 0:
        return None
    return start, end_at + len(closing)


def locate(item_text: bytes, opening_words: str, closing_words: str) -> Tuple[Optional[Tuple[int, int]], str]:
    """((start, end), 'exact' | 'collapsed') or (None, 'unlocated')."""
    o = opening_words.encode("utf-8")
    c = closing_words.encode("utf-8")
    hit = _find_once(item_text, o, c)
    if hit:
        return hit, "exact"
    col, idx = _collapse(item_text)
    oc, _ = _collapse(o.strip())
    cc, _ = _collapse(c.strip())
    hit = _find_once(col, oc, cc)
    if hit:
        return (idx[hit[0]], idx[hit[1]]), "collapsed"
    return None, "unlocated"


def project_labels(brackets: List[dict], unit_ranges: List[Tuple[int, int]]) -> List[str]:
    """brackets: [{start, end, label}] in item coordinates; unit_ranges likewise.
    Largest byte overlap wins; zero overlap or a tie between different labels → NO-LABEL."""
    out = []
    for a, b in unit_ranges:
        best = 0
        best_labels = set()
        for br in brackets:
            ov = min(b, br["end"]) - max(a, br["start"])
            if ov <= 0:
                continue
            if ov > best:
                best, best_labels = ov, {br["label"]}
            elif ov == best:
                best_labels.add(br["label"])
        if best == 0 or len(best_labels) != 1:
            out.append("NO-LABEL")
        else:
            out.append(next(iter(best_labels)))
    return out

# ------------------------------------------------------------------------------------------
# majority, family, final (SPEC §Projection, majority and the cross-family label)
# ------------------------------------------------------------------------------------------


def majority(votes):
    """score_extraction_panel.majority: the modal vote when strict, else None."""
    votes = [v for v in votes if v]
    if not votes:
        return None
    c = Counter(votes).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None
    return c[0][0]


def family_label(seat_labels: List[str], labels=LABELS_L) -> str:
    """Strict majority of three: at least two seats agreeing on a label in `labels`."""
    votes = [v for v in seat_labels if v in labels]
    if not votes:
        return "NO-MAJORITY"
    top, n = Counter(votes).most_common(1)[0]
    return top if n >= 2 else "NO-MAJORITY"


def final_label(a: str, b: str) -> str:
    if a in ("NO-MAJORITY",) or b in ("NO-MAJORITY",):
        return "NO-MAJORITY"
    return a if a == b else "FAMILY-SPLIT"


def cohen_kappa(a: List[str], b: List[str], exclude=EXCLUDED_LABELS) -> dict:
    pairs = [(x, y) for x, y in zip(a, b) if x not in exclude and y not in exclude]
    n = len(pairs)
    if n == 0:
        return {"kappa": float("nan"), "n": 0, "po": float("nan"), "pe": float("nan"),
                "excluded": len(a) - n}
    po = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    kappa = float("nan") if pe == 1 else (po - pe) / (1 - pe)
    return {"kappa": kappa, "n": n, "po": po, "pe": pe, "excluded": len(a) - n}


def smallest_n_clearing(bar: float, misses: int = 0, z: float = WILSON_Z, limit: int = 5000) -> Optional[int]:
    """The smallest n at which wilson(n - misses, n)[0] >= bar."""
    for n in range(max(1, misses + 1), limit + 1):
        if wilson(n - misses, n, z)[0] >= bar:
            return n
    return None

# ------------------------------------------------------------------------------------------
# lock (a local shim over git with styxx.receipt_binding's field names)
# ------------------------------------------------------------------------------------------


def bind_inputs(paths: List[Path], root=None) -> dict:
    root = Path(root or ROOT)
    head = head_commit(root)
    rows = []
    for p in paths:
        p = Path(p)
        rel = p.resolve().relative_to(root.resolve()).as_posix()
        raw = p.read_bytes()
        content = raw.replace(b"\r\n", b"\n")
        rc, out = git("hash-object", "--stdin", root=root, stdin=raw)
        blob = out.decode("ascii").strip() if rc == 0 else None
        rc, out = git("ls-tree", head, "--", rel, root=root)
        committed_blob = out.split()[2].decode("ascii") if rc == 0 and out.split() else None
        rows.append({"path": rel, "raw_sha256": sha256_bytes(raw), "content_sha256": sha256_bytes(content),
                     "blob": blob, "committed": bool(blob) and blob == committed_blob})
    return {"schema": "styxx-sworn/measurement-inputs/v1", "head": head, "inputs": rows}

# ------------------------------------------------------------------------------------------
# the PREREG refusal (SPEC §The directory and the ladder)
# ------------------------------------------------------------------------------------------


def prereg_at_head(root=None) -> Optional[str]:
    hits = ls_files("papers/sworn/PREREG_sworn_measurement_*.md", root=root)
    return sorted(hits)[0] if hits else None


def refuse_unless_prereg(dry_run: bool, key_digest_files: List[str], root=None) -> Optional[str]:
    """Returns the PREREG path, or None under --dry-run. SystemExit('REFUSED: …') otherwise."""
    if dry_run:
        return None
    prereg = prereg_at_head(root)
    if prereg is None:
        raise SystemExit("REFUSED: no papers/sworn/PREREG_sworn_measurement_*.md is committed at HEAD; "
                         "no seat reads a real document before the preregistration commit")
    head = head_commit(root)
    for rel in key_digest_files:
        if not tracked_at(head, rel, root=root):
            raise SystemExit("REFUSED: %s is not committed at HEAD; the key digest is committed "
                             "before any seat runs" % rel)
    return prereg

# ------------------------------------------------------------------------------------------
# documents: real (git at the sidecar's commit) or synthetic (files under dryrun/)
# ------------------------------------------------------------------------------------------


def open_document(entry: dict, root=None) -> Tuple[dict, Any, Optional[dict]]:
    """(sidecar, tree, committed receipt or None) for a population entry."""
    root = Path(root or ROOT)
    src = entry.get("source", {"kind": "git"})
    if src.get("kind") == "synthetic":
        side = json.loads((root / src["sidecar"]).read_text(encoding="utf-8"))
        files = json.loads((root / src["tree"]).read_text(encoding="utf-8"))
        tree = sworn.MemoryTree({k: v.encode("utf-8") for k, v in files.items()}, commit=side["commit"])
        rec = None
        if src.get("receipt"):
            rec = json.loads((root / src["receipt"]).read_text(encoding="utf-8"))
        return side, tree, rec
    # The sidecar and its receipt are read at the PINNED commit (the tree the population rule was
    # applied to); the sidecar's own `commit` is where its receipts resolve, and a sidecar is
    # routinely tracked later than the commit it names.
    pinned = entry.get("pinned_commit") or "HEAD"
    side_b = show_at(pinned, entry["stem"] + ".sworn.json", root=root)
    if side_b is None:
        raise SystemExit("REFUSED: %s.sworn.json is not tracked at %s" % (entry["stem"], pinned[:12]))
    rec_b = show_at(pinned, entry["stem"] + ".sworn-receipt.json", root=root)
    side = json.loads(side_b.decode("utf-8"))
    if entry.get("sidecar_commit") and side["commit"] != entry["sidecar_commit"]:
        raise SystemExit("REFUSED: %s names commit %s, population.json recorded %s"
                         % (entry["stem"], side["commit"], entry["sidecar_commit"]))
    rec = json.loads(rec_b.decode("utf-8")) if rec_b is not None else None
    return side, sworn.GitTree(root, side["commit"]), rec
