# -*- coding: utf-8 -*-
"""Synthetic documents, receipts, decoy picks and canned seat answers for the dry run (SPEC §The
dry run). Every byte here is invented in this file: no real document, stem or receipt name is
read, and no file written by it ends ``.sworn.json``, ``.sworn-receipt.json`` or ``PREREG_*.md``.

A synthetic population lives on a ``styxx.sworn.MemoryTree`` at a nominal commit, materialised
under a ``dryrun/`` directory as ``<id>.syn.json`` (sidecar), ``<id>.tree.json`` (the tree's files)
and ``<id>.receipt.json`` (the verifier's receipt over synthetic bytes — the committed-receipt
analogue). The canned answers are rules over the item text, so the pipeline's projection, decoy
gating (one family built to fail Panel R), unparsed and unlocated paths are all exercised.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C                                   # noqa: E402
import population as P                               # noqa: E402
from styxx import sworn                              # noqa: E402

C40 = "a" * 40
PREFIX = "SYN-"
EXCLUDED_PREFIX = "SYNX-"
FAIL_FAMILY = "local"          # the family the canned answers build to fail Panel R's decoys
_NAMES = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")


def _sp(text: str, receipt: str, kind: str) -> str:
    return '<sworn r="%s" k="%s">%s</sworn>' % (receipt, kind, text)


def make_document(k: int, rng: random.Random) -> Tuple[bytes, Dict[str, bytes]]:
    """One synthetic in-house document (inline bytes) and the files its receipts name."""
    base = "syn/%02d" % k
    prec = Decimal(rng.choice(["0.55", "0.61", "0.47", "0.72"]))
    rec = Decimal(rng.choice(["0.20", "0.31", "0.18"]))
    passed = rng.choice([296, 358, 120])
    log = ("synthetic log line one\nsynthetic log line seventeen\nsynthetic run complete with %d checks\n"
           % passed).encode("utf-8")
    result = {"rates": {"precision": str(prec), "recall": str(rec), "label": "held-out", "n": 38},
              "counts": {"passed": passed, "failed": 0, "note": "counts of a synthetic run"},
              "note": "a string leaf that says nothing"}
    result_b = (json.dumps(result, indent=1).replace('"%s"' % prec, str(prec)).replace('"%s"' % rec, str(rec))
                + "\n").encode("utf-8")
    log_sha = hashlib.sha256(log).hexdigest()
    doc = "\n".join([
        "# SYN document %02d: a synthetic report about nothing" % k,
        "",
        "Synthetic Lab, a header line with a date-shaped token 2026-09-05 and a version v0.%d." % k,
        "",
        "## What was measured",
        "",
        "The question was whether the synthetic gate holds under the synthetic bar. "
        + _sp("Precision was %s on the panel." % prec, "path:%s/result.json#/rates/precision" % base, "numeric")
        + " " + _sp("Recall was %s on the same items." % rec, "path:%s/result.json#/rates/recall" % base, "numeric")
        + " The suite ran for 42 minutes on the synthetic box. -",
        "",
        _sp("The harness ran the suite and %d checks passed." % passed, "path:%s/result.json#/counts/passed" % base, "numeric")
        + " " + _sp("The log reads `synthetic log line seventeen` near its middle.", "path:%s/log.txt" % base, "quote")
        + " " + _sp("The log carries no `unexpected failure marker`.", "path:%s/log.txt" % base, "absent")
        + " The runner printed `a phrase that is not in the log` twice.",
        "",
        _sp("The log's bytes hash to %s." % log_sha, "path:%s/log.txt" % base, "hash")
        + " This sentence restates the method without a number. That one is a hedge about scope.",
        "",
        "```",
        "code. fenced. 0.99 not a unit.",
        "```",
        "",
        "## What this does not say",
        "",
        "That any synthetic number means anything. The conclusion depends on the precision above and on nothing else.",
        "",
    ]).encode("utf-8")
    files = {"%s/result.json" % base: result_b, "%s/log.txt" % base: log}
    return doc, files


def make_excluded(k: int, rng: random.Random) -> Tuple[bytes, Dict[str, bytes]]:
    """A synthetic excluded document: eight pointer spans over a table with numeric siblings, eight
    number-bearing LOAD-BEARING sentences and eight digit-free NOT sentences, for the decoys.

    Every narrative sentence opens with three words no other sentence in the document opens with.
    That is not cosmetic: a Panel L bracket is located by exact byte search of its opening three
    words, so a document that repeats an opening leaves the bracket unlocated by the rule, and a
    decoy passage cut across two such paragraphs would gate the panel on the fixture's prose
    rather than on the seat's answer.
    """
    base = "synx/%02d" % k
    rows = []
    lines = ["# SYNX document %02d: a synthetic format document" % k, "",
             "A synthetic header line about the format, with no number in it.", ""]
    for j in range(8):
        v = Decimal(j) / Decimal(10) + Decimal("0.01")
        rows.append({"value": str(v), "other": str(v + Decimal("0.3")), "name": "row %s" % _NAMES[j]})
        lines.append(_sp("Row %s scored %s on the check." % (_NAMES[j], v), "path:%s/table.json#/rows/%d/value" % (base, j), "numeric")
                     + " Restatement %s carries the row without a number." % _NAMES[j])
        lines.append("")
    for j in range(8):
        lines.append("The %s gate cleared at 0.%d1 of the items in this paragraph. Method note %s follows it."
                     % (_NAMES[j], j + 1, _NAMES[j]))
        lines.append("")
    for j in range(8):
        lines.append("The %s paragraph explains a convention and nothing depends on it. Another %s sentence sits beside it."
                     % (_NAMES[j], _NAMES[j]))
        lines.append("")
    table = json.dumps({"rows": rows}, indent=1)
    for r in rows:
        table = table.replace('"%s"' % r["value"], r["value"]).replace('"%s"' % r["other"], r["other"])
    files = {"%s/table.json" % base: (table + "\n").encode("utf-8")}
    return "\n".join(lines).encode("utf-8"), files


def _materialise(out_dir: Path, root: Path, doc_id: str, raw: bytes, files: Dict[str, bytes]) -> dict:
    side = sworn.to_sidecar(raw, doc_id + ".md", commit=C40)
    tree = sworn.MemoryTree(files, commit=C40)
    core = sworn.verify(sidecar=side, tree=tree)
    receipt = sworn.issue_receipt(core, timestamp="2026-09-05T00:00:00Z")
    rel = out_dir.resolve().relative_to(root.resolve()).as_posix()
    paths = {"sidecar": "%s/%s.syn.json" % (rel, doc_id), "tree": "%s/%s.tree.json" % (rel, doc_id),
             "receipt": "%s/%s.receipt.json" % (rel, doc_id)}
    C.write_json_lf(out_dir / ("%s.syn.json" % doc_id), side)
    C.write_json_lf(out_dir / ("%s.tree.json" % doc_id), {k: v.decode("utf-8") for k, v in files.items()})
    C.write_json_lf(out_dir / ("%s.receipt.json" % doc_id), receipt)
    return {"stem": "%s/%s" % (rel, doc_id), "role": "synthetic", "source": dict(kind="synthetic", **paths)}


def write_population(out_dir, root, n_docs: int = 3, n_excluded: int = 2, seed: int = 1) -> dict:
    """Materialise a synthetic population under out_dir (a directory named dryrun) and return the
    population object in population.py's shape, doc_ids SYN-01.., excluded SYNX-01.. with sources."""
    out_dir, root = Path(out_dir), Path(root)
    if out_dir.name != "dryrun":
        raise SystemExit("REFUSED: synthetic items are written only under a directory named dryrun/")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    entries, excluded = [], []
    for k in range(1, n_docs + 1):
        raw, files = make_document(k, rng)
        entries.append(_materialise(out_dir, root, "%s%02d" % (PREFIX, k), raw, files))
    for k in range(1, n_excluded + 1):
        raw, files = make_excluded(k, rng)
        e = _materialise(out_dir, root, "%s%02d" % (EXCLUDED_PREFIX, k), raw, files)
        e["reason"] = "synthetic format document (the dry run's stand-in for the builder's documents)"
        excluded.append(e)
    pop = P.build(entries, C40, seed=seed, prefix=PREFIX, excluded=excluded, root=root)
    pop["synthetic"] = True
    pop["what_this_is_not"].append("synthetic: every byte invented by synthetic.py")
    return pop


def decoy_picks(pop: dict, root) -> List[Tuple[str, int, str]]:
    """Thirty picks over the synthetic excluded documents: number-bearing narrative units as
    LOAD-BEARING, digit-free ones as NOT."""
    lb, nt = [], []
    for e in P.iter_excluded(pop):
        side, _, _ = C.open_document(e, root=root)
        for i, u in enumerate(C.units_of(sworn.load_sidecar(side))):
            if u["sworn"] or u["fragment"] or i == 0:
                continue
            if "gate cleared" in u["text"]:
                lb.append((e["stem"], i, "LOAD-BEARING"))
            elif "explains a convention" in u["text"]:
                nt.append((e["stem"], i, "NOT"))
    n = C.N_DECOYS_PER_SIDE
    if len(lb) < n or len(nt) < n:
        raise SystemExit("REFUSED: synthetic excluded documents host %d/%d picks, %d needed" % (len(lb), len(nt), n))
    return lb[:n] + nt[:n]


# ------------------------------------------------------------------------------------------
# canned seat answers (--dry-run substitutes these; no transport is called)
# ------------------------------------------------------------------------------------------
_WORD = re.compile(r"\S+")


def _edge_words(sentence: str, n: int = C.EDGE_WORDS) -> Tuple[str, str]:
    words = _WORD.findall(sentence)
    if not words:
        return "", ""
    return " ".join(words[:n]), " ".join(words[-n:])


def _sentences(text: bytes) -> List[str]:
    out = []
    pos = 0
    for m in sworn._SENTENCE_SPLIT.finditer(text):
        out.append(text[pos:m.start()])
        pos = m.end()
    out.append(text[pos:])
    return [s.strip().decode("utf-8") for s in out if s.strip()]


def canned_answer(panel: str, item: dict, family: str, seat: int) -> str:
    """The raw text a seat would have returned, by rule. Panel L: LOAD-BEARING iff the sentence
    carries a digit, NOT otherwise; the failing family's third seat says UNSURE to everything;
    one item is returned unparseable, one bracket unlocatable, one only locatable after whitespace
    collapse. Panel R: a rule that reads the leaf; the failing family answers UNSURE throughout."""
    if panel == "L":
        if family == "claude" and seat == 2 and item["id"] == "L-0001":
            return "I would rather not answer in JSON today."
        brackets = []
        for n, s in enumerate(_sentences(item["text"].encode("utf-8"))):
            o, c = _edge_words(s)
            label = "LOAD-BEARING" if re.search(r"[0-9]", s) else "NOT"
            if family == FAIL_FAMILY and seat == 3:
                label = "UNSURE"
            if family == FAIL_FAMILY and seat == 1 and item["id"] == "L-0002" and n == 0:
                o = "zzz qqq www"
            if family == FAIL_FAMILY and seat == 2 and item["id"] == "L-0003" and n == 0:
                o = o.replace(" ", "  ")
            brackets.append({"opening_words": o, "closing_words": c, "label": label})
        return json.dumps({"brackets": brackets})
    if family == FAIL_FAMILY:
        return json.dumps({"answer": "UNSURE"})
    sent = item["sentence"]
    kind = item["kind"]
    value = item["leaf"]["value"]
    if kind == "numeric":
        why, tok, _ = sworn._number_token(sent)
        try:
            ok = why is None and Decimal(value) == Decimal(tok.replace(",", "").rstrip("%"))
        except (InvalidOperation, ValueError, AttributeError):
            ok = False
    elif kind in ("quote", "absent"):
        needle, _ = sworn._needle_in(sent.encode("utf-8"))
        present = needle is not None and needle.decode("utf-8", errors="replace") in value
        ok = present if kind == "quote" else not present
    else:
        runs = [m.group(0).lower() for m in sworn._HEXRUN.finditer(sent) if len(m.group(0)) == 64]
        ok = len(runs) == 1 and runs[0] == value
    return json.dumps({"answer": "YES" if ok else "NO"})


def canned_trivial_twin(side: dict) -> Optional[dict]:
    """A trivially-swearing twin by rule: the same text with only the last span kept."""
    side = sworn.load_sidecar(side)
    if not side["spans"]:
        return None
    twin = dict(side)
    twin["document"] = dict(side["document"])
    twin["spans"] = [dict(side["spans"][-1])]
    return twin
