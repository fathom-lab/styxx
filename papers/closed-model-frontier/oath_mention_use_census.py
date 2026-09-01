"""CENSUS — how big is the mention/use class, and what would silencing it cost?

**A census. It licenses no claim.** It sizes a class so a successor preregistration knows what to
freeze its bars against, exactly as `oath_v10_ordinal_census.py` did before v0.11.

## The defect being sized

`SYNTHESIS_mention_and_use_2026_08_26.md` records the same defect in four instruments: each
infers CLAIMHOOD FROM CO-OCCURRENCE, reading what sits near a token and concluding what the token
asserts. For the OATH verifier the sharpest form is quotation — a number you QUOTE is treated as a
number you CLAIM. The shipped verifier has one narrow escape (v0.1's quoted-historical rule, which
fires only on disclosure phrasing like *originally printed*) and it reaches nothing else.

Two documents in this corpus are OATH-FAILED because of it, both accused on tokens they quote as
examples of false accusations. That is not a rhetorical flourish: it is a measurable population,
and this file measures it.

## What is measured

For every token in the certified frame, which QUOTATION CONTEXTS it sits in, under several
candidate markers — and, crucially, **what its current status is**. Both halves matter and the
second is the one that kills designs:

* tokens currently UNGROUNDED in a quotation context are the class a mention/use predicate would
  RETRACT — the opportunity;
* tokens currently VERIFIED in a quotation context are what it would DESTROY — the cost.

v0.11 died three designs on exactly this arithmetic. A rule that silences quotations will also
silence every correct number a document quotes from its own receipts, and this corpus quotes its
own receipts constantly.

## What is NOT measured

Whether any individual token is genuinely a mention rather than a use. That is an adjudication and
it needs a panel with ties resolved AGAINST the clause, per the Retraction Protocol. This file
counts surfaces; it does not judge them.

  python papers/closed-model-frontier/oath_mention_use_census.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import certify_doc                                       # noqa: E402
from styxx.corpus_audit import _resolve_receipts                            # noqa: E402

OUT = HERE / "oath_mention_use_census.json"

# ---------------------------------------------------------------- candidate markers
#
# Each is a CANDIDATE, deliberately not a proposal. The census reports what each would reach so a
# prereg can pick one and freeze it, rather than a design being chosen because it sounded right.

_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
_BLOCKQUOTE = re.compile(r"^\s{0,3}>")
# A verb that introduces someone else's words. Narrow on purpose: the wide version of this idea is
# how `rate` came to fire on "learning rate".
_QUOTING_VERB = re.compile(
    r"\b(reads?|read|says?|said|prints?|printed|reported|quotes?|quoted|claims?|claimed|"
    r"asserts?|asserted|records?|recorded|wrote|writes)\b", re.I)
_LATEX = re.compile(r"\\[A-Za-z]+|\$")


def _inline_code_spans(line: str):
    """(start, end) spans of `inline code` — the commonest way this corpus quotes a value."""
    spans, i = [], 0
    while True:
        a = line.find("`", i)
        if a < 0:
            break
        b = line.find("`", a + 1)
        if b < 0:
            break
        spans.append((a, b))
        i = b + 1
    return spans


def _fenced_lines(lines) -> set:
    """1-based line numbers inside a fenced code block."""
    inside, out, fence = False, set(), None
    for i, ln in enumerate(lines, 1):
        m = _FENCE.match(ln)
        if m:
            if not inside:
                inside, fence = True, m.group(1)
            elif ln.strip().startswith(fence):
                inside = False
                continue
        if inside:
            out.add(i)
    return out


def markers_for(num: dict, lines, fenced: set) -> dict:
    ln_no = num["line"]
    line = lines[ln_no - 1] if ln_no - 1 < len(lines) else ""
    col = num.get("col")
    in_code = False
    if col is not None:
        norm = line.replace("\u2212", "-")
        in_code = any(a < col < b for a, b in _inline_code_spans(norm))
    return {
        "inline_code": in_code,
        "fenced_block": ln_no in fenced,
        "blockquote": bool(_BLOCKQUOTE.match(line)),
        "quoting_verb_on_line": bool(_QUOTING_VERB.search(line)),
        "latex_on_line": bool(_LATEX.search(line)),
    }


_INDEX_NAMES = frozenset({"line", "col", "seed", "token", "case", "i", "index", "idx", "step",
                          "epoch", "num", "count", "size", "id", "level", "rank"})
_SUBSCRIPT = re.compile(r"\[\d+\]$")


def _coincident(receipt_ref) -> bool:
    """Is this verification sworn to a POSITION rather than a measurement?

    Frozen definition, identical to `oath_v11_dogfood_selfcert.py`. It matters here because a
    candidate's COST column is only a real cost where the verifications it destroys are genuine.
    Destroying a coincidence is a gain wearing a loss's clothes.
    """
    path = (receipt_ref or "").partition(":")[2]
    if not path:
        return False
    term = _SUBSCRIPT.sub("", path.rsplit(".", 1)[-1]).lower()
    return bool(_SUBSCRIPT.search(path)) or term in _INDEX_NAMES


def resolvable_docs():
    out = []
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        if "anc" in cp.parts:
            continue
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        if not doc.exists():
            continue
        try:
            rec = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        receipts, missing, _ = _resolve_receipts(cp, rec, ROOT / "papers")
        if receipts and not missing:
            out.append((doc, receipts))
    return out


def main() -> int:
    docs = resolvable_docs()
    per_marker = collections.defaultdict(lambda: collections.Counter())
    totals = collections.Counter()
    any_marker = collections.Counter()
    accused_rows, verified_in_quote = [], []

    for doc, receipts in docs:
        try:
            cert = certify_doc(doc, receipts)
        except Exception:
            continue
        rel = doc.relative_to(ROOT).as_posix()
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        fenced = _fenced_lines(lines)
        for e in cert["ledger"]:
            st = e["status"]
            totals[st] += 1
            m = markers_for(e, lines, fenced)
            for name, hit in m.items():
                if hit:
                    per_marker[name][st] += 1
            if any(m.values()):
                any_marker[st] += 1
                if st == "VERIFIED":
                    verified_in_quote.append(
                        {"rel": rel, "line": e["line"], "token": e["token"],
                         "markers": [k for k, v in m.items() if v],
                         "receipt_ref": e["receipt_ref"],
                         "coincident": _coincident(e["receipt_ref"])})
            if st == "UNGROUNDED":
                accused_rows.append(
                    {"rel": rel, "line": e["line"], "token": e["token"],
                     "markers": [k for k, v in m.items() if v],
                     "context": e["context"][:150]})

    payload = {
        "census": "OATH mention/use — the quotation surface, sized",
        "status": "CENSUS. Licenses no claim. Sizes a class for a successor preregistration.",
        "verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "frame": {"documents": len(docs), "tokens": sum(totals.values()),
                  "status_counts": dict(totals)},
        "markers_are_candidates_not_a_proposal":
            "Each marker below is reported so a prereg can freeze ONE, with its measured cost in "
            "hand. None is recommended here. v0.11 killed three designs on exactly this "
            "arithmetic.",
        "per_marker": {k: dict(v) for k, v in sorted(per_marker.items())},
        "per_marker_cost_quality": {
            k: {"verified_destroyed": sum(1 for r in verified_in_quote if k in r["markers"]),
                "of_which_coincident":
                    sum(1 for r in verified_in_quote if k in r["markers"] and r["coincident"]),
                "of_which_nominal":
                    sum(1 for r in verified_in_quote if k in r["markers"] and not r["coincident"]),
                "accusations_reached":
                    sum(1 for r in accused_rows if k in r["markers"])}
            for k in sorted(per_marker)},
        "cost_quality_note":
            "A candidate's cost column is only a real cost where the verifications it destroys "
            "are GENUINE. Destroying a coincidence -- a token sworn to an index, a seed, a step "
            "counter -- is a gain wearing a loss's clothes. `of_which_nominal` is therefore the "
            "column a design lives or dies on, not `verified_destroyed`.",
        "any_marker": dict(any_marker),
        "opportunity": {
            "accused_total": totals["UNGROUNDED"],
            "accused_in_a_quotation_context":
                sum(1 for r in accused_rows if r["markers"]),
            "roster": accused_rows,
        },
        "cost": {
            "verified_total": totals["VERIFIED"],
            "verified_in_a_quotation_context": len(verified_in_quote),
            "note": "THE NUMBER THAT KILLS DESIGNS. A rule that silences quotations silences "
                    "every correct number a document quotes from its own receipts, and this "
                    "corpus quotes its own receipts constantly. Any candidate whose cost column "
                    "dwarfs its opportunity column is the broad-detector catastrophe again.",
            "sample": verified_in_quote[:40],
        },
        "what_this_does_not_show": [
            "Whether any individual token is genuinely a mention rather than a use. That is an "
            "adjudication needing a panel with ties resolved AGAINST the clause.",
            "That any marker is a good predicate. They are surfaces, not designs.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"frame {len(docs)} docs  {sum(totals.values())} tokens  {dict(totals)}")
    for k, v in sorted(per_marker.items()):
        print(f"  {k:<22} {dict(v)}")
    print(f"  {'ANY marker':<22} {dict(any_marker)}")
    print(f"opportunity: {payload['opportunity']['accused_in_a_quotation_context']} of "
          f"{totals['UNGROUNDED']} accusations sit in a quotation context")
    print(f"cost:        {len(verified_in_quote)} of {totals['VERIFIED']} verifications do too "
          f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
