"""Where an added obligation trigger can actually reach, given the status ladder.

Written to check a preregistration before freezing it, and it killed the preregistration.

`certify_doc` decides a token's status with an ordered ladder (certify.py ~887-917):

    if is_spec or is_hist:   -> ABSTAIN  "spec-or-historical"
    elif is_notation:        -> ABSTAIN  "v05-notation"
    elif derived_ref:        -> VERIFIED
    elif field_unbound_ref:  -> ABSTAIN  "unbound-field"
    elif hits:               -> VERIFIED          <-- BEFORE `bound` is consulted
    elif bound:              -> UNGROUNDED
    else:                    -> ABSTAIN  ref=None

Two consequences follow from the ORDER, and neither is obvious from the prose:

1. A token that is currently ABSTAIN with `receipt_ref: None` fell to the final `else`, which is
   reachable only when `hits` is empty. Adding a disjunct to `bound` moves it to `elif bound`, so
   it can only ever become **UNGROUNDED**. An added obligation trigger recovers nothing; it
   manufactures accusations.

2. Tokens abstained as `spec-or-historical` or `v05-notation` are intercepted at the TOP, above
   `bound`. Obligating them changes nothing at all.

And the deeper one, verified directly rather than inferred: because `elif hits` precedes
`elif bound`, **obligation does not gate verification. It gates accusation only.** A number whose
value matches a receipt leaf is sworn to whether or not anything obligated the verifier to look at
it.

  python papers/closed-model-frontier/oath_obligation_ladder_census.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import certify_doc                                    # noqa: E402
from styxx.corpus_audit import _doc_for, discover_certificates           # noqa: E402

OUT = HERE / "oath_obligation_ladder_census.json"
DEC = re.compile(r"^\d*\.\d{2,}$")


def code_spans(line: str):
    out, open_at = [], None
    for m in re.finditer("`", line):
        if open_at is None:
            open_at = m.end()
        else:
            out.append((open_at, m.start()))
            open_at = None
    return out


def find_col(src: str, tok: str):
    for m in re.finditer(re.escape(tok), src):
        a, b = m.start(), m.end()
        if (a == 0 or not (src[a - 1].isdigit() or src[a - 1] == ".")) and \
           (b == len(src) or not (src[b].isdigit() or src[b] == ".")):
            return a
    return None


def obligation_gates_verification() -> dict:
    """Does a value-match verify WITHOUT any obligation? Run it rather than reason about it."""
    d = Path(tempfile.mkdtemp())
    doc = d / "x.md"
    doc.write_text("Legal scholars have long argued about 0.4267 in the abstract.\n",
                   encoding="utf-8")
    rec = d / "r.json"
    rec.write_text(json.dumps({"whatever": 0.4267}), encoding="utf-8")
    e = certify_doc(doc, [rec])["ledger"][0]
    return {"line_has_no_measurement_vocabulary": True, "token": e["token"],
            "status": e["status"], "receipt_ref": e["receipt_ref"],
            "obligation_gates_verification": e["status"] != "VERIFIED"}


def main() -> int:
    by_ref = collections.Counter()
    movable_docs, total_abstain = set(), 0

    for cp in discover_certificates(ROOT / "papers"):
        try:
            cert = json.loads(cp.read_text(encoding="utf-8"))
            doc = _doc_for(cp)
            lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for e in cert.get("ledger", []):
            if e.get("status") != "ABSTAIN":
                continue
            total_abstain += 1
            if not DEC.match(e["token"]):
                continue
            src = lines[e["line"] - 1] if 0 < e["line"] <= len(lines) else ""
            col = e.get("col")
            # Certificates written before the v0.10 token-column work carry no `col`. A filter
            # that skips them silently drops most of the population -- an earlier version of this
            # count returned 5 instead of 493 for exactly that reason.
            if col is None or src[col:col + len(e["token"])] != e["token"]:
                col = find_col(src, e["token"])
            if col is None:
                continue
            if any(a <= col < b for a, b in code_spans(src)):
                continue
            ref = e.get("receipt_ref")
            by_ref[(ref or "NULL").split(":")[0]] += 1
            if ref is None:
                movable_docs.add(doc.name)

    movable = by_ref["NULL"]
    fires = sum(by_ref.values())
    gate = obligation_gates_verification()

    payload = {
        "census": "where an added obligation trigger can reach, given the ladder order",
        "status": "RECON. Killed a preregistration before it was frozen.",
        "candidate": ">= 2 decimal places AND not inside a backtick code span",
        "corpus_abstain_tokens": total_abstain,
        "predicate_fires_on": fires,
        "by_why_the_token_abstained": dict(by_ref.most_common()),
        "actually_movable": movable,
        "movable_across_documents": len(movable_docs),
        "intercepted_above_bound": fires - movable,
        "every_movable_token_becomes": "UNGROUNDED",
        "movable_tokens_that_could_become_VERIFIED": 0,
        "why_zero": ("`elif hits: VERIFIED` precedes `elif bound: UNGROUNDED`. A token currently "
                     "ABSTAIN with receipt_ref None reached the final else, which is reachable "
                     "only when hits is empty. Adding a disjunct to `bound` can therefore only "
                     "produce UNGROUNDED."),
        "obligation_gates_verification_probe": gate,
        "the_structural_finding": (
            "Obligation does NOT gate verification. It gates ACCUSATION only. A number whose "
            "value matches a receipt leaf is sworn to regardless of whether anything obligated "
            "the verifier to examine it -- demonstrated above on a line carrying no measurement "
            "vocabulary at all."),
        "what_this_corrects": (
            "Documents published on 2026-08-27 describe the abstained band as 'checkable claims "
            "the verifier declined to examine', which reads as claims it would have verified. It "
            "would not: those tokens abstained because no receipt holds their value, so obligating "
            "them yields accusations. The coverage gap is real and its CHARACTER is different -- "
            "it is unbacked claims that go unflagged, not backed claims that go unchecked."),
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"corpus ABSTAIN tokens        : {total_abstain}")
    print(f"predicate fires on           : {fires}")
    for k, v in by_ref.most_common():
        note = "CAN MOVE -> UNGROUNDED" if k == "NULL" else "intercepted above `bound`"
        print(f"    {v:>4}  {k:<22} {note}")
    print(f"actually movable             : {movable}  across {len(movable_docs)} documents")
    print("could become VERIFIED        : 0")
    print(f"obligation gates verification: {gate['obligation_gates_verification']} "
          f"(probe token {gate['token']} -> {gate['status']})")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
