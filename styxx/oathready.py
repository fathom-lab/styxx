"""styxx.oathready — is YOUR document ready to carry an OATH certificate?

`styxx.certify` answers a yes/no question about a finished document. This module answers the
question an author actually has while writing one: *which of my numbers carry receipts, which
do not, and what exactly do I change?*

## Why this exists, and what it does not claim

On 2026-08-26 the shipped verifier was pointed at twelve public repositories it had never seen
(`papers/closed-model-frontier/RECON_oath_external_reach_2026_08_26.md`). It abstained on 94% of
what it read, and every accusation it made was false. The conclusion was not that the instrument
is broken — it was that OATH is not a lie detector aimable at arbitrary prose.

**The second half of that is withdrawn** (`RESULT_oath_external_corpus_2026_08_27.md`). Against
140 repositories across seven filename conventions rather than two, the false-accusation rate is
0.2596: about three quarters of what the verifier accuses outside this lab are genuine claims, and
the original result replicates on its own query and nowhere else. The same cycle found that of
external tokens the verifier VERIFIED, a blind panel judged only about half to be claims at all —
the rest are command-line flags, link labels and hardware specs carrying an affirmative oath
because a value matched a receipt field. Read the "coincident" and "accused" sections of your own
report with that in mind: this module reports what the verifier did, and the verifier is noisy in
both directions on prose that was not written to carry receipts.

**OATH is a contract.** Proof-carrying code does not verify arbitrary binaries; it requires a
compiler that emits the proof alongside the program. Proof-carrying cognition is the same move:
it requires an author who emits receipts alongside the claims. The instrument works where the
contract is kept, and this module is the check that tells an author whether they have kept it.

So the report below is NOT a grade, a score, or a truth verdict. It cannot tell whether your
numbers are correct — only whether each one is *bound to something a reader could check*. A
document can be fully ready and completely wrong.

## The honest limits, stated up front because they will show up in your report

* **Mention is treated as use.** A number you QUOTE — from another paper, from a console
  transcript, from an error you are reporting — is treated as a number you CLAIM. The verifier
  has one narrow escape for this (disclosure phrasing like "originally printed") and it does not
  reach quotation in general. Expect false accusations on quoted figures.
* **The obligation predicate reads a LINE, not a claim.** A number is required to ground when its
  line carries measurement vocabulary. "The learning rate was tuned over 100,000 steps" obligates
  `100,000` because the line contains `rate`. Configuration values, hyperparameters and notation
  sitting on such a line will be accused.
* **Bulk arrays are not the claimable surface.** Per-item arrays are excluded on purpose: a
  receipt with a thousand row values covers most 2-decimal numbers in [0,1] by coincidence, so
  matching against them verifies nothing. Persist a SUMMARY field.

CLI:
  python -m styxx.oathready DOC.md receipt1.json [receipt2.json ...] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .certify import _TRIGGERS, _TRIGGERS_CORR, certify_doc

__all__ = ["readiness_report", "render"]

# A receipt path whose terminal segment is a position rather than a measurement. A claim that
# "verifies" against one of these is a coincidence: the v0.8 cycle closed the path-binding
# repair CLOSED_NEGATIVE after five design families failed to beat parity, and v0.11 retracted
# four accusations over the same channel. Reported so an author can see it, never auto-corrected.
_INDEX_NAMES = frozenset({"line", "col", "seed", "token", "case", "i", "index", "idx",
                          "step", "epoch", "num", "count", "size", "id", "level", "rank"})
_SUBSCRIPT = re.compile(r"\[\d+\]$")
_N_EQUALS = re.compile(r"\bn\s*=", re.I)

# Reason codes the shipped verifier writes into `receipt_ref`, mapped to what an author should
# do about them. Every entry names the cycle that established the behaviour.
_ABSTAIN_GUIDANCE = {
    "spec-or-historical": (
        "read as a preregistered bar, a CI level, or a value quoted inside a disclosure note. "
        "Specs have no receipt by design — their receipt is the prereg. Nothing to fix unless "
        "this is actually a measurement, in which case reword so it does not read as a bound."),
    "v05-notation": (
        "read as notation rather than a measurement — a unit-suffixed range, an arXiv id, or an "
        "@-glued parameter. Nothing to fix."),
    "row_ordinal_label": (
        "a markdown table row ordinal under an ordinal header. It has no truth condition, so it "
        "is deliberately silenced (v0.11). Nothing to fix."),
}


def _terminal(path: str) -> str:
    return _SUBSCRIPT.sub("", path.rsplit(".", 1)[-1]).lower()


def _coincident(path: str) -> bool:
    return bool(_SUBSCRIPT.search(path)) or _terminal(path) in _INDEX_NAMES


def _obligating_words(line: str) -> list:
    """The vocabulary that forced this token to ground. An accusation an author cannot trace to
    a word on their own line is one they cannot act on."""
    words = {m.group(0).lower() for m in _TRIGGERS.finditer(line)}
    words |= {m.group(0).lower() for m in _TRIGGERS_CORR.finditer(line)}
    if _N_EQUALS.search(line):
        words.add("n=")
    return sorted(words)


def _classify(entry: dict, line: str) -> dict:
    """One ledger row, turned into something an author can act on."""
    status, ref = entry["status"], entry.get("receipt_ref")
    out = {"line": entry["line"], "token": entry["token"], "status": status,
           "receipt_ref": ref, "context": entry.get("context", "")[:160]}

    if status == "VERIFIED":
        path = (ref or "").partition(":")[2]
        out["kind"] = "coincident" if _coincident(path) else "bound"
        out["advice"] = (
            "grounds at a leaf that is a POSITION, not a measurement — an index, a seed, a step "
            "counter. The value matched by arithmetic accident. Persist the quantity you mean as "
            "its own summary field, or reword the line to name it."
            if out["kind"] == "coincident" else
            "grounds at a receipt leaf whose path relates to this line. This is what a kept "
            "contract looks like.")
        return out

    if status == "UNGROUNDED":
        out["kind"] = "accused"
        words = _obligating_words(line)
        out["obligated_by"] = words
        out["advice"] = (
            "this line carries measurement vocabulary ({}), so every number on it must ground in "
            "a receipt — and this one does not. Three fixes, in order of honesty: persist the "
            "value as a summary field in a cited receipt; or, if the number is a configuration "
            "value, a quotation, or notation rather than a claim, move it off this line or "
            "reword so the vocabulary does not bind it; or, if it is a claim you cannot back, "
            "remove it.".format(", ".join(repr(w) for w in words) or "no word this tool can name"))
        return out

    out["kind"] = "abstained"
    if ref and ref.startswith("ulp-neighbour:"):
        out["advice"] = ("a receipt holds this measurement to within a few ULP — the same number "
                         "reached by differently ordered arithmetic. Quote the receipt value "
                         "verbatim to turn this into a verification.")
    elif ref and ref.startswith("unbound-field:"):
        out["advice"] = ("the value is in your receipts but not in any field this line names. "
                         "Name the quantity on the line, or persist it under a field whose path "
                         "says what it is.")
    elif ref in _ABSTAIN_GUIDANCE:
        out["advice"] = _ABSTAIN_GUIDANCE[ref]
    else:
        out["advice"] = ("nothing on this line names a quantity your receipts carry, so no oath "
                         "is taken either way. If this number IS a claim, name what it measures "
                         "on the same line and persist it as a summary field. Silence here is "
                         "honest, not a pass.")
    return out


def readiness_report(doc_path: Path, receipt_paths: list) -> dict:
    """Per-token readiness for one document against the receipts it cites."""
    cert = certify_doc(Path(doc_path), [Path(p) for p in receipt_paths])
    lines = Path(doc_path).read_text(encoding="utf-8", errors="replace").splitlines()

    rows = []
    for e in cert["ledger"]:
        raw = lines[e["line"] - 1] if e["line"] - 1 < len(lines) else e.get("context", "")
        rows.append(_classify(e, raw))

    bound = [r for r in rows if r["kind"] == "bound"]
    coincident = [r for r in rows if r["kind"] == "coincident"]
    accused = [r for r in rows if r["kind"] == "accused"]
    abstained = [r for r in rows if r["kind"] == "abstained"]

    return {
        "document": Path(doc_path).name,
        "receipts": [Path(p).name for p in receipt_paths],
        "verdict": cert["verdict"],
        "tokens": len(rows),
        "bound": len(bound),
        "coincident": len(coincident),
        "accused": len(accused),
        "abstained": len(abstained),
        "not_a_grade": "These are counts, not a score. This tool cannot tell whether a number is "
                       "CORRECT — only whether it is bound to something a reader could check. A "
                       "document can be fully ready and completely wrong.",
        "known_limits": [
            "Mention is treated as use: a number you QUOTE is treated as one you CLAIM.",
            "The obligation predicate reads a LINE, not a claim — configuration values and "
            "notation sharing a line with measurement vocabulary will be accused.",
            "Per-item bulk arrays are excluded from the claimable surface by design; persist a "
            "SUMMARY field.",
        ],
        "rows": rows,
    }


def render(rep: dict, show: str = "actionable") -> str:
    """Human-readable report. `show`: 'actionable' (accused + coincident) or 'all'."""
    out = []
    out.append(f"OATH readiness — {rep['document']}   [{rep['verdict']}]")
    out.append(f"  receipts: {', '.join(rep['receipts']) or '(none)'}")
    out.append(f"  {rep['tokens']} numeric tokens: {rep['bound']} bound, "
               f"{rep['coincident']} coincident, {rep['accused']} accused, "
               f"{rep['abstained']} abstained")
    out.append("")

    groups = [("ACCUSED — these block a certificate", "accused"),
              ("COINCIDENT — these verify by accident, which is worse than abstaining",
               "coincident")]
    if show == "all":
        groups.append(("ABSTAINED — no oath taken either way", "abstained"))
        groups.append(("BOUND — a kept contract", "bound"))

    for title, kind in groups:
        rows = [r for r in rep["rows"] if r["kind"] == kind]
        if not rows:
            continue
        out.append(title)
        for r in rows:
            out.append(f"  L{r['line']}  {r['token']}")
            out.append(f"      {r['context'][:110]}")
            out.append(f"      -> {r['advice']}")
        out.append("")

    if not rep["accused"] and not rep["coincident"]:
        out.append("No accusations and no coincidental bindings. Every number here either grounds")
        out.append("in a related receipt leaf or is honestly silent.")
        out.append("")
    out.append(rep["not_a_grade"])
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="styxx.oathready",
        description="Is your document ready to carry an OATH certificate?")
    ap.add_argument("doc")
    ap.add_argument("receipts", nargs="*")
    ap.add_argument("--json", default=None, help="also write the full report here")
    ap.add_argument("--all", action="store_true", help="show abstained and bound rows too")
    a = ap.parse_args(argv)

    rep = readiness_report(Path(a.doc), [Path(r) for r in a.receipts])
    print(render(rep, "all" if a.all else "actionable"))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    # Exit non-zero on accusations only. A coincidence is a warning an author should see, not a
    # build break — and abstention is honest behaviour, never a failure.
    return 1 if rep["accused"] else 0


if __name__ == "__main__":
    sys.exit(main())
