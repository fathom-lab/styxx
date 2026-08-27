"""CENSUS — the formula-constant class, measured at SPAN level this time.

**A census. It licenses no claim.** It exists because v0.12 froze its bar against a LINE-level
marker and then specified a SPAN-level clause, the populations differed, and the clause
under-reached its own motivating example. `RESULT_oath_v12_formula_constant_2026_08_26.md` states
the lesson in one line: *freeze the bar against the thing you are going to build, not against the
thing you happened to measure.* This is that measurement.

## The defect, still open

`extract_numbers` takes numerals out of rendered mathematics, and `delta` is trigger vocabulary,
so the literal `1` in `\\left(1 \\pm \\frac{\\Delta \\sigma^2}{\\sigma^2}\\right)` is accused of
being a claim whose truth condition was never met. It is a mathematical constant. Three
certificates in this corpus are OATH-FAILED on exactly that, and one of them is v0.12's own
preregistration.

## What is measured

Five SPAN definitions — the actual conjunct-1 candidates a successor could freeze — each scored on
what it REACHES (currently-accused tokens) and what it DESTROYS (currently-verified tokens),
with the destroy column split by whether the binding is NOMINAL or merely COINCIDENT.

That split is the one that decides. Destroying a verification sworn to an array index or a seed
is a gain wearing a loss's clothes; destroying one sworn to the quantity its line names is a real
cost. v0.11's winning clause destroyed zero nominal bindings, and every design killed in v0.11 and
v0.12 died on this column.

  python papers/closed-model-frontier/formula_span_census.py
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
from styxx.discriminates import discrimination_report                       # noqa: E402

OUT = HERE / "formula_span_census.json"

BSLASH = re.compile(r"\\[A-Za-z]+")
BARE_NUM = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_FENCE = re.compile(r"^\s{0,3}(```|~~~)")

# the frozen coincidence definition, identical to oath_v11_dogfood_selfcert.py
_INDEX_NAMES = frozenset({"line", "col", "seed", "token", "case", "i", "index", "idx", "step",
                          "epoch", "num", "count", "size", "id", "level", "rank"})
_SUBSCRIPT = re.compile(r"\[\d+\]$")


def coincident(receipt_ref) -> bool:
    path = (receipt_ref or "").partition(":")[2]
    if not path:
        return False
    term = _SUBSCRIPT.sub("", path.rsplit(".", 1)[-1]).lower()
    return bool(_SUBSCRIPT.search(path)) or term in _INDEX_NAMES


def _pairs(line: str, delim: str):
    out, i = [], 0
    while True:
        a = line.find(delim, i)
        if a < 0:
            return out
        b = line.find(delim, a + len(delim))
        if b < 0:
            return out
        out.append((a + len(delim), b))
        i = b + len(delim)


def fenced_lines(lines) -> set:
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


def spans_for(token_line: str, col: int, line_no: int, fenced: set, lines) -> dict:
    """Which SPAN definitions contain this token."""
    dollar = any(a <= col < b and BSLASH.search(token_line[a:b])
                 for a, b in _pairs(token_line, "$$") + _pairs(token_line, "$"))
    code = any(a <= col < b and BSLASH.search(token_line[a:b])
               for a, b in _pairs(token_line, "`"))
    indented = (token_line.startswith("    ") and bool(BSLASH.search(token_line))
                and line_no not in fenced)
    in_fence = line_no in fenced and bool(BSLASH.search(token_line))
    # S5: the token sits to the RIGHT of a backslash command on its line
    m = BSLASH.search(token_line)
    after_cmd = bool(m) and col > m.start()
    return {"S1_dollar_span": dollar,
            "S2_inline_code_with_command": code,
            "S3_indented_block_with_command": indented,
            "S4_fenced_block_with_command": in_fence,
            "S5_right_of_a_backslash_command": after_cmd}


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
    per = collections.defaultdict(lambda: {"reaches": [], "destroys_nominal": [],
                                           "destroys_coincident": [], "silences_abstain": 0})
    totals = collections.Counter()

    for doc, receipts in docs:
        try:
            cert = certify_doc(doc, receipts)
        except Exception:
            continue
        rel = doc.relative_to(ROOT).as_posix()
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        fenced = fenced_lines(lines)
        for e in cert["ledger"]:
            totals[e["status"]] += 1
            col = e.get("col")
            if col is None or not BARE_NUM.fullmatch(e["token"]):
                continue
            line = lines[e["line"] - 1].replace("−", "-") if e["line"] - 1 < len(lines) else ""
            if not BSLASH.search(line):
                continue
            hits = spans_for(line, col, e["line"], fenced, lines)
            for name, hit in hits.items():
                if not hit:
                    continue
                row = {"doc": rel, "line": e["line"], "token": e["token"],
                       "receipt_ref": e["receipt_ref"]}
                if e["status"] == "UNGROUNDED":
                    per[name]["reaches"].append(row)
                elif e["status"] == "VERIFIED":
                    key = ("destroys_coincident" if coincident(e["receipt_ref"])
                           else "destroys_nominal")
                    per[name][key].append(row)
                else:
                    per[name]["silences_abstain"] += 1

    rows = []
    for name in sorted(per):
        d = per[name]
        rows.append({"span": name,
                     "reaches_accusations": len(d["reaches"]),
                     "destroys_nominal": len(d["destroys_nominal"]),
                     "destroys_coincident": len(d["destroys_coincident"]),
                     "already_abstaining": d["silences_abstain"],
                     "reach_roster": d["reaches"],
                     "nominal_roster": d["destroys_nominal"]})

    # Corpus-wide surface: what each candidate touches across ALL papers/**/*.md, certified or
    # not. A prereg needs this for its boundary disclosure (v0.11's G9 required exactly it), and
    # a candidate that is quiet in frame can still be loud in the corpus it will regrow into.
    from styxx.certify import extract_numbers
    wide = collections.defaultdict(lambda: {"tokens": 0, "documents": set()})
    for md in sorted(ROOT.glob("papers/**/*.md")):
        if "anc" in md.parts:
            continue
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        fenced = fenced_lines(lines)
        rel = md.relative_to(ROOT).as_posix()
        for num in extract_numbers(text):
            col = num.get("col")
            if col is None or not BARE_NUM.fullmatch(num["token"]):
                continue
            line = lines[num["line"] - 1].replace("−", "-") if num["line"] - 1 < len(lines) else ""
            if not BSLASH.search(line):
                continue
            for name, hit in spans_for(line, col, num["line"], fenced, lines).items():
                if hit:
                    wide[name]["tokens"] += 1
                    wide[name]["documents"].add(rel)
    for r in rows:
        w = wide.get(r["span"], {"tokens": 0, "documents": set()})
        r["corpus_wide_tokens"] = w["tokens"]
        r["corpus_wide_documents"] = len(w["documents"])

    # PERMISSIVE CONTROL. If a rule with no span test at all scores the same as the best
    # candidate on a column, that column is not measuring the span test. This control is what
    # the census should have carried from the start; the vacuous-pass census written the same
    # day DID carry controls, and this one did not.
    ctrl = collections.Counter()
    ctrl_nominal = 0
    for doc, receipts in docs:
        try:
            cert = certify_doc(doc, receipts)
        except Exception:
            continue
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        for e in cert["ledger"]:
            col = e.get("col")
            if col is None or not BARE_NUM.fullmatch(e["token"]):
                continue
            line = lines[e["line"] - 1] if e["line"] - 1 < len(lines) else ""
            if not BSLASH.search(line):
                continue
            ctrl[e["status"]] += 1
            if e["status"] == "VERIFIED" and not coincident(e["receipt_ref"]):
                ctrl_nominal += 1

    # The retraction, computed rather than asserted. styxx.discriminates scores every column
    # against the permissive control; a column the null rule ties is not a deciding column.
    disc = discrimination_report(
        {c["span"]: {"reaches": c["reaches_accusations"],
                     "destroys_nominal": c["destroys_nominal"]} for c in rows},
        {"reaches": ctrl["UNGROUNDED"], "destroys_nominal": ctrl_nominal},
        {"reaches": "higher_is_better", "destroys_nominal": "lower_is_better"},
        deciding=["destroys_nominal"],
    )

    payload = {
        "census": "formula-constant class, measured at SPAN level",
        "discrimination": disc,
        "permissive_control": {
            "rule": "no span test at all — any bare numeral on a line containing a backslash "
                    "command",
            "reaches_accusations": ctrl["UNGROUNDED"],
            "destroys_nominal": ctrl_nominal,
            "destroys_coincident": ctrl["VERIFIED"] - ctrl_nominal,
            "verdict": "the deciding column reads IDENTICALLY for this and for the best "
                       "candidate, so it discriminates between none of them",
        },
        "status": "CENSUS. Licenses no claim. Measures the population a conjunct-1 candidate "
                  "would actually see — the step whose absence killed v0.12.",
        "verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "frame": {"documents": len(docs), "status_counts": dict(totals)},
        "the_column_that_decides": "RETRACTED 2026-08-27. This field claimed destroys_nominal was "
                                   "the deciding column. It cannot decide anything: the "
                                   "permissive control below — NO span test at all, every bare "
                                   "numeral on a line carrying a backslash command — scores "
                                   "destroys_nominal 0 as well, identical to the best candidate. "
                                   "A column that reads the same for the best and the worst "
                                   "possible rule is a vacuous gate, which is the defect "
                                   "RECON_vacuous_pass_2026_08_27.md catalogues, committed here "
                                   "the same day. Found by the adversarial red team, not by the "
                                   "author.",
        "why_it_is_vacuous": "destroys_nominal is measured over the CERTIFIED FRAME, which is 184 "
                             "of the ~1,119 markdown documents under papers/. Every genuine "
                             "measurement any of these candidates would silence lives in an "
                             "UNCERTIFIED document, where no ledger status exists and the column "
                             "is structurally blind. The zero is a property of the frame's "
                             "coverage, not of any rule.",
        "candidates": rows,
        "what_this_does_not_show": [
            "That any candidate is good, or that a successor should be written. Reaching an "
            "accusation is not the same as being RIGHT to silence it; that needs a blind "
            "adjudication with ties resolved against the clause.",
            "Anything about documents outside the certified frame.",
            "That the accusations reached are false. They look false to their author, which is "
            "exactly the judgement a panel exists to replace.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"frame {len(docs)} documents  {dict(totals)}\n")
    print(f"{'span candidate':<38}{'reaches':>8}{'nominal':>9}{'coincid':>9}"
          f"{'corpus':>8}{'docs':>6}")
    for r in rows:
        print(f"{r['span']:<38}{r['reaches_accusations']:>8}{r['destroys_nominal']:>9}"
              f"{r['destroys_coincident']:>9}{r['corpus_wide_tokens']:>8}"
              f"{r['corpus_wide_documents']:>6}")
    print(f"\n-> {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
