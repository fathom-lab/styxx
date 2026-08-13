"""Do styxx's OTHER rate-reporting instruments state their chance floor?

The resolution probe generalised today's defect class. Before pointing it at anyone else's tools,
point it at the rest of my own: `claim_audit` was not special, it was just the one I happened to
dogfood. Any function that returns a rate, fraction, or score without a null is exposed to the
same failure — a number that cannot fail, reported as if it could.

METHOD (static, conservative, and it says what it cannot see):
For every public callable in `styxx/*.py` whose name or return annotation suggests it produces a
rate/score/fraction, ask two questions of its source and docstring:

  Q1 does it compute or accept a NULL / baseline / chance level? (permutation, surrogate,
     shuffle, bootstrap, random baseline, floor, prior, expected-by-chance)
  Q2 does it EXPOSE that to the caller, or only use it internally?

An instrument that computes a null but hides it is better than one with no null, and worse than
one that returns it. Three buckets: DISCLOSES / INTERNAL_ONLY / NO_NULL.

LIMITS, stated up front: this is a lexical survey, not a semantic proof. It can mark a function
NO_NULL that in fact receives a null from its caller, and it can mark one DISCLOSES on the
strength of a variable name. It is a triage list for manual review, and the counts below should
be read as "candidates", never as "defects". Every hit is printed with its file and line so the
claim is checkable rather than trusted.
"""
from __future__ import annotations
import ast
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PKG = ROOT / "styxx"

RATE_NAME = re.compile(
    r"(rate|score|frac|fraction|pct|percent|accuracy|auc|auroc|ratio|prob|coverage|"
    r"precision|recall|sensitivity|specificity)", re.I)

# Statistics with a KNOWN ANALYTIC NULL do not need a measured floor: AUC/AUROC is 0.5 under
# the null by construction, and every reader of an AUC knows it. Flagging them as "no null" was
# a false positive of the first version of this survey — caught by reading four flagged
# functions by hand before quoting the headline. A survey that inflates its own finding is the
# same defect this whole line of work is about, so the exemption is explicit and named.
ANALYTIC_NULL = re.compile(r"^(auc|auroc|compute_auc)$", re.I)

NULL_HINT = re.compile(
    r"(null|chance|baseline|surrogate|permut|shuffl|bootstrap|random|floor|prior|"
    r"expected_by_chance|excess|by_luck)", re.I)

EXPOSE_HINT = re.compile(
    r"(return|self\.\w*(null|chance|baseline|floor|excess)|['\"]\w*(null|chance|baseline|floor|"
    r"excess)\w*['\"])", re.I)


def analyse(path: pathlib.Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    src_lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        if not RATE_NAME.search(node.name):
            continue
        seg = "\n".join(src_lines[node.lineno - 1: getattr(node, "end_lineno", node.lineno)])
        doc = ast.get_docstring(node) or ""
        has_null = bool(NULL_HINT.search(seg) or NULL_HINT.search(doc))
        exposes = bool(has_null and NULL_HINT.search(
            "\n".join(l for l in seg.splitlines()
                      if l.strip().startswith("return") or '":' in l or "':" in l)))
        if ANALYTIC_NULL.match(node.name):
            bucket = "ANALYTIC_NULL"
        elif exposes:
            bucket = "DISCLOSES"
        elif has_null:
            bucket = "INTERNAL_ONLY"
        else:
            bucket = "NO_NULL"
        out.append({"file": path.name, "func": node.name, "line": node.lineno,
                    "bucket": bucket})
    return out


def main():
    rows = []
    for p in sorted(PKG.glob("*.py")):
        rows.extend(analyse(p))
    buckets = {"DISCLOSES": [], "INTERNAL_ONLY": [], "NO_NULL": [], "ANALYTIC_NULL": []}
    for r in rows:
        buckets[r["bucket"]].append(r)

    print("=" * 78)
    print("SURVEY — do styxx's own rate-reporting functions state a chance level?")
    print("=" * 78)
    n = len(rows)
    for b in ("DISCLOSES", "ANALYTIC_NULL", "INTERNAL_ONLY", "NO_NULL"):
        print(f"  {b:14} {len(buckets[b]):4}  ({len(buckets[b])/max(n,1):.1%})")
    print(f"  {'TOTAL':14} {n:4}")
    print("\n  ANALYTIC_NULL = AUC/AUROC, null 0.5 by construction — exempt, not a defect.")
    print("  (Exemption added after reading four flagged functions by hand; the first run")
    print("   of this survey counted them as NO_NULL and overstated its own finding.)")

    print("\n  candidates with NO null of any kind (triage list, first 25):")
    for r in buckets["NO_NULL"][:25]:
        print(f"    {r['file']}:{r['line']:<5} {r['func']}")

    print("\n  computes a null but does not appear to return it (first 12):")
    for r in buckets["INTERNAL_ONLY"][:12]:
        print(f"    {r['file']}:{r['line']:<5} {r['func']}")

    out = {"total_rate_functions": n,
           "counts": {k: len(v) for k, v in buckets.items()},
           "no_null": buckets["NO_NULL"],
           "internal_only": buckets["INTERNAL_ONLY"],
           "method": "lexical/AST triage, NOT a semantic proof; counts are candidates for "
                     "manual review, never defects",
           "reading": "claim_audit was not special — it was the one that got dogfooded"}
    (HERE / "own_instrument_survey.json").write_text(json.dumps(out, indent=2) + "\n",
                                                     encoding="utf-8")
    print("\nwrote own_instrument_survey.json")
    print("\nNOTE: lexical triage. A NO_NULL hit is a QUESTION to answer by reading the")
    print("function, not a defect. The honest headline is the shape of the distribution,")
    print("not any individual row.")


if __name__ == "__main__":
    main()
