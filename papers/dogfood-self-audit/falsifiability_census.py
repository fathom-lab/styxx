"""falsifiability_census — how many gates in this suite can actually fail?

⚠️ STATUS RESOLVED 2026-08-13 by `census_discrimination_control.py`. THIS SCREEN
DETECTS SHAPE, NOT DEATH — measured, not assumed. Two gates were built with an
identical PRESENCE_TEST shape and exercised on one population: one matched the prompt
(fires 3/8, varies — ALIVE), one matched the reply (fires 0/8, constant — DEAD, the
memory_integrity defect). **The census flagged both.**

BINDING WORDING RULE, and it governs every downstream artifact:

    A census hit is a CANDIDATE. It may never be written as "dead", "broken",
    "unfalsifiable", or "confirmed" anywhere — not in the ledger, not in a commit
    message, not in a paper. Only PROBE E, run against a real population, may use
    those words. The 20.5% figure is a count of SHAPES, and any sentence that lets a
    reader hear "defects" is the same overstatement this program exists to catch.

That result also settles the circularity below in the screen's disfavour, which is the
honest direction: the anchors were never going to separate the two axes, and now they
do not have to, because the axis question was answered directly.

Original parking note, kept because the reasoning still holds — raised by the author
of the gates it audits:

    "its validation is a pair: flags memory_integrity, clears knowsay.datasheet.
     one of those two anchors is now the module under reconstruction."

The narrow version of that risk is survivable: the OLD memory_integrity's death is an
EMPIRICAL fact -- claims_past TRUE 0/24, recall_supported TRUE 24/24, measured against
real receipts -- and a rebuilt detector coming back alive would not retroactively
resurrect the old one. The anchor for "does the screen flag a gate independently proven
dead" holds.

The sharp version does not, and it is a defect in this screen's validation rather than
in the rebuild. The current pair is:

    positive: dead   AND carries at-risk shapes  (memory_integrity)
    negative: alive  AND carries only safe shapes (knowsay.datasheet, power floors)

Both anchors vary on BOTH axes at once, so the pair cannot separate "the screen detects
the shape" from "the screen detects death." A gate that is ALIVE while carrying a
PRESENCE_TEST or a text LENGTH_TEST is the missing cell, and it is exactly what a
rebuilt memory_integrity would be. Until that cell is filled the screen is only shown
to flag dead-and-risky and clear alive-and-safe, which is consistent with a screen that
measures nothing but shape.

Step 3 of that plan was run directly rather than waiting on the rebuild — a synthetic
alive-and-risky gate answered the same question and answered it faster. Remaining:

  1. memory_integrity rebuilt from source (reimplemented, not imported) — the author's
  2. the rebuild measured against real receipts — alive or dead, empirically
  3. the 143 get PROBE E, one module at a time, against the receipts that exist
  4. UNTESTABLE splits two ways: no receipts and no downstream claim is dead code;
     no receipts UNDER a published number is worse than confirmed-dead

The census screens. PROBE E confirms. Nothing in this file is a verdict, and after the
discrimination control that is a measured statement rather than a modest one.


PROBE E states the law: a gate is unfalsifiable when ANY term of its decision
expression is constant across the population. Stuck TRUE in an OR forces pass; stuck
FALSE in an AND forces silence. Either way the number restates a constant under the
name of the measured property.

Eight instances of that class were found by hand in one day, in three modules and in
both auditors. Hand-finding does not tell you the PREVALENCE. This walks every module
in the package, finds the boolean decision expressions that produce verdicts, and
classifies each term by whether it *structurally can* be constant.

It is a STATIC screen, and it is deliberately over-inclusive: it reports candidates,
not defects. A length comparison is not a bug — `len(recall) > 40` only became one
because it stood in for "the recall supports the claim" and entered a conjunction as
`and not supported`. The census finds the shape; a human or a dynamic PROBE E run
against real receipts decides which shapes are load-bearing.

    python falsifiability_census.py                # census of styxx/
    python falsifiability_census.py --json         # machine-readable
    python falsifiability_census.py --show MODULE  # every candidate in one module
"""
import argparse
import ast
import io
import json
import os
import sys

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "styxx")

# A function is treated as verdict-producing if its name or its returned/assigned
# names carry gate vocabulary. Deliberately broad — a missed gate is worse than an
# extra candidate, and the report separates the two.
GATE_WORDS = ("verdict", "gate", "pass", "fail", "refus", "admissib", "certif",
              "valid", "honest", "fired", "flag", "detect", "caught", "held",
              "survive", "closed", "score_ok", "ok")

# The at-risk set is DELIBERATELY NARROW, and the first version of this census got it
# wrong in exactly the way the census exists to catch. v1 counted CONST_COMPARE and
# FLAG_READ as at-risk and flagged 242 of 276 decision expressions -- 87.7%. A screen
# that fires on seven expressions in eight is not measuring a property, it is
# restating one: `if score > 0.5` is ordinary healthy code, and reading a precomputed
# flag is how every multi-stage instrument is built. That is the same
# non-discriminating shape as a conscience that fires on 94% of turns, committed
# inside the tool written to find it. Recorded rather than quietly narrowed.
#
# What remains are the three shapes where a term SUBSTITUTES for the property it
# claims to test, which is the mechanism in every confirmed instance:
#   LENGTH_TEST   `len(recall) > 40` standing in for "the recall supports the claim"
#   PRESENCE_TEST a regex/lookup matched against the wrong side of the exchange
#   LITERAL       a constant term, unfalsifiable by construction
AT_RISK = ("LENGTH_TEST", "PRESENCE_TEST", "LITERAL")


def _src(node):
    try:
        return ast.unparse(node)
    except Exception:                                        # noqa: BLE001
        return "<unparseable>"


def _producers(fn):
    """name -> the expression last assigned to it, within one function.

    Needed because the defect usually is NOT in the decision line. In the canonical
    case the decision reads `claims_past and not supported and not honest_out` --
    three ordinary flag reads -- while the two broken terms are a regex matched
    against the wrong side of the exchange and a length threshold standing in for a
    semantic property, both computed three lines above. A screen that classifies only
    the decision expression sees a clean conjunction and says nothing.
    """
    out = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = node.value
    return out


def classify_term(node, producers=None, _depth=0):
    """Why this term might be constant across a population.

    `producers` enables one level of substitution: a flag read resolves to the
    expression that computed it, so `supported` is judged as `len(recall) > 40`.
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        # producers MUST be threaded through the negation, or `not supported` stays an
        # opaque flag read and the length test one line up is never seen. This was the
        # bug that made the census miss its own canonical case a second time.
        inner = classify_term(node.operand, producers, _depth)
        return ("NOT " + inner[0], inner[1])
    # unwrap bool(x) / not not x style coercions -- the interesting term is inside
    if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "bool"
            and len(node.args) == 1):
        return classify_term(node.args[0], producers, _depth)
    # a producer that is ITSELF a boolean expression: report its most at-risk term.
    # `supported = bool(recall and len(recall.strip()) > 40)` hides the length test
    # one layer deeper than a single unwrap reaches.
    if isinstance(node, ast.BoolOp):
        best = ("OTHER", "")
        for v in node.values:
            k, w = classify_term(v, producers, _depth + 1)
            if k.replace("NOT ", "") in AT_RISK:
                best = (k, w)
                break
            if best[0] == "OTHER":
                best = (k, w)
        return best
    if isinstance(node, ast.Compare):
        left = node.left
        if isinstance(left, ast.Call) and getattr(left.func, "id", "") == "len":
            # A length test is only suspect when it measures TEXT. Measuring a
            # COLLECTION is a power floor -- `len(caved) >= MIN_CELL` is the correct
            # design, and it is what makes knowsay.datasheet refuse at n=3 and
            # measure at n=1100. The first version of this census flagged that gate,
            # which is the gold-standard refusal in the whole program: a screen that
            # cannot tell a sample-size check from a semantic substitution would have
            # condemned the one instrument proven to vary. Discriminator: string
            # methods / str literals on the operand mean text; a comparison against a
            # declared MIN_*/FLOOR constant means power.
            arg = left.args[0] if left.args else None
            txt = _src(arg) if arg is not None else ""
            is_text = any(m in txt for m in (".strip(", ".lower(", ".upper(",
                                             ".replace(", ".text", "str("))
            thresh = _src(node.comparators[0]) if node.comparators else ""
            is_power = (thresh.isupper() and len(thresh) > 2) or any(
                w in thresh.upper() for w in ("MIN_", "FLOOR", "_N", "SAMPLE"))
            if is_power and not is_text:
                return ("POWER_FLOOR",
                        "sample-size floor on a collection -- the correct use of a "
                        "length test; it is what makes an instrument able to refuse")
            if is_text:
                return ("LENGTH_TEST",
                        "a size threshold on TEXT standing in for a semantic "
                        "property is the G2 shape: it says 'non-empty', not "
                        "'supports the claim'")
            return ("LENGTH_TEST",
                    "length threshold; suspect if it substitutes for a semantic "
                    "property rather than bounding sample size")
        if all(isinstance(c, ast.Constant) for c in node.comparators):
            return ("CONST_COMPARE",
                    "compares against a literal; constant if the left side never "
                    "crosses it in practice")
        return ("COMPARE", "")
    if isinstance(node, ast.Call):
        fn = node.func
        name = getattr(fn, "attr", None) or getattr(fn, "id", "") or ""
        if name in ("get", "search", "match", "fullmatch"):
            return ("PRESENCE_TEST",
                    "regex/lookup presence: constant if the pattern is matched "
                    "against the wrong side of the exchange — the G2 claims_past bug")
        return ("CALL", "")
    if isinstance(node, ast.Name) and producers and _depth == 0:
        # Resolve one level: judge the flag by the expression that COMPUTED it.
        # `supported` looks innocent in the decision line; `len(recall) > 40` does not.
        prod = producers.get(node.id)
        if prod is not None:
            kind, why = classify_term(prod, producers, _depth + 1)
            if kind not in ("FLAG_READ", "OTHER"):
                return (kind, (why + f"  [via {node.id} = {_src(prod)[:60]}]").strip())
    if isinstance(node, ast.Attribute) or isinstance(node, ast.Name):
        return ("FLAG_READ",
                "reads a precomputed flag; constant if the producer is broken")
    if isinstance(node, ast.Constant):
        return ("LITERAL", "a literal term in a decision expression is constant "
                           "BY CONSTRUCTION — the strongest form of the defect")
    return ("OTHER", "")


def gate_functions(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name.lower()
        hit = any(w in name for w in GATE_WORDS)
        if not hit:                     # or it assigns/returns something gate-shaped
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and any(
                        w in sub.id.lower() for w in ("verdict", "gate", "fired")):
                    hit = True
                    break
        # ...or it simply HAS a boolean decision reaching its return value. This is
        # the clause that catches memory_integrity; without it the screen depends on
        # the author having named the thing like a gate.
        if not hit and decision_boolops(node):
            hit = True
        if hit:
            yield node


def _returned_names(fn):
    """Every bare name that flows into this function's return value, including
    through a returned dict/tuple/list. Naming is not consulted."""
    out = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Name):
                out.add(sub.id)
    return out


def decision_boolops(fn):
    """BoolOps in a decision position: a return, an if-test, or an assignment whose
    target REACHES the return value.

    The first version required the assignment target to be gate-NAMED, and that
    produced a false negative on the canonical case: `meta_audit.memory_integrity`
    assigns its verdict to `invented`, which matches no gate vocabulary, inside a
    function whose name matches none either. The screen missed the exact detector the
    law was derived from. Naming conventions were the wrong thing to trust — which is
    the same error, one level up, as trusting a term to mean what it is called.
    Dataflow replaces vocabulary here.
    """
    out = []
    returned = _returned_names(fn)
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.BoolOp):
            out.append(("return", node.value))
        elif isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp):
            out.append(("if", node.test))
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.BoolOp):
            tgt = node.targets[0]
            name = tgt.id if isinstance(tgt, ast.Name) else _src(tgt)
            if (isinstance(tgt, ast.Name) and tgt.id in returned) or \
                    any(w in str(name).lower() for w in GATE_WORDS):
                out.append((f"assign:{name}", node.value))
    return out


def census(pkg_dir):
    rows, files = [], 0
    for root, _dirs, names in os.walk(pkg_dir):
        for n in sorted(names):
            if not n.endswith(".py"):
                continue
            path = os.path.join(root, n)
            try:
                tree = ast.parse(io.open(path, encoding="utf-8").read())
            except (OSError, SyntaxError):
                continue
            files += 1
            mod = os.path.relpath(path, pkg_dir).replace("\\", "/")
            for fn in gate_functions(tree):
                prods = _producers(fn)
                for where, bo in decision_boolops(fn):
                    op = "AND" if isinstance(bo.op, ast.And) else "OR"
                    terms = []
                    for v in bo.values:
                        kind, why = classify_term(v, prods)
                        terms.append({"kind": kind, "why": why,
                                      "src": _src(v)[:110]})
                    risky = [t for t in terms
                             if t["kind"].replace("NOT ", "") in AT_RISK]
                    rows.append({
                        "module": mod, "function": fn.name, "line": bo.lineno,
                        "position": where, "op": op, "n_terms": len(terms),
                        "terms": terms, "n_at_risk": len(risky),
                        # the failure mode differs by operator: a stuck-TRUE term in
                        # an OR forces pass; a stuck-FALSE term in an AND forces silence
                        "forced_outcome_if_stuck": "PASS (never fails)" if op == "OR"
                                                   else "SILENCE (never fires)",
                    })
    return files, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show", default=None, help="print every candidate in a module")
    ap.add_argument("--pkg", default=PKG)
    args = ap.parse_args()

    files, rows = census(args.pkg)
    at_risk = [r for r in rows if r["n_at_risk"]]
    by_mod = {}
    for r in at_risk:
        by_mod.setdefault(r["module"], 0)
        by_mod[r["module"]] += 1

    if args.json:
        print(json.dumps({"files_scanned": files, "decision_expressions": len(rows),
                          "with_at_risk_term": len(at_risk), "rows": rows}, indent=1))
        return
    if args.show:
        for r in rows:
            if r["module"] != args.show:
                continue
            print(f"\n{r['module']}:{r['line']} {r['function']}() [{r['op']}, "
                  f"{r['position']}] -> {r['forced_outcome_if_stuck']}")
            for t in r["terms"]:
                mark = "!" if t["kind"].replace("NOT ", "") in AT_RISK else " "
                print(f"   {mark} {t['kind']:<14} {t['src']}")
        return

    print("=" * 74)
    print("FALSIFIABILITY CENSUS — decision expressions that could carry a")
    print("constant term (PROBE E law). Candidates, not defects.")
    print("=" * 74)
    print(f"  modules scanned          : {files}")
    print(f"  decision expressions     : {len(rows)}")
    print(f"  with >=1 at-risk term    : {len(at_risk)}"
          f"  ({100*len(at_risk)/len(rows):.1f}%)" if rows else "")
    ands = [r for r in at_risk if r["op"] == "AND"]
    ors = [r for r in at_risk if r["op"] == "OR"]
    print(f"    AND (stuck FALSE -> silence, the G2 shape) : {len(ands)}")
    print(f"    OR  (stuck TRUE  -> pass,    the G3 shape) : {len(ors)}")
    kinds = {}
    for r in at_risk:
        for t in r["terms"]:
            k = t["kind"].replace("NOT ", "")
            if k in AT_RISK:
                kinds[k] = kinds.get(k, 0) + 1
    print("\n  at-risk term kinds:")
    for k, c in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"    {k:<15} {c}")
    print("\n  top modules by at-risk decision expressions:")
    for m, c in sorted(by_mod.items(), key=lambda x: -x[1])[:12]:
        print(f"    {c:>3}  {m}")
    print("\n  NOTE: static screen, over-inclusive by design. A candidate becomes a")
    print("  defect only when a term is shown constant across a real population —")
    print("  run PROBE E (probe_term_variance) against that module's receipts.")


if __name__ == "__main__":
    sys.exit(main())
