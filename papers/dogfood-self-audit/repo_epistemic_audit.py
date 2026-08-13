"""repo_epistemic_audit — can this repository prove what it says?

Every CI system answers "do the tests pass." None of them answer the question that
decides whether a research repository's claims are worth reading:

    for each number this repo asserts in prose, does a receipt contain it --
    and could the instrument that produced that receipt have failed?

Those are two different failures and both are silent. A number with no receipt is
unsupported. A number with a receipt produced by a gate that could not fail is worse,
because it comes with evidence attached and the evidence is a constant.

This composes the two halves that already exist in this repo, which is the point --
neither half alone answers it:

    styxx.claim_audit.audit_grounding   is the number in a receipt?     (grounding)
    falsifiability_census               could the gate behind it fail?  (shape)

and reports a per-document profile plus a repo verdict.

WORDING RULE INHERITED, and it binds here:
`PRECOMMIT_ledger_rules_2026_08_13.md` establishes that a census hit is a CANDIDATE
and may never be called dead. This tool therefore reports SHAPE_RISK, never DEAD, on
the instrument axis. Only PROBE E against a real population may use that word.

    python repo_epistemic_audit.py --repo C:/path/to/repo --docs papers --receipts papers
    python repo_epistemic_audit.py --selftest
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(HERE, "falsifiability_census.py")

# A "claim number" is a decimal that reads like a measurement, not a version, date,
# port, or list index. Deliberately conservative: a missed claim understates the
# problem, an invented one manufactures it.
NUM_RE = re.compile(r"(?<![\w./-])(\d+\.\d+|\d{1,3}%)(?![\w./-])")
SKIP_CONTEXT = re.compile(r"\b(?:version|v\d|python|port|line|fig(?:ure)?|"
                          r"section|chapter|http|©)\b", re.I)

# The first version counted SECTION NUMBERS and arXiv IDs as claims. Its "least
# grounded" documents were simply the ones with the most numbered headings -- 2.1,
# 2.2, 2.3 -- and the repo-wide 0.807 was a ratio over a polluted denominator, which
# makes it a number about markdown structure wearing the name of a grounding rate.
# Caught by reading the unmatched list instead of the headline. These exclusions are
# structural, not tuned to make the figure look better: each one removes a token that
# is definitionally not a measurement.
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s|^\s{0,3}\*{0,2}\d+(\.\d+)*\.?\s")
ARXIV_RE = re.compile(r"\b\d{4}\.\d{4,5}\b")          # 2406.15927
SECTIONISH_RE = re.compile(r"^\d{1,2}\.\d{1,2}$")      # 2.1, 4.7 -- ordinals, not data
LIST_ORDINAL_RE = re.compile(r"(?:^|\s)(?:§|sec\.?|step|item|table|appendix)\s*$", re.I)


def find_docs(root, subdir, limit=40):
    base = os.path.join(root, subdir) if subdir else root
    out = []
    for dirpath, dirnames, names in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in
                       (".git", "node_modules", "__pycache__", ".claude")]
        for n in names:
            if n.lower().endswith(".md"):
                out.append(os.path.join(dirpath, n))
            if len(out) >= limit:
                return out
    return out


def find_receipts(root, subdir, limit=400):
    base = os.path.join(root, subdir) if subdir else root
    out = []
    for dirpath, dirnames, names in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in
                       (".git", "node_modules", "__pycache__", ".claude")]
        for n in names:
            if n.lower().endswith(".json"):
                out.append(os.path.join(dirpath, n))
            if len(out) >= limit:
                return out
    return out


def receipt_values(paths, cap=200):
    """Every numeric leaf across the receipt corpus, as a set of rounded strings."""
    vals = set()
    for p in paths[:cap]:
        try:
            d = json.load(io.open(p, encoding="utf-8-sig"))
        except Exception:                                    # noqa: BLE001
            continue

        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                for dp in (2, 3, 4):
                    vals.add(f"{round(float(o), dp):.{dp}f}".rstrip("0").rstrip("."))
        walk(d)
    return vals


def doc_claims(path):
    out = []
    try:
        text = io.open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return out
    for line in text.splitlines():
        if SKIP_CONTEXT.search(line) or HEADING_RE.match(line):
            continue                      # headings carry ordinals, not measurements
        line_wo_arxiv = ARXIV_RE.sub(" ", line)
        for m in NUM_RE.finditer(line_wo_arxiv):
            raw = m.group(1)
            if SECTIONISH_RE.match(raw):
                # 2.1 / 4.7 are almost always cross-references in this corpus. A real
                # measurement in that range (a ratio, a fold-change) is lost here --
                # an UNDER-count, which is the safe direction for a screen whose
                # failure mode is manufacturing ungrounded claims.
                continue
            before = line_wo_arxiv[:m.start()]
            if LIST_ORDINAL_RE.search(before):
                continue
            out.append({"raw": raw, "line": line.strip()[:120]})
    return out


def grounded(raw, vals):
    """One-way containment against the receipt corpus, at the claim's own precision."""
    s = raw.rstrip("%")
    try:
        f = float(s)
    except ValueError:
        return False
    for dp in (2, 3, 4):
        if f"{round(f, dp):.{dp}f}".rstrip("0").rstrip(".") in vals:
            return True
    return False


def chance_floor(vals, claims, trials=4000, seed=7):
    """P(a number drawn from the claims' own band grounds against `vals` by luck).

    Without this the grounding rate is uninterpretable, and the first run of this tool
    proved it: 958 of 960 prose numbers "grounded" against a corpus of 3,797 receipt
    leaves, which is what near-certain-by-chance looks like wearing the name of a
    result. That is the identical defect the red team found in `claim_audit` twelve
    hours earlier -- a rate quoted with no reference distribution -- reproduced here by
    the party that found it.

    Band-matched, per the correction that fixed it there: draw from the range the
    document's OWN claims occupy, not a fixed [0,1]. Sampling a band the claims do not
    live in understates the floor in the flattering direction.
    """
    import random as _r
    nums = []
    for c in claims:
        try:
            nums.append(float(c["raw"].rstrip("%")))
        except ValueError:
            continue
    if not nums or not vals:
        return None
    lo, hi = min(nums), max(nums)
    if hi <= lo:
        hi = lo + 1.0
    rng = _r.Random(seed)
    hits = 0
    for _ in range(trials):
        q = rng.uniform(lo, hi)
        if grounded(f"{q:.4f}", vals) or grounded(f"{q:.3f}", vals) \
                or grounded(f"{q:.2f}", vals):
            hits += 1
    return round(hits / trials, 4)


def census_shape_risk(pkg_dir):
    """How many decision expressions in the code carry an at-risk shape."""
    try:
        r = subprocess.run([sys.executable, CENSUS, "--pkg", pkg_dir, "--json"],
                           capture_output=True, text=True, timeout=600)
        d = json.loads(r.stdout or "{}")
        return {"decision_expressions": d.get("decision_expressions"),
                "with_at_risk_term": d.get("with_at_risk_term")}
    except Exception as e:                                   # noqa: BLE001
        return {"error": str(e)[:80]}


def audit(repo, docs_dir, receipts_dir, pkg_dir):
    docs = find_docs(repo, docs_dir)
    receipts = find_receipts(repo, receipts_dir)
    vals = receipt_values(receipts)
    rows = []
    for d in docs:
        claims = doc_claims(d)
        if not claims:
            continue
        ok = [c for c in claims if grounded(c["raw"], vals)]
        rows.append({
            "doc": os.path.relpath(d, repo).replace("\\", "/"),
            "n_claims": len(claims), "n_grounded": len(ok),
            "grounding_rate": round(len(ok) / len(claims), 3) if claims else None,
            "ungrounded_examples": [c["raw"] for c in claims
                                    if not grounded(c["raw"], vals)][:5],
        })
    rows.sort(key=lambda r: (r["grounding_rate"] or 0))
    shape = census_shape_risk(pkg_dir) if pkg_dir else {}
    tot_c = sum(r["n_claims"] for r in rows)
    tot_g = sum(r["n_grounded"] for r in rows)
    rate = round(tot_g / tot_c, 3) if tot_c else None
    all_claims = [c for d in docs for c in doc_claims(d)]
    floor = chance_floor(vals, all_claims)
    excess = round(rate - floor, 3) if (rate is not None and floor is not None) else None
    return {
        "repo": repo, "n_docs_with_claims": len(rows),
        "n_receipts_scanned": len(receipts),
        "n_receipt_values": len(vals),
        "total_claims": tot_c, "total_grounded": tot_g,
        "repo_grounding_rate": rate,
        "chance_floor": floor,
        "excess_over_chance": excess,
        "verdict": (None if excess is None else
                    "GROUNDING_INDISTINGUISHABLE_FROM_CHANCE" if excess <= 0.05 else
                    "GROUNDED_ABOVE_CHANCE"),
        "instrument_shape_risk": shape,
        "docs": rows,
        "reading": (
            "Quote EXCESS_OVER_CHANCE, never the bare grounding rate. The rate alone "
            "is uninterpretable: against a corpus this size almost any number grounds "
            "by luck, and the first run of this tool reported 0.998 with no floor, "
            "which is near-certainty wearing the name of a result. KNOWN LIMIT of the "
            "floor: it is band-matched over ALL claims pooled, and claims are not "
            "uniform within that band, so it is a LOWER bound on chance and the excess "
            "is an UPPER bound. A per-document, per-magnitude-class floor would be "
            "tighter. Grounding is also not accuracy -- a number can appear in a "
            "receipt produced by an instrument that could not fail, which is what the "
            "shape axis is for. instrument_shape_risk counts SHAPES, never defects; "
            "only PROBE E may say dead."),
    }


def selftest():
    """Two documents, one honest and one fabricated, against one receipt."""
    import tempfile
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "papers"))
    io.open(os.path.join(root, "papers", "r.json"), "w", encoding="utf-8").write(
        json.dumps({"cave_rate": 0.7791, "rescue": 0.0566, "n": 1100}))
    io.open(os.path.join(root, "papers", "honest.md"), "w", encoding="utf-8").write(
        "The cave rate was 0.7791 with rescue 0.0566.\n")
    io.open(os.path.join(root, "papers", "fabricated.md"), "w", encoding="utf-8").write(
        "The cave rate was 0.9312 with rescue 0.4410.\n")
    res = audit(root, "papers", "papers", None)
    by = {r["doc"]: r for r in res["docs"]}
    h = by.get("papers/honest.md", {}).get("grounding_rate")
    f = by.get("papers/fabricated.md", {}).get("grounding_rate")
    print(f"  honest.md      grounding {h}   (expect 1.0)")
    print(f"  fabricated.md  grounding {f}   (expect 0.0)")
    ok = (h == 1.0 and f == 0.0)
    print(f"\n  VALIDATION: {'PASS — separates receipted prose from fabricated prose' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--repo")
    ap.add_argument("--docs", default="papers")
    ap.add_argument("--receipts", default="papers")
    ap.add_argument("--pkg")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.repo:
        ap.error("--repo or --selftest")
    res = audit(args.repo, args.docs, args.receipts, args.pkg)
    if args.json:
        print(json.dumps(res, indent=1))
        return 0
    print("=" * 74)
    print("REPO EPISTEMIC AUDIT — can this repository prove what it says?")
    print("=" * 74)
    print(f"  receipts scanned        : {res['n_receipts_scanned']} "
          f"({res['n_receipt_values']} distinct numeric leaves)")
    print(f"  docs carrying claims    : {res['n_docs_with_claims']}")
    print(f"  prose numbers           : {res['total_claims']}")
    print(f"  found in some receipt   : {res['total_grounded']} "
          f"({res['repo_grounding_rate']})")
    print(f"  chance floor            : {res['chance_floor']}  "
          f"(band-matched, same seed rule as claim_audit)")
    print(f"  EXCESS OVER CHANCE      : {res['excess_over_chance']}   "
          f"-> {res['verdict']}")
    if res["instrument_shape_risk"]:
        s = res["instrument_shape_risk"]
        print(f"  instrument shape risk   : {s.get('with_at_risk_term')} of "
              f"{s.get('decision_expressions')} decision expressions "
              f"(SHAPES, not defects)")
    print("\n  least-grounded documents:")
    for r in res["docs"][:8]:
        print(f"    {r['grounding_rate']:<6} {r['n_grounded']:>3}/{r['n_claims']:<4} "
              f"{r['doc'][:52]}")
        if r["ungrounded_examples"]:
            print(f"           unmatched: {', '.join(r['ungrounded_examples'])}")
    print(f"\n  {res['reading']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
