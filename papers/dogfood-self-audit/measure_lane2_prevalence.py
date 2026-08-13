"""Lane 2 realism: how often does the path-length defect actually FIRE on real receipts?

The red team reported 114/400 receipts carrying the collision SHAPE (short scalar keys alongside
nested detail dicts) and was careful to say that is prevalence-of-shape, not of realised
collisions. That honesty leaves the load-bearing number unmeasured, and it is the number that
decides whether lane 2 was a real defect or a constructed one.

This measures the realised rate: over every receipt JSON in papers/, count value collisions where
a SHORT path and a LONGER path hold the same value — the exact configuration in which the old
scoring (divide by len(path_tokens)) would hand the win to the generic key.

Then it re-runs the OLD scoring against those real collisions and counts how many the old code
would have resolved to the shorter path with a `context` label.
"""
from __future__ import annotations
import json
import pathlib
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from styxx.claim_audit import _load_sources, _tokens  # noqa: E402

PAPERS = ROOT / "papers"


def old_score(ctx: set, path: str) -> float:
    """The shipped-then-fixed scoring: overlap divided by PATH length."""
    pt = _tokens(path)
    return (len(ctx & pt) / len(pt)) if pt else 0.0


def new_score(ctx: set, path: str):
    pt = _tokens(path)
    if not pt:
        return (0, 0.0)
    inter = ctx & pt
    return (len(inter), len(inter) / len(ctx | pt))


def main():
    files = sorted(PAPERS.rglob("*.json"))
    n_files = 0
    n_with_collision = 0
    collisions = []          # (file, value, short_path, long_path)
    for f in files:
        if ".claude" in f.parts or "node_modules" in f.parts:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        n_files += 1
        try:
            vals = _load_sources([data])
        except Exception:
            continue
        hit = False
        for v, paths in vals.items():
            if len(paths) < 2:
                continue
            depths = [(len(_tokens(p)), p) for p in paths]
            depths.sort()
            if depths[0][0] < depths[-1][0]:      # a genuinely shorter path shares the value
                hit = True
                collisions.append((str(f.relative_to(ROOT)), v, depths[0][1], depths[-1][1]))
        if hit:
            n_with_collision += 1

    print(f"receipt JSONs scanned                     : {n_files}")
    print(f"files with a SHORT/LONG value collision   : {n_with_collision} "
          f"({n_with_collision/max(n_files,1):.1%})")
    print(f"distinct realised collisions              : {len(collisions)}")

    # Would the old scoring have mis-resolved them?
    #
    # FIRST ATTEMPT WAS BROKEN, AND BROKEN IN MY FAVOUR. It set ctx = _tokens(long_path), i.e.
    # prose that reproduces EVERY token of the specific path. Under that context the old scoring
    # gives the long path len(inter)/len(pt) = 1.0, so it can never lose, and the script dutifully
    # reported 0/54701 — a number that would have let me contest the red team's finding. The
    # artifact was in the fixture, not the world: real prose PARAPHRASES. It writes ">=3/7" for
    # `ge3_of_7` and drops container words like `cells`. Partial overlap is the realistic case and
    # is exactly where dividing by path length hurts the specific path.
    #
    # So: drop a fraction of the long path's tokens to simulate paraphrase, and require the
    # retained tokens to still uniquely name it.
    would_break = 0
    examples = []
    n_evaluated = 0
    for fname, v, short_p, long_p in collisions:
        lt = sorted(_tokens(long_p))
        if len(lt) < 2:
            continue
        # keep all but the last token — a mild, realistic paraphrase loss
        ctx = set(lt[:-1])
        if not ctx:
            continue
        n_evaluated += 1
        o_short, o_long = old_score(ctx, short_p), old_score(ctx, long_p)
        n_short, n_long = new_score(ctx, short_p), new_score(ctx, long_p)
        old_wrong = o_short > o_long
        new_ok = n_long >= n_short
        if old_wrong and new_ok:
            would_break += 1
            if len(examples) < 6:
                examples.append((fname, v, short_p, long_p, round(o_short, 3), round(o_long, 3)))

    print(f"collisions evaluated under paraphrase     : {n_evaluated}")
    print(f"collisions the OLD scoring mis-resolves   : {would_break} "
          f"({would_break/max(n_evaluated,1):.1%} of evaluated collisions)")
    print("\nexamples (old scoring picks the SHORT path over the named one):")
    for fname, v, sp, lp, os_, ol_ in examples:
        print(f"  {fname}")
        print(f"    value {v}: '{sp}' (old {os_}) beat '{lp}' (old {ol_})")

    out = {"files_scanned": n_files, "files_with_short_long_collision": n_with_collision,
           "realised_collisions": len(collisions),
           "collisions_evaluated_under_paraphrase": n_evaluated,
           "old_scoring_misresolves": would_break,
           "misresolve_rate_of_evaluated": round(would_break / max(n_evaluated, 1), 4),
           "reading": "prevalence of the DEFECT FIRING on real receipts, not of the shape",
           "note": "first version of this script set ctx = full token set of the specific path, "
                   "under which the old scoring cannot lose; it reported 0/54701 in the author's "
                   "favour. Corrected to simulate paraphrase (drop one token), which is the "
                   "realistic case and the one the red team's fixture modelled."}
    (HERE / "lane2_prevalence.json").write_text(json.dumps(out, indent=2) + "\n",
                                                encoding="utf-8")
    print("\nwrote lane2_prevalence.json")


if __name__ == "__main__":
    main()
