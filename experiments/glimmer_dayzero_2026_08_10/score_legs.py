"""Aggregate per-fixture styxx audit JSONs into a leg-level verdict table.

Replaces the runbook's `styxx ci-test --window N` step, which cannot see this
run: ci-test windows chart.jsonl through load_audit's default live_only filter
(LIVE_SOURCES = live/self-report/guardian/None) and `styxx audit` persists with
source="preflight". Scoring directly from the per-fixture receipts is
windowing-proof and keeps every number traceable to a file in the runs/ tree.

    python score_legs.py runs/<legA> [runs/<legB> ...] --out verdict.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_leg(leg_dir: Path) -> dict:
    scores = []
    for p in sorted(leg_dir.glob("*.styxx.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_id"] = p.stem.replace(".styxx", "")
        scores.append(d)
    failures = []
    fpath = leg_dir / "failures.jsonl"
    if fpath.exists():
        failures = [json.loads(l) for l in fpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = {}
    mpath = leg_dir / "run_meta.json"
    if mpath.exists():
        meta = json.loads(mpath.read_text(encoding="utf-8"))
    return {"scores": scores, "failures": failures, "meta": meta}


def category_of(fixture_id: str, fixtures_index: dict) -> str:
    return fixtures_index.get(fixture_id, "uncategorized")


def build_fixture_index(*paths: Path) -> dict:
    idx = {}
    for path in paths:
        if not path or not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            idx[d["id"]] = d.get("category", "uncategorized")
    return idx


def summarize(leg: dict, fixtures_index: dict) -> dict:
    scores = leg["scores"]
    n = len(scores)
    if n == 0:
        return {"n": 0, "note": "no scored fixtures"}
    passed = sum(1 for s in scores if not s.get("needs_revision"))
    composites = [s.get("composite") for s in scores if isinstance(s.get("composite"), (int, float))]
    by_cat = defaultdict(list)
    for s in scores:
        by_cat[category_of(s["_id"], fixtures_index)].append(s)
    per_cat = {}
    for cat, items in sorted(by_cat.items()):
        cat_pass = sum(1 for s in items if not s.get("needs_revision"))
        cat_comp = [s.get("composite") for s in items if isinstance(s.get("composite"), (int, float))]
        per_cat[cat] = {
            "n": len(items),
            "pass_rate": round(cat_pass / len(items), 4),
            "mean_composite": round(sum(cat_comp) / len(cat_comp), 4) if cat_comp else None,
        }
    return {
        "n": n,
        "n_failed_endpoint": len(leg["failures"]),
        "pass_rate": round(passed / n, 4),
        "mean_composite": round(sum(composites) / len(composites), 4) if composites else None,
        "max_composite": round(max(composites), 4) if composites else None,
        "per_category": per_cat,
        "run_meta": {k: leg["meta"].get(k) for k in
                     ("model", "endpoint", "started_iso", "ended_iso", "styxx_version",
                      "chart_appended", "appended_this_run") if k in leg["meta"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("legs", nargs="+", help="run directories to score")
    ap.add_argument("--fixtures", action="append", default=[],
                    help="fixture jsonl(s) supplying id->category mapping")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    idx = build_fixture_index(*[Path(f) for f in args.fixtures])
    result = {}
    for leg_path in args.legs:
        leg_dir = Path(leg_path)
        if not leg_dir.is_dir():
            print(f"skip (not a dir): {leg_dir}", file=sys.stderr)
            continue
        result[leg_dir.name] = summarize(load_leg(leg_dir), idx)

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()
