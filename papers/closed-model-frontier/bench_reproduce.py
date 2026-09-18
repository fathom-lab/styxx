"""Re-fetch, verify and re-score the styxx PR-claim dataset. Nothing here trusts us.

    python bench_reproduce.py --fetch                 # re-fetch every diff, check its sha256
    python bench_reproduce.py --oracle                # re-run bench2_oracle.py over the fetched diffs
    python bench_reproduce.py --score styxx           # score styxx.diffgate against the oracle
    python bench_reproduce.py --score mytool:check    # score YOUR checker (module:callable)

The published rows carry no verdict of ours. That is deliberate: hand adjudication found 9 of our
11 accusations on `only_touches` were wrong (RESULT_bench2_INVALID_2026_09_17.md), so ours are not
worth taking on trust. `--score` regenerates them locally in one command, and takes anyone else's
checker on the same footing.

A checker is any callable `f(claim_text, diff_text, kind) -> str` returning one of
"CONTRADICTED", "SUPPORTED" or an abstention (anything else, including None).

Both benchmarks in this directory were declared INVALID by their own audit gates. These rows are
data, not a validated benchmark, and the ground-truth labels carry the oracle's known error rate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE / "bench2_dataset.jsonl"
DIFFS = HERE / "_diffs"
RAW = "https://patch-diff.githubusercontent.com/raw/{owner}/{repo}/pull/{n}.diff"


def rows() -> list[dict]:
    out = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith('{"_header"'):
            continue
        out.append(json.loads(line))
    return out


def _slug(url: str) -> tuple[str, str, str]:
    p = url.rstrip("/").split("/")
    return p[-4], p[-3], p[-1]


def fetch(sleep: float = 0.4) -> int:
    """Re-fetch every diff and check it against the sha256 in the row. Mismatch is loud."""
    DIFFS.mkdir(exist_ok=True)
    seen, ok, bad, gone = set(), 0, 0, Counter()
    for r in rows():
        if r["pr_id"] in seen:
            continue
        seen.add(r["pr_id"])
        dest = DIFFS / f"{r['pr_id']}.diff"
        if not dest.exists():
            owner, repo, n = _slug(r["url"])
            try:
                with urllib.request.urlopen(RAW.format(owner=owner, repo=repo, n=n), timeout=60) as fh:
                    dest.write_bytes(fh.read())
            except urllib.error.HTTPError as e:
                gone[f"http_{e.code}"] += 1
                continue
            except Exception as e:                       # noqa: BLE001 - reported, never swallowed
                gone[type(e).__name__] += 1
                continue
            time.sleep(sleep)
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        if got == r["diff_sha256"]:
            ok += 1
        else:
            bad += 1
            print(f"SHA MISMATCH {r['url']}\n  published {r['diff_sha256']}\n  fetched   {got}")
    print(f"{ok} diffs match their published sha256, {bad} mismatch, {sum(gone.values())} unreachable {dict(gone)}")
    print("A mismatch means the pull request changed after we read it, or we are wrong. Either is worth knowing.")
    return 1 if bad else 0


def _load_checker(spec: str):
    if spec == "styxx":
        from styxx.diffgate import gate_diff_text

        def f(claim: str, diff: str, kind: str):
            g = gate_diff_text(claim, diff, run=None, strict=False)
            hit = next((c for c in g.claims if c.kind == kind), None)
            return hit.verdict if hit else None
        return f
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn or "check")


def score(spec: str) -> int:
    """Score a checker against the oracle's labels. Reports the abstention rate beside the rest."""
    sys.path.insert(0, str(HERE))
    import bench2_oracle as oracle                       # noqa: F401 - re-derives truth locally

    check = _load_checker(spec)
    per = {}
    for r in rows():
        p = DIFFS / f"{r['pr_id']}.diff"
        if not p.exists():
            continue
        diff = p.read_text(encoding="utf-8", errors="replace")
        truth, _ = oracle.label(r["kind"], r.get("claim_detail") or {}, diff, r["claim_text"])
        if truth not in ("CONTRADICTED", "SUPPORTED"):
            continue
        got = check(r["claim_text"], diff, r["kind"])
        d = per.setdefault(r["kind"], Counter())
        d["n"] += 1
        if got == "CONTRADICTED":
            d["tp" if truth == "CONTRADICTED" else "fp"] += 1
        elif got == "SUPPORTED" or got == "VERIFIED":
            d["fn" if truth == "CONTRADICTED" else "tn"] += 1
        else:
            d["abstained"] += 1
    for kind, d in sorted(per.items()):
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        print(f"{kind:22s} n={d['n']:4d}  precision={prec if prec is None else round(prec,4)}  "
              f"recall={rec if rec is None else round(rec,4)}  abstained={d['abstained']}")
    print("\nPrecision without the abstention count beside it is not a number worth reporting: a checker "
          "that says nothing has perfect precision. That is the whole argument of this programme, and it "
          "cuts against us too.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--score", metavar="CHECKER")
    a = ap.parse_args()
    if not (a.fetch or a.oracle or a.score):
        ap.print_help()
        return 0
    rc = 0
    if a.fetch:
        rc |= fetch()
    if a.oracle:
        sys.path.insert(0, str(HERE))
        import bench2_oracle as oracle
        c = Counter()
        for r in rows():
            p = DIFFS / f"{r['pr_id']}.diff"
            if not p.exists():
                continue
            t, _ = oracle.label(r["kind"], r.get("claim_detail") or {}, p.read_text(encoding="utf-8", errors="replace"), r["claim_text"])
            c[(r["kind"], t)] += 1
        for k, v in sorted(c.items()):
            print(f"  {k[0]:22s} {k[1]:14s} {v}")
    if a.score:
        rc |= score(a.score)
    return rc


if __name__ == "__main__":
    sys.exit(main())
