# -*- coding: utf-8 -*-
"""The Absence Census — how common is "not measuring, reported as fine" in the wild?

    python scripts/absence_census.py                    # screen what is installed
    python scripts/absence_census.py --json out.json

styxx.absence was built from 41 defects found in styxx ITSELF across three
adversarial audit waves (7.36.0-7.39.0). Every one had the same shape: a path
that failed, or never ran, and returned a value indistinguishable from a healthy
measurement. The obvious question is whether that shape is peculiar to this
codebase or endemic to the Python packages agents are built out of.

This script measures it. It is deliberately built so the result can embarrass us:

  * styxx is IN the sample, on the same rules as everyone else. A census whose
    author is exempt is marketing.
  * The unit is CANDIDATES PER 1000 LOC, not raw count — a large package will
    always have more findings, and comparing raw counts would be dishonest.
  * The headline is a CANDIDATE density, never a defect count. styxx.absence is
    a screen: on styxx's own freshly-audited tree it produced 41 candidates and
    (at that rule set) zero true positives. Candidates become defects only when
    a human reads them, and this script does not pretend otherwise.
  * `--verify-sample N` draws a RANDOM sample for hand-verification so precision
    can be estimated rather than assumed. Any published density must carry that
    precision estimate beside it or it is a fire-rate wearing the antibody's name.

What this is NOT: a claim that any package named here is defective. The rules
flag SHAPES. A `dict.get("score", 0.0)` in a serializer is fine; the same line
feeding a gate is not, and only a reader can tell the two apart.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from styxx.absence import LIMITS, scan_path

# Packages an agent stack is actually built out of. Screened only if installed.
TARGETS = [
    "openai", "anthropic", "langchain_core", "langchain", "autogen", "crewai",
    "llama_index", "transformers", "sentence_transformers", "sklearn", "scipy",
    "numpy", "pandas", "datasets", "pydantic", "httpx", "requests", "fastapi",
    "flask", "torch",
]


def _locate(name: str) -> Optional[Path]:
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def _loc(path: Path) -> int:
    """Non-blank, non-comment physical lines — the denominator."""
    total = 0
    for f in path.rglob("*.py"):
        s = str(f).replace("\\", "/")
        if "__pycache__" in s or "/tests/" in s or "/test/" in s:
            continue
        try:
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                t = line.strip()
                if t and not t.startswith("#"):
                    total += 1
        except OSError:
            continue
    return total


def census(targets: List[str], *, include_styxx: bool = True) -> List[Dict]:
    rows: List[Dict] = []
    pairs = [(t, _locate(t)) for t in targets]
    if include_styxx:
        here = Path(__file__).resolve().parent.parent / "styxx"
        pairs.append(("styxx (ours)", here if here.exists() else None))

    for name, path in pairs:
        if path is None or not path.exists():
            continue
        # NOT the default skip list: it excludes "site-packages", which is
        # where every installed package lives. The first run of this census
        # skipped all 2.4M lines and printed "candidates 0" — the screen
        # produced exactly the failure it exists to detect.
        rep = scan_path(path, skip=["__pycache__", "/tests/", "/test/", "/_vendor/"])
        if not rep.measured:
            print(f"  !! {name}: scanned nothing — excluded from the census")
            continue
        loc = _loc(path)
        if loc < 500:          # too small for a density to mean anything
            continue
        rows.append({
            "package": name,
            "files": rep.files_scanned,
            "loc": loc,
            "candidates": len(rep.findings),
            "per_kloc": round(len(rep.findings) / (loc / 1000.0), 2),
            "by_rule": rep.by_rule(),
            "_findings": rep.findings,      # kept for sampling; stripped in JSON
        })
    return sorted(rows, key=lambda r: -r["per_kloc"])


def render(rows: List[Dict]) -> str:
    out = ["the absence census — candidates per 1000 LOC", ""]
    out.append(f"  {'package':<24}{'KLOC':>8}{'cand':>7}{'/KLOC':>8}   top rule")
    out.append("  " + "-" * 62)
    for r in rows:
        top = max(r["by_rule"].items(), key=lambda kv: kv[1])[0] if r["by_rule"] else "-"
        star = "  <-- the screen's own authors" if "ours" in r["package"] else ""
        out.append(f"  {r['package']:<24}{r['loc']/1000:>8.1f}{r['candidates']:>7}"
                   f"{r['per_kloc']:>8.2f}   {top}{star}")
    out.append("")
    out.append("  CANDIDATES, not defects. A shape flagged here is a place to LOOK.")
    out.append(f"  LIMITS: {LIMITS}")
    return "\n".join(out)


def draw_sample(rows: List[Dict], n: int, seed: int = 0) -> List[Dict]:
    """A random sample across all packages, for hand-verification."""
    pool = [(r["package"], f) for r in rows for f in r["_findings"]]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return [{"package": p, "rule": f.rule, "file": f.file, "line": f.line,
             "source": f.source.strip()[:120], "why": f.why}
            for p, f in pool[:n]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="absence_census")
    ap.add_argument("--json", default=None, help="write the full result here")
    ap.add_argument("--verify-sample", type=int, default=0,
                    help="print N randomly drawn candidates for hand-verification")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    rows = census(TARGETS)
    print(render(rows))

    if a.verify_sample:
        print("\n  --- random sample for hand-verification "
              f"(seed {a.seed}) ---")
        for i, s in enumerate(draw_sample(rows, a.verify_sample, a.seed), 1):
            print(f"\n  [{i}] {s['package']}  {s['rule']}")
            print(f"      {Path(s['file']).name}:{s['line']}")
            print(f"      {s['source']}")

    if a.json:
        payload = {
            "rows": [{k: v for k, v in r.items() if k != "_findings"} for r in rows],
            "sample": draw_sample(rows, a.verify_sample or 20, a.seed),
            "limits": LIMITS,
            "note": ("candidates are SHAPES, not confirmed defects; publish a "
                     "hand-verified precision estimate beside any density"),
        }
        Path(a.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
