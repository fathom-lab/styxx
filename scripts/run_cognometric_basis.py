# -*- coding: utf-8 -*-
"""
run_cognometric_basis.py — apply PCA to the cross-instrument fingerprint
matrix on the full single-turn benchmarks/data corpus.

Output: how many independent dimensions does cognometric measurement
actually have? If rank95 < n_instruments, the instruments are
collapsing onto fewer latent factors. That's the quantitative form
of the rc2 non-orthogonality finding.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from styxx.attack import cognometric_basis

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "benchmarks" / "data"

SINGLE_TURN_CORPORA = {
    "sycophancy":     ("sycophancy/responses_v0.jsonl",   "label_sycophantic"),
    "deception":      ("deception/responses_v0.jsonl",    "label_dishonest"),
    "overconfidence": ("overconfidence/pairs_v0.jsonl",   "label_overconfident"),
}


def _load(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    samples = []
    for inst_name, (rel_path, _) in SINGLE_TURN_CORPORA.items():
        for row in _load(DATA / rel_path):
            samples.append({
                "prompt": row.get("question", ""),
                "response": row.get("response", ""),
            })
    print(f"loaded {len(samples)} single-turn samples from 3 corpora")

    result = cognometric_basis(samples)

    print("\n" + "=" * 78)
    print(" COGNOMETRIC BASIS DECOMPOSITION (SVD-PCA, z-scored) ")
    print("=" * 78 + "\n")
    print(f"  n_samples      : {result.n_samples}")
    print(f"  instruments    : {result.instruments}")
    print(f"  total dims     : {len(result.instruments)}")
    print(f"  rank @ 95% var : {result.rank95}  ({result.rank95}/{len(result.instruments)})")
    print(f"  rank @ 99% var : {result.rank99}  ({result.rank99}/{len(result.instruments)})")
    print()
    print(f"  {'PC':>4}  {'EVR':>7}  {'cumEVR':>7}  loadings (instrument weights)")
    print(f"  {'----':>4}  {'-------':>7}  {'-------':>7}  {'-' * 60}")
    for i, (e, c, row) in enumerate(zip(result.evr, result.cumulative_evr, result.loadings)):
        load_str = "  ".join(
            f"{name[:6]}={w:+.3f}" for name, w in zip(result.instruments, row)
        )
        print(f"  PC{i+1:>2}  {e:>7.3f}  {c:>7.3f}  {load_str}")

    print()
    if result.rank95 < len(result.instruments):
        gap = len(result.instruments) - result.rank95
        print(f"  >>> FINDING: {len(result.instruments)} instruments collapse to "
              f"{result.rank95} effective dimensions at 95% variance.")
        print(f"  >>> The cognometric system measures {gap} fewer "
              f"independent factor(s) than its surface count suggests.")
    else:
        print(f"  >>> No collapse: each of {len(result.instruments)} instruments "
              f"contributes meaningfully to variance.")

    # write the analysis result for downstream use
    out = REPO_ROOT / "benchmarks" / "cognometric_basis_v0.json"
    out.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"\n  full result written to {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
