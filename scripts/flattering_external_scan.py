# -*- coding: utf-8 -*-
"""Run the FROZEN flattering screen over every third-party package on this box.

Prereg: papers/PREREG_flattering_external_2026_08_21.md
Detector frozen at the commit recorded in the output. G3 voids this run if
styxx/flattering.py is edited afterwards.
"""
from __future__ import annotations

import json
import random
import site
import subprocess
import sys
from pathlib import Path

from styxx.flattering import scan_path

SEED = 20260821
MIN_FILES = 40
SAMPLE_CAP = 60
OUT = Path(__file__).resolve().parent.parent / "papers" / "out_flattering_external.json"


def frozen_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parent.parent)
    dirty = subprocess.run(["git", "status", "--porcelain", "styxx/flattering.py"],
                           capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parent.parent).stdout.strip()
    return r.stdout.strip()[:9] + (" DIRTY-RUN-IS-VOID" if dirty else "")


def main() -> int:
    sp = [Path(p) for p in site.getsitepackages() if p.endswith("site-packages")]
    pkgs = []
    for base in sp:
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            if d.name.endswith(("dist-info", "egg-info")) or d.name.startswith((".", "_")):
                continue
            n = sum(1 for _ in d.rglob("*.py"))
            if n >= MIN_FILES:
                pkgs.append((d, n))

    print(f"detector frozen at {frozen_commit()}")
    print(f"{len(pkgs)} third-party packages with >= {MIN_FILES} .py files\n")

    per_pkg, all_a, all_b = [], [], []
    files = unparsed = 0
    for d, n in pkgs:
        rep = scan_path(d)
        if not rep.measured:
            print(f"  {d.name:22} SCANNED NOTHING: {rep.why}")
            continue
        files += rep.files_scanned
        unparsed += rep.files_unparsed
        a = [h.as_dict() | {"package": d.name} for h in rep.tier_a]
        b = [h.as_dict() | {"package": d.name} for h in rep.tier_b]
        all_a.extend(a)
        all_b.extend(b)
        per_pkg.append({"package": d.name, "files": rep.files_scanned,
                        "unparsed": rep.files_unparsed,
                        "tier_a": len(a), "tier_b": len(b)})
        if a:
            print(f"  {d.name:22} {rep.files_scanned:5d} files   "
                  f"TIER-A {len(a):3d}   TIER-B {len(b):4d}")

    print(f"\n{'='*66}")
    print(f"files scanned      {files}  ({unparsed} unparseable)")
    print(f"TIER-A (claimed)   {len(all_a)}")
    print(f"TIER-B (counted)   {len(all_b)}")
    print(f"packages with >=1 TIER-A: "
          f"{sum(1 for p in per_pkg if p['tier_a'])} / {len(per_pkg)}")

    # G2 -- power precondition, checked before anything is claimed
    if len(all_a) < 15:
        print(f"\nG2: {len(all_a)} < 15 -> INVALID__UNDERPOWERED. Not a null.")

    sampled = all_a
    if len(all_a) > SAMPLE_CAP:
        rng = random.Random(SEED)
        sampled = rng.sample(all_a, SAMPLE_CAP)
        print(f"\nG-sampling: {len(all_a)} > {SAMPLE_CAP}, adjudicating a random "
              f"{SAMPLE_CAP} (seed {SEED}); disclosed in the RESULT.")

    OUT.write_text(json.dumps({
        "frozen_commit": frozen_commit(),
        "seed": SEED,
        "files_scanned": files, "files_unparsed": unparsed,
        "n_tier_a": len(all_a), "n_tier_b": len(all_b),
        "per_package": per_pkg,
        "tier_a": all_a,
        "tier_a_sampled": sampled,
        "tier_b_sample": all_b[:200],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT.name}  ({len(sampled)} hits queued for adjudication)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
