"""Qualify `styxx.claim_audit.audit_grounding` against planted controls.

STANDING RULE (MEMORY.md, 2026-08-13): any favorable number I publish about myself gets a judge
qualified against planted controls FIRST. The judge here told me my C6 prereg is 65/76 grounded.
Before that number goes anywhere, the judge itself gets tested.

THE SUSPICION, stated before the run:
`_match` grounds a claimed number if ANY value anywhere in the flattened source dict is within
tolerance. It never checks that the matched path has anything to do with the sentence the number
appeared in. With a large receipt (c6_basis_v2.json flattens to hundreds of numbers in [0,1]),
a two-decimal quantity should find a coincidental match almost always. If that is true, then
"grounded" is a statement about SOURCE CARDINALITY, not about the document's honesty, and a
fabricated document will score about as well as the real one.

PRE-STATED PREDICTION (frozen before running):
  P1. The FABRICATED document (every statistic replaced with a wrong-but-plausible value) will
      score a grounded rate within 15 points of the REAL document.
  P2. Grounded rate on a fixed document will RISE monotonically as the source dict grows,
      despite the document not changing.
  If P1 and P2 both hold, "grounded" is not evidence of grounding, and I will say so about my
  own favorable result rather than quoting the 65/76.
  If P1 fails (fabricated scores much lower), the judge discriminates and my 65/76 stands.
"""
from __future__ import annotations
import json
import pathlib
import random
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from styxx.claim_audit import audit_grounding, _flatten  # noqa: E402

FA = HERE.parent / "first-afference"
basis = json.loads((FA / "c6_basis_v2.json").read_text(encoding="utf-8"))
power = json.loads((FA / "c6_power.json").read_text(encoding="utf-8"))
SOURCES = {"basis_v2": basis, "power": power}

PREREG = (FA / "PREREG_c6_derived_bar_2026_08_13.md").read_text(encoding="utf-8")


def rate(text, sources):
    rep = audit_grounding(text, sources)
    items = rep.items if hasattr(rep, "items") else rep.__dict__["items"]
    n = len(items)
    if n == 0:
        return 0.0, 0, 0
    g = sum(1 for i in items if i.status in ("grounded", "derived"))
    return g / n, g, n


def fabricate(text, seed=11):
    """Replace every decimal in [0,1] with a DIFFERENT plausible value. Same prose, false numbers.

    This is the planted control: a document whose every quantitative claim is wrong. An honest
    grounding judge must score it far below the real document.
    """
    rng = random.Random(seed)

    def sub(m):
        v = float(m.group())
        if 0.0 <= v <= 1.0:
            new = round(rng.uniform(0.01, 0.99), len(m.group().split(".")[-1]))
            if abs(new - v) < 0.02:
                new = round(min(0.99, new + 0.17), 3)
            return f"{new:.{len(m.group().split('.')[-1])}f}"
        return m.group()

    return re.sub(r"(?<![\w.])\d\.\d+(?![\w.])", sub, text)


def prune_sources(frac, seed=5):
    """Keep a random fraction of the leaf numbers, to vary source cardinality only."""
    flat = _flatten(SOURCES, "", {})
    keys = list(flat)
    rng = random.Random(seed)
    rng.shuffle(keys)
    keep = keys[: max(1, int(len(keys) * frac))]
    return {f"k{i}": v for i, v in enumerate(keep)}


def main():
    print("=" * 74)
    print("QUALIFYING THE GROUNDING JUDGE — planted controls")
    print("=" * 74)

    real_r, real_g, real_n = rate(PREREG, SOURCES)
    fab = fabricate(PREREG)
    fab_r, fab_g, fab_n = rate(fab, SOURCES)

    print(f"\nP1 — does the judge separate truth from fabrication?")
    print(f"  REAL prereg        : {real_g}/{real_n} = {real_r:.3f} grounded")
    print(f"  FABRICATED prereg  : {fab_g}/{fab_n} = {fab_r:.3f} grounded")
    gap = real_r - fab_r
    print(f"  separation         : {gap:+.3f}")
    p1_holds = gap < 0.15
    print(f"  P1 (gap < 0.15, judge does NOT discriminate): {'HOLDS' if p1_holds else 'FAILS'}")

    print(f"\nP2 — does grounded rate track SOURCE SIZE on an unchanged document?")
    curve = []
    for frac in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        src = prune_sources(frac)
        r, g, n = rate(PREREG, src)
        curve.append((frac, len(src), r))
        print(f"  sources={len(src):5d} ({frac:>5.0%})  grounded={g:3d}/{n}  rate={r:.3f}")
    rates = [c[2] for c in curve]
    p2_holds = rates[-1] - rates[0] > 0.30 and all(
        rates[i] <= rates[i + 1] + 0.05 for i in range(len(rates) - 1))
    print(f"  P2 (rate rises with cardinality): {'HOLDS' if p2_holds else 'FAILS'}")

    # How many distinct source values does a random 2-decimal number hit?
    flat = _flatten(SOURCES, "", {})
    rng = random.Random(99)
    hits = 0
    trials = 2000
    for _ in range(trials):
        q = round(rng.uniform(0, 1), 2)
        if any(abs(v - q) <= 0.005 for v in flat):
            hits += 1
    print(f"\n  BASELINE: a RANDOM 2-decimal number in [0,1] grounds "
          f"{hits}/{trials} = {hits/trials:.3f} of the time against this receipt")
    print(f"  (source has {len(flat)} distinct leaf values)")

    verdict = ("JUDGE_IS_DECORATION__grounding_measures_source_cardinality"
               if (p1_holds and p2_holds) else
               "JUDGE_DISCRIMINATES__grounded_rate_is_evidence")
    print(f"\nVERDICT: {verdict}")

    out = {"real_rate": round(real_r, 4), "real": [real_g, real_n],
           "fabricated_rate": round(fab_r, 4), "fabricated": [fab_g, fab_n],
           "separation": round(gap, 4), "P1_judge_fails_to_discriminate": p1_holds,
           "cardinality_curve": [{"frac": f, "n_sources": n, "rate": round(r, 4)}
                                 for f, n, r in curve],
           "P2_rate_tracks_cardinality": p2_holds,
           "random_number_ground_rate": round(hits / trials, 4),
           "n_source_leaf_values": len(flat),
           "verdict": verdict}
    (HERE / "grounding_judge_qualification.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote grounding_judge_qualification.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
