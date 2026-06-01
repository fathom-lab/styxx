"""Mechanistic depth: are the cave-directions the SAME direction across pressure types?
PREREG_intent_direction_alignment_2026_05_31.

For each pressure, the cave direction at a fixed layer = normalize(mean(lie residuals) - mean(mistake
residuals)) — the pressure-prompt effect cancels (both classes share it), isolating lie-vs-mistake.
Cosine matrix across pressures, vs the same-pressure split-half CEILING (max achievable given data noise)
and the random-direction FLOOR (~0). High cross-pressure cosine -> one geometric feature, not N probes.

  python probe_alignment.py [layer=36]
"""
from __future__ import annotations
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 36
PRESSURE_TAGS = {
    "default":    ["full", "bc", "bc2"],
    "authority":  ["pr_authority"],
    "social":     ["pr_social"],
    "insistence": ["pr_insistence"],
}


def load(tags):
    X, y = [], []
    for t in tags:
        mp = os.path.join(HERE, f"intent_meta{t}.json")
        rp = os.path.join(HERE, f"residuals_intent{t}.npz")
        if not (os.path.exists(mp) and os.path.exists(rp)):
            return None, None
        meta = json.load(open(mp, encoding="utf-8"))
        R = np.load(rp)["residuals"]
        for i, r in enumerate(meta["rows"]):
            if r["cls"] in ("lie", "mistake"):
                X.append(R[i, LAYER, :].astype(np.float64))
                y.append(1 if r["cls"] == "lie" else 0)
    return np.array(X), np.array(y)


def direction(X, y):
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-9)


dirs, counts = {}, {}
for p, tags in PRESSURE_TAGS.items():
    X, y = load(tags)
    if X is not None and (y == 1).sum() >= 20 and (y == 0).sum() >= 20:
        dirs[p] = direction(X, y)
        counts[p] = (int((y == 1).sum()), int((y == 0).sum()))

keys = list(dirs)
print(f"cave-direction cosine matrix @ layer {LAYER}  (pressures present: {keys})\n")
print("          " + "".join(f"{k[:8]:>9}" for k in keys))
for a in keys:
    print(f"{a[:9]:9} " + "".join(f"{float(dirs[a] @ dirs[b]):9.3f}" for b in keys))

# same-pressure split-half ceiling (default) + random floor
Xd, yd = load(PRESSURE_TAGS["default"])
rng = np.random.RandomState(0)
idx = rng.permutation(len(yd))
h = len(yd) // 2
ceil = float(direction(Xd[idx[:h]], yd[idx[:h]]) @ direction(Xd[idx[h:]], yd[idx[h:]]))
rv = rng.randn(Xd.shape[1]); rv /= np.linalg.norm(rv)
floor = float(dirs["default"] @ rv)

cross = {f"{a}-{b}": float(dirs[a] @ dirs[b]) for a, b in itertools.combinations(keys, 2)}
mean_cross = float(np.mean(list(cross.values()))) if cross else None
print(f"\nsame-pressure split-half CEILING (default): {ceil:.3f}")
print(f"random-direction FLOOR:                     {floor:.3f}")
print(f"mean cross-pressure cosine:                 {mean_cross:.3f}  ({len(cross)} pairs)")
if mean_cross is not None:
    ratio = mean_cross / ceil if ceil > 0 else None
    aligned = mean_cross >= 0.5 and (ratio is not None and ratio >= 0.6)
    print(f"cross/ceiling ratio: {ratio:.2f}   ALIGNED (>=0.5 & >=0.6*ceiling): {aligned}")
    print("  -> the cave is ONE geometric direction across pressures" if aligned
          else "  -> the directions are partly pressure-specific (report honestly)")

json.dump({"experiment": "cave-direction alignment across pressure types",
           "prereg": "papers/grounded-honesty-axis/PREREG_intent_direction_alignment_2026_05_31.md",
           "layer": LAYER, "counts": counts, "cosines": cross, "mean_cross": mean_cross,
           "ceiling": ceil, "floor": floor},
          open(os.path.join(HERE, "intent_alignment_result.json"), "w"), indent=2)
print("\nwrote intent_alignment_result.json")
