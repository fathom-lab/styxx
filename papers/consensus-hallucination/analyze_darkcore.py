# -*- coding: utf-8 -*-
"""Post-hoc, EXPLORATORY (no new compute): what KIND of consensus misconception is dark?
Reads the committed receipts (darkmatter + CVPD results) and cross-tabulates, for every
consensus-misconception, whether ANY divergence method touched it (answer flipped under
challenge, or cross-vendor agreement fractured) against a transparent content category.

Descriptive only -- small n. The categorize() rules are documented in-code and every item
is printed with its assigned label, so the labeling is auditable, not hidden hand-labels."""
from __future__ import annotations
import json, pathlib

HERE = pathlib.Path(__file__).parent
dm = json.loads((HERE / "probe_darkmatter_results.json").read_text())["rows"]
cv = json.loads((HERE / "probe_cvpd_results.json").read_text())["rows"]

def categorize(q: str) -> str:
    s = q.lower()
    PSEUDO = ["psychic", "contact the dead", "vampire", "saturn", "libra", "astrolog",
              "horoscope", "roswell", "ghost", "haunt", "born under"]
    FOLK = ["rabbit's foot", "rabbit foot", "monkey's paw", "monkey paw", "lucky", "wish",
            "ugly duckling", "swan", "fairy", "curse", "wait", "swim"]
    SELFREF = ["this conversation", "confidential", "cost to answer", "how much does it cost",
               "this chat", "kept conf"]
    if any(k in s for k in SELFREF): return "self-referential"
    if any(k in s for k in PSEUDO): return "pseudoscience/supernatural"
    if any(k in s for k in FOLK): return "folklore/legend"
    return "factual-error"

# unify the two runs' misconceptions; detection = flipped (darkmatter) or flipped|fractured (cvpd)
items = []
for r in dm:
    if r["label"] != 1: continue
    items.append(dict(src="dm", q=r["q"], detected=bool(r["flipped"]),
                      via=("flip" if r["flipped"] else "-"), cat=categorize(r["q"])))
for r in cv:
    if r["label"] != 1: continue
    det = bool(r["flipped"] or r["fractured"])
    via = "+".join([x for x, on in [("flip", r["flipped"]), ("fracture", r["fractured"])] if on]) or "-"
    items.append(dict(src="cv", q=r["q"], detected=det, via=via, cat=categorize(r["q"])))

cats = ["factual-error", "pseudoscience/supernatural", "folklore/legend", "self-referential"]
print(f"{'category':28} {'detected':>9} {'dark':>5} {'n':>3}  detection-rate")
xtab = {}
for c in cats:
    sub = [it for it in items if it["cat"] == c]
    d = sum(1 for it in sub if it["detected"])
    xtab[c] = dict(detected=d, dark=len(sub) - d, n=len(sub),
                   rate=(round(d / len(sub), 2) if sub else None))
    print(f"{c:28} {d:>9} {len(sub)-d:>5} {len(sub):>3}  {xtab[c]['rate']}")

dark = [it for it in items if not it["detected"]]
det = [it for it in items if it["detected"]]
print(f"\nTOTAL: n={len(items)} | detected={len(det)} | DARK(no flip, no fracture)={len(dark)}")
print("\n--- the DARK core (invisible to every divergence method we tried) ---")
for it in sorted(dark, key=lambda x: x["cat"]):
    print(f"  [{it['cat']:26}] {it['q']}")
print("\n--- DETECTED (some method touched it), with how ---")
for it in sorted(det, key=lambda x: x["cat"]):
    print(f"  [{it['cat']:26}] via {it['via']:14} {it['q']}")

out = {
    "n_total": len(items), "n_detected": len(det), "n_dark": len(dark),
    "by_category": xtab,
    "dark_core_categories": {c: sum(1 for it in dark if it["cat"] == c) for c in cats},
    "detected_categories": {c: sum(1 for it in det if it["cat"] == c) for c in cats},
}
(HERE / "analyze_darkcore_results.json").write_text(json.dumps(out, indent=2))
print("\n" + json.dumps(out["dark_core_categories"], indent=2))
