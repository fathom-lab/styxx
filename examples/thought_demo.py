# -*- coding: utf-8 -*-
"""
examples/thought_demo.py — the .fathom file format and the Thought type.

A demonstration that cognitive content is now a portable data type.

For every category in the atlas v0.3 fixture set this script:
  1. Reads the bundled demo trajectory through styxx.Raw()
  2. Projects the resulting Vitals into a styxx.Thought
  3. Saves the Thought to a .fathom file
  4. Loads it back and checks round-trip integrity
  5. Computes the cognitive distance to every other category
  6. Demonstrates the Thought algebra (interpolate, mix, delta)

Runs on bundled fixtures only — no API key, no network, no model
download. Output .fathom files are written to demo/thoughts/ in
the styxx repo so you can inspect them by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import styxx
from styxx import Raw, Thought
from styxx.cli import _load_demo_trajectories
from styxx.thought import CATEGORIES, PHASE_ORDER


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "demo" / "thoughts"


# ══════════════════════════════════════════════════════════════════
# Pretty printing
# ══════════════════════════════════════════════════════════════════

def hr(char: str = "─", width: int = 72) -> str:
    return char * width


def header(title: str) -> None:
    print()
    print(hr("═"))
    print(f"  {title}")
    print(hr("═"))


def section(title: str) -> None:
    print()
    print(hr("─"))
    print(f"  {title}")
    print(hr("─"))


# ══════════════════════════════════════════════════════════════════
# 1. Read every demo trajectory into a Thought
# ══════════════════════════════════════════════════════════════════

header(f"styxx {styxx.__version__} — the .fathom format demo")

print()
print("  PNG is the format for images.")
print("  JSON is the format for data.")
print("  .fathom is the format for thoughts.")
print()

print(f"  output dir: {OUT_DIR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = _load_demo_trajectories()
source_model = data.get("source_model", "unknown")
print(f"  source model: {source_model}")
print(f"  atlas version: {data.get('source_atlas_version', '?')}")
print(f"  categories:   {', '.join(CATEGORIES)}")

section("1. read each demo trajectory into a Thought")

thoughts: dict[str, Thought] = {}
for cat in CATEGORIES:
    traj = data["trajectories"][cat]
    vitals = Raw().read(
        entropy=traj["entropy"],
        logprob=traj["logprob"],
        top2_margin=traj["top2_margin"],
    )
    t = Thought.from_vitals(
        vitals,
        source_text=traj.get("text_preview", ""),
        source_model=source_model,
        tags={"demo_category": cat, "probe_id": traj.get("probe_id")},
    )
    thoughts[cat] = t
    primary = t.primary_category or "?"
    conf = t.primary_confidence
    n = t.n_tokens_observed or 0
    print(f"  {cat:<14} → primary={primary:<14} conf={conf:.2f}  n_tokens={n}  hash={t.content_hash()[:12]}")


# ══════════════════════════════════════════════════════════════════
# 2. Save every Thought as a .fathom file
# ══════════════════════════════════════════════════════════════════

section("2. save each Thought as a .fathom file")

paths: dict[str, Path] = {}
for cat, t in thoughts.items():
    path = OUT_DIR / f"{cat}.fathom"
    saved = t.save(path)
    size = saved.stat().st_size
    print(f"  {cat:<14} → {saved.name}  ({size:>5} bytes)")
    paths[cat] = saved


# ══════════════════════════════════════════════════════════════════
# 3. Round-trip integrity: load each .fathom and compare
# ══════════════════════════════════════════════════════════════════

section("3. round-trip every .fathom file (load, compare to original)")

for cat, path in paths.items():
    loaded = Thought.load(path)
    ok = loaded == thoughts[cat]
    hash_ok = loaded.content_hash() == thoughts[cat].content_hash()
    status = "OK" if (ok and hash_ok) else "FAIL"
    print(f"  {cat:<14} {status}  cognitive_eq={ok}  hash_match={hash_ok}")


# ══════════════════════════════════════════════════════════════════
# 4. Cognitive distance matrix
# ══════════════════════════════════════════════════════════════════

section("4. cognitive distance matrix (euclidean in eigenvalue space)")

# Header row
header_cells = ["".ljust(14)] + [c[:6].ljust(7) for c in CATEGORIES]
print("  " + "".join(header_cells))
for c1 in CATEGORIES:
    row = [c1.ljust(14)]
    for c2 in CATEGORIES:
        d = thoughts[c1].distance(thoughts[c2])
        if c1 == c2:
            row.append(" .     ")
        else:
            row.append(f"{d:5.2f}  ")
    print("  " + "".join(row))

print()
print("  reading: rows are 'from', columns are 'to'.")
print("  diagonal is 0 by definition. lower = more cognitively similar.")


# ══════════════════════════════════════════════════════════════════
# 5. Cognitive similarity matrix (1 - normalized distance)
# ══════════════════════════════════════════════════════════════════

section("5. cognitive similarity matrix (1.00 = identical, 0.00 = orthogonal)")

header_cells = ["".ljust(14)] + [c[:6].ljust(7) for c in CATEGORIES]
print("  " + "".join(header_cells))
for c1 in CATEGORIES:
    row = [c1.ljust(14)]
    for c2 in CATEGORIES:
        s = thoughts[c1].similarity(thoughts[c2])
        if c1 == c2:
            row.append("1.00   ")
        else:
            row.append(f"{s:5.2f}  ")
    print("  " + "".join(row))


# ══════════════════════════════════════════════════════════════════
# 6. Algebra: interpolate
# ══════════════════════════════════════════════════════════════════

section("6. algebra — interpolate(reasoning, creative) along alpha")

a = thoughts["reasoning"]
b = thoughts["creative"]
print(f"  endpoints:  a = {a}")
print(f"              b = {b}")
print()
print(f"  alpha   primary             distance(a)   distance(b)")
for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
    m = a.interpolate(b, alpha=alpha)
    da = a.distance(m)
    db = b.distance(m)
    print(f"  {alpha:.2f}    {(m.primary_category or '?'):<19} {da:>9.4f}     {db:>9.4f}")
print()
print("  invariant: at alpha=0 the result is b; at alpha=1 it is a.")
print("             at alpha=0.5 distances should be equal.")


# ══════════════════════════════════════════════════════════════════
# 7. Algebra: mix
# ══════════════════════════════════════════════════════════════════

section("7. algebra — mix(reasoning, creative, retrieval) with weights")

mix_inputs = [thoughts["reasoning"], thoughts["creative"], thoughts["retrieval"]]
weights = [0.5, 0.3, 0.2]
mixed = Thought.mix(mix_inputs, weights=weights)
print(f"  mixture (50% reasoning, 30% creative, 20% retrieval):")
print(f"    primary  = {mixed.primary_category}")
print(f"    conf     = {mixed.primary_confidence:.3f}")
print(f"    mean probs:")
for cat, p in zip(CATEGORIES, mixed.mean_probs()):
    bar = "█" * int(p * 30)
    print(f"      {cat:<14} {p:.3f}  {bar}")


# ══════════════════════════════════════════════════════════════════
# 8. Algebra: delta (signed difference)
# ══════════════════════════════════════════════════════════════════

section("8. algebra — delta (reasoning - creative)")

delta = thoughts["reasoning"] - thoughts["creative"]
print(f"  {delta}")
print()
print(f"  biggest movers (signed):")
for phase, cat, val in delta.biggest_movers(top_k=6):
    sign = "+" if val >= 0 else " "
    print(f"    {phase:<18} {cat:<14} {sign}{val:+.3f}")


# ══════════════════════════════════════════════════════════════════
# 9. Cognitive equivalence — load a .fathom from disk into a fresh process
# ══════════════════════════════════════════════════════════════════

section("9. portability proof — every saved .fathom loads back identically")

all_match = True
for cat in CATEGORIES:
    src = thoughts[cat]
    loaded = Thought.load(paths[cat])
    eq_cog = (loaded == src)
    eq_hash = (loaded.content_hash() == src.content_hash())
    if not (eq_cog and eq_hash):
        all_match = False
        print(f"  {cat:<14} MISMATCH cog={eq_cog} hash={eq_hash}")
    else:
        print(f"  {cat:<14} cognitive_eq=True  content_hash={loaded.content_hash()[:16]}...")

print()
if all_match:
    print("  >> all 6 .fathom files round-trip with bit-perfect cognitive content.")
    print("  >> .fathom is a portable cognitive data type. demo complete.")
else:
    print("  !! one or more files failed round-trip. investigate.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# 10. Where to look next
# ══════════════════════════════════════════════════════════════════

section("10. inspect the artifacts")
print(f"  open the .fathom files in your editor:")
for cat in CATEGORIES:
    print(f"    {paths[cat]}")
print()
print(f"  each is canonical sort_keys JSON, no BOM, ~1-2 KB.")
print(f"  the schema is documented in: docs/fathom-spec-v0.md")
print()
