#!/usr/bin/env bash
# reproduce.sh — one-command reproduction of the styxx v3.4.0
# anthropic_hack benchmarks and confabulation finding.
#
# Requirements:
#   - Python 3.9+
#   - ANTHROPIC_API_KEY in env
#   - ~$1 of Anthropic credit
#   - ~20 minutes wall clock on a modern laptop
#
# Outputs in benchmarks/:
#   anthropic_hack_real_text.json        — text-mode on 84 fixtures
#   anthropic_hack_real_consensus.json   — consensus-N=5 on 84 fixtures
#   confabulation_results_v3.json        — alignment-inversion finding
#   confab_entropy_trajectories.svg      — plot of the finding

set -euo pipefail

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: set ANTHROPIC_API_KEY first"
  exit 1
fi

echo "=== styxx v3.4.0 reproduction ==="
echo "target: claude-haiku-4-5"
echo "estimated cost: ~\$1"
echo "estimated wall clock: ~20 min"
echo

pip install -e . --quiet

echo "=== (1/4) text-mode on 84 fixtures (~2 min, ~\$0.10) ==="
python benchmarks/anthropic_hack_real.py \
  --mode text \
  --out benchmarks/anthropic_hack_real_text.json

echo
echo "=== (2/4) consensus-mode N=5 on 84 fixtures (~10 min, ~\$0.50) ==="
python benchmarks/anthropic_hack_real.py \
  --mode consensus --n 5 \
  --out benchmarks/anthropic_hack_real_consensus.json

echo
echo "=== (3/4) confabulation probe n=96 (~5 min, ~\$0.30) ==="
python benchmarks/confabulation_claude.py \
  --n 5 \
  --fixtures confabulation_fixtures_v3.jsonl \
  --out benchmarks/confabulation_results_v3.json

echo
echo "=== (4/4) plot trajectories ==="
python benchmarks/plot_confab_trajectories.py

echo
echo "=== summary ==="
python - <<'PY'
import json
from pathlib import Path
R = Path("benchmarks")

t = json.load(open(R / "anthropic_hack_real_text.json"))["text"]
c = json.load(open(R / "anthropic_hack_real_consensus.json"))["consensus"]
f = json.load(open(R / "confabulation_results_v3.json"))

print(f"\n[text-heuristic vs fixture label, n={t['n']}]")
print(f"  category accuracy = {t['category_accuracy']:.3f}")
print(f"  gate agreement    = {t['gate_agreement']:.3f}")

print(f"\n[consensus N=5 vs fixture label, n={c['n']}]")
print(f"  category accuracy = {c['category_accuracy']:.3f}")

es = f.get("effect_sizes", {})
print(f"\n[confabulation finding, n_confab={f['summary_confab']['n']} n_real={f['summary_real']['n']}]")
for metric, info in es.items():
    print(f"  {metric:22s} d={info['d']:+.3f}  95% CI [{info['ci95_lo']:+.3f}, {info['ci95_hi']:+.3f}]")
PY

echo
echo "done. see benchmarks/confab_entropy_trajectories.svg for the plot."
