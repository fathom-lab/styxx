"""Generation-matched cross-vendor cliff (PREREG_genmatch_xvendor_2026_06_23).

Re-sample the 3 open families at max_new_tokens=32 (matching gpt-4o-mini's only unmatched generation
knob; prompt/T/n/top_p already match), identical NLI judge, recompute each family's per-domain cliff.
Monkeypatches run_local_cliff so NOTHING clobbers the committed 24-token results (outputs get a _gm32
suffix). No API key.

Usage:  python run_genmatch_cliff.py [--smoke]
"""
from __future__ import annotations
import run_local_cliff as L

# match gpt-4o-mini's generation exactly: the only unmatched knob is max tokens (24 -> 32)
L.MAX_NEW = 32

# retag every output file so the committed 24-token gates/benchmarks are untouched
_orig_slug = L._slug
L._slug = lambda m: _orig_slug(m) + "_gm32"
L.RESULT = L.HERE / "genmatch_cliff_result.json"

if __name__ == "__main__":
    raise SystemExit(L.main())
