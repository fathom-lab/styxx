"""Demo for styxx.probe_validity — runs the probe-validation battery on today's truth data and reproduces the
SURFACE-ARTIFACT verdict (NOTE_probe_orthogonality_2026_06_24). The reusable API lives in
`styxx/probe_validity.py`:

    from styxx.probe_validity import validate_probe
    report = validate_probe(construct_rows, natural_rows, get_acts)   # any concept, any model
    print(report.summary())

Run: python scripts/probe_validity.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from styxx.probe_validity import validate_probe

D = ROOT / "benchmarks" / "data" / "deception"
def load(p): return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

if __name__ == "__main__":
    for tag, L in [("qwen", 19), ("llama", 14)]:
        wa = np.load(D / f"_settling_wide_{tag}.npz"); oa = np.load(D / f"_settling_ood_{tag}.npz")
        wide = load(D / "wide_truthset.jsonl"); ood = load(D / "ood_naturals.jsonl")
        cache = {r["statement"]: wa["acts"][i] for i, r in enumerate(wide)}
        cache.update({r["statement"]: oa["acts"][i] for i, r in enumerate(ood)})
        get_acts = lambda texts: np.array([cache[t] for t in texts])
        crows = [{"text": r["statement"], "label": r["label_false"], "group": int(r["domain"])} for r in wide]
        nrows = [{"text": r["statement"], "label": r["label_false"]} for r in ood]
        print(f"\n###### DEMO: truth-template probe on {tag} (expect SURFACE-ARTIFACT) ######")
        print(validate_probe(crows, nrows, get_acts).summary())
