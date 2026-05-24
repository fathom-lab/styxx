# -*- coding: utf-8 -*-
"""
Bet-0 H1 — refusal holdout construction (2026-05-24).

Builds the pre-registered refusal holdout and writes its SHA-256 hash to
holdout_corpora.json. This script does the CONSTRUCTION only — it computes
gold labels (vendor-robust labeler, prereg §7) and overlap-bin stratification,
but does NOT run the instrument under test (refuse_check). The instrument
scoring + ρ test happen AFTER this holdout is hashed and committed
(run_refusal_h1.py), enforcing the no-peeking discipline.

Calibration corpus (locked, prereg §6): corpus_coverage_law._domain_pool()
— the threshold-law's security/safety domain corpus, embedded with the locked
text-embedding-3-large. Distances d were computed in the embed step and cached
in _build/xstest_gpt4_candidates.json.

Holdout source: XSTest-v2 (natolambert/xstest-v2-copy), gpt4 completion split —
450 (prompt, completion, human final_label). Spans d=0.62..0.98 → 4 quartile
overlap bins. n=450 ≥ pre-registered 400.

Gold: prereg-primary via vendor-robust detect_refusal(response) (offline).
XSTest human final_label retained as a rigor cross-check (NOT the primary).
"""
import hashlib
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/dogfood")
from vendor_robust_refusal_label import detect_refusal  # noqa: E402

HERE = Path("papers/grounded-arc")
cand = json.loads((HERE / "_build" / "xstest_gpt4_candidates.json").read_text(encoding="utf-8"))

# gold labels (construction — not the instrument under test)
for r in cand:
    r["gold"] = int(detect_refusal(r["response"]))  # prereg §7 primary
    r["gold_human"] = int(r["xstest_label"] in ("2_full_refusal", "3_partial_refusal"))

# stratify into 4 overlap bins by distance quartiles
qs = st.quantiles(sorted(r["d"] for r in cand), n=4)
def bin_of(d: float) -> str:
    return "low" if d <= qs[0] else "med" if d <= qs[1] else "high" if d <= qs[2] else "vhigh"
for r in cand:
    r["bin"] = bin_of(r["d"])

# write the locked holdout
hd = HERE / "holdout"; hd.mkdir(exist_ok=True)
with open(hd / "refusal.jsonl", "w", encoding="utf-8") as f:
    for r in cand:
        f.write(json.dumps({
            "id": r["id"], "prompt": r["prompt"], "response": r["response"],
            "gold": r["gold"], "gold_human": r["gold_human"],
            "bin": r["bin"], "d": r["d"],
        }) + "\n")

prompts = [r["prompt"] for r in cand]
sha = hashlib.sha256("\n".join(sorted(prompts)).encode("utf-8")).hexdigest()
(HERE / "holdout_corpora.json").write_text(json.dumps({
    "refusal": {
        "source": "XSTest-v2 (natolambert/xstest-v2-copy) gpt4 split; gold=vendor_robust "
                  "detect_refusal (prereg §7); human final_label retained as rigor check; "
                  "calibration corpus=corpus_coverage_law._domain_pool() (threshold-law, prereg §6)",
        "n_prompts": len(cand),
        "sha256": sha,
        "bins": ["low", "med", "high", "vhigh"],
    }
}, indent=2) + "\n", encoding="utf-8")

print(f"n = {len(cand)}  | holdout sha256 = {sha}")
print("bin counts:", dict(Counter(r["bin"] for r in cand)))
print("gold (labeler):", dict(Counter(r["gold"] for r in cand)))
print("gold (human)  :", dict(Counter(r["gold_human"] for r in cand)))
agree = sum(r["gold"] == r["gold_human"] for r in cand) / len(cand)
print(f"labeler vs human gold agreement: {agree:.3f}")
print("NO instrument scoring performed. holdout is now hashable + committable.")
