"""Atlas of Minds v0 — the first table of certified mind profiles (moonshot M2, rung 1).

Ten minds profiled with the PUBLISHED instrument (styxx.mind, 7.14.0; equivalence-gated ports of the
frozen apparatus):

  - 6 anchor models: geometry citizenship from the stored convergence reps (leave-FAMILY-out: a
    candidate scores only against anchors of OTHER families — but disclosed: these models also
    DEFINE the anchor set, so their entries are not independent of its construction).
  - 4 out-of-anchor models (gpt2, gpt2-large, pythia-410m, Qwen2.5-0.5B-Instruct): reps computed
    live on the frozen 96-concept battery (8 contextual templates, fixed 0.66-layer — the exact
    confirm-run convention). The gpt2/pythia FAMILIES played no part in anchor construction: these
    are the genuinely independent entries.
  - Behavioral conduct axis where frozen receipts exist (Qwen2.5-3B: B22 silent regime).

Geometry-only entries say so; axes without receipts stay absent (no score without an instrument).
Output: atlas_v0_result.json (machine receipt) — the ATLAS doc cites it and must pass styxx.certify.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "papers" / "real-convergence"))

from styxx import mind  # noqa: E402

ANCHORS = REPO / "papers" / "real-convergence" / "contextual_reps.npz"
B22 = REPO / "papers" / "closed-model-frontier" / "behavioral_sycophancy_b22_result.json"

LIVE = [
    ("gpt2", "gpt2", "gpt2"),
    ("gpt2-large", "gpt2-large", "gpt2"),
    ("pythia-410m", "EleutherAI/pythia-410m", "pythia"),
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "qwen"),
]


def live_reps(repo: str) -> np.ndarray:
    """(96 x d) battery reps at the fixed 0.66 layer — the frozen confirm-run convention."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_real_convergence_v2 import concept_all_layers
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16,
                                               trust_remote_code=True).to(dev).eval()
    fl = max(1, int(0.66 * mdl.config.num_hidden_layers))
    allrep = np.stack([concept_all_layers(mdl, tok, w) for w in mind.BATTERY])
    out = allrep[:, fl, :]
    del mdl, tok, allrep
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> int:
    z = np.load(ANCHORS)
    entries = []
    for name, family in mind.ANCHOR_MODELS:
        g = mind.geometry_citizenship(z[f"fixed__{name}"].astype(float), ANCHORS, family)
        entries.append({"subject": name, "family": family, "class": "anchor (in-set, disclosed)",
                        "geometry": g})
        print(f"[anchor] {name:14s} citizenship={g['citizenship_xfam_partial_lex']}", flush=True)
    for name, repo, family in LIVE:
        reps = live_reps(repo)
        g = mind.geometry_citizenship(reps, ANCHORS, family)
        entries.append({"subject": name, "family": family, "class": "out-of-anchor (independent)",
                        "geometry": g})
        print(f"[live]   {name:14s} citizenship={g['citizenship_xfam_partial_lex']}", flush=True)

    behavioral = {"subject": "Qwen2.5-3B-Instruct", "regime": "B22 silent bare-term",
                  **mind.load_behavioral_receipt(B22)}
    cert = mind.mind_certificate("Atlas of Minds v0", {"atlas_geometry": entries,
                                                       "behavioral_qwen3b_b22": behavioral})
    cert["battery"] = {"n_concepts": len(mind.BATTERY), "n_templates": len(mind.TEMPLATES),
                       "layer_convention": 0.66}
    out = HERE / "atlas_v0_result.json"
    out.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
    ranked = sorted(entries, key=lambda e: -(e["geometry"]["citizenship_xfam_partial_lex"] or -9))
    print("\n=== ATLAS v0 (citizenship, cross-family partial-lexical RSA) ===")
    for e in ranked:
        print(f"  {e['geometry']['citizenship_xfam_partial_lex']:+.4f}  {e['subject']:14s} [{e['class'][:12]}]")
    print(f"-> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
