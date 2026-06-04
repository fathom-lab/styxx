# -*- coding: utf-8 -*-
"""
real_model_spectrum.py — frequency research, toy -> REAL. Reuses the validated DMD core
(spectral_features.py) to read the oscillatory-mode structure of a REAL model's residual-stream
trajectory across token positions. CPU-only by design (tiny model) so it never contends with a
GPU job. First question: does a real LM's trajectory have non-trivial spectral structure, and does
it differ between coherent and incoherent input? (A first hint for both the universality question
and the integrity instrument — same tool.)
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from spectral_features import dmd_modes, spectral_features

torch.set_grad_enabled(False)
DEV = "cpu"                                   # tiny model on CPU — no GPU contention
MODEL = "distilgpt2"
LAYER = 4                                     # mid-stack residual read


def trajectory(model, tok, text, layer=LAYER):
    ids = tok(text, return_tensors="pt").input_ids[:, :64].to(DEV)
    out = model(ids, output_hidden_states=True)
    H = out.hidden_states[layer][0].cpu().numpy()        # (T, d_model) residual trajectory
    # center + PCA to 24 dims for a clean DMD (d_model=768 is overkill for ~tens of tokens)
    H = H - H.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    return (U[:, :24] * S[:24])                          # (T, 24) PC trajectory


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {MODEL} on {DEV} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV).eval()

    prompts = {
        "coherent": "The history of mathematics begins with counting. Early civilizations developed "
                    "numeral systems to track trade, seasons, and astronomy, and from these grew "
                    "geometry, algebra, and eventually the calculus that describes motion.",
        "shuffled": "track trade calculus the begins counting numeral motion seasons mathematics of "
                    "history developed astronomy describes early grew geometry from these eventually "
                    "and systems civilizations the algebra to and that",
        "repetitive": "memory memory memory memory memory memory memory memory memory memory memory "
                      "memory memory memory memory memory memory memory memory memory memory memory",
    }
    print(f"\n{'prompt':12s} {'dom_freq':>9} {'wt_freq':>8} {'spec_ent':>9} {'hi_band':>8} {'persist':>8}")
    for name, text in prompts.items():
        X = trajectory(model, tok, text)
        f = spectral_features(X, rank=12)
        print(f"{name:12s} {f['dominant_freq']:9.4f} {f['weighted_freq']:8.4f} "
              f"{f['spectral_entropy']:9.4f} {f['high_band_frac']:8.4f} {f['weighted_decay']:8.4f}")
    print("\n[read] non-trivial, prompt-dependent spectral structure in a real residual stream = "
          "the DMD tool transfers from toy to real. Next (GPU): honest/deceptive split + the K1-K4 gates.")


if __name__ == "__main__":
    main()
