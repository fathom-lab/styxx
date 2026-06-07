# -*- coding: utf-8 -*-
"""diag_transfer.py — diagnose WHY zero-anchor direction transfer underperformed.
Q1: are concept steering vectors IN the top-k concept subspace the map operates on?
Q2: can the map transfer CONCEPT directions on a KNOWN A->A-rotation (fair positive control)?
If Q2 is high but cross-model is null, the bottleneck is cross-model Q precision (read!=write).
"""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "introspection-gate"))
from styxx_transfer import pca_basis, self_test, TransferMap
from run_thought_transfer import extract, CONCEPTS, TRAIN, HELDOUT
from introspection_gate import load_model

SRC = "meta-llama/Llama-3.2-3B-Instruct"
from transformers import AutoConfig
nl = AutoConfig.from_pretrained(SRC).num_hidden_layers
L = round(0.60 * nl)
tok, m = load_model(SRC)
pts, vecs = extract(m, tok, L)
del m; torch.cuda.empty_cache()

Rtrain = np.array([pts[c] for c in TRAIN])
Vheld = np.array([vecs[c] for c in HELDOUT])

for k in [40, 60, 85]:
    _, VAk = pca_basis(Rtrain, k)
    frac = np.mean([np.linalg.norm(v @ VAk) for v in Vheld])     # unit vecs -> ||proj||
    cos_concept, _ = self_test(Rtrain, k=k, test_dirs=Vheld)      # transfer CONCEPT dirs, A->A-rot
    cos_insub, _ = self_test(Rtrain, k=k, in_subspace=True)       # in-subspace random dirs
    cos_rand, _ = self_test(Rtrain, k=k, in_subspace=False)       # full-dim random (unfair)
    print(f"k={k:3d}  concept-vec subspace frac={frac:.3f} | A->A-rot transfer |cos|: "
          f"concept-dirs={cos_concept:.3f}  in-subspace-rand={cos_insub:.3f}  full-rand={cos_rand:.3f}")
