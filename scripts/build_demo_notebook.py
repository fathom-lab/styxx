"""Generate the probe-validity Colab notebook (examples/probe_validity_colab.ipynb) — the friction-free
adoption artifact: install styxx, see it catch a surface-artifact probe in ~60s (no GPU), then a real-model
reproduction, then a paste-in-your-own-probe template. Deterministic builder so the .ipynb JSON is valid.

Run: python scripts/build_demo_notebook.py
"""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "probe_validity_colab.ipynb"


def md(*lines): return {"cell_type": "markdown", "metadata": {}, "source": [l if l.endswith("\n") else l + "\n" for l in lines]}
def code(*lines): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": [l if l.endswith("\n") else l + "\n" for l in lines]}

cells = [
    md("# Is your oversight probe real — or a surface artifact?",
       "",
       "**`styxx.validate_probe`** — a 60-second check for anyone training linear probes on model activations "
       "(deception / truthfulness / harm detectors).",
       "",
       "A probe can hit **0.98 accuracy**, generalize across domains, and pass a text-silence gate — and still "
       "be measuring a surface pattern *orthogonal to the concept*, failing silently on real inputs. This "
       "notebook lets you catch that on **your own** probe.",
       "",
       "Paper: `papers/grounded-honesty-axis/NOTE_probe_orthogonality_2026_06_24.md` · "
       "Repo: https://github.com/fathom-lab/styxx · MIT"),

    md("## 1. Install (≈20s — pure-Python core, no GPU needed for the headline demo)"),
    code("!pip install -q \"git+https://github.com/fathom-lab/styxx.git\"",
         "from styxx import validate_probe, ProbeValidityReport",
         "import numpy as np"),

    md("## 2. The headline demo (no model needed)",
       "",
       "Two synthetic probes on a concept axis `u`. A **valid** probe's direction aligns with `u`. A "
       "**surface‑artifact** probe is trained on data where the label is driven by an *orthogonal* direction "
       "`v` — it scores great in‑construct but is blind to the real concept. The battery tells them apart."),
    code("rng = np.random.default_rng(0); dim = 64",
         "u = rng.standard_normal(dim); u /= np.linalg.norm(u)                    # the concept axis",
         "v = rng.standard_normal(dim); v -= (v@u)*u; v /= np.linalg.norm(v)      # an orthogonal surface axis",
         "",
         "def make(drive):",
         "    acts, crows, nrows = {}, [], []",
         "    for i in range(80):   # construct: label driven by `drive`",
         "        y=i%2; acts[f'c{i}']=(2*y-1)*3*drive+rng.standard_normal(dim); crows.append({'text':f'c{i}','label':y,'group':i//20})",
         "    for i in range(40):   # natural test: concept always on u",
         "        y=i%2; acts[f'n{i}']=(2*y-1)*3*u+rng.standard_normal(dim); nrows.append({'text':f'n{i}','label':y})",
         "    return crows, nrows, (lambda ts: np.array([acts[t] for t in ts]))",
         "",
         "print('=== a VALID probe (trained on the real concept axis u) ===')",
         "c,n,g = make(u);  print(validate_probe(c, n, g).summary())",
         "print('\\n=== a SURFACE-ARTIFACT probe (trained on an orthogonal axis v) ===')",
         "c,n,g = make(v);  print(validate_probe(c, n, g).summary())"),
    md("Both probes hit ~1.0 in‑construct. Only the battery — via **natural‑OOD transfer (permutation‑tested)** "
       "and **cosine to the natural‑data direction** — exposes the second one as a surface artifact."),

    md("## 3. Reproduce it on a real model (needs a GPU runtime)",
       "",
       "Builds a template true/false construct, reads `gemma-2-2b-it` activations, and shows the 0.98 probe is "
       "orthogonal to the model's natural truth axis — the paper's result, live."),
    code("!pip install -q transformers accelerate",
         "import torch; from transformers import AutoModelForCausalLM, AutoTokenizer",
         "REPO='google/gemma-2-2b-it'; LAYER=14",
         "tok=AutoTokenizer.from_pretrained(REPO)",
         "model=AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float16, device_map='cuda', output_hidden_states=True).eval()",
         "def get_acts(texts):",
         "    out=[]",
         "    for t in texts:",
         "        ids=tok(t, return_tensors='pt').to('cuda')",
         "        with torch.no_grad(): hs=model(**ids).hidden_states",
         "        out.append(hs[LAYER][0,-1,:].float().cpu().numpy())",
         "    return np.array(out)"),
    code("# tiny template construct (true/false, balanced answer tokens) + natural OOD",
         "caps=[('France','Paris'),('Germany','Berlin'),('Spain','Madrid'),('Italy','Rome'),('Japan','Tokyo'),('Egypt','Cairo'),('Canada','Ottawa'),('Greece','Athens')]",
         "construct=[]",
         "for i,(c,a) in enumerate(caps):",
         "    wrong=caps[(i+1)%len(caps)][1]",
         "    construct.append({'text':f'The capital of {c} is {a}.','label':0,'group':i})",
         "    construct.append({'text':f'The capital of {c} is {wrong}.','label':1,'group':i})",
         "natural=[{'text':'Humans use only ten percent of their brains.','label':1},",
         "         {'text':'Octopuses have three hearts.','label':0},",
         "         {'text':'The Great Wall of China is visible from space with the naked eye.','label':1},",
         "         {'text':'Bananas are botanically classified as berries.','label':0},",
         "         {'text':'Bats are completely blind.','label':1},",
         "         {'text':'A day on Venus is longer than a year on Venus.','label':0},",
         "         {'text':'Lightning never strikes the same place twice.','label':1},",
         "         {'text':'Honey can remain edible for thousands of years.','label':0}]",
         "print(validate_probe(construct, natural, get_acts, perm_iters=500).summary())",
         "# (small n here — for a real verdict use the full 61-item OOD set from the repo)"),

    md("## 4. ▶ Use it on YOUR probe",
       "",
       "Drop in your own data and activation function. `construct_rows` = your probe's training/validation set; "
       "`natural_rows` = natural held-out examples of the same concept (the OOD test); `get_acts` = your model "
       "+ layer.",
       "",
       "Verdict logic: **VALID** = transfers to natural data (permutation p<0.05) AND its direction is aligned "
       "with the natural-data direction (cosine ≥ 0.5). **SURFACE-ARTIFACT** = high in-construct AUC but "
       "non-transferring / orthogonal. The `summary()` prints every diagnostic."),
    code("# construct_rows = [{'text': '<statement>', 'label': 0|1, 'group': <optional domain id>}, ...]",
         "# natural_rows   = [{'text': '<natural statement>', 'label': 0|1}, ...]",
         "# def get_acts(texts): return np.ndarray [len(texts), hidden_dim]   # your model, your layer",
         "#",
         "# report = validate_probe(construct_rows, natural_rows, get_acts)",
         "# print(report.summary())",
         "# report.as_dict()   # machine-readable"),

    md("---",
       "*If `validate_probe` flags your probe SURFACE-ARTIFACT, the concept axis may still exist — fit on "
       "natural data and re-check the cosine. Built by styxx / fathom-lab. We caught our own 0.98 truth-probe "
       "with this. MIT.*"),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")
