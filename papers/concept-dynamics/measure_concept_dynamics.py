# -*- coding: utf-8 -*-
"""
measure_concept_dynamics.py — collect the raw CONCEPT-SIGNAL TIME SERIES s(t) as an LLM
generates, token by token.

For each (model, prompt): autoregressive GREEDY generation for T steps (KV-cached). At
each step we read the residual stream and project it onto every trained concept
direction (at that concept's probe layer) -> s_c(t), and onto K random unit directions
at each probe layer -> the NULL. We store the raw trajectories + generated text + the
eos position, for OFFLINE spectral analysis.

This is the MEASUREMENT engine (GPU, expensive). The analysis (trend, autocorrelation,
spectral peak vs AR/phase-randomized surrogates, the gate) lives in a separate script so
the null models can be refined without re-generating. Greedy = deterministic, so the raw
signal is reproducible.

Reuses the concept-balanced prompts from the convergence study (48 prompts, 4 concepts).
"""
from __future__ import annotations

import gc, json, os, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from styxx.residual_probe import StyxxProbe, ProbeNotAvailable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "representational-convergence"))
from convergence_eval_set import get_convergence_eval

MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
          "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
          "microsoft/Phi-3.5-mini-instruct", "google/gemma-2-2b-it"]
CONCEPTS = ["comply_refuse", "deception", "corrigibility", "truthfulness"]
T = 64            # generation steps (uniform length; we generate past eos and record eos_pos)
N_RAND = 8        # random null directions
SEED = 0


def is_cached(mid):
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    return (hub / ("models--" + mid.replace("/", "--"))).exists()


def main():
    rows = get_convergence_eval()                       # (id, concept, polarity, prompt)
    out = {"T": T, "n_rand": N_RAND, "seed": SEED, "concepts": CONCEPTS,
           "prompts": [(r[0], r[1], r[2]) for r in rows], "models": {}}
    rng = np.random.default_rng(SEED)

    for mid in MODELS:
        short = mid.split("/")[-1]
        if not is_cached(mid):
            out["models"][mid] = {"status": "skip_uncached"}; continue
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to("cuda").eval()
            eos = tok.eos_token_id; eos = eos[0] if isinstance(eos, list) else eos

            probes = {}
            for c in CONCEPTS:
                try:
                    p = StyxxProbe.from_pretrained(model=mid, task=c)
                    p.weight = p.weight.to("cuda", dtype=torch.float16)
                    probes[c] = p
                except ProbeNotAvailable:
                    pass
            if not probes:
                out["models"][mid] = {"status": "no_probes"}; del mdl; gc.collect(); torch.cuda.empty_cache(); continue

            layers = sorted({p.layer for p in probes.values()})
            hsz = mdl.config.hidden_size
            rand = torch.tensor(rng.standard_normal((N_RAND, hsz)), dtype=torch.float16, device="cuda")
            rand = rand / rand.norm(dim=1, keepdim=True)

            rec = {"status": "ok", "concept_layers": {c: probes[c].layer for c in probes},
                   "rand_layers": layers, "traj": []}

            for pid, con, pol, prompt in rows:
                ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                              add_generation_prompt=True, return_tensors="pt").to("cuda")
                sig = {c: [] for c in probes}
                rsig = {L: [] for L in layers}             # rsig[L][t] = [N_RAND]

                def read(o):
                    hs = o.hidden_states
                    for c, p in probes.items():
                        h = hs[p.layer][0, -1, :]
                        sig[c].append(float((h @ p.weight).item() + p.bias))
                    for L in layers:
                        hr = hs[L][0, -1, :]
                        rsig[L].append((rand @ hr).float().cpu().numpy().round(4).tolist())

                with torch.no_grad():
                    o = mdl(input_ids=ids, output_hidden_states=True, use_cache=True)
                    past = o.past_key_values
                    read(o)
                    nxt = o.logits[0, -1].argmax().view(1, 1)
                    gen = [int(nxt)]
                    eos_pos = -1
                    for t in range(T):
                        if int(nxt) == eos and eos_pos < 0:
                            eos_pos = t
                        o = mdl(input_ids=nxt, past_key_values=past, output_hidden_states=True, use_cache=True)
                        past = o.past_key_values
                        read(o)
                        nxt = o.logits[0, -1].argmax().view(1, 1)
                        gen.append(int(nxt))

                rec["traj"].append({"pid": pid, "concept": con, "polarity": pol, "eos_pos": eos_pos,
                                    "sig": {c: [round(x, 4) for x in sig[c]] for c in sig},
                                    "rand": {str(L): rsig[L] for L in layers},
                                    "text": tok.decode(gen, skip_special_tokens=True)[:300]})
            out["models"][mid] = rec
            del mdl, tok; gc.collect(); torch.cuda.empty_cache()
            (HERE / "concept_dynamics_raw.json").write_text(json.dumps(out), encoding="utf-8")
            print(f"{short}: {len(rec['traj'])} trajectories, layers={layers}", flush=True)
        except Exception as e:
            out["models"][mid] = {"status": "error", "error": repr(e)[:300]}
            print(f"{short}: ERROR {e!r}", flush=True)
            try:
                del mdl, tok
            except Exception:
                pass
            gc.collect(); torch.cuda.empty_cache()

    (HERE / "concept_dynamics_raw.json").write_text(json.dumps(out), encoding="utf-8")
    print("wrote concept_dynamics_raw.json")


if __name__ == "__main__":
    main()
