# -*- coding: utf-8 -*-
"""
measure_ssm_contrast.py — the DECISIVE rhythm experiment: does a concept signal s(t)
oscillate during generation in a TRANSFORMER (no recurrence over steps) vs a STATE-SPACE
model (Mamba — genuine recurrence, complex-eigenvalue state rotation)?

Brains oscillate; SSMs *can* (complex eigenvalues); transformers *can't* (no state
recursion over tokens). If the SSM signal oscillates and the transformer's only commits,
that isolates the architectural origin of rhythm in a machine mind.

Fairness by design: SAME diff-in-means concept direction method, SAME prompts, SAME T,
SAME spectral analysis on both. We generate greedily, then read every per-position hidden
state in ONE forward pass — for a causal model that equals the autoregressive value, so
the ONLY thing differing between the two arms is the architecture.

Output: ssm_contrast_raw.json (analyzed by analyze_ssm_contrast.py with the validated
spectral battery from analyze_concept_dynamics.py).
"""
from __future__ import annotations

import gc, json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "representational-convergence"))
sys.path.insert(0, str(HERE.parent.parent / "scripts" / "dogfood"))
from convergence_eval_set import get_convergence_eval
try:
    from aggressive_borderline_eval_set import get_aggressive_borderlines
except Exception:
    get_aggressive_borderlines = None

ARCHS = [("transformer", "Qwen/Qwen2.5-1.5B-Instruct"),
         ("ssm", "state-spaces/mamba-1.4b-hf")]
T = 64
N_RAND = 8
SEED = 0


def format_ids(tok, prompt):
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, return_tensors="pt")
    return tok(prompt, return_tensors="pt").input_ids


def last_resid(mdl, ids, layer):
    with torch.no_grad():
        o = mdl(input_ids=ids.to(mdl.device), output_hidden_states=True)
    return o.hidden_states[layer][0, -1, :].float().cpu().numpy()


def main():
    conv = get_convergence_eval()
    harm = [p for (_, c, pol, p) in conv if c == "comply_refuse" and pol == 1]   # 6 harmful
    benign = [p for (_, c, pol, p) in conv if c == "comply_refuse" and pol == 0]  # 6 benign
    dyn_prompts = list(harm)
    if get_aggressive_borderlines:
        dyn_prompts += [p for (_, _, p) in get_aggressive_borderlines()[:14]]      # +14 -> 20
    rng = np.random.default_rng(SEED)
    out = {"T": T, "n_rand": N_RAND, "dir_prompts": {"harmful": len(harm), "benign": len(benign)},
           "n_dyn": len(dyn_prompts), "archs": {}}

    for arch, mid in ARCHS:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to("cuda").eval()
            eos = tok.eos_token_id; eos = eos[0] if isinstance(eos, list) else eos
            pad = tok.pad_token_id if tok.pad_token_id is not None else eos
            nL = mdl.config.num_hidden_layers
            layer = nL // 2

            # diff-in-means concept direction (last-prompt-token residual)
            Hh = np.stack([last_resid(mdl, format_ids(tok, p), layer) for p in harm])
            Hb = np.stack([last_resid(mdl, format_ids(tok, p), layer) for p in benign])
            v = Hh.mean(0) - Hb.mean(0)
            v = v / (np.linalg.norm(v) + 1e-9)
            vt = torch.tensor(v, dtype=torch.float16, device=mdl.device)
            R = torch.tensor(rng.standard_normal((N_RAND, len(v))), dtype=torch.float16, device=mdl.device)
            R = R / R.norm(dim=1, keepdim=True)

            traj = []
            for p in dyn_prompts:
                ids = format_ids(tok, p).to(mdl.device)
                plen = ids.shape[1]
                with torch.no_grad():
                    g = mdl.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=T,
                                     do_sample=False, pad_token_id=pad)
                    o = mdl(input_ids=g, output_hidden_states=True)
                h = o.hidden_states[layer][0]                      # [full_len, hidden]
                s = (h @ vt).float().cpu().numpy()                 # projection per position
                rs = (h @ R.T).float().cpu().numpy()               # [full_len, N_RAND]
                # generation span: from last prompt token through generated tokens
                sig = s[plen - 1:].round(4).tolist()
                rsig = rs[plen - 1:].round(4).tolist()
                traj.append({"prompt": p[:80], "gen_len": g.shape[1] - plen,
                             "sig": sig, "rand": rsig})
            out["archs"][arch] = {"status": "ok", "model": mid, "layer": layer, "n_layers": nL,
                                  "traj": traj}
            print(f"{arch} ({mid.split('/')[-1]}): {len(traj)} trajectories, layer {layer}/{nL}", flush=True)
            del mdl, tok; gc.collect(); torch.cuda.empty_cache()
            (HERE / "ssm_contrast_raw.json").write_text(json.dumps(out), encoding="utf-8")
        except Exception as e:
            out["archs"][arch] = {"status": "error", "model": mid, "error": repr(e)[:400]}
            print(f"{arch} ({mid}): ERROR {e!r}", flush=True)
            try:
                del mdl, tok
            except Exception:
                pass
            gc.collect(); torch.cuda.empty_cache()

    (HERE / "ssm_contrast_raw.json").write_text(json.dumps(out), encoding="utf-8")
    print("wrote ssm_contrast_raw.json")


if __name__ == "__main__":
    main()
