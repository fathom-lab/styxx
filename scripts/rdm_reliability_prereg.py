# -*- coding: utf-8 -*-
"""Is representational reliability a validity channel?

Runs the frozen preregistration in
``papers/PREREG_rdm_reliability_error_predictor_2026_08_21.md``. Every parameter
below is the one committed before any result was seen; changing one makes this a
different experiment, not this one.

    python scripts/rdm_reliability_prereg.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch

# ── frozen parameters ──────────────────────────────────────────────────────
MODEL = "Qwen2.5-1.5B-Instruct"
MODEL_ID = f"Qwen/{MODEL}"
N_ITEMS = 500
SEED = 20260821
LAYER_FRAC = 0.75
N_SPLITS = 20
MAX_NEW = 24
POOL = "last"   # attempt 2, frozen: final prompt token, not mean-pooled
OUT = Path(__file__).resolve().parent.parent / "papers" / f"out_rdm_reliability_{POOL}_2026_08_21.json"

_norm_re = re.compile(r"[^a-z0-9 ]+")


def _norm(s: str) -> str:
    return _norm_re.sub(" ", (s or "").lower()).strip()


def load_items():
    from datasets import load_dataset
    ds = load_dataset("akariasai/popqa", split="test")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(ds), size=N_ITEMS, replace=False)
    items = []
    for i in idx:
        r = ds[int(i)]
        try:
            answers = json.loads(r["possible_answers"])
        except Exception:
            answers = [r["obj"]]
        items.append({
            "question": r["question"],
            "answers": [a for a in answers if a],
            "s_pop": float(r.get("s_pop") or 1.0),
        })
    return items


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    items = load_items()
    print(f"loaded {len(items)} PopQA items (seed {SEED})")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    n_layers = model.config.num_hidden_layers
    layer = int(LAYER_FRAC * n_layers)
    print(f"{MODEL}: {n_layers} layers, reading layer {layer}")

    reps, correct, conf_lp, conf_ent, conf_margin, lengths, degenerate = [], [], [], [], [], [], 0

    for n, it in enumerate(items):
        msgs = [{"role": "user",
                 "content": f"{it['question']} Answer with the name only, no sentence."}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to("cuda")
        lengths.append(int(enc.input_ids.shape[1]))

        with torch.no_grad():
            # representation of the QUESTION, before any answer exists
            hs = model(**enc, output_hidden_states=True).hidden_states[layer]
            vec = hs[0, -1] if POOL == "last" else hs[0].mean(0)
            reps.append(vec.float().cpu().numpy())

            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True,
                                 pad_token_id=tok.eos_token_id)

        new_ids = gen.sequences[0, enc.input_ids.shape[1]:]
        text = tok.decode(new_ids, skip_special_tokens=True)
        if not text.strip():
            degenerate += 1

        # baseline confidence, from the answer's own distribution
        lps, ents, margins = [], [], []
        for step, score in enumerate(gen.scores):
            if step >= len(new_ids):
                break
            probs = torch.softmax(score[0].float(), dim=-1)
            tokid = new_ids[step]
            lps.append(float(torch.log(probs[tokid] + 1e-12)))
            ents.append(float(-(probs * torch.log(probs + 1e-12)).sum()))
            top2 = torch.topk(probs, 2).values
            margins.append(float(top2[0] - top2[1]))
        conf_lp.append(float(np.mean(lps)) if lps else -20.0)
        conf_ent.append(float(np.mean(ents)) if ents else 10.0)
        conf_margin.append(float(np.mean(margins)) if margins else 0.0)

        g = _norm(text)
        correct.append(int(any(_norm(a) and _norm(a) in g for a in it["answers"])))

        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(items)}  acc so far {np.mean(correct):.3f}")

    R = np.stack(reps)                                    # (N, d)
    y = np.array(correct)
    acc = float(y.mean())
    print(f"\naccuracy {acc:.3f} | degenerate {degenerate} | {time.time()-t0:.0f}s")

    # ── per-item representational reliability (frozen: 20 splits) ─────────
    rng = np.random.default_rng(SEED)
    d = R.shape[1]
    Rz = (R - R.mean(0)) / (R.std(0) + 1e-8)

    def rdm(X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return 1.0 - (Xn @ Xn.T)

    from scipy.stats import spearmanr
    per_item = np.zeros((N_SPLITS, len(items)))
    for s in range(N_SPLITS):
        perm = rng.permutation(d)
        A, B = perm[: d // 2], perm[d // 2:]
        Da, Db = rdm(Rz[:, A]), rdm(Rz[:, B])
        for i in range(len(items)):
            mask = np.ones(len(items), dtype=bool)
            mask[i] = False
            per_item[s, i] = spearmanr(Da[i, mask], Db[i, mask]).correlation
    reliability = np.nanmean(per_item, axis=0)

    payload = {
        "model": MODEL, "n_items": len(items), "seed": SEED, "layer": layer,
        "pool": POOL,
        "n_layers": n_layers, "n_splits": N_SPLITS,
        "accuracy": acc, "degenerate": degenerate,
        "runtime_s": round(time.time() - t0, 1),
        "correct": y.tolist(),
        "reliability": reliability.tolist(),
        "conf_logprob": conf_lp, "conf_entropy": conf_ent,
        "conf_margin": conf_margin,
        "prompt_len": lengths,
        "s_pop": [it["s_pop"] for it in items],
    }
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
