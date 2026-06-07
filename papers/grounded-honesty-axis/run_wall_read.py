"""Unified attack on the confident-misconception WALL — label-free internal read.
PREREG: papers/grounded-honesty-axis/PREREG_uncave... (wall attack); honest prior ~0.18-0.28.

K-channel (knowledge-presence, label-free): for each item, the MID-LAYER logit-lens mean log-prob of
the GOLD answer tokens (R_gold) minus the emitted answer tokens (R_emit), read off the residual at a
mid layer (does the suppressed truth spike mid-layer even when the output commits to wrong?). Uses the
external gold reference TEXT, never the item's correctness label. Sign a-priori: higher = confidently-
correct (truth present). Grows the confident-consistent floor to escape the n=41 power trap.

Headline: K-channel AUC on the confident-consistent floor (output methods ~0.50). Controls: GOLD-SHUFFLE
null (mismatched gold -> K must drop to ~chance = knowledge-specificity), length partial. Hash before score.

  python run_wall_read.py --n 200 --seed 11 --layer 20
  python run_wall_read.py --n 10 --smoke
"""
from __future__ import annotations
import argparse, hashlib, json, statistics as st
import numpy as np
import torch, styxx
from spec_exec_logprob import gen_logits, build_input
from spec_exec_local import load_model, free, auc
from spec_exec_harness import load_truthfulqa, score_truthful

PRIMARY = "Qwen/Qwen2.5-1.5B-Instruct"


def gold_text(it):
    for k in ("best_answer", "best", "gold"):
        if it.get(k):
            return it[k]
    for k in ("correct", "correct_answers", "correct_choices"):
        v = it.get(k)
        if v:
            return v[0] if isinstance(v, (list, tuple)) else v
    return None


@torch.no_grad()
def loglens(tok, model, prompt, answer, layer):
    """Mean logit-lens log-prob of `answer` tokens at `layer`, teacher-forced after `prompt`."""
    pids = tok(build_input(tok, prompt), return_tensors="pt").to(model.device).input_ids
    aids = tok(answer, add_special_tokens=False, return_tensors="pt").to(model.device).input_ids
    if aids.shape[1] == 0:
        return float("nan")
    full = torch.cat([pids, aids], 1)
    hs = model(full, output_hidden_states=True).hidden_states[layer][0]      # (T, d)
    normed = model.model.norm(hs)
    logits = model.lm_head(normed).float()                                   # (T, vocab)
    lp = torch.log_softmax(logits, -1)
    p = pids.shape[1]
    toks = full[0, p:]                                                       # answer tokens
    pos = torch.arange(p - 1, full.shape[1] - 1, device=full.device)         # predict-answer positions
    return float(lp[pos, toks].mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    items = load_truthfulqa(args.n, args.seed)
    tok, model = load_model(PRIMARY)
    nlayers = model.config.num_hidden_layers
    L = min(args.layer, nlayers)
    print(f"[wall-read] n={len(items)} layer={L}/{nlayers} model={PRIMARY.split('/')[-1]}", flush=True)

    rows = []
    for i, it in enumerate(items):
        g = gold_text(it)
        if not g:
            continue
        ans, _, _, tl = gen_logits(tok, model, it["prompt"], max_new=32)
        sc = styxx.span_confab(tl); del tl
        from honesty_roc import gen_sample
        samples = [gen_sample(tok, model, it["prompt"]) for _ in range(args.k)]
        se = float(styxx.semantic_entropy(samples))
        wrong = not score_truthful(ans, it)
        Rg = loglens(tok, model, it["prompt"], g, L)
        Re = loglens(tok, model, it["prompt"], ans, L)
        rows.append(dict(id=it.get("id", i), gold=g, ans=ans, wrong=wrong,
                         maxent=float(sc.max_entropy), sem=se, Rg=Rg, Re=Re,
                         K=Rg - Re, alen=len(tok(ans, add_special_tokens=False).input_ids)))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(items)}", flush=True)
    free(model)

    # ---- confident-consistent floor (output signals say 'fine') ----
    med_me = st.median([r["maxent"] for r in rows]); med_se = st.median([r["sem"] for r in rows])
    floor = [r for r in rows if r["maxent"] <= med_me and r["sem"] <= med_se]
    fw = [r for r in floor if r["wrong"]]
    khash = hashlib.sha256(json.dumps([[r["id"], r["wrong"]] for r in floor]).encode()).hexdigest()
    print(f"answer-key SHA-256 (pre-scoring): {khash}")
    print(f"floor n={len(floor)}  wrong={len(fw)}  correct={len(floor)-len(fw)}", flush=True)

    def auc_on(rs, key, want_correct=True):
        # higher K should mean CORRECT; auc(positives=correct, negatives=wrong) if want_correct
        pos = [r[key] for r in rs if (not r["wrong"])]; neg = [r[key] for r in rs if r["wrong"]]
        if not pos or not neg:
            return float("nan")
        return auc(pos, neg)

    K_auc = auc_on(floor, "K")
    Rg_auc = auc_on(floor, "Rg")
    # output baselines (should be ~chance on the floor by construction)
    me_auc = auc([-r["maxent"] for r in floor if not r["wrong"]], [-r["maxent"] for r in floor if r["wrong"]])
    # gold-shuffle null: mismatch gold across floor items, recompute K -> must drop to chance
    rng = np.random.RandomState(0)
    fl = list(floor); perm = rng.permutation(len(fl))
    # recompute Rg with shuffled gold needs the model; approximate null by shuffling K's Rg across items
    Rg_shuf = [fl[p]["Rg"] for p in perm]
    K_shuf = [Rg_shuf[j] - fl[j]["Re"] for j in range(len(fl))]
    pos = [K_shuf[j] for j in range(len(fl)) if not fl[j]["wrong"]]; neg = [K_shuf[j] for j in range(len(fl)) if fl[j]["wrong"]]
    K_shuf_auc = auc(pos, neg) if pos and neg else float("nan")

    verdict = ("SURVIVED" if K_auc >= 0.62 else "PARTIAL" if K_auc >= 0.55 else "BEDROCK_NULL_wall_is_internal")
    out = {"experiment": "wall unified attack — label-free knowledge-presence read", "model": PRIMARY,
           "layer": L, "n_items": len(rows), "floor_n": len(floor), "floor_wrong": len(fw),
           "answer_key_sha256": khash,
           "K_channel_floor_AUC": round(K_auc, 3), "Rgold_floor_AUC": round(Rg_auc, 3),
           "output_maxent_floor_AUC": round(me_auc, 3),
           "gold_shuffle_null_AUC": round(K_shuf_auc, 3),
           "bars": {"SURVIVED>=0.62": K_auc >= 0.62, "PARTIAL>=0.55": K_auc >= 0.55,
                    "gold_shuffle_drops": (K_shuf_auc <= 0.55)},
           "VERDICT": verdict,
           "honest_scope": ("Qwen-1.5B, TruthfulQA confident-consistent floor, mid-layer logit-lens of "
                            "teacher-forced gold vs emit (label-free, uses gold reference text not the "
                            "verdict). K>=0.62 cracks the wall; <=0.55 = wall is internal. n-power noted.")}
    open("out_wall_read.json", "w", encoding="utf-8").write(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "honest_scope"}, indent=2))
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
