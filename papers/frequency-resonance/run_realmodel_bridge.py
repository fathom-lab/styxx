# -*- coding: utf-8 -*-
"""
run_realmodel_bridge.py -- frozen by PREREG_realmodel_bridge_2026_07_24.

THE REAL-MODEL RUNG. Every prior result in this arc was a controlled SSM. Mamba's state matrix
A = -exp(A_log) is REAL and strictly negative -- pure decay, no phase -- i.e. the CLAMPED arm of our own
ablation, shipped as a production LM. A transformer's attention is global (no decay horizon). Our
mechanism therefore predicts a DISTANCE-DEPENDENT deficit SPECIFIC TO COMPARISON, not recall, for Mamba
vs a same-scale transformer.

Zero-shot likelihood scoring, no fine-tuning, matched prompts. RECALL (storage only) is the control;
COMPARE (relate a planted fact to a later claim) is the treatment.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = "--smoke" in sys.argv

MODELS = {"decay_ssm": "state-spaces/mamba-130m-hf", "attention": "EleutherAI/pythia-160m"}
DISTANCES = [16, 128] if SMOKE else [16, 64, 128, 256]
N_ITEMS = 24 if SMOKE else 200
SEED = 606

CODES = ["4271", "8395", "1638", "5024", "9713", "3186", "6450", "2879"]
FILLER = ("The weather report mentioned scattered clouds and a mild breeze across the valley. "
          "Local markets opened as usual and traffic moved steadily through the main streets. ")


def build_items(n, dist, seed):
    """Returns list of dicts with prompt variants for both families. Chance = 0.5 by construction."""
    g = np.random.default_rng(seed)
    items = []
    # filler sized in WORDS to approximate the token distance; trimmed per-tokenizer later
    reps = max(1, dist // 12)
    filler = (FILLER * (reps + 2))
    for i in range(n):
        true_code, distractor = g.choice(CODES, size=2, replace=False)
        claim_matches = bool(g.random() < 0.5)
        claimed = true_code if claim_matches else distractor
        stem = f"The secret code is {true_code}. {filler}"
        items.append({
            "true": str(true_code), "distractor": str(distractor),
            "claim_matches": claim_matches, "claimed": str(claimed),
            "recall_prompt": stem + "The secret code is",
            "compare_prompt": stem + f"Someone claims the secret code is {claimed}. That claim is",
        })
    return items


@torch.no_grad()
def score_continuation(model, tok, prompt, continuation):
    """Length-normalized log-likelihood of `continuation` given `prompt` (identical for both models)."""
    p_ids = tok(prompt, return_tensors="pt").input_ids
    c_ids = tok(continuation, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([p_ids, c_ids], dim=1).to(DEV)
    logits = model(ids).logits.float()
    logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[:, 1:]
    n_cont = c_ids.shape[1]
    sel = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0, -n_cont:]
    return float(sel.mean())            # length-normalized, same rule for both models


def eval_model(tag, hf_id, items_by_dist, res):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"  loading {hf_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(DEV).eval()
    # red-team assert 1: on the LOADED checkpoint, is the decay-SSM's A real and negative?
    if tag == "decay_ssm":
        A = -torch.exp(model.backbone.layers[0].mixer.A_log.float())
        real_neg = (not A.is_complex()) and bool((A < 0).all())
        res["redteam"]["mamba_A_real_and_negative_on_weights"] = real_neg
        print(f"    [redteam] loaded Mamba A real&negative: {real_neg} (sample {[round(x,2) for x in A[0,:3].tolist()]})", flush=True)
    for dist, items in items_by_dist.items():
        rec_hits, cmp_hits = 0, 0
        t0 = time.time()
        for it in items:
            # RECALL: true code should beat the distractor as the continuation
            lp_true = score_continuation(model, tok, it["recall_prompt"], " " + it["true"])
            lp_dist = score_continuation(model, tok, it["recall_prompt"], " " + it["distractor"])
            rec_hits += int(lp_true > lp_dist)
            # COMPARE: " correct" vs " incorrect" must match whether the claim was true
            lp_c = score_continuation(model, tok, it["compare_prompt"], " correct")
            lp_i = score_continuation(model, tok, it["compare_prompt"], " incorrect")
            pred_match = lp_c > lp_i
            cmp_hits += int(pred_match == it["claim_matches"])
        n = len(items)
        res["acc"][f"{tag}_recall_{dist}"] = round(rec_hits / n, 4)
        res["acc"][f"{tag}_compare_{dist}"] = round(cmp_hits / n, 4)
        print(f"    {tag} D={dist:>3}: recall {rec_hits/n:.3f}  compare {cmp_hits/n:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()


def main():
    print(f"device={DEV} smoke={SMOKE} distances={DISTANCES} n_items={N_ITEMS}", flush=True)
    items_by_dist = {d: build_items(N_ITEMS, d, SEED + d) for d in DISTANCES}
    # red-team assert 4/5: balance and non-inferability
    for d, its in items_by_dist.items():
        frac = np.mean([it["claim_matches"] for it in its])
        assert abs(frac - 0.5) < 0.18, f"claim balance off at D={d}: {frac}"
        assert all(it["true"] not in FILLER for it in its), "planted fact leaks into filler"
    res = {"config": {"models": MODELS, "distances": DISTANCES, "n_items": N_ITEMS, "seed": SEED},
           "acc": {}, "redteam": {}}
    for tag, hf_id in MODELS.items():
        eval_model(tag, hf_id, items_by_dist, res)

    D0, D1 = DISTANCES[0], DISTANCES[-1]
    a = res["acc"]
    Rm0, Cm0 = a[f"decay_ssm_recall_{D0}"], a[f"decay_ssm_compare_{D0}"]
    Rm1, Cm1 = a[f"decay_ssm_recall_{D1}"], a[f"decay_ssm_compare_{D1}"]
    Rt1, Ct1 = a[f"attention_recall_{D1}"], a[f"attention_compare_{D1}"]
    between = (Ct1 - Cm1) - (Rt1 - Rm1)          # compare-deficit minus recall-deficit, at longest D
    within = (Cm0 - Cm1) - (Rm0 - Rm1)           # Mamba's own compare-degradation minus recall-degradation
    floor_ok = max(Rm0, Cm0) >= 0.60 and max(a[f"attention_recall_{D0}"], a[f"attention_compare_{D0}"]) >= 0.60

    if not floor_ok:
        verdict = "ABSTAIN__probe_below_floor_in_these_small_models"
    elif between >= 0.10 and within >= 0.10:
        verdict = "SUPPORT__compare_specific_deficit_in_deployed_decay_ssm"
    elif between <= 0.03:
        verdict = "NULL__dissociation_does_not_transfer_to_deployed_models"
    else:
        verdict = "PARTIAL__reported_verbatim"

    res["result"] = {
        "between_model_compare_minus_recall_deficit": round(between, 4),
        "within_mamba_compare_minus_recall_degradation": round(within, 4),
        "mamba_recall": {str(d): a[f"decay_ssm_recall_{d}"] for d in DISTANCES},
        "mamba_compare": {str(d): a[f"decay_ssm_compare_{d}"] for d in DISTANCES},
        "attn_recall": {str(d): a[f"attention_recall_{d}"] for d in DISTANCES},
        "attn_compare": {str(d): a[f"attention_compare_{d}"] for d in DISTANCES},
        "floor_control_passed": bool(floor_ok), "verdict": verdict,
    }
    out = HERE / ("realmodel_bridge_smoke.json" if SMOKE else "realmodel_bridge_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  distances:      " + "  ".join(f"{d:>6}" for d in DISTANCES))
    for tag in ("decay_ssm", "attention"):
        print(f"  {tag:9} recall : " + "  ".join(f"{a[f'{tag}_recall_{d}']:>6.3f}" for d in DISTANCES))
        print(f"  {tag:9} compare: " + "  ".join(f"{a[f'{tag}_compare_{d}']:>6.3f}" for d in DISTANCES))
    print(f"\n  between-model (compare-deficit - recall-deficit) @D={D1}: {between:+.3f}")
    print(f"  within-Mamba  (compare-decay  - recall-decay):          {within:+.3f}")
    print("  ===== VERDICT:", verdict, "=====")
    print("  wrote", out.name, flush=True)


if __name__ == "__main__":
    main()
