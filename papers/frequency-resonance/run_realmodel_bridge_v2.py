# -*- coding: utf-8 -*-
"""
run_realmodel_bridge_v2.py -- frozen by PREREG_realmodel_bridge_v2_2026_07_24.

THE REAL-MODEL RUNG, at a scale where the operation exists. v1 was uninterpretable (recall at ceiling,
compare at chance in 130M models). v2: parameter-matched 1.4B pair, a compare-SPECIFIC floor gate, and a
non-trivial recall control (three labelled codes, query one by name).

Mamba's A = -exp(A_log) is real & strictly negative (verified on loaded weights) -- pure decay, no phase,
i.e. the CLAMPED arm of our ablation shipped as a production LM. Attention is global. Prediction: a
distance-dependent deficit SPECIFIC to comparison, not recall.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = "--smoke" in sys.argv

MODELS = {"decay_ssm": "state-spaces/mamba-1.4b-hf", "attention": "EleutherAI/pythia-1.4b"}
DISTANCES = [16, 256] if SMOKE else [16, 64, 128, 256]
N_ITEMS = 20 if SMOKE else 200
SEED = 707
DTYPE = torch.float16

CODES = ["4271", "8395", "1638", "5024", "9713", "3186", "6450", "2879", "7104", "5567"]
NAMES = ["alpha", "bravo", "delta"]
FILLER = ("The weather report mentioned scattered clouds and a mild breeze across the valley. "
          "Local markets opened as usual and traffic moved steadily through the main streets. ")


def build_items(n, dist, seed):
    g = np.random.default_rng(seed)
    reps = max(1, dist // 12)
    filler = FILLER * (reps + 2)
    items = []
    for _ in range(n):
        codes = list(g.choice(CODES, size=3, replace=False))
        qi = int(g.integers(0, 3))
        qname, qcode = NAMES[qi], codes[qi]
        others = [c for j, c in enumerate(codes) if j != qi]
        stem = ("Here are three codes. " +
                " ".join(f"The {NAMES[j]} code is {codes[j]}." for j in range(3)) +
                " " + filler)
        claim_matches = bool(g.random() < 0.5)
        claimed = qcode if claim_matches else str(others[0])
        items.append({
            "q_code": str(qcode), "others": [str(o) for o in others],
            "claim_matches": claim_matches,
            "recall_prompt": stem + f"The {qname} code is",
            "compare_prompt": stem + f"Someone claims the {qname} code is {claimed}. That claim is",
        })
    return items


@torch.no_grad()
def score(model, tok, prompt, continuation):
    p = tok(prompt, return_tensors="pt").input_ids
    c = tok(continuation, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([p, c], 1).to(DEV)
    lg = model(ids).logits.float()
    lp = torch.log_softmax(lg[:, :-1], -1)
    sel = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0, -c.shape[1]:]
    return float(sel.mean())


def eval_model(tag, hf_id, items_by_dist, res):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"  loading {hf_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=DTYPE).to(DEV).eval()
    if tag == "decay_ssm":
        A = -torch.exp(model.backbone.layers[0].mixer.A_log.float())
        ok = (not A.is_complex()) and bool((A < 0).all())
        res["redteam"]["mamba_A_real_and_negative_on_weights"] = ok
        print(f"    [redteam] Mamba A real&negative: {ok} sample={[round(x,2) for x in A[0,:3].tolist()]}", flush=True)
    for dist, items in items_by_dist.items():
        rh, ch = 0, 0
        t0 = time.time()
        for it in items:
            lp_q = score(model, tok, it["recall_prompt"], " " + it["q_code"])
            lp_o = max(score(model, tok, it["recall_prompt"], " " + o) for o in it["others"])
            rh += int(lp_q > lp_o)
            lc = score(model, tok, it["compare_prompt"], " correct")
            li = score(model, tok, it["compare_prompt"], " incorrect")
            ch += int((lc > li) == it["claim_matches"])
        n = len(items)
        res["acc"][f"{tag}_recall_{dist}"] = round(rh / n, 4)
        res["acc"][f"{tag}_compare_{dist}"] = round(ch / n, 4)
        print(f"    {tag} D={dist:>3}: recall {rh/n:.3f}  compare {ch/n:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()


def main():
    print(f"device={DEV} smoke={SMOKE} distances={DISTANCES} n={N_ITEMS} dtype={DTYPE}", flush=True)
    items_by_dist = {d: build_items(N_ITEMS, d, SEED + d) for d in DISTANCES}
    for d, its in items_by_dist.items():
        assert abs(np.mean([i["claim_matches"] for i in its]) - 0.5) < 0.2, f"balance off D={d}"
        assert all(i["q_code"] not in FILLER for i in its), "fact leaks into filler"
    res = {"config": {"models": MODELS, "distances": DISTANCES, "n_items": N_ITEMS, "seed": SEED,
                      "chance": {"recall": 0.3333, "compare": 0.5}},
           "acc": {}, "redteam": {}}
    for tag, hf in MODELS.items():
        eval_model(tag, hf, items_by_dist, res)

    a, D0, D1 = res["acc"], DISTANCES[0], DISTANCES[-1]
    Cm0, Ct0 = a[f"decay_ssm_compare_{D0}"], a[f"attention_compare_{D0}"]
    Cm1, Ct1 = a[f"decay_ssm_compare_{D1}"], a[f"attention_compare_{D1}"]
    Rm0, Rm1 = a[f"decay_ssm_recall_{D0}"], a[f"decay_ssm_recall_{D1}"]
    Rt1 = a[f"attention_recall_{D1}"]
    between = (Ct1 - Cm1) - (Rt1 - Rm1)
    within = (Cm0 - Cm1) - (Rm0 - Rm1)
    floor_ok = min(Cm0, Ct0) >= 0.60                      # CORRECTED: compare-specific floor

    if not floor_ok:
        verdict = "ABSTAIN__comparison_absent_at_this_scale"
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
        "compare_floor_passed": bool(floor_ok), "verdict": verdict,
    }
    out = HERE / ("realmodel_bridge_v2_smoke.json" if SMOKE else "realmodel_bridge_v2_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  distances:        " + "  ".join(f"{d:>6}" for d in DISTANCES))
    for t in ("decay_ssm", "attention"):
        print(f"  {t:9} recall : " + "  ".join(f"{a[f'{t}_recall_{d}']:>6.3f}" for d in DISTANCES))
        print(f"  {t:9} compare: " + "  ".join(f"{a[f'{t}_compare_{d}']:>6.3f}" for d in DISTANCES))
    print(f"\n  between-model (compare-def - recall-def) @D={D1}: {between:+.3f}")
    print(f"  within-Mamba  (compare-decay - recall-decay):     {within:+.3f}")
    print("  ===== VERDICT:", verdict, "=====")
    print("  wrote", out.name, flush=True)


if __name__ == "__main__":
    main()
