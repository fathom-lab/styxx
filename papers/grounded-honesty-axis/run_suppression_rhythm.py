"""Is the truth-flash TRUTH-specific, and is the overwrite RHYTHMIC?
PREREG_suppression_rhythm_2026_05_29.md.

The spectral run's headline F3 (78% of confabs: correct token leads mid-network then is
overwritten late) had an untested control: maybe MANY tokens lead-then-lose, and the
correct token is nothing special. This run runs that control (D1: does the correct digit
lead more than MATCHED non-correct digits?) and characterizes the overwrite's layer
localization (D2 late-localized, D3 tight/rhythmic) -- the one falsifiable, non-circular
form of the "frequency/rhythm" question (global-beta-as-truth is closed, K-failed).

Ground truth is computed in-code (arithmetic) and hashed before scoring; correctness is
exact integer match (no judge). Greedy/deterministic -> reproduces the white-box answer key.

Usage:
    python papers/grounded-honesty-axis/run_suppression_rhythm.py --n 6   # pilot
    python papers/grounded-honesty-axis/run_suppression_rhythm.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_competence_cliff import SPECS, _eval, _expr, parse_int  # noqa: E402
from run_depth_grounding_whitebox import MODEL_NAME, DEVICE, generate  # noqa: E402

RECEIPT = HERE / "suppression_rhythm_result.json"


def single_digit_token_ids(tok):
    """Token ids for '0'..'9' that tokenize to exactly one token (no special tokens)."""
    out = {}
    for d in range(10):
        ids = tok(str(d), add_special_tokens=False).input_ids
        if len(ids) == 1:
            out[d] = ids[0]
    return out


@torch.no_grad()
def suppression_profile(model, tok, prompt_text, realized_text, correct_str, digit_ids):
    """At the first divergent answer position, return per-layer lead info.

    Returns dict or None (unalignable). Keys:
      crossed     : correct leads at some L<final AND realized wins at final (the F3 def)
      flip_layer  : last layer (index, <final) where correct is still ahead, or None
      n_layers    : L (number of layers incl. embeddings)
      digit_pos   : True iff BOTH correct & realized tokens are single-digit tokens
      corr_lead   : correct token leads realized at some layer < final (bool)
      distractors : list of (token_id, leads_bool) for matched non-correct single digits
    """
    plen = tok(prompt_text, return_tensors="pt").input_ids.shape[1]
    fids = tok(prompt_text + realized_text, return_tensors="pt").input_ids.to(DEVICE)
    T = fids.shape[1]
    if T <= plen:
        return None
    real_ids = fids[0, plen:T].tolist()
    corr_ids = tok(correct_str, add_special_tokens=False).input_ids
    div = next((i for i in range(min(len(real_ids), len(corr_ids)))
                if real_ids[i] != corr_ids[i]), None)
    if div is None:
        return None
    pos = plen - 1 + div
    out = model(fids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0)[:, 0, :, :]
    vec = model.model.norm(hs)[:, pos, :].float()          # (L, d)
    W = model.lm_head.weight.float()
    L = vec.shape[0]

    cid, rid = corr_ids[div], real_ids[div]
    lc = vec @ W[cid]                                      # (L,)
    lr = vec @ W[rid]                                      # (L,)
    lead = (lc > lr)                                       # correct ahead, per layer
    corr_lead = bool(lead[:-1].any().item())
    crossed = corr_lead and bool((lr[-1] >= lc[-1]).item())
    ahead_idx = torch.nonzero(lead[:-1], as_tuple=False).flatten()
    flip_layer = int(ahead_idx[-1].item()) if ahead_idx.numel() > 0 else None

    digit_token_set = set(digit_ids.values())
    digit_pos = (cid in digit_token_set) and (rid in digit_token_set)

    distractors = []
    if digit_pos:
        for tid in digit_token_set:
            if tid == cid or tid == rid:
                continue
            lx = vec @ W[tid]
            distractors.append((int(tid), bool((lx[:-1] > lr[:-1]).any().item())))
    return {"crossed": crossed, "flip_layer": flip_layer, "n_layers": L,
            "digit_pos": digit_pos, "corr_lead": corr_lead, "distractors": distractors}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(SPECS))
    args = ap.parse_args(argv)

    items = []
    for i, (form, delta, subset) in enumerate(SPECS[: args.n]):
        items.append((form, _eval(form), _expr(form), subset, i))

    key_blob = json.dumps([(e, c) for _, c, e, _, _ in items], ensure_ascii=False)
    key_hash = hashlib.sha256(key_blob.encode("utf-8")).hexdigest()
    print(f"answer-key SHA-256 (pre-scoring): {key_hash}")
    print(f"model={MODEL_NAME} device={DEVICE} items={len(items)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16).to(DEVICE).eval()
    digit_ids = single_digit_token_ids(tok)
    print(f"model loaded; single-digit token ids: "
          f"{ {d: i for d, i in digit_ids.items()} }\n")

    rows = []
    for form, correct, expr, subset, idx in items:
        p1, a1 = generate(model, tok, "Answer with only the final number, nothing else.",
                          f"What is {expr}?", max_new_tokens=16)
        v1 = parse_int(a1); ok1 = (v1 == correct)
        prof = None
        if not ok1 and v1 is not None:
            prof = suppression_profile(model, tok, p1, a1, str(correct), digit_ids)
        rows.append({"subset": subset, "expr": expr, "correct": correct,
                     "v1": v1, "ok1": ok1, "prof": prof})
        fl = prof["flip_layer"] if prof else None
        dp = prof["digit_pos"] if prof else None
        cr = prof["crossed"] if prof else None
        print(f"[{idx:2d}|{subset:9}] {expr:>14}={correct:<9} | "
              f"1shot {str(v1):>9} {'OK ' if ok1 else 'BAD'} | "
              f"cross={cr} flip={fl} digit_pos={dp}")

    # ---- scoring ----
    profs = [r["prof"] for r in rows if r["prof"] is not None]
    L = profs[0]["n_layers"] if profs else 29

    # D1: truth-specificity of the flash vs matched non-correct digits (digit positions).
    dpos = [p for p in profs if p["digit_pos"] and p["distractors"]]
    per_item = []   # (correct_lead 0/1, mean_distractor_lead) for paired test
    for p in dpos:
        c = 1.0 if p["corr_lead"] else 0.0
        dl = float(np.mean([1.0 if leads else 0.0 for _, leads in p["distractors"]]))
        per_item.append((c, dl))
    if len(per_item) >= 8:
        cvec = np.array([c for c, _ in per_item])
        dvec = np.array([d for _, d in per_item])
        d1_corr_rate = float(cvec.mean())
        d1_dist_rate = float(dvec.mean())
        d1_delta = d1_corr_rate - d1_dist_rate
        d1_t, d1_p = stats.ttest_rel(cvec, dvec)
        d1 = (d1_delta >= 0.20) and (d1_p < 0.05) and (d1_delta > 0)
        d1_powered = True
    else:
        d1_corr_rate = d1_dist_rate = d1_delta = d1_p = float("nan")
        d1 = False; d1_powered = False

    # D2/D3: layer-localization of the overwrite among genuine crossings.
    flips = [p["flip_layer"] for p in profs if p["crossed"] and p["flip_layer"] is not None]
    late_thresh = int(np.ceil(2 * (L - 1) / 3))   # 2L/3 on 0..L-1 index scale
    if flips:
        fa = np.array(flips, float)
        d2_late_frac = float(np.mean(fa >= late_thresh))
        d2 = d2_late_frac >= 0.60
        d3_iqr = float(np.percentile(fa, 75) - np.percentile(fa, 25))
        d3 = d3_iqr <= 5
        flip_median = float(np.median(fa))
    else:
        d2_late_frac = d3_iqr = flip_median = float("nan")
        d2 = d3 = False

    n_confab = sum(1 for r in rows if not r["ok1"])
    n_align = len(profs)
    cross_rate = (float(np.mean([1.0 if p["crossed"] else 0.0 for p in profs]))
                  if profs else float("nan"))

    receipt = {
        "experiment": "truth-specificity of the flash + layer-frequency of the overwrite",
        "prereg": "papers/grounded-honesty-axis/PREREG_suppression_rhythm_2026_05_29.md",
        "answer_key_sha256_pre_scoring": key_hash,
        "model": MODEL_NAME, "device": DEVICE, "n_items": len(rows),
        "core_signal": "exact integer parse vs in-code arithmetic truth (no judge)",
        "n_oneshot_confab": n_confab, "n_alignable_confab": n_align,
        "replicated_crossing_rate": round(cross_rate, 4) if cross_rate == cross_rate else None,
        "n_layers": L, "late_threshold_layer_index": late_thresh,
        "D1_truth_specific_flash": {
            "n_digit_pos_confab": len(per_item),
            "correct_digit_lead_rate": round(d1_corr_rate, 4) if d1_corr_rate == d1_corr_rate else None,
            "matched_distractor_lead_rate": round(d1_dist_rate, 4) if d1_dist_rate == d1_dist_rate else None,
            "delta": round(d1_delta, 4) if d1_delta == d1_delta else None,
            "paired_p": round(float(d1_p), 5) if d1_p == d1_p else None,
            "powered": d1_powered},
        "D2_late_localized": {
            "n_crossings": len(flips),
            "late_fraction": round(d2_late_frac, 4) if d2_late_frac == d2_late_frac else None,
            "median_flip_layer": flip_median if flip_median == flip_median else None},
        "D3_rhythmic_tight_band": {
            "flip_layer_iqr": round(d3_iqr, 4) if d3_iqr == d3_iqr else None},
        "rows": rows,
        "D1": bool(d1), "D1_powered": d1_powered, "D2": bool(d2), "D3": bool(d3),
        "RESULT": ("SURVIVED" if (d1 and d2) else "REPORT_AS_LANDED"),
        "honest_scope": (
            "single open model Qwen2.5-1.5B-Instruct, SAE-free full-vocab logit-lens, "
            "feasibility-grade n=36, one confirmatory run; arithmetic ground truth computed "
            "in-code then hashed pre-scoring; exact-integer correctness (no judge). Tests "
            "(a) whether the F3 truth-flash is specific to the correct token vs matched digit "
            "distractors and (b) the layer-localization/consistency of the overwrite. A "
            "logit-lens phenomenon on the realized run; NOT a causal demonstration that "
            "dampening the overwrite recovers the answer (the named disinhibition test, which "
            "D2/D3 would target). A D1 null deflates F3's truth-specificity, reported as such."),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2))
    print(f"\nD1={d1}(powered={d1_powered} delta={d1_delta:.3f} p={d1_p:.4f}) "
          f"D2={d2}(late {d2_late_frac:.2f}) D3={d3}(iqr {d3_iqr:.1f}) -> {receipt['RESULT']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
