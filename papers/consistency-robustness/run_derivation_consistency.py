# -*- coding: utf-8 -*-
"""
run_derivation_consistency.py — frozen by PREREG_derivation_consistency_2026_06_04.

A new orthogonal readout: what the model DERIVES (step-by-step) vs what it RETRIEVES (quick answer).
A quick answer that doesn't survive reasoning = confabulation. Tests whether reasoning breaks the
retrieval-consistency floor and decomposes it into shared-ignorance (fixable) vs irreducible.
"""
from __future__ import annotations
import sys, os, json, re
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = "--smoke" in sys.argv
MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LETTERS = ["A", "B", "C", "D"]
N_ITEMS = 12 if SMOKE else 100
SYS_D = "You are answering multiple-choice questions. Reply with ONLY the single letter (A, B, C, or D)."
SYS_C = "You are answering multiple-choice questions. Reason step by step BRIEFLY, then end with a line `Answer: <letter>`."
# derivation-vs-retrieval is meaningful for KNOWLEDGE questions; skip computation-heavy subjects where a
# 3B model just can't compute (runaway reasoning, not a retrieval/derivation distinction).
EXCLUDE = ("math", "algebra", "physics", "chemistry", "computer", "engineering", "statistics",
           "econometrics", "accounting")


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    p, n = s[y == 1], s[y == 0]
    return float(sum((a > b) + 0.5 * (a == b) for a in p for b in n) / (len(p) * len(n))) if len(p) and len(n) else float("nan")


def parse_letter(text):
    m = re.findall(r"[Aa]nswer\s*(?:is|:)?\s*\(?\*?\*?([A-D])\b", text)
    if m:
        return m[-1].upper()
    m2 = re.findall(r"\b([A-D])\b", text)
    return m2[-1].upper() if m2 else None


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    np.random.seed(0); torch.set_grad_enabled(False)
    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
    lids = [tok(L, add_special_tokens=False).input_ids[0] for L in LETTERS]

    def chat(s, u):
        return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}],
                                       tokenize=False, add_generation_prompt=True)

    def direct(u):
        ids = tok(chat(SYS_D, u), return_tensors="pt").input_ids.to(DEV)
        lg = model(ids).logits[0, -1, lids].float().cpu().numpy()
        return LETTERS[int(lg.argmax())]

    def cot(u):
        ids = tok(chat(SYS_C, u), return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids, max_new_tokens=640, do_sample=False, pad_token_id=tok.eos_token_id)
        return parse_letter(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))

    ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
    rows, parse_fail = [], 0
    for ex in ds:
        if len(ex["choices"]) != 4:
            continue
        if any(k in ex["subject"] for k in EXCLUDE):
            continue
        gold = LETTERS[int(ex["answer"])]
        u = ex["question"] + "\n" + "\n".join(f"{L}) {c}" for L, c in zip(LETTERS, ex["choices"])) + "\nAnswer with a single letter."
        d = direct(u); c = cot(u)
        if c is None:
            parse_fail += 1; c = "?"
        rows.append({"gold": gold, "direct": d, "cot": c})
        if len(rows) % 10 == 0:
            print(f"  {len(rows)}: gold={gold} direct={d} cot={c}", flush=True)
        if len(rows) >= N_ITEMS:
            break

    dca = float(np.mean([r["direct"] == r["gold"] for r in rows]))
    cca = float(np.mean([r["cot"] == r["gold"] for r in rows]))
    dwrong = [1 if r["direct"] != r["gold"] else 0 for r in rows]
    disagree = [1 if r["direct"] != r["cot"] else 0 for r in rows]
    p2_auc = auroc(dwrong, disagree)
    wrong = [r for r in rows if r["direct"] != r["gold"]]
    cot_fixes = float(np.mean([r["cot"] == r["gold"] for r in wrong])) if wrong else float("nan")
    residual_floor = float(np.mean([(r["direct"] != r["gold"] and r["cot"] != r["gold"] and r["cot"] != "?") for r in rows]))

    p1 = (cca - dca) >= 0.05
    p2 = p2_auc >= 0.65
    p3 = (cot_fixes >= 0.30) if not np.isnan(cot_fixes) else False
    if p1 and p2 and p3:
        verdict = (f"DERIVATION IS A FLOOR-BREAKING READOUT — reasoning lifts accuracy {dca:.2f}->{cca:.2f}; "
                   f"direct-vs-derived disagreement flags confabulation (AUROC {p2_auc:.2f}); and reasoning "
                   f"CORRECTS {cot_fixes:.0%} of the quick-answer errors. The retrieval 'floor' is largely shared "
                   f"IGNORANCE, breakable by an orthogonal readout; the residual (cot also wrong) is {residual_floor:.0%} "
                   f"= the harder floor that still needs external truth.")
    else:
        verdict = (f"PARTIAL — P1(reasoning helps {cca-dca:+.2f}):{p1}, P2(disagreement flags {p2_auc:.2f}):{p2}, "
                   f"P3(cot fixes {cot_fixes:.0%}):{p3}. Floor decomposes: reasoning fixes {cot_fixes:.0%} of direct errors, "
                   f"residual floor {residual_floor:.0%}.")

    res = {"model": MODEL, "n": len(rows), "direct_acc": dca, "cot_acc": cca,
           "derivation_disagreement_auroc": p2_auc, "cot_fixes_direct_errors": cot_fixes,
           "residual_floor_both_wrong": residual_floor, "parse_fail": parse_fail,
           "P1_reasoning_helps": bool(p1), "P2_disagreement_flags": bool(p2), "P3_floor_decomposes": bool(p3),
           "verdict": verdict}
    out = os.path.join(HERE, "derivation_consistency_smoke.json" if SMOKE else "derivation_consistency_result.json")
    json.dump(res, open(out, "w"), indent=2)
    print("\n=== RESULT ===")
    print(f"  accuracy: direct {dca:.3f} -> CoT {cca:.3f}  (P1 reasoning helps: {p1})")
    print(f"  derivation-disagreement AUROC (flags direct-wrong): {p2_auc:.3f}  (P2: {p2})")
    print(f"  reasoning corrects {cot_fixes:.1%} of direct errors | residual floor {residual_floor:.1%} | parse-fail {parse_fail}")
    print("\n===== " + verdict)
    print("wrote", os.path.basename(out))


if __name__ == "__main__":
    main()
