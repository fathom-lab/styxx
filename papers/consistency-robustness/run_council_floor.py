# -*- coding: utf-8 -*-
"""
run_council_floor.py — frozen by PREREG_council_floor_2026_06_04. The bottom of the defense-in-depth
stack. 3 models / 2 vendors answer the same MMLU items; council_support(claim) = fraction of models
that answer it. Tests: does cross-model agreement separate truth from a tempting lie (P1), cover
single-model error (P2), and what is the irreducible floor of SHARED confabulations (P3)?
"""
from __future__ import annotations
import sys, os, json
from collections import Counter
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

SMOKE = "--smoke" in sys.argv
MODELS = ["Qwen/Qwen2.5-3B-Instruct", "google/gemma-2-2b-it", "Qwen/Qwen2.5-1.5B-Instruct"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LETTERS = ["A", "B", "C", "D"]
N_ITEMS = 12 if SMOKE else 100
SYS_H = "You are answering multiple-choice questions. Reply with ONLY the single letter (A, B, C, or D) of the correct option."


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    p, n = s[y == 1], s[y == 0]
    return float(sum((a > b) + 0.5 * (a == b) for a in p for b in n) / (len(p) * len(n))) if len(p) and len(n) else float("nan")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    np.random.seed(0); torch.set_grad_enabled(False)

    # ---- fixed item set (gold) ----
    ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
    items = []
    for ex in ds:
        if len(ex["choices"]) == 4:
            items.append({"q": ex["question"], "choices": ex["choices"], "gold": LETTERS[int(ex["answer"])]})
        if len(items) >= N_ITEMS:
            break

    def run_model(name, first):
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
        lids = [tok(L, add_special_tokens=False).input_ids[0] for L in LETTERS]

        def chat(s, u):
            try:
                return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}],
                                               tokenize=False, add_generation_prompt=True)
            except Exception:
                return tok.apply_chat_template([{"role": "user", "content": s + "\n\n" + u}],
                                               tokenize=False, add_generation_prompt=True)
        for it in items:
            u = it["q"] + "\n" + "\n".join(f"{L}) {c}" for L, c in zip(LETTERS, it["choices"])) + "\nAnswer with a single letter."
            ids = tok(chat(SYS_H, u), return_tensors="pt").input_ids.to(DEV)
            lg = model(ids).logits[0, -1, lids].float().cpu().numpy()
            it.setdefault("answers", []).append(LETTERS[int(lg.argmax())])
            if first:
                it["lie"] = LETTERS[int(np.argsort(lg)[::-1][1])]      # runner-up = tempting wrong option
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()
        print(f"  ran {name}", flush=True)

    for i, name in enumerate(MODELS):
        run_model(name, first=(i == 0))

    # ---- council metrics ----
    def support(it, claim):
        return float(np.mean([a == claim for a in it["answers"]]))
    sg = [support(it, it["gold"]) for it in items]
    sl = [support(it, it["lie"]) for it in items]
    p1_auc = auroc([0] * len(sg) + [1] * len(sl), [1 - x for x in sg] + [1 - x for x in sl])   # lie = low support = high fabrication

    def majority(ans):
        c = Counter(ans).most_common()
        return c[0][0] if (len(c) == 1 or c[0][1] > c[1][1]) else None      # None on a 3-way tie
    err_items = [it for it in items if any(a != it["gold"] for a in it["answers"])]   # >=1 model wrong
    covered = [it for it in err_items if majority(it["answers"]) == it["gold"]]
    p2_cover = len(covered) / len(err_items) if err_items else float("nan")
    floor = [it for it in items if (majority(it["answers"]) is not None and majority(it["answers"]) != it["gold"])]
    p3_floor = len(floor) / len(items)
    allcorrect = float(np.mean([all(a == it["gold"] for a in it["answers"]) for it in items]))

    p1 = p1_auc >= 0.80
    p2 = (p2_cover >= 0.70) if not np.isnan(p2_cover) else False
    council_covers = p1 and p2
    verdict = (
        (f"COUNCIL COVERS — cross-model agreement separates truth from the tempting lie (AUROC {p1_auc:.2f}) and "
         f"outvotes single-model error on {p2_cover:.0%} of items where >=1 model was wrong. A valid orthogonal "
         f"layer over single-model consistency. " if council_covers else
         f"COUNCIL WEAK — P1 AUROC {p1_auc:.2f} (bar 0.80), single-model-error coverage {p2_cover:.0%} (bar 0.70). ")
        + f"IRREDUCIBLE FLOOR: {p3_floor:.0%} of items have the council AGREEING on a WRONG answer (shared "
          f"confabulation no consistency/council check can catch; upper bound — shrinks with model diversity). "
          f"All-3-correct {allcorrect:.0%}.")

    res = {"models": MODELS, "n": len(items), "p1_auroc_gold_vs_lie": p1_auc,
           "p2_single_model_error_coverage": p2_cover, "p3_irreducible_floor_shared_confab": p3_floor,
           "all_three_correct": allcorrect, "council_covers": bool(council_covers), "verdict": verdict}
    out = os.path.join(HERE, "council_floor_smoke.json" if SMOKE else "council_floor_result.json")
    json.dump(res, open(out, "w"), indent=2)
    print("\n=== RESULT ===")
    print(f"  P1 council AUROC(gold vs lie): {p1_auc:.3f}")
    print(f"  P2 single-model-error coverage: {p2_cover:.3f}  (council outvotes a wrong model)")
    print(f"  P3 IRREDUCIBLE FLOOR (council agrees WRONG): {p3_floor:.3f}  | all-3-correct {allcorrect:.3f}")
    print("\n===== " + verdict)
    print("wrote", os.path.basename(out))


if __name__ == "__main__":
    main()
