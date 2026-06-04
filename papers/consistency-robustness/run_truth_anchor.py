# -*- coding: utf-8 -*-
"""
run_truth_anchor.py — frozen by PREREG_truth_anchor_2026_06_04. The framework's 6th readout: the only
one that reads OUTSIDE the model. Closed-book errors = the knowledge floor; the gold context is the
external-truth anchor. Does it break the floor that no self/peer/derivation readout could?
"""
from __future__ import annotations
import sys, os, json, re
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = "--smoke" in sys.argv
MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_ITEMS = 12 if SMOKE else 100
SYS = "Answer the question in a few words. Reply with ONLY the answer, no explanation."


def norm(s):
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def correct(out, golds):
    no = norm(out)
    return any(g and norm(g) in no for g in golds)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    np.random.seed(0); torch.set_grad_enabled(False)
    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()

    def ask(user):
        ids = tok(tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                          tokenize=False, add_generation_prompt=True), return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]

    ds = load_dataset("rajpurkar/squad_v2", split="validation", streaming=True)
    rows = []
    for ex in ds:
        golds = ex["answers"]["text"]
        if not golds:                      # skip unanswerable
            continue
        q, ctx = ex["question"], ex["context"]
        cb = ask(f"Question: {q}")
        ob = ask(f"Context: {ctx}\n\nUsing only the context above, answer.\nQuestion: {q}")
        rows.append({"q": q, "golds": golds, "cb": cb, "ob": ob,
                     "cb_ok": correct(cb, golds), "ob_ok": correct(ob, golds)})
        if len(rows) % 10 == 0:
            print(f"  {len(rows)}: cb_ok={rows[-1]['cb_ok']} ob_ok={rows[-1]['ob_ok']} gold={golds[0][:30]!r}", flush=True)
        if len(rows) >= N_ITEMS:
            break

    cb_acc = float(np.mean([r["cb_ok"] for r in rows]))
    ob_acc = float(np.mean([r["ob_ok"] for r in rows]))
    floor = [r for r in rows if not r["cb_ok"]]                          # the knowledge floor (closed-book wrong)
    floor_correction = float(np.mean([r["ob_ok"] for r in floor])) if floor else float("nan")
    residual = float(np.mean([(not r["cb_ok"]) and (not r["ob_ok"]) for r in rows]))   # context couldn't fix

    p1 = (ob_acc - cb_acc) >= 0.20
    p2 = (floor_correction >= 0.50) if not np.isnan(floor_correction) else False
    if p1 and p2:
        verdict = (f"ANCHOR BREAKS THE FLOOR — external trusted context lifts accuracy {cb_acc:.2f}->{ob_acc:.2f} and "
                   f"CORRECTS {floor_correction:.0%} of the closed-book knowledge floor that no self/peer/derivation "
                   f"readout could. The only readout that reads OUTSIDE the model grounds the stack in truth. Residual "
                   f"{residual:.0%} = even-with-context failure (extraction/reasoning). The trust commitment now lives "
                   f"in the SOURCE — guarded by detect_context_injection (91% this arc). The stack closes on itself.")
    else:
        verdict = (f"PARTIAL — P1(anchor lifts {ob_acc-cb_acc:+.2f}):{p1}, P2(floor correction {floor_correction:.0%}):{p2}. "
                   f"cb {cb_acc:.2f} -> ob {ob_acc:.2f}, residual {residual:.0%}.")

    res = {"model": MODEL, "n": len(rows), "closed_book_acc": cb_acc, "open_book_acc": ob_acc,
           "floor_correction_rate": floor_correction, "residual_even_with_context": residual,
           "P1_anchor_lifts": bool(p1), "P2_breaks_floor": bool(p2), "verdict": verdict}
    out = os.path.join(HERE, "truth_anchor_smoke.json" if SMOKE else "truth_anchor_result.json")
    json.dump(res, open(out, "w"), indent=2)
    print("\n=== RESULT ===")
    print(f"  accuracy: closed-book {cb_acc:.3f} -> open-book {ob_acc:.3f}  (P1 lift>=0.20: {p1})")
    print(f"  FLOOR CORRECTION: external context fixes {floor_correction:.1%} of closed-book errors  (P2>=50%: {p2})")
    print(f"  residual (wrong even with context): {residual:.1%}")
    print("\n===== " + verdict)
    print("wrote", os.path.basename(out))


if __name__ == "__main__":
    main()
