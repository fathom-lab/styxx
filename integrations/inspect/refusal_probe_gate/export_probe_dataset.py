# -*- coding: utf-8 -*-
"""
export_probe_dataset.py — generate the (prompt, refused, probe_score) CSV that the
`refusal_probe_gate` Inspect eval consumes, from the REAL styxx pre-output refusal probe.

For each borderline prompt:
  - probe_score = StyxxProbe(comply_refuse).predict_before_generation -> P(refuse),
    read from the residual stream at end-of-prefill (NO token is generated);
  - refused (gold)  = vendor-robust detect_refusal on an actual greedy generation.

Run once per model (needs the model + a GPU). Output feeds `inspect eval`.

  python export_probe_dataset.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --out data/qwen2.5-1.5b_probe_scores.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]                 # styxx repo root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "dogfood"))

from styxx.residual_probe.probe import StyxxProbe                  # noqa: E402
from aggressive_borderline_eval_set import get_aggressive_borderlines  # noqa: E402
from vendor_robust_refusal_label import detect_refusal            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--task", default="comply_refuse")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    out = args.out or str(Path(__file__).resolve().parent / "data" /
                          f"{args.model.split('/')[-1].lower()}_probe_scores.csv")

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
    probe = StyxxProbe.from_pretrained(args.model, args.task)     # raises + lists atlas keys if absent
    refuse_is_positive = "refus" in probe.positive_class.lower()
    print(f"probe {args.model}::{args.task} layer={probe.layer} "
          f"positive_class={probe.positive_class!r} refuse_is_positive={refuse_is_positive}")

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows, n_ref = [], 0
    for sid, cat, prompt in get_aggressive_borderlines():
        v = probe.predict_before_generation(mdl, tok, prompt)
        p_refuse = v.p_positive if refuse_is_positive else (1.0 - v.p_positive)
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            g = mdl.generate(ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=pad)
        refused = int(detect_refusal(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)))
        n_ref += refused
        rows.append((prompt, refused, round(float(p_refuse), 6)))
        print(f"  {sid:6s} P(refuse)={p_refuse:.3f} gold_refused={refused}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "refused", "probe_score"])
        w.writerows(rows)
    print(f"\nwrote {out}: {len(rows)} rows, {n_ref} refused / {len(rows) - n_ref} allowed")


if __name__ == "__main__":
    main()
