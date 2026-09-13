#!/usr/bin/env python3
"""run_observatory_demo.py — three "days" of watching one model on cpu, so the observatory's log,
plates and verification exist before a frontier model is ever watched.

    python papers/checksum/run_observatory_demo.py

day 1: SmolLM2-135M float32 → baseline.   day 2: the same weights → SAME.
day 3: the same weights, int8-quantized → DRIFT.   then Observatory.verify() over the log.
Nothing here is a claim about the model; it is the observatory exercising every path it has.
"""
import os, sys, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from styxx import checksum as ck
from styxx.observatory import Observatory

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "HuggingFaceTB/SmolLM2-135M"
ROOT = os.path.join(HERE, "observatory_demo")


def main():
    tok = AutoTokenizer.from_pretrained(NAME)
    obs = Observatory(ROOT, NAME, tokenizer_id=NAME)
    for when, label in (("2026-09-13T00:00:00Z", "day 1"), ("2026-09-14T00:00:00Z", "day 2"), ("2026-09-15T00:00:00Z", "day 3")):
        m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
        if label == "day 3":
            m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        e = obs.observe(ck.hf_probe(m, tok), n_null=2, when=when, note=label + (" (int8 dynamic)" if label == "day 3" else ""))
        print(f"  {label}: {e['kind']:11s} floor={e['null_floor_nats']:.2e} "
              + (f"vs baseline {e['vs_baseline']['verdict']} {e['vs_baseline']['mean_abs_nats']:.4f} | vs previous {e['vs_previous']['verdict']}" if e.get('vs_baseline') else ""))
        obs.render()
    v = Observatory.verify(ROOT)
    print("  verify:", v)
    open(os.path.join(ROOT, "STATUS.md"), "w").write(f"# observatory: {NAME}\n\n" + Observatory.status(ROOT) + "\n")
    print(Observatory.status(ROOT))


if __name__ == "__main__":
    main()
