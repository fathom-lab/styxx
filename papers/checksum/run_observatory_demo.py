#!/usr/bin/env python3
"""run_observatory_demo.py — three "days" of watching one model on cpu, so the observatory's log,
plates and verification exist before a frontier model is ever watched.

    python papers/checksum/run_observatory_demo.py [--root papers/checksum/observatory_demo_v1]

day 1: SmolLM2-135M float32 → baseline.   day 2: the same weights → SAME.
day 3: the same weights, int8-quantized → DRIFT.   then Observatory.verify() over the log.
Nothing here is a claim about the model; it is the observatory exercising every path it has.

The three dates are LABELS passed to observe(); the clock at each observation is written beside
them as `taken`, and the whole demo runs in about a minute on one day. The first demo directory
(observatory_demo, 2026-09-13, v0) is kept as history with a CORRECTION.md; this script writes a
new directory and refuses to append to an existing one.
"""
import argparse
import os
import sys

ROOT_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT_REPO not in sys.path:
    sys.path.insert(0, ROOT_REPO)

import torch  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

from styxx import checksum as ck  # noqa: E402
from styxx.observatory import Observatory  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "HuggingFaceTB/SmolLM2-135M"
DAYS = (("2026-09-13T00:00:00Z", "day 1"), ("2026-09-14T00:00:00Z", "day 2"), ("2026-09-15T00:00:00Z", "day 3"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "observatory_demo_v1"))
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.root, "log.jsonl")):
        raise SystemExit(f"REFUSED: {a.root} already holds a log; a demo is history too — choose a new --root")
    tok = AutoTokenizer.from_pretrained(NAME)
    obs = Observatory(a.root, NAME, tokenizer_id=NAME)
    for when, label in DAYS:
        m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
        if label == "day 3":
            m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        note = label + (" (the same weights, int8 dynamic quantization)" if label == "day 3" else " (the same weights)")
        e = obs.observe(ck.hf_probe(m, tok), n_null=2, when=when, note=note)
        print(f"  {label}: {e['kind']:11s} taken={e['taken']} floor={e['null_floor_nats']:.2e} applied={e['floor_applied_nats']:.2e} "
              + (f"vs baseline {e['vs_baseline']['verdict']} {e['vs_baseline']['mean_abs_nats']:.4f} | vs previous {e['vs_previous']['verdict']}"
                 if e.get('vs_baseline') else ""))
        obs.render()
    last = obs.entries()[-1]
    v = Observatory.verify(a.root, expect_head=last["entry_hash"], expect_entries=len(obs.entries()))
    print("  verify:", v)
    header = (f"# observatory: {NAME}\n\n"
              "A demo, not an observation of anything: the three `when` dates are labels handed to observe() by\n"
              "`run_observatory_demo.py`; `taken` is the clock, and all three lines were taken within a minute of\n"
              "each other. Day 3 is the same weights int8-quantized. The head hash below is what a ledger anchor\n"
              f"would pin: `{last['entry_hash']}` over {len(obs.entries())} lines.\n\n")
    with open(os.path.join(a.root, "STATUS.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header + Observatory.status(a.root) + "\n")
    print(Observatory.status(a.root))


if __name__ == "__main__":
    main()
