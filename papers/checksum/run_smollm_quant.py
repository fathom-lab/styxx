#!/usr/bin/env python3
"""run_smollm_quant.py — the quantization plate, on a model small enough for any laptop.

    pip install -e '.[plate]' torch transformers
    python papers/checksum/run_smollm_quant.py

Four fingerprints of HuggingFaceTB/SmolLM2-135M on the 48 hashed canaries, CPU, deterministic:
  A   float32, loaded once
  A'  float32, loaded again (the null pair — must read SAME at distance exactly 0)
  Q   int8 dynamic quantization of every nn.Linear (torch.ao.quantization.quantize_dynamic)
  R   the same architecture with random weights (the far control)
and the three distances from A, each with a bootstrap 95% interval, written as certs, plus the
plates: each model's 48 x 48 belief geometry drawn as sand, A' and Q next to A.
"""
import json, os, sys, time
import numpy as np
import torch

from styxx import checksum as ck
from styxx.geoplate import coefficients, render_grid, render_drift, coefficients_sha256

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "HuggingFaceTB/SmolLM2-135M"


def load(name=NAME):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    m.eval()
    return tok, m


def main():
    torch.manual_seed(0)
    t0 = time.time()
    tok, mA = load()
    fpA = ck.fingerprint(ck.hf_probe(mA, tok), f"{NAME} float32 #1", tokenizer_id=NAME)
    _, mA2 = load()
    fpA2 = ck.fingerprint(ck.hf_probe(mA2, tok), f"{NAME} float32 #2 (reloaded)", tokenizer_id=NAME)
    mQ = torch.ao.quantization.quantize_dynamic(mA2, {torch.nn.Linear}, dtype=torch.qint8)
    fpQ = ck.fingerprint(ck.hf_probe(mQ, tok), f"{NAME} int8 dynamic (Linear)", tokenizer_id=NAME)
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(343)
    mR = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(NAME)).eval()
    fpR = ck.fingerprint(ck.hf_probe(mR, tok), f"{NAME} architecture, random init (seed 343)", tokenizer_id=NAME)
    print(f"four fingerprints in {time.time() - t0:.0f}s; canary set {fpA.canary_sha256[:12]}")

    def top1(m):
        hits = 0
        for _, p, c in ck.CANARIES:
            ids = tok(p, return_tensors="pt").input_ids
            with torch.no_grad():
                pred = m(input_ids=ids).logits[0, -1].argmax().item()
            hits += int(pred == tok(c, add_special_tokens=False).input_ids[0])
        return hits
    sanity = {"first_token_top1_hits": {"A": top1(mA), "reloaded": top1(mA2), "int8": top1(mQ), "random": top1(mR)},
              "n_canaries": len(ck.CANARIES)}
    print("  sanity, first-token top-1 hits /48:", sanity["first_token_top1_hits"])

    floor = ck.null_floor([fpA, fpA2])          # deterministic cpu: expected exactly 0
    sanity["null_floor_nats"] = floor
    out = {"sanity": sanity}
    for tag, fpX in (("reloaded", fpA2), ("int8", fpQ), ("random", fpR)):
        d = ck.distance(fpA, fpX, floor_nats=floor)
        c = ck.cert(fpA, fpX, d, note="SmolLM2-135M on CPU; teacher-forced; deterministic")
        out[tag] = c
        print(f"  A vs {tag:8s}: {d.verdict:12s} mean|Δlogp| = {d.mean_abs_nats:.5f} nats/token "
              f"[{d.ci_mean_abs[0]:.5f}, {d.ci_mean_abs[1]:.5f}]   belief-geometry r = {d.rdm_r:.4f} "
              f"[{d.ci_rdm_r[0]:.3f}, {d.ci_rdm_r[1]:.3f}]   corr-dist = {d.corr_dist:.4f}")
    out["coefficients_sha256"] = {k: coefficients_sha256(coefficients(f.rdm)) for k, f in
                                  (("A", fpA), ("A2", fpA2), ("Q", fpQ), ("R", fpR))}
    json.dump(out, open(os.path.join(HERE, "smollm_quant_certs.json"), "w"), indent=1)
    json.dump({"A": fpA.to_json(), "A2": fpA2.to_json(), "Q": fpQ.to_json(), "R": fpR.to_json()},
              open(os.path.join(HERE, "smollm_quant_fingerprints.json"), "w"))

    items = [
        ("smollm2-135m  float32", "the model", coefficients(fpA.rdm)),
        ("same weights, reloaded", f"vs A: {out['reloaded']['distance']['mean_abs_nats']:.5f} nats/token, "
                                   f"r = {out['reloaded']['distance']['rdm_r']:.3f}  ({out['reloaded']['distance']['verdict']})",
         coefficients(fpA2.rdm)),
        ("int8 quantized", f"vs A: {out['int8']['distance']['mean_abs_nats']:.4f} nats/token, "
                           f"r = {out['int8']['distance']['rdm_r']:.3f}  ({out['int8']['distance']['verdict']})",
         coefficients(fpQ.rdm)),
        ("control: random weights", f"vs A: {out['random']['distance']['mean_abs_nats']:.3f} nats/token, "
                                    f"r = {out['random']['distance']['rdm_r']:.3f}  ({out['random']['distance']['verdict']})",
         coefficients(fpR.rdm)),
    ]
    png = render_grid(items, os.path.join(HERE, "smollm_quant_plates.png"),
                      title="checksum: one model, 48 hashed canaries, cpu — what quantization moves, as sand", ncols=2)
    print("wrote", png)
    lab2 = lambda t: f"{out[t]['distance']['mean_abs_nats']:.4f} nats/token, r = {out[t]['distance']['rdm_r']:.3f}  ({out[t]['distance']['verdict']})"
    render_drift([("same weights, reloaded", lab2("reloaded"), coefficients(fpA.rdm), coefficients(fpA2.rdm)),
                  ("int8 quantized", lab2("int8"), coefficients(fpA.rdm), coefficients(fpQ.rdm)),
                  ("random weights", lab2("random"), coefficients(fpA.rdm), coefficients(fpR.rdm))],
                 os.path.join(HERE, "smollm_quant_drift_plates.png"),
                 title="drift plates: sand lands only where the two geometries disagree — no sand is what SAME looks like")


if __name__ == "__main__":
    main()
