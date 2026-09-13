#!/usr/bin/env python3
"""run_deploy_quant.py — the sealed deploy-scale checksum run (see PREREG_checksum_deploy_quant).

    # real run, GPU, after the prereg hash is anchored:
    python papers/checksum/run_deploy_quant.py --model Qwen/Qwen2.5-1.5B
    # pipeline smoke on cpu (no bitsandbytes; dynamic int8 stands in for the two bnb arms;
    # results are NOT the experiment and are written to *_smoke.* files):
    python papers/checksum/run_deploy_quant.py --smoke

Arms: A bf16, A' bf16 reloaded, A'' bf16 reloaded (floor = worst pairwise), Q4 bitsandbytes NF4,
Q8 bitsandbytes LLM.int8, R random init (seed 343). Every number the prereg names is written to
deploy_quant_certs.json; the RESULT document swears to those bytes, not to this printout.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

from styxx import checksum as ck
from styxx.geoplate import coefficients, render_grid

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, kind, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    tok = AutoTokenizer.from_pretrained(name)
    if kind == "bf16":
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device)
    elif kind == "fp32":
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device)
    elif kind == "nf4":
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        m = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map={"": 0})
    elif kind == "int8":
        from transformers import BitsAndBytesConfig
        m = AutoModelForCausalLM.from_pretrained(name, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                                 device_map={"": 0})
    elif kind == "int8-dynamic-cpu":
        base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
        m = torch.ao.quantization.quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8)
    elif kind == "random":
        torch.manual_seed(343)
        m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name)).to(device)
    else:
        raise ValueError(kind)
    return tok, m.eval()


def hf_probe_on(model, tokenizer, device):
    base = ck.hf_probe(model, tokenizer)

    def probe(prompt, continuation):
        p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        c_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        ids = torch.cat([p_ids, c_ids], dim=1)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        n_p = p_ids.shape[1]
        cont = [logp[n_p - 1 + t, c_ids[0, t]].item() for t in range(c_ids.shape[1])]
        return ck.Probe(cont_logprobs=cont, next_logprobs=logp[n_p - 1].cpu().numpy())
    return probe


def top1(model, tok, device):
    hits = 0
    for _, p, c in ck.CANARIES:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            pred = model(input_ids=ids).logits[0, -1].argmax().item()
        hits += int(pred == tok(c, add_special_tokens=False).input_ids[0])
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    smoke = args.smoke
    name = "HuggingFaceTB/SmolLM2-135M" if smoke else args.model
    device = "cpu" if smoke or not torch.cuda.is_available() else "cuda"
    suffix = "_smoke" if smoke else ""
    base_kind = "fp32" if smoke else "bf16"
    arms = [("A", base_kind), ("A2", base_kind), ("A3", base_kind),
            ("Q4", "int8-dynamic-cpu" if smoke else "nf4"),
            ("Q8", "int8-dynamic-cpu" if smoke else "int8"),
            ("R", "random")]
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    t0 = time.time()
    fps, hits = {}, {}
    for tag, kind in arms:
        tok, m = load(name, kind, device)
        fps[tag] = ck.fingerprint(hf_probe_on(m, tok, device), f"{name} {kind} [{tag}]", tokenizer_id=name)
        hits[tag] = top1(m, tok, device)
        print(f"  {tag:3s} {kind:18s} top-1 {hits[tag]}/48   [{time.time() - t0:.0f}s]", flush=True)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    floor = ck.null_floor([fps["A"], fps["A2"], fps["A3"]])
    out = {"prereg": "PREREG_checksum_deploy_quant_2026_09_13.md", "smoke": smoke, "model": name, "device": device,
           "sanity": {"first_token_top1_hits": hits, "n_canaries": len(ck.CANARIES), "null_floor_nats": floor}}
    print(f"  null floor (worst of 3 bf16 loads): {floor:.6f} nats/token")
    for tag in ("A2", "Q4", "Q8", "R"):
        d = ck.distance(fps["A"], fps[tag], floor_nats=floor)
        out[tag] = ck.cert(fps["A"], fps[tag], d, note=f"{name} on {device}; teacher-forced; deterministic best effort")
        print(f"  A vs {tag:3s}: {d.verdict:12s} mean|Δlogp| = {d.mean_abs_nats:.5f} [{d.ci_mean_abs[0]:.5f}, {d.ci_mean_abs[1]:.5f}]"
              f"   r = {d.rdm_r:.4f}   top-1 {hits['A']}->{hits[tag]}")
    json.dump(out, open(os.path.join(HERE, f"deploy_quant_certs{suffix}.json"), "w"), indent=1)
    json.dump({k: v.to_json() for k, v in fps.items()}, open(os.path.join(HERE, f"deploy_quant_fingerprints{suffix}.json"), "w"))
    lab = lambda t: f"vs A: {out[t]['distance']['mean_abs_nats']:.4f} nats/token, r = {out[t]['distance']['rdm_r']:.3f}  ({out[t]['distance']['verdict']})"
    items = [(f"{name.split('/')[-1]}  {base_kind}", "the model", coefficients(fps["A"].rdm)),
             ("same weights, reloaded", lab("A2"), coefficients(fps["A2"].rdm)),
             ("nf4 (4-bit)" if not smoke else "int8 dynamic (stand-in)", lab("Q4"), coefficients(fps["Q4"].rdm)),
             ("llm.int8" if not smoke else "int8 dynamic (stand-in)", lab("Q8"), coefficients(fps["Q8"].rdm)),
             ("control: random weights", lab("R"), coefficients(fps["R"].rdm))]
    png = render_grid(items, os.path.join(HERE, f"deploy_quant_plates{suffix}.png"),
                      title=f"checksum: {name}, 48 hashed canaries, {device} — what quantization moves, as sand", ncols=3)
    print("wrote", png)


if __name__ == "__main__":
    main()
