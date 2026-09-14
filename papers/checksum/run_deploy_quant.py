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

What this runner binds, so a stranger can tell the sealed run from any other (added 2026-09-13 after
the red team found the run could import a styxx from outside this checkout and record nothing about
it): the styxx package it imported must live inside this checkout or it refuses to start; the certs
carry `provenance` — git HEAD, whether the tree was dirty, the sha256 of the PREREG's git blob (the
sealed bytes, LF as stored), the sha256 of the checksum.py that ran, the torch/transformers/
bitsandbytes versions, the CUDA device, CUBLAS_WORKSPACE_CONFIG — and `k1`, the PREREG's first kill
gate, evaluated in code: if the null floor exceeds 1e-2 nats/token no comparison is written, as the
PREREG demands. `distance_params` records n_boot, seed and the floor used. `top1_loss_vs_A` is
hits[A] − hits[arm], the definition the RESULT uses for H2/K3.
"""
import argparse, hashlib, json, os, re, subprocess, sys, time


def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the output files of an INSTRUMENT CHECK that is not the "
                    "experiment (e.g. _dryrun_qwen0.5b); the experiment writes untagged files and only for the PREREG's model")
    ap.add_argument("--beacon", default="", help="64-hex beacon (a slot blockhash as styxx.clock reports it): draw the 48 "
                    "canaries from the committed pool instead of the hand set. The 2026-09-13 PREREG froze the hand set, so "
                    "a beacon-drawn run is never that experiment and must carry --tag; the next PREREG is written for this.")
    args = ap.parse_args(argv)
    # refusals that need no model stack: decided before torch is imported
    if args.beacon and not args.tag:
        raise SystemExit("a beacon-drawn run is not the sealed experiment (the 2026-09-13 PREREG froze the hand-written 48); "
                         "pass --tag, or write the next PREREG with the draw in it")
    if args.beacon and not re.fullmatch(r"[0-9a-f]{64}", args.beacon):
        raise SystemExit("--beacon must be 64 lowercase hex characters (styxx.clock.blockhash_to_beacon converts the chain's base58)")
    return args


ARGS = _parse_args() if __name__ == "__main__" else None

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)                      # the checksum that runs is the one this commit carries
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before CUDA initialises: cuBLAS determinism

import numpy as np  # noqa: E402
import torch  # noqa: E402

import styxx  # noqa: E402
from styxx import checksum as ck  # noqa: E402
from styxx.geoplate import coefficients, render_grid  # noqa: E402

_STYXX_FILE = os.path.abspath(styxx.__file__)
if os.path.commonpath([_STYXX_FILE, ROOT]) != ROOT:
    raise SystemExit(f"styxx resolved to {_STYXX_FILE}, outside this checkout {ROOT}; "
                     "refusing to run the sealed experiment with another package")

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = "PREREG_checksum_deploy_quant_2026_09_13.md"
K1_FLOOR_NATS = 1e-2
N_BOOT, SEED = 2000, 20260913


def _git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, check=True)
    except Exception:
        return None


def provenance(device):
    def sha(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    blob = _git("show", f"HEAD:papers/checksum/{PREREG}")
    versions = {"torch": torch.__version__}
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover
        versions["transformers"] = None
    try:
        import bitsandbytes
        versions["bitsandbytes"] = bitsandbytes.__version__
    except Exception:
        versions["bitsandbytes"] = None
    return {
        "git_head": head.stdout.decode().strip() if head else None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty else None,
        "prereg": PREREG,
        "prereg_blob_sha256": hashlib.sha256(blob.stdout).hexdigest() if blob else None,
        "styxx_file": _STYXX_FILE,
        "checksum_py_sha256": sha(os.path.join(ROOT, "styxx", "checksum.py")),
        "versions": versions,
        "python": sys.version.split()[0],
        "device": device,
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_quantized_engine": torch.backends.quantized.engine,
    }


def load(name, kind, device, dtype):
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
        m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name), dtype=dtype).to(device)
    else:
        raise ValueError(kind)
    return tok, m.eval()


def hf_probe_on(model, tokenizer, device):
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


def top1(model, tok, device, canaries):
    hits = 0
    for _, p, c in canaries:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            pred = model(input_ids=ids).logits[0, -1].argmax().item()
        hits += int(pred == tok(c, add_special_tokens=False).input_ids[0])
    return hits


def _write_json(path, obj, indent=None):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=indent)
        fh.write("\n")


def main():
    args = ARGS if ARGS is not None else _parse_args()
    smoke = args.smoke
    name = "HuggingFaceTB/SmolLM2-135M" if smoke else args.model
    device = "cpu" if smoke or not torch.cuda.is_available() else "cuda"
    is_the_experiment = (not smoke) and (not args.tag) and name == "Qwen/Qwen2.5-1.5B"
    if not smoke and not args.tag and not is_the_experiment:
        raise SystemExit("an untagged, non-smoke run is the sealed experiment and its model is Qwen/Qwen2.5-1.5B; "
                         "pass --tag <suffix> for an instrument check on another model")
    suffix = ("_smoke" if smoke else "") + args.tag
    base_kind = "fp32" if smoke else "bf16"
    dtype = torch.float32 if base_kind == "fp32" else torch.bfloat16
    arms = [("A", base_kind), ("A2", base_kind), ("A3", base_kind),
            ("Q4", "int8-dynamic-cpu" if smoke else "nf4"),
            ("Q8", "int8-dynamic-cpu" if smoke else "int8"),
            ("R", "random")]
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    prov = provenance(device)
    print(f"  provenance: head {prov['git_head']} dirty={prov['git_dirty']} prereg blob {str(prov['prereg_blob_sha256'])[:12]} "
          f"styxx {prov['styxx_file']}")
    canaries, draw_record = ck.CANARIES, None
    if args.beacon:
        from styxx import beacon as _beacon
        canaries, draw_record = _beacon.draw(args.beacon, 48)
        print(f"  canaries drawn by beacon {draw_record['beacon'][:12]}… from pool {draw_record['pool_sha256'][:12]}… "
              f"(canary set {draw_record['canary_sha256'][:12]}…)")
    t0 = time.time()
    fps, hits = {}, {}
    for tag, kind in arms:
        tok, m = load(name, kind, device, dtype)
        fps[tag] = ck.fingerprint(hf_probe_on(m, tok, device), f"{name} {kind} [{tag}]", canaries=canaries,
                                  tokenizer_id=name, draw=draw_record)
        hits[tag] = top1(m, tok, device, canaries)
        print(f"  {tag:3s} {kind:18s} top-1 {hits[tag]}/48   [{time.time() - t0:.0f}s]", flush=True)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    floor = ck.null_floor([fps["A"], fps["A2"], fps["A3"]])
    k1 = {"threshold_nats": K1_FLOOR_NATS, "floor_nats": floor, "fired": bool(floor > K1_FLOOR_NATS)}
    out = {"prereg": PREREG, "smoke": smoke, "tag": args.tag, "is_the_experiment": is_the_experiment,
           "model": name, "device": device,
           "provenance": prov,
           "canaries": {"n": len(canaries), "canary_sha256": ck.canary_sha256(canaries), "draw": draw_record,
                        "source": "styxx.beacon draw" if draw_record else "styxx.checksum.CANARIES (the hand-written set)"},
           "sanity": {"first_token_top1_hits": hits, "n_canaries": len(canaries), "null_floor_nats": floor,
                      "top1_loss_vs_A": {t: hits["A"] - hits[t] for t in ("A2", "A3", "Q4", "Q8", "R")}},
           "distance_params": {"n_boot": N_BOOT, "seed": SEED, "floor_nats": floor},
           "k1": k1}
    print(f"  null floor (worst of 3 {base_kind} loads): {floor:.6f} nats/token")
    if k1["fired"]:
        # PREREG K1: the serving is not deterministic enough for this probe; the run is INCONCLUSIVE and
        # says so; no other hypothesis is evaluated. The fingerprints are still written so a NEW prereg
        # can look at them; no comparison cert exists for this run.
        out["verdict"] = "INCONCLUSIVE"
        print(f"  K1 FIRED: null floor {floor:.6f} > {K1_FLOOR_NATS} nats/token — INCONCLUSIVE; no hypothesis evaluated")
    else:
        for tag in ("A2", "Q4", "Q8", "R"):
            d = ck.distance(fps["A"], fps[tag], n_boot=N_BOOT, seed=SEED, floor_nats=floor)
            out[tag] = ck.cert(fps["A"], fps[tag], d, note=f"{name} on {device}; teacher-forced; deterministic best effort")
            print(f"  A vs {tag:3s}: {d.verdict:12s} mean|Δlogp| = {d.mean_abs_nats:.5f} [{d.ci_mean_abs[0]:.5f}, {d.ci_mean_abs[1]:.5f}]"
                  f"   r = {d.rdm_r:.4f}   top-1 {hits['A']}->{hits[tag]}")
        # CORRECTION_prereg_deploy_quant_H1: the frozen H1 grades A vs A' against a floor that includes
        # that very pair. Beside it, unpreregistered and labelled so, the held-out reading: the floor
        # from the two pairs that do not contain A' (A-A'' and A'-A''), which is what the clause meant.
        held_out_floor = max(float(np.abs(fps["A"].mean_lp - fps["A3"].mean_lp).mean()),
                             float(np.abs(fps["A2"].mean_lp - fps["A3"].mean_lp).mean()))
        d_ho = ck.distance(fps["A"], fps["A2"], n_boot=N_BOOT, seed=SEED, floor_nats=held_out_floor)
        out["h1_held_out"] = {"unpreregistered": True, "floor_from": ["A-A3", "A2-A3"], "floor_nats": held_out_floor,
                              "cert": ck.cert(fps["A"], fps["A2"], d_ho, note="held-out floor reading of H1; decides nothing in this run")}
        print(f"  H1 held-out reading (unpreregistered): A vs A2 {d_ho.verdict} against floor {held_out_floor:.6f}")
    _write_json(os.path.join(HERE, f"deploy_quant_certs{suffix}.json"), out, indent=1)
    _write_json(os.path.join(HERE, f"deploy_quant_fingerprints{suffix}.json"), {k: v.to_json() for k, v in fps.items()})
    if k1["fired"]:
        print("wrote certs (K1 fired: no comparisons) and fingerprints; no plates for an INCONCLUSIVE run")
        return
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
