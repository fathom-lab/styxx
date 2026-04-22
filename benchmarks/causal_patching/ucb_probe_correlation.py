# -*- coding: utf-8 -*-
"""
benchmarks/causal_patching/ucb_probe_correlation.py

Universal Cognitive Basis — per-probe cross-model correlation.

The cleaner UCB test than pooled-PCA CCA:

  For each pair of models (A, B) with probes trained on the same
  concept, and a shared prompt set P, compute:

    s_A[p] = sigmoid(w_A . h_A(p) + b_A)   for each prompt p
    s_B[p] = sigmoid(w_B . h_B(p) + b_B)

  Then Pearson correlation rho(s_A, s_B) is the **cross-model
  probe agreement** for that concept. High rho means: when A's
  probe says 'truthful', B's probe also says 'truthful'. They are
  measuring the same concept.

If rho > 0.5 across all pairs of 4+ models → UCB Phase 2 supported.
If rho < 0.2 → concept is vendor-specific.
Anywhere in between → partial UCB.

This is the right test. Pooled-PCA CCA was finding high-variance
pooled directions, not shared-semantics directions — even though
Phase 1 ridge transfers clearly showed alignment existed.

Usage
-----
  python benchmarks/causal_patching/ucb_probe_correlation.py \\
    --manifests styxx/residual_probe/atlas/*_truthfulness.json \\
    --n_prompts 80 --seed 11 \\
    --out_file benchmarks/causal_patching/runs/ucb_probe_correlation.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract_and_train import build_probe_set  # noqa: E402


def _load_manifest_and_weight(manifest_path: Path):
    import torch
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weight_fp = manifest_path.parent / manifest["weight_file"]
    state = torch.load(weight_fp, map_location="cpu", weights_only=True)
    return (manifest,
            manifest["layer"],
            state["weight"].to(torch.float32),
            float(state["bias"]) if "bias" in state else 0.0)


def _extract_residuals(model_name: str, prompts: List[str], layer: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)

    # Install capture hook at the specified layer
    for candidate in (mdl, getattr(mdl, "model", None),
                      getattr(getattr(mdl, "model", None), "model", None)):
        if candidate is None:
            continue
        layers = getattr(candidate, "layers", None)
        if layers is not None and len(layers) > layer:
            target = layers[layer]
            break
    else:
        raise RuntimeError("could not find decoder layer")

    captured = {"h": None}
    def _h(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["h"] = hs[:, -1, :].detach()
        return out
    handle = target.register_forward_hook(_h)

    out = []
    try:
        for prompt in prompts:
            input_ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                _ = mdl(input_ids=input_ids)
            out.append(captured["h"][0].to(torch.float32).cpu())
    finally:
        handle.remove()
    X = torch.stack(out)

    del mdl
    torch.cuda.empty_cache()
    return X


def pearson(a, b):
    import numpy as np
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean()
                 / (a.std() * b.std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--n_prompts", type=int, default=80)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dataset", default="jbb")
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    # Expand any globs
    manifest_paths = []
    for m in args.manifests:
        expanded = glob.glob(m)
        if expanded:
            manifest_paths.extend(Path(e) for e in expanded)
        else:
            manifest_paths.append(Path(m))
    manifest_paths = sorted(set(manifest_paths))

    import torch
    import numpy as np

    # Build shared prompt set
    n_half = args.n_prompts // 2
    rows = build_probe_set(n_half, n_half, args.seed,
                           dataset=args.dataset, split="test")
    prompts = [r["prompt"] for r in rows]
    print(f"shared prompt set: {len(prompts)}")

    # Load each probe + extract residuals at that probe's layer
    probes = []
    for mp in manifest_paths:
        m, L, w, b = _load_manifest_and_weight(mp)
        print(f"\n=== {m['model']} ===  layer {L}  AUC {m['auc_validation']}")
        print(f"extracting residuals @ layer {L} ...")
        X = _extract_residuals(m["model"], prompts, L)
        # Apply probe to get per-prompt scores
        # score = sigmoid(w . h + b)
        logits = (X @ w) + b
        scores = 1.0 / (1.0 + (-logits).exp())
        probes.append({
            "manifest": m,
            "layer": L,
            "scores": scores.tolist(),
        })

    # Pairwise Pearson correlation of per-prompt scores
    N = len(probes)
    rho = [[1.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            r = pearson(probes[i]["scores"], probes[j]["scores"])
            rho[i][j] = rho[j][i] = r

    # Report
    print("\n" + "=" * 72)
    print("Pairwise probe-agreement (Pearson rho of per-prompt scores)")
    print("=" * 72)
    tasks = [p["manifest"]["task"] for p in probes]
    short_names = [p["manifest"]["model"].split("/")[-1][:18] for p in probes]
    hdr = "               " + "  ".join(f"{n:>18s}" for n in short_names)
    print(hdr)
    for i, n in enumerate(short_names):
        row = "  ".join(f"{rho[i][j]:>+18.3f}" for j in range(N))
        print(f"  {n:>12s}  {row}")

    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append({
                "a": probes[i]["manifest"]["model"],
                "b": probes[j]["manifest"]["model"],
                "rho": rho[i][j],
            })

    min_rho = min(p["rho"] for p in pairs) if pairs else float("nan")
    mean_rho = sum(p["rho"] for p in pairs) / len(pairs) if pairs else float("nan")

    print(f"\npairwise-rho min:  {min_rho:+.3f}")
    print(f"pairwise-rho mean: {mean_rho:+.3f}")

    verdict = None
    if min_rho > 0.5:
        verdict = (f"UCB Phase 2 SUPPORTED: min pairwise rho "
                   f"{min_rho:.3f} > 0.5 across {N} models. "
                   f"Probes trained independently on different models "
                   f"agree on which prompts are {tasks[0]}.")
    elif mean_rho > 0.3:
        verdict = (f"UCB Phase 2 PARTIAL: mean pairwise rho "
                   f"{mean_rho:.3f} > 0.3; min = {min_rho:.3f}. "
                   f"Some model pairs agree, others disagree.")
    else:
        verdict = (f"UCB Phase 2 null: min pairwise rho {min_rho:.3f} "
                   f"and mean {mean_rho:.3f} too low. Probes do not "
                   f"strongly agree across models at the concept level.")

    print(f"\n{verdict}")

    out = {
        "concept_task": tasks[0] if len(set(tasks)) == 1 else "mixed",
        "models": [p["manifest"]["model"] for p in probes],
        "layers": [p["layer"] for p in probes],
        "n_prompts": len(prompts),
        "prompt_seed": args.seed,
        "pairwise_rho": rho,
        "pairs": pairs,
        "min_rho": min_rho,
        "mean_rho": mean_rho,
        "verdict": verdict,
    }
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_file).write_text(json.dumps(out, indent=2),
                                    encoding="utf-8")
    print(f"\nwrote {args.out_file}")


if __name__ == "__main__":
    main()
