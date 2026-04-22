# -*- coding: utf-8 -*-
"""
benchmarks/causal_patching/extract_and_train.py

Day-1 prep for the causal-patching experiment.

1. Load Llama-3.2-1B-Instruct.
2. Build a balanced probe set: unsafe prompts (HarmBench) + safe
   prompts (Alpaca). Unsafe elicit refusal, safe elicit compliance —
   that's what gives us two classes to probe.
3. For each prompt, run prefill and capture the final-token residual
   activation at every layer.
4. Train a per-layer linear probe (logistic regression with L2) to
   predict comply-vs-refuse, using Llama's actual observed behavior
   as labels.
5. Pick the best-AUC layer. Save that probe as a .pt + manifest
   compatible with styxx.residual_probe.

Datasets:
  Default (open, no gating):
    - JailbreakBench/JBB-Behaviors[behaviors/harmful]  (unsafe class)
    - JailbreakBench/JBB-Behaviors[behaviors/benign]   (safe class,
      category-matched to the harmful set — a cleaner control than
      raw Alpaca)

  Legacy (gated, requires HF access):
    - walledai/HarmBench  (unsafe class)
    - tatsu-lab/alpaca   (safe class — benign instructions)

Usage:
  python benchmarks/causal_patching/extract_and_train.py \
    --out_dir styxx/residual_probe/atlas \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --n_unsafe 50 --n_safe 50 --seed 0

  # Force the original HarmBench+Alpaca pair:
  python benchmarks/causal_patching/extract_and_train.py \
    --dataset harmbench --out_dir ...

  # Optional: override with a local jsonl (rows: {"id","kind","prompt"})
  python benchmarks/causal_patching/extract_and_train.py \
    --prompts path/to/custom.jsonl --out_dir ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_prompts_from_jsonl(path: Path) -> List[Dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


JBB_MASTER_SEED = 42   # fixed deterministic split across train/test
JBB_TRAIN_HALF = 50    # items from each class assigned to the train half


def _jbb_split_pools(split: str):
    """Return (harmful_pool, benign_pool) for the requested split.

    JBB has only 100 harmful + 100 benign. To keep train and test
    disjoint, we deterministically shuffle once (master_seed=42) and
    reserve the first 50 of each class for 'train' and the last 50 for
    'test'. Sampling within a split is then a sub-sample of that half.
    """
    from datasets import load_dataset
    import random

    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    harmful = [row["Goal"] for row in ds["harmful"] if row.get("Goal")]
    benign = [row["Goal"] for row in ds["benign"] if row.get("Goal")]

    # deterministic canonical shuffle, then split
    master = random.Random(JBB_MASTER_SEED)
    master.shuffle(harmful)
    master.shuffle(benign)

    if split == "train":
        return harmful[:JBB_TRAIN_HALF], benign[:JBB_TRAIN_HALF]
    if split == "test":
        return harmful[JBB_TRAIN_HALF:], benign[JBB_TRAIN_HALF:]
    raise ValueError(f"unknown split {split!r}")


def _build_jbb(n_unsafe: int, n_safe: int, seed: int,
               split: str = "train") -> List[Dict]:
    """JailbreakBench/JBB-Behaviors — open, 100 harmful + 100 benign,
    category-matched, deterministically halved into train/test.
    Field is `Goal`."""
    import random

    harmful_pool, benign_pool = _jbb_split_pools(split)
    if n_unsafe > len(harmful_pool) or n_safe > len(benign_pool):
        raise ValueError(
            f"JBB {split} half has {len(harmful_pool)} harmful + "
            f"{len(benign_pool)} benign; requested "
            f"n_unsafe={n_unsafe}, n_safe={n_safe}"
        )

    rng = random.Random(seed)
    unsafe = rng.sample(harmful_pool, n_unsafe)
    safe = rng.sample(benign_pool, n_safe)

    tag = split[0]  # 't' or 't' — disambiguated by id prefix
    rows: List[Dict] = []
    for i, p in enumerate(unsafe):
        rows.append({"id": f"jbb-harm-{split}-{i:03d}",
                     "kind": "unsafe", "prompt": p})
    for i, p in enumerate(safe):
        rows.append({"id": f"jbb-ben-{split}-{i:03d}",
                     "kind": "safe", "prompt": p})
    rng.shuffle(rows)
    return rows


def _build_harmbench_alpaca(n_unsafe: int, n_safe: int,
                            seed: int) -> List[Dict]:
    """Original HarmBench + Alpaca pair. Gated; requires HF access."""
    from datasets import load_dataset
    import random

    rng = random.Random(seed)
    hb = load_dataset("walledai/HarmBench", "standard", split="train")
    hb_prompts = [row["prompt"] for row in hb if row.get("prompt")]
    rng.shuffle(hb_prompts)
    unsafe = hb_prompts[:n_unsafe]

    alp = load_dataset("tatsu-lab/alpaca", split="train")
    alp_prompts = [
        row["instruction"] for row in alp
        if row.get("instruction") and not row.get("input")
    ]
    rng.shuffle(alp_prompts)
    safe = alp_prompts[:n_safe]

    rows: List[Dict] = []
    for i, p in enumerate(unsafe):
        rows.append({"id": f"hb-{i:03d}", "kind": "unsafe", "prompt": p})
    for i, p in enumerate(safe):
        rows.append({"id": f"alp-{i:03d}", "kind": "safe", "prompt": p})
    rng.shuffle(rows)
    return rows


def build_probe_set(n_unsafe: int, n_safe: int, seed: int,
                    dataset: str = "jbb",
                    split: str = "train") -> List[Dict]:
    """Stratified sample of unsafe + safe prompts.

    Returns a list of dicts: {"id", "kind", "prompt"}.
    kind is "unsafe" or "safe" — this is the *input* label, not the
    compliance label. Compliance is measured from the model's response.

    dataset="jbb"       : JBB-Behaviors harmful + benign (open, default).
                          split={"train","test"} picks disjoint halves.
    dataset="harmbench" : HarmBench + Alpaca (gated, legacy).
                          split arg is ignored — relies on seed alone.
    """
    if dataset == "jbb":
        return _build_jbb(n_unsafe, n_safe, seed, split=split)
    if dataset == "harmbench":
        return _build_harmbench_alpaca(n_unsafe, n_safe, seed)
    raise ValueError(f"unknown dataset {dataset!r} "
                     f"(expected 'jbb' or 'harmbench')")


from refusal_heuristic import REFUSAL_MARKERS, detect_compliance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=None,
                    help="Optional JSONL override. If omitted, sample "
                         "from HarmBench + Alpaca.")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--out_dir", required=True,
                    help="atlas directory for probe artifacts")
    ap.add_argument("--n_unsafe", type=int, default=50,
                    help="HarmBench samples (refuse class)")
    ap.add_argument("--n_safe", type=int, default=50,
                    help="Alpaca samples (comply class)")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed for reproducibility")
    ap.add_argument("--dataset", default="jbb",
                    choices=["jbb", "harmbench"],
                    help="'jbb' (open JBB-Behaviors, default) or "
                         "'harmbench' (gated HarmBench+Alpaca, legacy)")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import LeaveOneOut
        import numpy as np
    except Exception as e:
        print(f"FATAL: dependency missing: {e}", file=sys.stderr)
        sys.exit(1)

    if args.prompts:
        prompts = load_prompts_from_jsonl(Path(args.prompts))
        probe_set_source = Path(args.prompts).name
    else:
        try:
            prompts = build_probe_set(args.n_unsafe, args.n_safe,
                                      args.seed, dataset=args.dataset)
        except Exception as e:
            print(f"FATAL: could not build probe set from HF datasets: {e}",
                  file=sys.stderr)
            print("  install with: pip install datasets", file=sys.stderr)
            sys.exit(1)
        source_label = ("jbb-harmful+jbb-benign" if args.dataset == "jbb"
                        else "harmbench+alpaca")
        probe_set_source = (
            f"{source_label}[{args.n_unsafe}/{args.n_safe}]"
            f"@seed={args.seed}"
        )

    print(f"[1/5] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=(torch.bfloat16 if args.device == "cuda" else torch.float32),
        output_hidden_states=True,
    )
    mdl.eval()
    device = args.device if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    n_layers = mdl.config.num_hidden_layers
    hidden_size = mdl.config.hidden_size
    print(f"  loaded on {device} - layers={n_layers} hidden={hidden_size}")

    print(f"[2/5] probe set: {len(prompts)} prompts from {probe_set_source}; "
          f"generating + extracting residuals ...")

    residuals: List[List[torch.Tensor]] = [[] for _ in range(n_layers + 1)]
    labels: List[int] = []  # 1 = refused, 0 = complied
    response_records: List[Dict] = []

    t0 = time.time()
    for i, row in enumerate(prompts):
        prompt = row["prompt"]
        input_ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        prefill_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = mdl(input_ids=input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            for layer_idx, h in enumerate(hidden_states):
                last_token = h[0, -1, :].to(torch.float32).cpu()
                residuals[layer_idx].append(last_token)

            gen = mdl.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            new_tokens = gen[0, prefill_len:].tolist()
            response_text = tok.decode(new_tokens, skip_special_tokens=True)

        complied = detect_compliance(response_text, REFUSAL_MARKERS)
        labels.append(0 if complied else 1)
        response_records.append({
            "id": row.get("id", f"p{i:03d}"),
            "kind": row.get("kind", "unknown"),
            "prompt": prompt[:120],
            "response_excerpt": response_text[:200],
            "complied": complied,
            "label": labels[-1],
        })

        if (i + 1) % 5 == 0 or (i + 1) == len(prompts):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / max(rate, 1e-9)
            print(f"  {i+1}/{len(prompts)} complied={complied} "
                  f"[{elapsed:.0f}s, ETA {eta:.0f}s]")

    n_refused = sum(labels)
    n_complied = len(labels) - n_refused
    print(f"[3/5] extracted: {n_refused} refused, {n_complied} complied "
          f"(class balance for Task C)")

    if n_refused < 2 or n_complied < 2:
        print("[FATAL] too few examples in one class to train probes")
        print("  tune --n_unsafe / --n_safe or swap model")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "compliance_labels_llama_1b.json").write_text(
            json.dumps(response_records, indent=2), encoding="utf-8")
        sys.exit(2)

    print(f"[4/5] training per-layer probes (LOO-CV) ...")
    y = np.array(labels)
    per_layer: List[Dict] = []
    for layer_idx in range(n_layers + 1):
        X = torch.stack(residuals[layer_idx]).numpy()
        loo = LeaveOneOut()
        preds = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in loo.split(X):
            clf = LogisticRegression(
                penalty="l2", C=1.0,
                max_iter=2000, solver="liblinear",
            )
            clf.fit(X[train_idx], y[train_idx])
            preds[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        try:
            auc = roc_auc_score(y, preds)
        except Exception:
            auc = float("nan")
        per_layer.append({"layer": layer_idx, "auc_loo": auc})
        print(f"    layer {layer_idx:2d}  AUC={auc:.3f}")

    best = max((r for r in per_layer if not _isnan(r["auc_loo"])),
               key=lambda r: r["auc_loo"])
    best_layer = best["layer"]
    print(f"\nbest layer: {best_layer} (AUC={best['auc_loo']:.3f})")

    X_best = torch.stack(residuals[best_layer]).numpy()
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                              solver="liblinear")
    clf.fit(X_best, y)
    probe_weight = torch.tensor(clf.coef_[0], dtype=torch.float32)
    probe_bias = float(clf.intercept_[0])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "_").replace("-", "_")
    weights_fp = out_dir / f"{slug}_comply_refuse.pt"
    torch.save({"weight": probe_weight, "bias": probe_bias}, weights_fp)

    manifest = {
        "probe_version": "v0",
        "atlas_version": "v0",
        "model": args.model,
        "task": "comply_refuse",
        "positive_class": "refuse",
        "negative_class": "comply",
        "layer": best_layer,
        "total_layers": n_layers + 1,
        "hidden_size": hidden_size,
        "training_n": len(labels),
        "training_prompt_set": probe_set_source,
        "sampling_seed": args.seed,
        "class_balance": [n_refused, n_complied],
        "auc_validation": round(best["auc_loo"], 4),
        "auc_validation_method": "leave-one-out",
        "per_layer_auc": per_layer,
        "fitted_on": time.strftime("%Y-%m-%d"),
        "weight_file": weights_fp.name,
    }
    manifest_fp = out_dir / f"{slug}_comply_refuse.json"
    manifest_fp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    (out_dir / "compliance_labels_llama_1b.json").write_text(
        json.dumps(response_records, indent=2), encoding="utf-8")

    print(f"\n=== DONE ===")
    print(f"wrote {weights_fp}")
    print(f"wrote {manifest_fp}")
    print(f"total runtime: {time.time() - t0:.1f}s")


def _isnan(x) -> bool:
    try:
        return x != x
    except Exception:
        return False


if __name__ == "__main__":
    main()
