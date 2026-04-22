# -*- coding: utf-8 -*-
"""
benchmarks/causal_patching/train_concept_probe.py

Generalized concept-probe trainer for the Cognitive State Vector (CSV)
research sprint. Same mechanics as `extract_and_train.py` (residual
extraction + per-layer linear probe), but:

  1. Pluggable concept registry (refuse, sycophant, confab, ...).
  2. Always saves the per-layer weight directions, not just the
     best-AUC layer. This is what enables the probe-geometry experiment
     in `measure_probe_geometry.py` — we compare probe directions at
     the same layer across different concepts.

Each concept declares:
  - positive / negative class names
  - a prompt-set builder with "train" / "test" splits
  - a labeling strategy: either "input" (labels are derived from which
    dataset the prompt came from — no generation needed) or "behavioral"
    (we generate, then classify the response)

Usage
-----
  # train refuse concept probes on Llama-1B
  python benchmarks/causal_patching/train_concept_probe.py \
    --concept refuse --model meta-llama/Llama-3.2-1B-Instruct \
    --out_dir styxx/residual_probe/atlas --n_pos 40 --n_neg 40

  # train sycophant concept probes (uses meg-tong/sycophancy-eval)
  python benchmarks/causal_patching/train_concept_probe.py \
    --concept sycophant --out_dir styxx/residual_probe/atlas

  # train confab concept probes (uses local fixtures + JBB-benign)
  python benchmarks/causal_patching/train_concept_probe.py \
    --concept confab --out_dir styxx/residual_probe/atlas
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract_and_train import _jbb_split_pools, JBB_TRAIN_HALF  # noqa: E402
from refusal_heuristic import REFUSAL_MARKERS, detect_compliance  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Concept: refuse (JBB-harmful vs JBB-benign; behavioral labeling)
# ─────────────────────────────────────────────────────────────────────

def _refuse_build(n_pos: int, n_neg: int, seed: int,
                  split: str) -> List[Dict]:
    """Positive class = unsafe (JBB harmful), negative = safe (JBB benign).
    Behavioral labeling happens at runtime from model response."""
    harmful_pool, benign_pool = _jbb_split_pools(split)
    if n_pos > len(harmful_pool) or n_neg > len(benign_pool):
        raise ValueError(
            f"JBB {split} half has {len(harmful_pool)}/{len(benign_pool)} "
            f"pos/neg, requested {n_pos}/{n_neg}"
        )
    rng = random.Random(seed)
    pos = rng.sample(harmful_pool, n_pos)
    neg = rng.sample(benign_pool, n_neg)

    rows: List[Dict] = []
    for i, p in enumerate(pos):
        rows.append({"id": f"jbb-harm-{split}-{i:03d}",
                     "kind": "unsafe", "prompt": p})
    for i, p in enumerate(neg):
        rows.append({"id": f"jbb-ben-{split}-{i:03d}",
                     "kind": "safe", "prompt": p})
    rng.shuffle(rows)
    return rows


def _refuse_label_behavioral(row: Dict, response_text: str) -> int:
    """1 = refused, 0 = complied."""
    return 0 if detect_compliance(response_text, REFUSAL_MARKERS) else 1


# ─────────────────────────────────────────────────────────────────────
# Concept: sycophant (meg-tong/sycophancy-eval; input-template labeling)
# ─────────────────────────────────────────────────────────────────────

SYCO_MASTER_SEED = 43    # distinct from JBB master seed
SYCO_TRAIN_CAP = 200     # items from each class for the train half


def _syco_split_pools(split: str) -> Tuple[List[str], List[str]]:
    """Return (pressure_pool, neutral_pool) for the requested split.

    meg-tong/sycophancy-eval has paired rows: a neutral question and a
    sycophancy-pressure version of the same question. We classify by
    prompt template — 'pressure' if it contains user-expressed belief
    ('I think', 'I don't think', 'My professor said', etc.); 'neutral'
    otherwise.
    """
    from datasets import load_dataset

    # meg-tong/sycophancy-eval has heterogeneous metadata schemas across
    # rows (prompt_template_type only on some). Non-streaming load
    # fails schema-unification; streaming bypasses the cast.
    ds = load_dataset("meg-tong/sycophancy-eval", split="train",
                      streaming=True)
    # Conservative marker list — every marker must be a fairly unambiguous
    # first-person *belief injection*. We deliberately exclude weak
    # markers like "please" and "i feel" because they occur in neutral
    # questions too and would inflate the pressure pool.
    pressure_markers = (
        "i don't think the answer",
        "i do not think the answer",
        "i think the answer is",
        "i'm pretty sure the answer",
        "i am pretty sure the answer",
        "my professor said",
        "my professor told",
        "my teacher said",
        "my teacher told",
        "but i'm really not sure",
        "but i'm not sure",
    )

    pressure: List[str] = []
    neutral: List[str] = []
    seen = 0
    MAX_SCAN = 4000  # plenty of paired prompts
    for row in ds:
        seen += 1
        if seen > MAX_SCAN:
            break
        prompt_list = row.get("prompt")
        if not prompt_list or not isinstance(prompt_list, list):
            continue
        content = next(
            (seg.get("content") for seg in prompt_list
             if isinstance(seg, dict) and seg.get("type") == "human"),
            None,
        )
        if not content:
            continue
        cl = content.lower()
        if any(m in cl for m in pressure_markers):
            pressure.append(content)
        else:
            neutral.append(content)

    master = random.Random(SYCO_MASTER_SEED)
    master.shuffle(pressure)
    master.shuffle(neutral)

    if split == "train":
        return pressure[:SYCO_TRAIN_CAP], neutral[:SYCO_TRAIN_CAP]
    if split == "test":
        return pressure[SYCO_TRAIN_CAP:], neutral[SYCO_TRAIN_CAP:]
    raise ValueError(f"unknown split {split!r}")


def _syco_build(n_pos: int, n_neg: int, seed: int,
                split: str) -> List[Dict]:
    pressure_pool, neutral_pool = _syco_split_pools(split)
    if n_pos > len(pressure_pool) or n_neg > len(neutral_pool):
        raise ValueError(
            f"sycophancy-eval {split} half has "
            f"{len(pressure_pool)}/{len(neutral_pool)} pressure/neutral, "
            f"requested {n_pos}/{n_neg}"
        )
    rng = random.Random(seed)
    pos = rng.sample(pressure_pool, n_pos)
    neg = rng.sample(neutral_pool, n_neg)

    rows: List[Dict] = []
    for i, p in enumerate(pos):
        rows.append({"id": f"syco-press-{split}-{i:03d}",
                     "kind": "pressure", "prompt": p,
                     "_input_label": 1})
    for i, p in enumerate(neg):
        rows.append({"id": f"syco-neut-{split}-{i:03d}",
                     "kind": "neutral", "prompt": p,
                     "_input_label": 0})
    rng.shuffle(rows)
    return rows


def _syco_label_input(row: Dict, response_text: str) -> int:
    """Label comes from the prompt template (pressure-eliciting vs not),
    not the response. We're training a "detect social pressure"
    direction — the residual-stream readout of user-injected belief,
    independent of whether the model caved to it."""
    return int(row["_input_label"])


# ─────────────────────────────────────────────────────────────────────
# Concept: confab (confabulation_fixtures_v3.jsonl + JBB-benign)
# ─────────────────────────────────────────────────────────────────────

CONFAB_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "confabulation_fixtures_v3.jsonl"
)
CONFAB_MASTER_SEED = 44


def _confab_split_pools(split: str) -> Tuple[List[str], List[str]]:
    """Positive = confab-eliciting prompts (fake papers/people from the
    v3 fixture file). Negative = benign answerable questions (JBB
    benign half). Both halves are deterministically split into
    train/test to keep probes disjoint."""
    # Positives: fake-entity elicitors
    fixtures = []
    with CONFAB_FIXTURE_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("should_confabulate"):
                fixtures.append(row["prompt"])

    # Negatives: JBB benign (real, answerable)
    harmful_pool, benign_pool = _jbb_split_pools("train")
    harmful_pool2, benign_pool2 = _jbb_split_pools("test")
    negatives_all = benign_pool + benign_pool2

    master = random.Random(CONFAB_MASTER_SEED)
    master.shuffle(fixtures)
    master.shuffle(negatives_all)

    # Keep the pool halves symmetric: 50% of smallest pool per split.
    half_pos = len(fixtures) // 2
    half_neg = len(negatives_all) // 2
    if split == "train":
        return fixtures[:half_pos], negatives_all[:half_neg]
    if split == "test":
        return fixtures[half_pos:], negatives_all[half_neg:]
    raise ValueError(f"unknown split {split!r}")


def _confab_build(n_pos: int, n_neg: int, seed: int,
                  split: str) -> List[Dict]:
    pos_pool, neg_pool = _confab_split_pools(split)
    if n_pos > len(pos_pool) or n_neg > len(neg_pool):
        raise ValueError(
            f"confab {split} half has {len(pos_pool)}/{len(neg_pool)} "
            f"pos/neg, requested {n_pos}/{n_neg}"
        )
    rng = random.Random(seed)
    pos = rng.sample(pos_pool, n_pos)
    neg = rng.sample(neg_pool, n_neg)

    rows: List[Dict] = []
    for i, p in enumerate(pos):
        rows.append({"id": f"confab-fake-{split}-{i:03d}",
                     "kind": "fake", "prompt": p,
                     "_input_label": 1})
    for i, p in enumerate(neg):
        rows.append({"id": f"confab-real-{split}-{i:03d}",
                     "kind": "real", "prompt": p,
                     "_input_label": 0})
    rng.shuffle(rows)
    return rows


def _confab_label_input(row: Dict, response_text: str) -> int:
    """Train on input-label for simplicity: does the prompt ask about
    a fabricated entity? The direction we learn is "prompt is about
    something non-existent" — the residual-stream readout of the
    model detecting fiction-request. Behavioral labeling (did it
    actually confabulate?) is a follow-up experiment."""
    return int(row["_input_label"])


# ─────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────

CONCEPT_REGISTRY = {
    "refuse": {
        "task": "comply_refuse",
        "positive_class": "refuse",
        "negative_class": "comply",
        "build": _refuse_build,
        "label_fn": _refuse_label_behavioral,
        "label_mode": "behavioral",
        "default_n_pos": 40,
        "default_n_neg": 40,
    },
    "sycophant": {
        "task": "sycophant_pressure",
        "positive_class": "pressure",
        "negative_class": "neutral",
        "build": _syco_build,
        "label_fn": _syco_label_input,
        "label_mode": "input",
        "default_n_pos": 60,
        "default_n_neg": 60,
    },
    "confab": {
        "task": "confab_prompt",
        "positive_class": "fake",
        "negative_class": "real",
        "build": _confab_build,
        "label_fn": _confab_label_input,
        "label_mode": "input",
        "default_n_pos": 40,
        "default_n_neg": 40,
    },
}


# ─────────────────────────────────────────────────────────────────────
# Core training loop
# ─────────────────────────────────────────────────────────────────────

def train_concept_probe(
    *,
    concept: str,
    model_name: str,
    out_dir: Path,
    n_pos: int,
    n_neg: int,
    seed: int,
    device: str,
    max_new_tokens: int,
) -> Dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneOut
    import numpy as np

    if concept not in CONCEPT_REGISTRY:
        raise ValueError(
            f"unknown concept {concept!r}; "
            f"available: {list(CONCEPT_REGISTRY)}"
        )
    spec = CONCEPT_REGISTRY[concept]

    print(f"[concept={concept}] task={spec['task']} "
          f"pos={spec['positive_class']} neg={spec['negative_class']} "
          f"label_mode={spec['label_mode']}")

    prompts = spec["build"](n_pos, n_neg, seed, "train")

    print(f"[1/5] loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(torch.bfloat16 if device == "cuda" else torch.float32),
        output_hidden_states=True,
    )
    mdl.eval()
    device_real = device if torch.cuda.is_available() else "cpu"
    mdl.to(device_real)
    n_layers = mdl.config.num_hidden_layers
    hidden_size = mdl.config.hidden_size
    print(f"  loaded on {device_real} - layers={n_layers} hidden={hidden_size}")

    print(f"[2/5] extracting residuals from {len(prompts)} prompts "
          f"(label_mode={spec['label_mode']}) ...")

    residuals: List[List[torch.Tensor]] = [[] for _ in range(n_layers + 1)]
    labels: List[int] = []
    records: List[Dict] = []

    t0 = time.time()
    need_generate = spec["label_mode"] == "behavioral"

    for i, row in enumerate(prompts):
        prompt = row["prompt"]
        input_ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device_real)
        prefill_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = mdl(input_ids=input_ids, output_hidden_states=True)
            for layer_idx, h in enumerate(outputs.hidden_states):
                last_token = h[0, -1, :].to(torch.float32).cpu()
                residuals[layer_idx].append(last_token)

            if need_generate:
                gen = mdl.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
                new_tokens = gen[0, prefill_len:].tolist()
                response_text = tok.decode(new_tokens, skip_special_tokens=True)
            else:
                response_text = ""

        label = spec["label_fn"](row, response_text)
        labels.append(label)
        records.append({
            "id": row.get("id", f"p{i:03d}"),
            "kind": row.get("kind", "unknown"),
            "prompt": prompt[:120],
            "response_excerpt": response_text[:200] if response_text else "",
            "label": label,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
            elapsed = time.time() - t0
            eta = (len(prompts) - i - 1) * (elapsed / (i + 1))
            print(f"  {i+1}/{len(prompts)} label={label} "
                  f"[{elapsed:.0f}s, ETA {eta:.0f}s]")

    n_pos_obs = sum(labels)
    n_neg_obs = len(labels) - n_pos_obs
    print(f"[3/5] labeled: {n_pos_obs} pos, {n_neg_obs} neg")
    if n_pos_obs < 2 or n_neg_obs < 2:
        raise RuntimeError(
            f"[{concept}] too few examples in one class ({n_pos_obs}/{n_neg_obs})"
        )

    print(f"[4/5] training per-layer probes (LOO-CV) ...")
    y = np.array(labels)
    per_layer_records: List[Dict] = []
    per_layer_weights: Dict[str, List[float]] = {}
    per_layer_bias: Dict[str, float] = {}

    for layer_idx in range(n_layers + 1):
        X = torch.stack(residuals[layer_idx]).numpy()
        loo = LeaveOneOut()
        preds = np.zeros_like(y, dtype=float)
        for tr, te in loo.split(X):
            clf = LogisticRegression(
                penalty="l2", C=1.0, max_iter=2000, solver="liblinear",
            )
            clf.fit(X[tr], y[tr])
            preds[te] = clf.predict_proba(X[te])[:, 1]
        try:
            auc = float(roc_auc_score(y, preds))
        except Exception:
            auc = float("nan")
        per_layer_records.append({"layer": layer_idx, "auc_loo": auc})

        # Also refit on the full set to get the "final" direction for
        # this layer (used by probe geometry analysis).
        clf_full = LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, solver="liblinear",
        )
        clf_full.fit(X, y)
        per_layer_weights[str(layer_idx)] = clf_full.coef_[0].tolist()
        per_layer_bias[str(layer_idx)] = float(clf_full.intercept_[0])

        print(f"    layer {layer_idx:2d}  AUC={auc:.3f}")

    valid = [r for r in per_layer_records
             if r["auc_loo"] == r["auc_loo"]]  # not-NaN check
    if not valid:
        raise RuntimeError(f"[{concept}] no layer produced a valid AUC")
    best = max(valid, key=lambda r: r["auc_loo"])
    best_layer = best["layer"]
    print(f"\nbest layer: {best_layer} (AUC={best['auc_loo']:.3f})")

    best_weight = torch.tensor(
        per_layer_weights[str(best_layer)], dtype=torch.float32
    )
    best_bias = per_layer_bias[str(best_layer)]

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = model_name.replace("/", "_").replace("-", "_")
    stem = f"{slug}_{spec['task']}"
    weights_fp = out_dir / f"{stem}.pt"

    # Save a multi-layer artifact. 'weight' + 'bias' keep compatibility
    # with existing styxx.residual_probe loaders (they read the best
    # layer); 'weight_per_layer' / 'bias_per_layer' are the new payload
    # the geometry analysis consumes.
    torch.save({
        "weight": best_weight,
        "bias": best_bias,
        "weight_per_layer": {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in per_layer_weights.items()
        },
        "bias_per_layer": per_layer_bias,
    }, weights_fp)

    manifest = {
        "probe_version": "v1",                 # bumped: has per-layer weights
        "atlas_version": "v0",
        "concept": concept,
        "model": model_name,
        "task": spec["task"],
        "positive_class": spec["positive_class"],
        "negative_class": spec["negative_class"],
        "layer": best_layer,
        "total_layers": n_layers + 1,
        "hidden_size": hidden_size,
        "training_n": len(labels),
        "training_seed": seed,
        "class_balance": [n_pos_obs, n_neg_obs],
        "label_mode": spec["label_mode"],
        "auc_validation": round(best["auc_loo"], 4),
        "auc_validation_method": "leave-one-out",
        "per_layer_auc": per_layer_records,
        "fitted_on": time.strftime("%Y-%m-%d"),
        "weight_file": weights_fp.name,
    }
    manifest_fp = out_dir / f"{stem}.json"
    manifest_fp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    records_fp = out_dir / f"{stem}_records.json"
    records_fp.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(f"\n=== DONE: {concept} ===")
    print(f"wrote {weights_fp}")
    print(f"wrote {manifest_fp}")
    print(f"wrote {records_fp}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True,
                    choices=sorted(CONCEPT_REGISTRY))
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_pos", type=int, default=None)
    ap.add_argument("--n_neg", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    args = ap.parse_args()

    spec = CONCEPT_REGISTRY[args.concept]
    n_pos = args.n_pos if args.n_pos is not None else spec["default_n_pos"]
    n_neg = args.n_neg if args.n_neg is not None else spec["default_n_neg"]

    train_concept_probe(
        concept=args.concept,
        model_name=args.model,
        out_dir=Path(args.out_dir),
        n_pos=n_pos,
        n_neg=n_neg,
        seed=args.seed,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
