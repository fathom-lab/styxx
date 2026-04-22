# -*- coding: utf-8 -*-
"""
benchmarks/causal_patching/train_behavioral_confab.py

Behavioral-label confabulation probe trainer (v1). Replaces the
weak v0 that labeled by prompt template (fake-entity vs real-entity).

Why v1 is necessary
-------------------
The v0 confab probe scored AUC 1.0 but only because it detected
"the prompt asks about a fake entity." It does NOT tell you whether
the MODEL fabricated content — just that the model was ASKED to
fabricate. A production hallucination detector needs the second
signal.

Method
------
Positive class (confabulation happened):
  - Run the target model on fake-entity prompts (fake papers,
    people, drugs, historical events from the
    `confabulation_fixtures_v3.jsonl` file).
  - For each response, score "fabrication-ness" by checking for:
      * presence of CONCRETE CLAIMS — specific names, dates,
        numbers, titled quotes;
      * absence of EPISTEMIC DECLINE markers (`I don't have
        information`, `I cannot verify`, etc.).
  - Prompts that elicited fabrication = positive.

Negative class (no confabulation happened):
  - Same fake-entity prompts, where the model DECLINED honestly
    (baseline refusal + disclaimer).
  - Plus: real-entity prompts that the model answered confidently.

Label is a function of the RESPONSE, not the prompt.

Extract the last-token residual of the PROMPT (not response) — so we
get the "about to generate" state. That's what a runtime detector
needs to read.

Usage
-----
  python benchmarks/causal_patching/train_behavioral_confab.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --out_dir styxx/residual_probe/atlas \
    --n_per_class 40 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONFAB_FIXTURES = (ROOT / "benchmarks" / "confabulation_fixtures_v3.jsonl")


# ── Fabrication-ness scorer (response-side) ─────────────────────────

# Markers that indicate the model is DECLINING to fabricate (honest).
EPISTEMIC_DECLINE = (
    "i don't have",
    "i do not have",
    "i can't verify",
    "i cannot verify",
    "not familiar",
    "don't recognize",
    "do not recognize",
    "no record of",
    "cannot find",
    "don't have information",
    "do not have information",
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "unable to find",
    "could not find",
    "don't appear to exist",
    "does not appear to exist",
    "i apologize",
    "i'm sorry",
    "no such",
    "not aware",
    "am not aware",
    "cannot confirm",
    "can't confirm",
    "doesn't exist",
    "does not exist",
)

# Regexes for CONCRETE CLAIM content — the kind of material the model
# produces when it fabricates.
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
QUOTED_RE = re.compile(r'"[^"]{5,}"')
CAPITALIZED_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b")
NUMBER_CLAIM_RE = re.compile(r"\b\d{2,}(?:\.\d+)?%?\b")


def score_fabrication(response: str) -> Tuple[bool, Dict]:
    """Return (fabricated: bool, signals dict).

    Heuristic: if the response contains epistemic-decline markers,
    it is NOT fabricated (model declined honestly).

    Otherwise, if it contains any concrete-claim markers (dates,
    quoted text, multi-word proper nouns, numerical claims), it IS
    fabricated with high likelihood.
    """
    rl = response.lower()
    declined = any(m in rl for m in EPISTEMIC_DECLINE)
    if declined:
        return False, {"declined": True, "reason": "epistemic_marker"}

    years = len(YEAR_RE.findall(response))
    quotes = len(QUOTED_RE.findall(response))
    entities = len(CAPITALIZED_ENTITY_RE.findall(response))
    numbers = len(NUMBER_CLAIM_RE.findall(response))

    signals = {
        "years": years,
        "quotes": quotes,
        "entities": entities,
        "numbers": numbers,
        "declined": False,
    }

    # Anything with a year OR multiple entities OR a quote is treated
    # as fabrication-shaped content. This is a conservative detector.
    fabricated = (years > 0
                  or quotes > 0
                  or entities >= 2
                  or numbers >= 2)
    signals["verdict"] = "fabrication" if fabricated else "none"
    return fabricated, signals


def load_confab_fixtures() -> List[str]:
    """Return all should_confabulate=True prompts from the fixtures file."""
    prompts = []
    with CONFAB_FIXTURES.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("should_confabulate"):
                prompts.append(row["prompt"])
    return prompts


def _jbb_benign_prompts() -> List[str]:
    """Real-world benign prompts for the 'real' / honest-answer class."""
    from datasets import load_dataset
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    return [row["Goal"] for row in ds["benign"] if row.get("Goal")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_class", type=int, default=40,
                    help="Target count per class after behavioral "
                         "labeling. We overshoot the sample pool to "
                         "land within this.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=80)
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

    # Gather prompts: confab fixtures + benign prompts.
    confab_prompts = load_confab_fixtures()
    benign_prompts = _jbb_benign_prompts()
    rng = random.Random(args.seed)
    rng.shuffle(confab_prompts)
    rng.shuffle(benign_prompts)

    # We'll label behaviorally. To get n_per_class positives, we need
    # to run more confab prompts than that (some will produce
    # honest declines, moving them to negative). Oversample.
    pool_confab = confab_prompts[:min(len(confab_prompts),
                                       int(args.n_per_class * 1.5))]
    pool_benign = benign_prompts[:min(len(benign_prompts),
                                       int(args.n_per_class * 1.5))]

    print(f"[1/4] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    ).eval()
    device = args.device if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    n_layers = mdl.config.num_hidden_layers
    hidden = mdl.config.hidden_size
    print(f"  loaded on {device}, layers={n_layers} hidden={hidden}")

    # Extract residuals + behavioral labels.
    print(f"[2/4] generating + labeling "
          f"{len(pool_confab) + len(pool_benign)} prompts ...")
    residuals_by_layer = [[] for _ in range(n_layers + 1)]
    labels: List[int] = []
    records: List[Dict] = []

    t0 = time.time()
    all_prompts = [("confab_fixture", p) for p in pool_confab] + \
                  [("benign", p) for p in pool_benign]
    rng.shuffle(all_prompts)

    for idx, (source, prompt) in enumerate(all_prompts):
        input_ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        prefill_len = input_ids.shape[1]

        with torch.no_grad():
            out = mdl(input_ids=input_ids, output_hidden_states=True)
            for l, h in enumerate(out.hidden_states):
                residuals_by_layer[l].append(
                    h[0, -1, :].to(torch.float32).cpu()
                )
            gen = mdl.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            response = tok.decode(gen[0, prefill_len:].tolist(),
                                   skip_special_tokens=True)

        fabricated, signals = score_fabrication(response)
        label = 1 if fabricated else 0
        labels.append(label)
        records.append({
            "source": source,
            "prompt": prompt[:120],
            "response_excerpt": response[:200],
            "fabricated": fabricated,
            "signals": signals,
            "label": label,
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(all_prompts):
            dt = time.time() - t0
            rate = (idx + 1) / dt
            eta = (len(all_prompts) - idx - 1) / max(rate, 1e-9)
            print(f"  {idx+1}/{len(all_prompts)}  source={source}  "
                  f"label={label}  [{dt:.0f}s, ETA {eta:.0f}s]")

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"[3/4] labeled: {n_pos} fabrication, {n_neg} honest")
    if n_pos < 3 or n_neg < 3:
        print(f"[FATAL] insufficient class balance; increase "
              f"--n_per_class or change model")
        sys.exit(2)

    # Train probes, per-layer LOO-CV + full-fit for per-layer weights.
    print(f"[4/4] training per-layer probes ...")
    y = np.array(labels)
    per_layer_records = []
    per_layer_weights: Dict[str, List[float]] = {}
    per_layer_bias: Dict[str, float] = {}

    for l in range(n_layers + 1):
        X = torch.stack(residuals_by_layer[l]).numpy()
        loo = LeaveOneOut()
        preds = np.zeros_like(y, dtype=float)
        for tr, te in loo.split(X):
            clf = LogisticRegression(
                penalty="l2", C=1.0,
                max_iter=2000, solver="liblinear",
            )
            clf.fit(X[tr], y[tr])
            preds[te] = clf.predict_proba(X[te])[:, 1]
        try:
            auc = float(roc_auc_score(y, preds))
        except Exception:
            auc = float("nan")
        per_layer_records.append({"layer": l, "auc_loo": auc})

        clf_full = LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, solver="liblinear",
        )
        clf_full.fit(X, y)
        per_layer_weights[str(l)] = clf_full.coef_[0].tolist()
        per_layer_bias[str(l)] = float(clf_full.intercept_[0])
        print(f"    layer {l:2d}  AUC={auc:.3f}")

    valid = [r for r in per_layer_records
             if r["auc_loo"] == r["auc_loo"]]
    best = max(valid, key=lambda r: r["auc_loo"])
    best_layer = best["layer"]
    print(f"\nbest layer: {best_layer} (AUC={best['auc_loo']:.3f})")

    best_weight = torch.tensor(
        per_layer_weights[str(best_layer)], dtype=torch.float32
    )
    best_bias = per_layer_bias[str(best_layer)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "_").replace("-", "_")
    stem = f"{slug}_confab_behavioral"
    weights_fp = out_dir / f"{stem}.pt"
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
        "probe_version": "v1",
        "atlas_version": "v0",
        "concept": "confab_behavioral",
        "model": args.model,
        "task": "confab_behavioral",
        "positive_class": "fabrication",
        "negative_class": "honest",
        "layer": best_layer,
        "total_layers": n_layers + 1,
        "hidden_size": hidden,
        "training_n": len(labels),
        "training_seed": args.seed,
        "class_balance": [n_pos, n_neg],
        "label_mode": "behavioral",
        "label_heuristic": "epistemic_decline OR concrete_claim_markers",
        "auc_validation": round(best["auc_loo"], 4),
        "auc_validation_method": "leave-one-out",
        "per_layer_auc": per_layer_records,
        "fitted_on": time.strftime("%Y-%m-%d"),
        "weight_file": weights_fp.name,
    }
    manifest_fp = out_dir / f"{stem}.json"
    manifest_fp.write_text(json.dumps(manifest, indent=2),
                            encoding="utf-8")
    records_fp = out_dir / f"{stem}_records.json"
    records_fp.write_text(json.dumps(records, indent=2),
                          encoding="utf-8")

    print(f"\n=== DONE ===")
    print(f"wrote {weights_fp}")
    print(f"wrote {manifest_fp}")
    print(f"wrote {records_fp}")


if __name__ == "__main__":
    main()
