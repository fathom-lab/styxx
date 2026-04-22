# -*- coding: utf-8 -*-
"""
benchmarks/causal_patching/run_patching.py

Day-2 causal claim for the v3.5.0 research sprint.

Where `extract_and_train.py` produces a probe that *predicts*
comply-vs-refuse from prefill residuals, this script tests whether
the probe direction is *causally* load-bearing: does adding alpha *
unit(probe_direction) to the residual stream at the trained layer
actually change the model's behavior?

Method
------
1. Load the trained probe via InterveneProbe.from_pretrained.
2. Build a held-out test set (different seed from training) from
   HarmBench (unsafe) + Alpaca (safe).
3. For each prompt, sweep alpha in a configurable grid and target
   each class. At alpha=0 we recover the baseline. At alpha > 0 we
   push toward the target class.
4. For every (prompt, alpha, target), record:
     - probe score before/after patch
     - actual generation text
     - behavioral label from the same refusal heuristic used in
       training (keeps labels consistent)
5. Aggregate:
     - Probe flip rate     : did the linear classifier change its
                             prediction? (mechanical)
     - Behavioral flip rate: did the observed generation change
                             comply <-> refuse? (the real causal
                             claim)
     - Capability cost    : for safe prompts, how often does
                            patching *break* an originally-complying
                            output into a refusal (false-positive
                            steer)?

Output
------
JSON with per-prompt rows + per-alpha aggregates, plus a short
markdown report for the paper.

Usage
-----
  python benchmarks/causal_patching/run_patching.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --task comply_refuse \
    --out_dir benchmarks/causal_patching/runs/v0 \
    --n_unsafe 40 --n_safe 40 --test_seed 1 \
    --alphas 0 0.5 1.0 1.5 2.0 2.5 3.0

Notes
-----
- test_seed MUST differ from the seed used in extract_and_train.py
  so the test set is disjoint from the probe's training data.
- Runs are idempotent: existing results/<alpha>.json is skipped
  unless --force is passed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refusal_heuristic import REFUSAL_MARKERS, detect_compliance  # noqa: E402


def build_test_set(n_unsafe: int, n_safe: int, seed: int,
                   dataset: str = "jbb") -> List[Dict]:
    """Held-out test set. Uses the same dataset family as training but
    the complementary half (when dataset='jbb') or a different seed
    (legacy dataset='harmbench')."""
    # Delegate to the training-side helper so both sides stay in sync.
    from extract_and_train import build_probe_set
    return build_probe_set(n_unsafe, n_safe, seed,
                           dataset=dataset, split="test")


def aggregate(rows: List[Dict]) -> Dict:
    """Per-alpha aggregate stats."""
    by_alpha: Dict[float, Dict[str, List]] = {}
    for r in rows:
        key = (round(r["alpha"], 4), r["target_class"])
        slot = by_alpha.setdefault(key, {
            "probe_flipped": [],
            "behavior_refused": [],
            "behavior_complied": [],
            "kinds": [],
            "pre_scores": [],
            "post_scores": [],
        })
        slot["probe_flipped"].append(1 if r["flipped"] else 0)
        slot["behavior_refused"].append(1 if r["refused_label"] else 0)
        slot["behavior_complied"].append(1 if not r["refused_label"] else 0)
        slot["kinds"].append(r["kind"])
        slot["pre_scores"].append(r["pre_score"])
        slot["post_scores"].append(r["post_score"])

    out = []
    for (alpha, target), slot in sorted(by_alpha.items()):
        n = len(slot["probe_flipped"])
        unsafe_idx = [i for i, k in enumerate(slot["kinds"]) if k == "unsafe"]
        safe_idx = [i for i, k in enumerate(slot["kinds"]) if k == "safe"]

        def _rate(mask_idx, key):
            if not mask_idx:
                return None
            vals = [slot[key][i] for i in mask_idx]
            return sum(vals) / len(vals)

        row = {
            "alpha": alpha,
            "target_class": target,
            "n": n,
            "probe_flip_rate": sum(slot["probe_flipped"]) / n,
            "refuse_rate_unsafe": _rate(unsafe_idx, "behavior_refused"),
            "refuse_rate_safe":   _rate(safe_idx,   "behavior_refused"),
            "mean_pre_score":     sum(slot["pre_scores"]) / n,
            "mean_post_score":    sum(slot["post_scores"]) / n,
        }
        out.append(row)
    return {"per_alpha": out}


def render_report(manifest: Dict, agg: Dict) -> str:
    """Short markdown report — one table per target_class."""
    lines = []
    lines.append(f"# Causal Patching Sweep")
    lines.append("")
    lines.append(f"- model: `{manifest['model']}`")
    lines.append(f"- task: `{manifest['task']}`")
    lines.append(f"- patched layer: {manifest['layer']}")
    lines.append(f"- probe AUC (LOO train): "
                 f"{manifest.get('auc_validation', 'n/a')}")
    lines.append("")

    by_target: Dict[str, List] = {}
    for r in agg["per_alpha"]:
        by_target.setdefault(r["target_class"], []).append(r)

    for target, rows in by_target.items():
        lines.append(f"## target_class = `{target}`")
        lines.append("")
        lines.append("| alpha | n | probe_flip | "
                     "refuse@unsafe | refuse@safe | pre -> post |")
        lines.append("|-------|---|-----------|"
                     "---------------|-------------|-------------|")
        for r in rows:
            ru = (f"{r['refuse_rate_unsafe']:.2f}"
                  if r['refuse_rate_unsafe'] is not None else "-")
            rs = (f"{r['refuse_rate_safe']:.2f}"
                  if r['refuse_rate_safe'] is not None else "-")
            lines.append(
                f"| {r['alpha']:.2f} | {r['n']} | "
                f"{r['probe_flip_rate']:.2f} | {ru} | {rs} | "
                f"{r['mean_pre_score']:.2f} -> {r['mean_post_score']:.2f} |"
            )
        lines.append("")

    lines.append("## reading the table")
    lines.append("")
    lines.append("- **probe_flip**: mechanical — always increases with alpha. "
                 "Confirms the hook fires.")
    lines.append("- **refuse@unsafe**: should INcrease when "
                 "target_class=refuse. That's the causal claim.")
    lines.append("- **refuse@safe**: should stay LOW under safe prompts. "
                 "If it rises with alpha, the direction isn't "
                 "refusal-specific — it's a generic 'say no' push, "
                 "which would be a negative result.")
    lines.append("- Capability cost = refuse@safe at the alpha that "
                 "maximizes refuse@unsafe.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--task", default="comply_refuse",
                    help="must match the trained probe's task")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_unsafe", type=int, default=40)
    ap.add_argument("--n_safe",   type=int, default=40)
    ap.add_argument("--test_seed", type=int, default=1,
                    help="MUST differ from training seed")
    ap.add_argument("--dataset", default="jbb",
                    choices=["jbb", "harmbench"],
                    help="match the dataset used at training time")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--targets", nargs="+",
                    default=["refuse", "comply"],
                    help="class names to steer toward")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing result files")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from styxx.residual_probe.intervene import InterveneProbe
    except Exception as e:
        print(f"FATAL: dependency missing: {e}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = InterveneProbe.from_pretrained(model=args.model, task=args.task)
    manifest = {
        "model": probe.model,
        "task": probe.task,
        "layer": probe.layer,
        "total_layers": probe.total_layers,
        "positive_class": probe.positive_class,
        "negative_class": probe.negative_class,
        "auc_validation": probe.auc_validation,
        "probe_version": probe.probe_version,
        "atlas_version": probe.atlas_version,
    }
    print(f"probe: {args.task} @ layer {probe.layer}/{probe.total_layers} "
          f"(AUC {probe.auc_validation})")

    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=(torch.bfloat16 if args.device == "cuda" else torch.float32),
        output_hidden_states=True,
    )
    mdl.eval()
    device = args.device if torch.cuda.is_available() else "cpu"
    mdl.to(device)

    test_rows = build_test_set(args.n_unsafe, args.n_safe, args.test_seed,
                               dataset=args.dataset)
    (out_dir / "test_set.json").write_text(
        json.dumps(
            [{"id": r["id"], "kind": r["kind"],
              "prompt": r["prompt"][:200]} for r in test_rows],
            indent=2), encoding="utf-8")
    print(f"test set: {len(test_rows)} prompts "
          f"(seed={args.test_seed}, held out from training)")

    all_results: List[Dict] = []
    t0 = time.time()
    total_steps = len(args.alphas) * len(args.targets) * len(test_rows)
    step = 0

    for target in args.targets:
        if target not in (probe.positive_class, probe.negative_class):
            print(f"  skipping target={target!r} "
                  f"(probe classes are {probe.positive_class}/"
                  f"{probe.negative_class})")
            continue
        for alpha in args.alphas:
            shard_fp = out_dir / f"results_target-{target}_alpha-{alpha:.2f}.jsonl"
            if shard_fp.exists() and not args.force:
                print(f"  skip existing: {shard_fp.name}")
                for line in shard_fp.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        all_results.append(json.loads(line))
                step += len(test_rows)
                continue

            shard_lines: List[str] = []
            for row in test_rows:
                step += 1
                try:
                    res = probe.intervene_and_generate(
                        model=mdl, tokenizer=tok,
                        prompt=row["prompt"],
                        target_class=target,
                        alpha=alpha,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                except Exception as e:
                    print(f"    [error] {row['id']} alpha={alpha} "
                          f"target={target}: {e}")
                    continue

                complied = detect_compliance(res.output_text, REFUSAL_MARKERS)
                refused_label = 0 if complied else 1

                rec = {
                    "id": row["id"],
                    "kind": row["kind"],
                    "prompt": row["prompt"][:200],
                    "target_class": target,
                    "alpha": alpha,
                    "layer_patched": res.layer_patched,
                    "pre_score": res.original_score,
                    "post_score": res.patched_score,
                    "flipped": res.flipped,
                    "output_excerpt": res.output_text[:200],
                    "refused_label": refused_label,
                    "generation_completed": res.generation_completed,
                }
                shard_lines.append(json.dumps(rec))
                all_results.append(rec)

                if step % 20 == 0 or step == total_steps:
                    elapsed = time.time() - t0
                    rate = step / max(elapsed, 1e-9)
                    eta = (total_steps - step) / max(rate, 1e-9)
                    print(f"  {step}/{total_steps} "
                          f"[target={target} alpha={alpha:.2f}] "
                          f"{elapsed:.0f}s ETA {eta:.0f}s")
            shard_fp.write_text("\n".join(shard_lines) + "\n",
                                encoding="utf-8")

    agg = aggregate(all_results)
    (out_dir / "aggregate.json").write_text(
        json.dumps({"manifest": manifest, "aggregate": agg}, indent=2),
        encoding="utf-8")
    (out_dir / "report.md").write_text(
        render_report(manifest, agg), encoding="utf-8")

    print(f"\n=== DONE ===")
    print(f"wrote {out_dir/'aggregate.json'}")
    print(f"wrote {out_dir/'report.md'}")
    print(f"total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
