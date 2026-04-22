#!/usr/bin/env bash
# reproduce-cis-v0.sh — full reproduction of the Cognitive Instruction
# Set v0 results on Llama-3.2-1B-Instruct with open-source datasets.
#
# Requires a CUDA-capable GPU with ≥6GB VRAM.
#
# What this does, end to end:
#   1. Train the refusal probe (JBB-Behaviors).
#   2. Run the causal α-sweep for target ∈ {refuse, comply}.
#   3. Train the sycophancy-pressure probe (meg-tong/sycophancy-eval).
#   4. Train the confabulation-elicitation probe (local fixtures).
#   5. Measure pairwise probe geometry.
#   6. Run the full multi-concept / cogvm demo.
#
# Expected wall-clock on an RTX 4070 laptop GPU: ~20-25 min.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ATLAS="$ROOT/styxx/residual_probe/atlas"
SWEEP="$ROOT/benchmarks/causal_patching/runs/v0"

echo "[1/6] train refusal probe"
python benchmarks/causal_patching/extract_and_train.py \
    --out_dir "$ATLAS" \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --n_unsafe 40 --n_safe 40 --seed 0 --dataset jbb

echo "[2/6] α-sweep causal patching"
python benchmarks/causal_patching/run_patching.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --task comply_refuse \
    --out_dir "$SWEEP" \
    --n_unsafe 30 --n_safe 30 --test_seed 1 --dataset jbb \
    --alphas 0.0 0.5 1.0 1.5 2.0 2.5 3.0 --max_new_tokens 80

echo "[3/6] train sycophant_pressure probe"
python benchmarks/causal_patching/train_concept_probe.py \
    --concept sycophant --out_dir "$ATLAS" \
    --model meta-llama/Llama-3.2-1B-Instruct --n_pos 60 --n_neg 60

echo "[4/6] train confab_prompt probe"
python benchmarks/causal_patching/train_concept_probe.py \
    --concept confab --out_dir "$ATLAS" \
    --model meta-llama/Llama-3.2-1B-Instruct --n_pos 20 --n_neg 40

echo "[5/6] measure probe geometry"
python benchmarks/causal_patching/measure_probe_geometry.py \
    --probes "$ATLAS"/meta_llama_Llama_3.2_1B_Instruct_*.json \
    --out_file "$SWEEP/geometry.json"

echo "[6/6] run cogvm demo"
python benchmarks/cogvm_demo/demo_multi_concept.py

echo
echo "=== CIS v0 reproduction complete ==="
echo "Artifacts:"
echo "  $ATLAS/*.pt, *.json  — probe weights + manifests"
echo "  $SWEEP/*              — α-sweep results + geometry"
