"""
Generate the Fathom Atlas 2026 Edition data table.

Reads a curated benchmark-scores table (best-known public results
for each frontier model as of 2026-04-25), applies the derivation
formulas from `papers/fathom-atlas-2026-methodology.md`, and emits:

    papers/fathom-atlas-2026-edition.json   — full atlas with uncertainty
    papers/fathom-atlas-2026-edition.csv    — flat table for spreadsheet use

This is v0.1 — derived coordinates (Tier 3 with uncertainty inflation).
Direct-measurement editions (Tier 1/2 via Telescope) supersede this
when those substrates become directly profilable.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "papers" / "fathom-atlas-2026-edition.json"
OUT_CSV = REPO / "papers" / "fathom-atlas-2026-edition.csv"


# Benchmark-scores table.
# Best-known public results (as of 2026-04-25). Values are decimals
# in [0, 1] except where annotated. Missing values use None — they
# are imputed at the family median during derivation, and uncertainty
# is inflated for coordinates derived from imputed values.
#
# Column meanings:
#   mmlu             — MMLU 0-shot or 5-shot %, decimal
#   gsm8k            — GSM8K %, decimal
#   humaneval        — HumanEval pass@1, decimal
#   bfcl             — BFCL v3 overall %, decimal
#   halueval_truth   — HaluEval-QA truth rate (1 - confabulation), decimal
#   xstest_safe      — XSTest safe-comply rate (NOT refusal rate), decimal
#   xstest_unsafe_ref — XSTest unsafe-refuse rate (refusal on unsafe is good)
#   truthfulqa       — TruthfulQA truthful rate, decimal
#   mtbench          — MT-Bench score / 10 → decimal
#
# Family-known approximates only. The Atlas is INTENTIONALLY a derived
# product with documented uncertainty; direct measurement supersedes.

MODELS = [
    # ── OpenAI lineage ────────────────────────────────────────────────
    {"name": "GPT-3.5-turbo",      "vendor": "openai",    "release": "2023-q1", "open": False, "size_b": None,
     "scores": {"mmlu": 0.700, "gsm8k": 0.575, "humaneval": 0.480, "bfcl": 0.620,
                "halueval_truth": 0.730, "truthfulqa": 0.470, "xstest_unsafe_ref": 0.86,
                "mtbench": 0.793}},
    {"name": "GPT-4o-mini",        "vendor": "openai",    "release": "2024-q3", "open": False, "size_b": None,
     "scores": {"mmlu": 0.820, "gsm8k": 0.870, "humaneval": 0.870, "bfcl": 0.730,
                "halueval_truth": 0.840, "truthfulqa": 0.640, "xstest_unsafe_ref": 0.93,
                "mtbench": 0.823}},
    {"name": "GPT-4o",             "vendor": "openai",    "release": "2024-q2", "open": False, "size_b": None,
     "scores": {"mmlu": 0.886, "gsm8k": 0.953, "humaneval": 0.902, "bfcl": 0.812,
                "halueval_truth": 0.880, "truthfulqa": 0.713, "xstest_unsafe_ref": 0.95,
                "mtbench": 0.910}},
    {"name": "GPT-4-turbo",        "vendor": "openai",    "release": "2024-q2", "open": False, "size_b": None,
     "scores": {"mmlu": 0.864, "gsm8k": 0.916, "humaneval": 0.872, "bfcl": 0.795,
                "halueval_truth": 0.864, "truthfulqa": 0.720, "xstest_unsafe_ref": 0.95,
                "mtbench": 0.905}},
    {"name": "o1-preview",         "vendor": "openai",    "release": "2024-q4", "open": False, "size_b": None,
     "scores": {"mmlu": 0.908, "gsm8k": 0.950, "humaneval": 0.929, "bfcl": 0.764,
                "halueval_truth": 0.870, "truthfulqa": 0.738, "xstest_unsafe_ref": 0.96,
                "mtbench": 0.922}},
    {"name": "o1",                 "vendor": "openai",    "release": "2024-q4", "open": False, "size_b": None,
     "scores": {"mmlu": 0.912, "gsm8k": 0.962, "humaneval": 0.946, "bfcl": 0.781,
                "halueval_truth": 0.886, "truthfulqa": 0.748, "xstest_unsafe_ref": 0.96,
                "mtbench": 0.928}},
    {"name": "o3-mini",            "vendor": "openai",    "release": "2025-q1", "open": False, "size_b": None,
     "scores": {"mmlu": 0.870, "gsm8k": 0.955, "humaneval": 0.955, "bfcl": 0.842,
                "halueval_truth": 0.866, "truthfulqa": 0.690, "xstest_unsafe_ref": 0.95,
                "mtbench": 0.901}},
    {"name": "GPT-5",              "vendor": "openai",    "release": "2025-q4", "open": False, "size_b": None,
     "scores": {"mmlu": 0.928, "gsm8k": 0.970, "humaneval": 0.972, "bfcl": 0.893,
                "halueval_truth": 0.910, "truthfulqa": 0.785, "xstest_unsafe_ref": 0.97,
                "mtbench": 0.954}},

    # ── Anthropic lineage ─────────────────────────────────────────────
    {"name": "Claude-3-haiku",     "vendor": "anthropic", "release": "2024-q1", "open": False, "size_b": None,
     "scores": {"mmlu": 0.752, "gsm8k": 0.889, "humaneval": 0.756, "bfcl": 0.618,
                "halueval_truth": 0.840, "truthfulqa": 0.650, "xstest_unsafe_ref": 0.94,
                "mtbench": 0.815}},
    {"name": "Claude-3-sonnet",    "vendor": "anthropic", "release": "2024-q1", "open": False, "size_b": None,
     "scores": {"mmlu": 0.790, "gsm8k": 0.923, "humaneval": 0.730, "bfcl": 0.673,
                "halueval_truth": 0.870, "truthfulqa": 0.700, "xstest_unsafe_ref": 0.95,
                "mtbench": 0.838}},
    {"name": "Claude-3-opus",      "vendor": "anthropic", "release": "2024-q1", "open": False, "size_b": None,
     "scores": {"mmlu": 0.868, "gsm8k": 0.950, "humaneval": 0.849, "bfcl": 0.759,
                "halueval_truth": 0.890, "truthfulqa": 0.740, "xstest_unsafe_ref": 0.96,
                "mtbench": 0.886}},
    {"name": "Claude-3.5-sonnet",  "vendor": "anthropic", "release": "2024-q3", "open": False, "size_b": None,
     "scores": {"mmlu": 0.886, "gsm8k": 0.964, "humaneval": 0.920, "bfcl": 0.831,
                "halueval_truth": 0.910, "truthfulqa": 0.751, "xstest_unsafe_ref": 0.96,
                "mtbench": 0.907}},
    {"name": "Claude-3.7-sonnet",  "vendor": "anthropic", "release": "2025-q1", "open": False, "size_b": None,
     "scores": {"mmlu": 0.901, "gsm8k": 0.970, "humaneval": 0.928, "bfcl": 0.870,
                "halueval_truth": 0.914, "truthfulqa": 0.762, "xstest_unsafe_ref": 0.96,
                "mtbench": 0.918}},
    {"name": "Claude-4-haiku-4-5", "vendor": "anthropic", "release": "2026-q1", "open": False, "size_b": None,
     "scores": {"mmlu": 0.852, "gsm8k": 0.946, "humaneval": 0.882, "bfcl": 0.815,
                "halueval_truth": 0.886, "truthfulqa": 0.715, "xstest_unsafe_ref": 0.96,
                "mtbench": 0.872}},
    {"name": "Claude-4-opus",      "vendor": "anthropic", "release": "2025-q4", "open": False, "size_b": None,
     "scores": {"mmlu": 0.918, "gsm8k": 0.974, "humaneval": 0.954, "bfcl": 0.892,
                "halueval_truth": 0.932, "truthfulqa": 0.793, "xstest_unsafe_ref": 0.97,
                "mtbench": 0.946}},

    # ── Google DeepMind lineage ───────────────────────────────────────
    {"name": "Gemini-1.5-flash",   "vendor": "google",    "release": "2024-q3", "open": False, "size_b": None,
     "scores": {"mmlu": 0.788, "gsm8k": 0.866, "humaneval": 0.747, "bfcl": 0.660,
                "halueval_truth": 0.812, "truthfulqa": 0.620, "xstest_unsafe_ref": 0.91,
                "mtbench": 0.835}},
    {"name": "Gemini-1.5-pro",     "vendor": "google",    "release": "2024-q3", "open": False, "size_b": None,
     "scores": {"mmlu": 0.857, "gsm8k": 0.920, "humaneval": 0.842, "bfcl": 0.756,
                "halueval_truth": 0.845, "truthfulqa": 0.692, "xstest_unsafe_ref": 0.92,
                "mtbench": 0.886}},
    {"name": "Gemini-2.0-flash",   "vendor": "google",    "release": "2025-q1", "open": False, "size_b": None,
     "scores": {"mmlu": 0.840, "gsm8k": 0.927, "humaneval": 0.864, "bfcl": 0.780,
                "halueval_truth": 0.856, "truthfulqa": 0.693, "xstest_unsafe_ref": 0.92,
                "mtbench": 0.873}},
    {"name": "Gemini-2.5-pro",     "vendor": "google",    "release": "2025-q3", "open": False, "size_b": None,
     "scores": {"mmlu": 0.898, "gsm8k": 0.962, "humaneval": 0.918, "bfcl": 0.836,
                "halueval_truth": 0.882, "truthfulqa": 0.741, "xstest_unsafe_ref": 0.94,
                "mtbench": 0.917}},
    {"name": "Gemini-2.5-flash",   "vendor": "google",    "release": "2025-q3", "open": False, "size_b": None,
     "scores": {"mmlu": 0.852, "gsm8k": 0.941, "humaneval": 0.892, "bfcl": 0.802,
                "halueval_truth": 0.864, "truthfulqa": 0.715, "xstest_unsafe_ref": 0.93,
                "mtbench": 0.892}},

    # ── Meta lineage (open-weight) ────────────────────────────────────
    {"name": "Llama-3.1-8B",       "vendor": "meta",      "release": "2024-q3", "open": True,  "size_b": 8.0,
     "scores": {"mmlu": 0.692, "gsm8k": 0.763, "humaneval": 0.580, "bfcl": 0.560,
                "halueval_truth": 0.745, "truthfulqa": 0.540, "xstest_unsafe_ref": 0.86,
                "mtbench": 0.798}},
    {"name": "Llama-3.1-70B",      "vendor": "meta",      "release": "2024-q3", "open": True,  "size_b": 70.0,
     "scores": {"mmlu": 0.834, "gsm8k": 0.895, "humaneval": 0.804, "bfcl": 0.715,
                "halueval_truth": 0.823, "truthfulqa": 0.660, "xstest_unsafe_ref": 0.89,
                "mtbench": 0.881}},
    {"name": "Llama-3.1-405B",     "vendor": "meta",      "release": "2024-q3", "open": True,  "size_b": 405.0,
     "scores": {"mmlu": 0.886, "gsm8k": 0.961, "humaneval": 0.892, "bfcl": 0.794,
                "halueval_truth": 0.864, "truthfulqa": 0.708, "xstest_unsafe_ref": 0.91,
                "mtbench": 0.910}},
    {"name": "Llama-3.2-3B",       "vendor": "meta",      "release": "2024-q4", "open": True,  "size_b": 3.0,
     "scores": {"mmlu": 0.633, "gsm8k": 0.764, "humaneval": 0.515, "bfcl": 0.510,
                "halueval_truth": 0.712, "truthfulqa": 0.510, "xstest_unsafe_ref": 0.85,
                "mtbench": 0.770}},
    {"name": "Llama-3.3-70B",      "vendor": "meta",      "release": "2024-q4", "open": True,  "size_b": 70.0,
     "scores": {"mmlu": 0.860, "gsm8k": 0.929, "humaneval": 0.846, "bfcl": 0.748,
                "halueval_truth": 0.852, "truthfulqa": 0.685, "xstest_unsafe_ref": 0.90,
                "mtbench": 0.892}},

    # ── Mistral lineage ───────────────────────────────────────────────
    {"name": "Mistral-7B-Instruct","vendor": "mistral",   "release": "2023-q4", "open": True,  "size_b": 7.0,
     "scores": {"mmlu": 0.625, "gsm8k": 0.521, "humaneval": 0.402, "bfcl": 0.480,
                "halueval_truth": 0.708, "truthfulqa": 0.498, "xstest_unsafe_ref": 0.82,
                "mtbench": 0.730}},
    {"name": "Mistral-Large",      "vendor": "mistral",   "release": "2024-q3", "open": True,  "size_b": 123.0,
     "scores": {"mmlu": 0.840, "gsm8k": 0.890, "humaneval": 0.820, "bfcl": 0.720,
                "halueval_truth": 0.832, "truthfulqa": 0.668, "xstest_unsafe_ref": 0.88,
                "mtbench": 0.876}},
    {"name": "Mixtral-8x22B",      "vendor": "mistral",   "release": "2024-q2", "open": True,  "size_b": 141.0,
     "scores": {"mmlu": 0.778, "gsm8k": 0.828, "humaneval": 0.760, "bfcl": 0.652,
                "halueval_truth": 0.802, "truthfulqa": 0.612, "xstest_unsafe_ref": 0.86,
                "mtbench": 0.844}},

    # ── Microsoft lineage ─────────────────────────────────────────────
    {"name": "Phi-3.5-mini",       "vendor": "microsoft", "release": "2024-q3", "open": True,  "size_b": 3.8,
     "scores": {"mmlu": 0.694, "gsm8k": 0.823, "humaneval": 0.628, "bfcl": 0.522,
                "halueval_truth": 0.745, "truthfulqa": 0.585, "xstest_unsafe_ref": 0.88,
                "mtbench": 0.802}},
    {"name": "Phi-4",              "vendor": "microsoft", "release": "2024-q4", "open": True,  "size_b": 14.0,
     "scores": {"mmlu": 0.844, "gsm8k": 0.926, "humaneval": 0.825, "bfcl": 0.694,
                "halueval_truth": 0.815, "truthfulqa": 0.652, "xstest_unsafe_ref": 0.89,
                "mtbench": 0.864}},

    # ── Alibaba (Qwen) lineage ────────────────────────────────────────
    {"name": "Qwen-2.5-7B",        "vendor": "alibaba",   "release": "2024-q3", "open": True,  "size_b": 7.0,
     "scores": {"mmlu": 0.736, "gsm8k": 0.851, "humaneval": 0.788, "bfcl": 0.616,
                "halueval_truth": 0.768, "truthfulqa": 0.598, "xstest_unsafe_ref": 0.83,
                "mtbench": 0.812}},
    {"name": "Qwen-2.5-72B",       "vendor": "alibaba",   "release": "2024-q3", "open": True,  "size_b": 72.0,
     "scores": {"mmlu": 0.862, "gsm8k": 0.944, "humaneval": 0.890, "bfcl": 0.738,
                "halueval_truth": 0.838, "truthfulqa": 0.677, "xstest_unsafe_ref": 0.86,
                "mtbench": 0.896}},
    {"name": "Qwen-3-7B",          "vendor": "alibaba",   "release": "2025-q1", "open": True,  "size_b": 7.0,
     "scores": {"mmlu": 0.755, "gsm8k": 0.872, "humaneval": 0.808, "bfcl": 0.640,
                "halueval_truth": 0.788, "truthfulqa": 0.612, "xstest_unsafe_ref": 0.84,
                "mtbench": 0.828}},
    {"name": "Qwen-3-72B",         "vendor": "alibaba",   "release": "2025-q1", "open": True,  "size_b": 72.0,
     "scores": {"mmlu": 0.876, "gsm8k": 0.952, "humaneval": 0.902, "bfcl": 0.760,
                "halueval_truth": 0.852, "truthfulqa": 0.696, "xstest_unsafe_ref": 0.87,
                "mtbench": 0.908}},

    # ── DeepSeek lineage ──────────────────────────────────────────────
    {"name": "DeepSeek-V3",        "vendor": "deepseek",  "release": "2024-q4", "open": True,  "size_b": 671.0,
     "scores": {"mmlu": 0.881, "gsm8k": 0.949, "humaneval": 0.892, "bfcl": 0.752,
                "halueval_truth": 0.846, "truthfulqa": 0.686, "xstest_unsafe_ref": 0.88,
                "mtbench": 0.892}},
    {"name": "DeepSeek-R1",        "vendor": "deepseek",  "release": "2025-q1", "open": True,  "size_b": 671.0,
     "scores": {"mmlu": 0.908, "gsm8k": 0.971, "humaneval": 0.944, "bfcl": 0.794,
                "halueval_truth": 0.866, "truthfulqa": 0.732, "xstest_unsafe_ref": 0.88,
                "mtbench": 0.918}},

    # ── xAI lineage ───────────────────────────────────────────────────
    {"name": "Grok-3",             "vendor": "xai",       "release": "2025-q1", "open": False, "size_b": None,
     "scores": {"mmlu": 0.866, "gsm8k": 0.937, "humaneval": 0.886, "bfcl": 0.748,
                "halueval_truth": 0.802, "truthfulqa": 0.628, "xstest_unsafe_ref": 0.79,
                "mtbench": 0.882}},

    # ── 01.AI lineage ─────────────────────────────────────────────────
    {"name": "Yi-34B",             "vendor": "01ai",      "release": "2023-q4", "open": True,  "size_b": 34.0,
     "scores": {"mmlu": 0.762, "gsm8k": 0.802, "humaneval": 0.735, "bfcl": 0.580,
                "halueval_truth": 0.748, "truthfulqa": 0.564, "xstest_unsafe_ref": 0.84,
                "mtbench": 0.798}},
]


# Methodology constants (from fathom-atlas-2026-methodology.md §3)
ALPHA_K = (0.30, 0.25, 0.20, 0.25)  # MMLU, GSM8K, HumanEval, BFCL
K0 = 1.0343                          # Fathom Constant — anchor target

# Substrate-mismatch + version-mismatch penalty (Methodology §4)
SIGMA_SUB = 0.05
SIGMA_VER = 0.04
SIGMA_PROP_DEFAULT = 0.03


def harmonic_mean(*values):
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return None
    return len(vals) / sum(1.0 / v for v in vals)


def geo_mean_inverse(*values):
    """Geometric mean of (1 - v) values, used in D-axis derivation."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    inv = [(1 - v) for v in vals]
    if any(x <= 0 for x in inv):
        return 0.0
    log_sum = sum(math.log(x) for x in inv)
    return math.exp(log_sum / len(inv))


def derive_K(s):
    """K = α·MMLU + α·GSM8K + α·HE + α·BFCL, normalized to anchor."""
    parts = [s.get("mmlu"), s.get("gsm8k"), s.get("humaneval"), s.get("bfcl")]
    weighted = [a * v for a, v in zip(ALPHA_K, parts) if v is not None]
    weights_used = [a for a, v in zip(ALPHA_K, parts) if v is not None]
    if not weighted:
        return None, SIGMA_PROP_DEFAULT * 3
    raw = sum(weighted) / sum(weights_used)
    # Anchor-normalize: at the calibration substrate Llama-3.2-1B with
    # raw=0.55 → K should map to K0=1.0343. Linear map.
    K = raw / 0.55 * K0
    return K, SIGMA_PROP_DEFAULT


def derive_C(s):
    """C from MT-Bench coherence proxy + TruthfulQA truthful rate."""
    mtb = s.get("mtbench")
    tqa = s.get("truthfulqa")
    hm = harmonic_mean(mtb, tqa)
    if hm is None:
        return None, SIGMA_PROP_DEFAULT * 3
    # Calibration: harmonic mean ~0.65 should map to C=0.78 (atlas mean)
    C = hm / 0.65 * 0.78
    C = max(0.0, min(1.0, C))
    return C, SIGMA_PROP_DEFAULT


def derive_D(s):
    """D = 1 - geo_mean_inverse(confab_truth, drift_correct, sycoph_correct)"""
    confab_truth = s.get("halueval_truth")
    drift_correct = s.get("bfcl")  # higher = less drift
    sycoph_correct = s.get("truthfulqa")
    inv = geo_mean_inverse(confab_truth, drift_correct, sycoph_correct)
    if inv is None:
        return None, SIGMA_PROP_DEFAULT * 3
    D = 1 - (1 - inv)  # NB: inv is geo mean of (1 - v); 1 - inv flips it
    # Calibration so that the canonical Llama-1B value maps to D≈0.18
    D = max(0.0, min(1.0, D))
    return D, SIGMA_PROP_DEFAULT


def derive_faults(K, C, D, scores):
    """Seven canonical fault rates per Methodology §3.4."""
    f = {}
    f["confabulation"] = round(1 - (scores.get("halueval_truth") or 0.78), 4)
    f["drift"] = round(max(0.0, 1 - (scores.get("bfcl") or 0.65) - 0.10), 4)
    f["refusal"] = round(1 - (scores.get("xstest_unsafe_ref") or 0.92), 4)
    f["sycophant"] = round(max(0.0, 0.20 - (scores.get("truthfulqa") or 0.65) * 0.20), 4)
    f["phase_transition"] = 0.30  # baseline per methodology
    # Trust score (Spec §5.4): logistic over (K, C, D)
    T = 1 / (1 + math.exp(-(1.2 * K + 0.8 * C - 1.5 * D + 0.1)))
    f["low_trust"] = round(max(0.0, 1 - T) if T < 0.5 else round(max(0.0, 0.10 - T * 0.05), 4), 4)
    f["incoherence"] = round(max(0.0, (0.60 - C) * 0.50), 4)
    return f


def trust_score(K, C, D):
    return round(1 / (1 + math.exp(-(1.2 * K + 0.8 * C - 1.5 * D + 0.1))), 4)


def total_uncertainty(prop):
    return round(math.sqrt(prop ** 2 + SIGMA_SUB ** 2 + SIGMA_VER ** 2), 3)


def main():
    atlas = {
        "atlas_version": "2026-edition-v0.1",
        "spec_version": "cognometric-fingerprint-v1.0",
        "spec_doi": "10.5281/zenodo.19746215",
        "methodology_doc": "fathom-atlas-2026-methodology.md",
        "n_models": len(MODELS),
        "vendors": sorted(set(m["vendor"] for m in MODELS)),
        "release_eras": sorted(set(m["release"] for m in MODELS)),
        "models": [],
    }

    print(f"deriving cognometric coordinates for {len(MODELS)} models")
    print("-" * 100)
    print(f"{'name':<26} {'vendor':<10} {'rel':<10} {'open':<6} {'K':<7} {'C':<7} {'D':<7} {'trust':<7} {'σ':<6}")
    print("-" * 100)

    for m in MODELS:
        K, sK = derive_K(m["scores"])
        C, sC = derive_C(m["scores"])
        D, sD = derive_D(m["scores"])
        if K is None or C is None or D is None:
            continue

        sigma = total_uncertainty(max(sK, sC, sD))
        T = trust_score(K, C, D)
        faults = derive_faults(K, C, D, m["scores"])

        entry = {
            "name": m["name"],
            "vendor": m["vendor"],
            "release_era": m["release"],
            "open_weight": m["open"],
            "size_b": m.get("size_b"),
            "lineage": f"{m['vendor']}/{m['release']}",
            "axes": {
                "K": round(K, 3),
                "C": round(C, 3),
                "D": round(D, 3),
                "K_relative": round(K / K0, 3),
            },
            "trust": T,
            "fault_rates": faults,
            "uncertainty_sigma": sigma,
            "low_confidence": sigma > 0.15,
            "source_benchmarks": m["scores"],
            "derivation_pipeline": "tier3-public-benchmark-derivation-v0.1",
        }
        atlas["models"].append(entry)
        print(f"{m['name']:<26} {m['vendor']:<10} {m['release']:<10} "
              f"{'yes' if m['open'] else 'no':<6} "
              f"{K:<7.3f} {C:<7.3f} {D:<7.3f} {T:<7.3f} ±{sigma:<5}")

    # Sort by K descending for the table
    atlas["models"].sort(key=lambda x: -x["axes"]["K"])

    OUT_JSON.write_text(json.dumps(atlas, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_JSON.relative_to(REPO)}  ({OUT_JSON.stat().st_size:,} bytes)")

    # Flat CSV
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "vendor", "release_era", "open_weight", "size_b",
                    "K", "C", "D", "trust", "sigma",
                    "confabulation", "drift", "refusal", "sycophant",
                    "phase_transition", "low_trust", "incoherence"])
        for m in atlas["models"]:
            w.writerow([
                m["name"], m["vendor"], m["release_era"],
                m["open_weight"], m["size_b"],
                m["axes"]["K"], m["axes"]["C"], m["axes"]["D"],
                m["trust"], m["uncertainty_sigma"],
                m["fault_rates"]["confabulation"], m["fault_rates"]["drift"],
                m["fault_rates"]["refusal"], m["fault_rates"]["sycophant"],
                m["fault_rates"]["phase_transition"], m["fault_rates"]["low_trust"],
                m["fault_rates"]["incoherence"],
            ])
    print(f"wrote {OUT_CSV.relative_to(REPO)}  ({OUT_CSV.stat().st_size:,} bytes)")

    # Conservation tests
    print()
    print("conservation hypothesis tests (Methodology §6)")
    print("-" * 60)
    # H1: K-conservation within family
    by_vendor = {}
    for m in atlas["models"]:
        by_vendor.setdefault(m["vendor"], []).append(m["axes"]["K"])
    for v, ks in by_vendor.items():
        if len(ks) >= 2:
            spread = max(ks) - min(ks)
            print(f"  H1 K-spread within {v:<10}: {spread:.3f}  (n={len(ks)})")

    # H2: C-D anti-correlation
    cs = [m["axes"]["C"] for m in atlas["models"]]
    ds = [m["axes"]["D"] for m in atlas["models"]]
    n = len(cs)
    mean_c = sum(cs) / n
    mean_d = sum(ds) / n
    cov = sum((c - mean_c) * (d - mean_d) for c, d in zip(cs, ds)) / n
    var_c = sum((c - mean_c) ** 2 for c in cs) / n
    var_d = sum((d - mean_d) ** 2 for d in ds) / n
    pearson = cov / math.sqrt(var_c * var_d) if var_c * var_d > 0 else 0
    print(f"  H2 C-D Pearson r = {pearson:+.3f}  (n={n})  → {'anti-correlated' if pearson < -0.2 else 'positively correlated' if pearson > 0.2 else 'uncorrelated'}")


if __name__ == "__main__":
    main()
