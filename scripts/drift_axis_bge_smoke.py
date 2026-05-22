#!/usr/bin/env python3
"""
drift_axis_bge_smoke.py
========================

§7 pilot question 1 + 2 verification for the BGE embedding provider:

  1. Does the BGE embedding loader run end-to-end?
  2. Are DAA values produced under BGE structurally compatible with
     values produced under OpenAI on the same N=5 cooperative corpus?

This script does NOT validate H_drift_axis. It validates that the
locked methodology's second embedding pathway (§6 cross-vendor) is
operational before any new data is pulled through it.

Outputs:
  - per-conversation DAA under both providers
  - rank correlation (Spearman) between OpenAI-DAA and BGE-DAA
  - structural sanity check: are both providers in roughly the same
    DAA regime? (not a hypothesis test — just "did one of them break?")
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
from drift_axis_scorer import (
    OpenAIEmbeddings,
    BGEEmbeddings,
    conversation_embeddings,
    drift_axis_alignment,
)


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Simple Spearman rank correlation."""
    n = len(x)
    if n != len(y) or n < 2:
        return float("nan")
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for rank, i in enumerate(order, 1):
            r[i] = rank
        return r
    rx, ry = ranks(x), ranks(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = (sum((r - mean_rx) ** 2 for r in rx)) ** 0.5
    den_y = (sum((r - mean_ry) ** 2 for r in ry)) ** 0.5
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def main() -> int:
    coop_manifest_path = REPO / "papers/cooperative-agent-regime/corpus_manifest.json"
    if not coop_manifest_path.exists():
        print(f"FAIL: {coop_manifest_path} not found")
        return 1
    manifest = json.loads(coop_manifest_path.read_text(encoding="utf-8"))
    convs = manifest.get("conversation_metadata", manifest["conversations"])

    print("=" * 70)
    print("Drift-Axis BGE smoke (§7 pilot question 1+2)")
    print("=" * 70)
    print(f"manifest: {coop_manifest_path.relative_to(REPO)}")
    print(f"n_conversations in manifest: {len(convs)}")
    print()

    # Load BGE first — if it doesn't initialize, abort before spending OpenAI calls.
    print("[1/4] initializing BGE provider (model download if first run)...")
    try:
        bge = BGEEmbeddings()
    except Exception as e:
        print(f"FAIL: BGE provider failed to initialize: {e}")
        return 1
    print(f"  OK: {bge.name}")
    print()

    print("[2/4] initializing OpenAI provider...")
    try:
        oai = OpenAIEmbeddings()
    except Exception as e:
        print(f"FAIL: OpenAI provider failed to initialize: {e}")
        return 1
    print(f"  OK: {oai.name}")
    print()

    print("[3/4] scoring DAA per conversation under both providers...")
    print()
    print(f"  {'conv':<6} {'oai_daa':>10} {'bge_daa':>10} {'|diff|':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8}")

    oai_daa_list: list[float] = []
    bge_daa_list: list[float] = []

    for c in convs:
        tx_path = Path(c.get("transcript_path") or "")
        if not tx_path.is_absolute():
            tx_path = REPO / tx_path
        if not tx_path.exists():
            # fall back like score_corpus does
            base = coop_manifest_path.parent / "corpus"
            cid = c.get("conv_id")
            if cid is not None:
                tx_path = base / f"conv{cid}_transcript.json"
            if not tx_path.exists():
                print(f"  conv{c.get('conv_id','?'):<2}    MISSING TRANSCRIPT")
                continue

        # OpenAI
        embs_a_oai, embs_b_oai = conversation_embeddings(tx_path, oai)
        d_oai = drift_axis_alignment(embs_a_oai, embs_b_oai)

        # BGE
        embs_a_bge, embs_b_bge = conversation_embeddings(tx_path, bge)
        d_bge = drift_axis_alignment(embs_a_bge, embs_b_bge)

        oai_daa_list.append(d_oai)
        bge_daa_list.append(d_bge)
        diff = abs(d_oai - d_bge)
        print(f"  conv{c.get('conv_id','?'):<2}   {d_oai:+.4f}    {d_bge:+.4f}    {diff:.4f}")

    print()
    print("[4/4] structural compatibility checks")
    print()

    oai_median = float(np.median(oai_daa_list))
    bge_median = float(np.median(bge_daa_list))
    rho = spearman_rho(oai_daa_list, bge_daa_list)

    print(f"  OpenAI median DAA   : {oai_median:+.4f}")
    print(f"  BGE    median DAA   : {bge_median:+.4f}")
    print(f"  median |diff|       : {abs(oai_median - bge_median):.4f}")
    print(f"  Spearman rho        : {rho:+.4f}")
    print()

    # Interpretive heuristics (NOT hypothesis tests — §7 says smoke only)
    print("  interpretation (smoke, non-evidentiary):")
    if all(d > 0.3 for d in bge_daa_list):
        print("    [OK]   BGE values land in the positive-DAA regime on cooperative data")
    else:
        below = [i for i, d in enumerate(bge_daa_list) if d <= 0.3]
        print(f"    [NOTE] BGE produced DAA <= 0.3 on convs at positions {below}")
    if rho > 0.5:
        print(f"    [OK]   BGE and OpenAI rank-agree on these 5 conversations (rho > 0.5)")
    else:
        print(f"    [NOTE] BGE and OpenAI rank-disagree on these 5 conversations (rho = {rho:.3f})")
        print(f"           This is allowed under the locked methodology — bar requires")
        print(f"           BOTH providers clear independently, NOT that they agree on")
        print(f"           individual conversations. But worth flagging for the writeup.")

    # Save smoke results
    out = REPO / "papers/cooperative-agent-regime/results/drift_axis_bge_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kind": "bge_smoke",
        "purpose": "§7 pilot questions 1+2 — methodology validation only",
        "evidentiary": False,
        "oai_per_conv_daa": oai_daa_list,
        "bge_per_conv_daa": bge_daa_list,
        "oai_median": oai_median,
        "bge_median": bge_median,
        "spearman_rho": rho,
    }, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
