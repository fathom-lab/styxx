#!/usr/bin/env python3
"""
embedding_coupling.py — EXPLORATORY (NOT PREREGISTERED)
========================================================

Post-locked-experiment probe asking: if cooperative agents' pulse-traces
don't couple in the cogn-text channel (closed-negative tonight), do they
couple at the *latent-geometry* channel?

Methodology: embed each turn of each conversation in
text-embedding-3-large (validated in styxx universal-probe work,
commit 496a8b8). For each agent extract a 1-D scalar trajectory from
the per-turn embedding sequence, then compute Pearson r at lag 0
between the two agents' scalar series, conversation by conversation.

Three scalar trajectories per agent:
  TRAJ-STEP    s_t = 1 - cos(emb_t, emb_{t-1})   "trajectory step size"
  CENTROID     d_t = 1 - cos(emb_t, mean(emb))   "drift from own centroid"
  ALIGNMENT    a_t = cos(emb_t_A, emb_t_B)       "pairwise alignment" (shared series)

Plus a per-conversation aggregate:
  DRIFT-AXIS   cos(centroid_A_drift_vec, centroid_B_drift_vec)
               — do the two agents' overall trajectories point the same
               way in latent space?

ALL of this is exploratory. The preregistered hypothesis test was the
cogn-text channel (median CC 0.111, closed-negative). NONE of the
results here modify the closed-negative or its bar.

The honest framing: this probe tells us *where to look next*. If
embedding coupling is also low, the negative generalizes across
channels. If embedding coupling is high while cogn-text was low,
that's a structural finding worth preregistering as bet 1 of the
cognitive-coupling program.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI


CHANNELS = ("traj_step", "centroid_drift", "pairwise_alignment")


def _ensure_styxx_importable() -> None:
    here = Path(__file__).resolve().parent.parent.parent
    if (here / "styxx").is_dir() and str(here) not in sys.path:
        sys.path.insert(0, str(here))


_ensure_styxx_importable()


def load_transcripts(manifest_path: Path) -> list[dict]:
    """Read the corpus manifest's conversation metadata + open each
    transcript JSON so we have the actual turn-by-turn text content."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = []
    for c in manifest.get("conversation_metadata", manifest["conversations"]):
        transcript_path = Path(c["transcript_path"]) if "transcript_path" in c else None
        if transcript_path is None or not transcript_path.exists():
            # Fall back: derive from session_id pattern
            possible = (
                Path("papers/cooperative-agent-regime/corpus") /
                f"conv{c.get('conv_id', '?')}_transcript.json"
            )
            transcript_path = possible if possible.exists() else None
        if transcript_path is None or not transcript_path.exists():
            raise FileNotFoundError(
                f"transcript JSON not found for session {c.get('session_id')}"
            )
        out.append({
            "session_id": c["session_id"],
            "transcript": json.loads(transcript_path.read_text(encoding="utf-8")),
        })
    return out


def embed_batch(texts: list[str], model: str = "text-embedding-3-large") -> np.ndarray:
    """Batch-embed texts. Returns (N, D) numpy array, L2-normalized."""
    client = OpenAI()
    last_err = None
    for attempt in range(3):
        try:
            r = client.embeddings.create(model=model, input=texts)
            vecs = np.array([d.embedding for d in r.data], dtype=np.float64)
            # L2 normalize so cosine = dot
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vecs / norms
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"embed_batch failed: {last_err}")


def per_agent_embeddings(transcript: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (turn_indices, embeddings_A, embeddings_B) for one conversation.

    Embeddings are turn-ordered and aligned (kth-of-A with kth-of-B,
    truncated to shorter — matches preregistration §4 corrigendum).
    """
    role_a = transcript["task"]["role_a"]
    role_b = transcript["task"]["role_b"]
    texts_a = [t["content"] for t in transcript["turns"] if t["sender"] == role_a]
    texts_b = [t["content"] for t in transcript["turns"] if t["sender"] == role_b]
    n = min(len(texts_a), len(texts_b))
    texts_a = texts_a[:n]
    texts_b = texts_b[:n]

    # Single batched embed call per agent.
    embs_a = embed_batch(texts_a)
    embs_b = embed_batch(texts_b)
    return [str(i) for i in range(n)], embs_a, embs_b


def trajectory_step_series(embs: np.ndarray) -> np.ndarray:
    """s_t = 1 - cos(emb_t, emb_{t-1}). Returns length n-1 series."""
    sims = (embs[1:] * embs[:-1]).sum(axis=1)
    return 1.0 - sims


def centroid_drift_series(embs: np.ndarray) -> np.ndarray:
    """d_t = 1 - cos(emb_t, centroid). Returns length n series."""
    centroid = embs.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    return 1.0 - embs @ centroid


def pairwise_alignment_series(embs_a: np.ndarray, embs_b: np.ndarray) -> np.ndarray:
    """a_t = cos(emb_t_A, emb_t_B). Returns length n series."""
    return (embs_a * embs_b).sum(axis=1)


def drift_axis_alignment(embs_a: np.ndarray, embs_b: np.ndarray) -> float:
    """Cosine of A's overall trajectory direction vs B's.

    Trajectory direction = unit vector from first-half centroid to
    second-half centroid.
    """
    n = embs_a.shape[0]
    half = n // 2
    if half < 2:
        return float("nan")
    a_dir = embs_a[half:].mean(0) - embs_a[:half].mean(0)
    b_dir = embs_b[half:].mean(0) - embs_b[:half].mean(0)
    a_dir /= np.linalg.norm(a_dir) + 1e-12
    b_dir /= np.linalg.norm(b_dir) + 1e-12
    return float(a_dir @ b_dir)


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r — matches styxx.coherence._pearson_r semantics."""
    if len(a) != len(b) or len(a) < 3:
        return float("nan")
    a_mean = a.mean(); b_mean = b.mean()
    cov = ((a - a_mean) * (b - b_mean)).mean()
    sd_a = a.std(); sd_b = b.std()
    if sd_a == 0 or sd_b == 0:
        return 0.0
    return float(cov / (sd_a * sd_b))


def shuffled_pairs_null(
    per_conv_pairs: list[tuple[np.ndarray, np.ndarray]],
    n_resamples: int = 5000,
    seed: int = 1729,
) -> list[float]:
    """Mirror of locked-scorer null model on the trajectory_step channel."""
    import random
    rng = random.Random(seed)
    nulls: list[float] = []
    n_convs = len(per_conv_pairs)
    if n_convs < 2:
        return []
    for _ in range(n_resamples):
        i = rng.randrange(n_convs)
        j = rng.randrange(n_convs)
        while j == i:
            j = rng.randrange(n_convs)
        a_series = per_conv_pairs[i][0]
        b_series = per_conv_pairs[j][1]
        n = min(len(a_series), len(b_series))
        if n < 3:
            continue
        nulls.append(pearson_r(a_series[:n], b_series[:n]))
    return nulls


def analyze_corpus(manifest_path: Path, label: str) -> dict:
    print(f"\n=== {label} corpus from {manifest_path.name} ===", flush=True)
    transcripts = load_transcripts(manifest_path)
    per_conv: list[dict] = []
    per_conv_traj_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for t in transcripts:
        sid = t["session_id"]
        print(f"  embedding {sid}...", flush=True)
        turn_idx, embs_a, embs_b = per_agent_embeddings(t["transcript"])
        n = embs_a.shape[0]
        if n < 5:
            print(f"    skipped (n={n})")
            continue

        # Three scalar trajectories per agent.
        traj_a = trajectory_step_series(embs_a)
        traj_b = trajectory_step_series(embs_b)
        cent_a = centroid_drift_series(embs_a)
        cent_b = centroid_drift_series(embs_b)
        align = pairwise_alignment_series(embs_a, embs_b)

        # Pearson r per channel.
        cc_traj_step = pearson_r(traj_a, traj_b)
        cc_centroid_drift = pearson_r(cent_a, cent_b)
        # Pairwise alignment is already a shared scalar series; we report
        # its mean (turn-by-turn semantic alignment) and stdev.
        alignment_mean = float(align.mean())
        alignment_std = float(align.std())

        # Drift-axis alignment (single number per conversation).
        drift_axis = drift_axis_alignment(embs_a, embs_b)

        per_conv.append({
            "session_id": sid,
            "n_turns": int(n),
            "cc_traj_step": cc_traj_step,
            "cc_centroid_drift": cc_centroid_drift,
            "pairwise_alignment_mean": alignment_mean,
            "pairwise_alignment_std": alignment_std,
            "drift_axis_alignment": drift_axis,
        })
        per_conv_traj_pairs.append((traj_a, traj_b))

    # Aggregate
    def _stats(values):
        v = [x for x in values if x == x]  # drop NaN
        return {
            "median": statistics.median(v) if v else float("nan"),
            "mean": statistics.fmean(v) if v else float("nan"),
            "n": len(v),
        }

    aggregate = {
        "cc_traj_step":        _stats([d["cc_traj_step"] for d in per_conv]),
        "cc_centroid_drift":   _stats([d["cc_centroid_drift"] for d in per_conv]),
        "pairwise_alignment_mean": _stats([d["pairwise_alignment_mean"] for d in per_conv]),
        "drift_axis_alignment":    _stats([d["drift_axis_alignment"] for d in per_conv]),
    }

    # Null model on the trajectory-step channel (the closest analog to
    # the locked cogn-text channel).
    null_dist = shuffled_pairs_null(per_conv_traj_pairs, n_resamples=5000)
    observed_median_traj = aggregate["cc_traj_step"]["median"]
    p_traj_step = float("nan")
    if null_dist:
        # bootstrap null medians at corpus N
        import random
        rng = random.Random(3142)
        n_pool = len(null_dist); corpus_n = len(per_conv_traj_pairs)
        null_meds = []
        for _ in range(5000):
            sample = [null_dist[rng.randrange(n_pool)] for _ in range(corpus_n)]
            null_meds.append(statistics.median(sample))
        n_ge = sum(1 for m in null_meds if m >= observed_median_traj)
        p_traj_step = (n_ge + 1) / (5000 + 1)

    return {
        "label": label,
        "manifest": str(manifest_path),
        "n_conversations": len(per_conv),
        "per_conversation": per_conv,
        "aggregate": aggregate,
        "trajectory_step_pvalue_vs_shuffled_pairs_null": p_traj_step,
        "note": (
            "EXPLORATORY — NOT preregistered. The closed-negative phase-coherence "
            "finding (2026-05-20) is in the cogn-text channel. This probe asks "
            "whether the embedding-trajectory channel shows coupling that the "
            "cogn-text channel did not. Results here have no preregistered bar "
            "and cannot modify the closed-negative."
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--coop-manifest", type=Path,
                   default=Path("papers/cooperative-agent-regime/corpus_manifest.json"))
    p.add_argument("--noncoop-manifest", type=Path,
                   default=Path("papers/cooperative-agent-regime/noncoop_corpus_manifest.json"))
    p.add_argument("--output", type=Path,
                   default=Path("papers/cooperative-agent-regime/results/embedding_coupling.json"))
    args = p.parse_args(argv)

    payload = {
        "cooperative": analyze_corpus(args.coop_manifest, "cooperative"),
        "noncooperative": analyze_corpus(args.noncoop_manifest, "noncooperative"),
    }
    # Cross-regime comparison
    payload["regime_comparison"] = {
        "cc_traj_step_delta":
            payload["cooperative"]["aggregate"]["cc_traj_step"]["median"]
            - payload["noncooperative"]["aggregate"]["cc_traj_step"]["median"],
        "drift_axis_alignment_delta":
            payload["cooperative"]["aggregate"]["drift_axis_alignment"]["median"]
            - payload["noncooperative"]["aggregate"]["drift_axis_alignment"]["median"],
        "pairwise_alignment_mean_delta":
            payload["cooperative"]["aggregate"]["pairwise_alignment_mean"]["median"]
            - payload["noncooperative"]["aggregate"]["pairwise_alignment_mean"]["median"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Compact stdout
    print(json.dumps({
        "cooperative": {
            "cc_traj_step_median": payload["cooperative"]["aggregate"]["cc_traj_step"]["median"],
            "cc_centroid_drift_median": payload["cooperative"]["aggregate"]["cc_centroid_drift"]["median"],
            "drift_axis_alignment_median": payload["cooperative"]["aggregate"]["drift_axis_alignment"]["median"],
            "pairwise_alignment_mean_median": payload["cooperative"]["aggregate"]["pairwise_alignment_mean"]["median"],
            "p_traj_step": payload["cooperative"]["trajectory_step_pvalue_vs_shuffled_pairs_null"],
        },
        "noncooperative": {
            "cc_traj_step_median": payload["noncooperative"]["aggregate"]["cc_traj_step"]["median"],
            "cc_centroid_drift_median": payload["noncooperative"]["aggregate"]["cc_centroid_drift"]["median"],
            "drift_axis_alignment_median": payload["noncooperative"]["aggregate"]["drift_axis_alignment"]["median"],
            "pairwise_alignment_mean_median": payload["noncooperative"]["aggregate"]["pairwise_alignment_mean"]["median"],
        },
        "regime_comparison": payload["regime_comparison"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
