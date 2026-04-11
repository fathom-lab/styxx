# -*- coding: utf-8 -*-
"""
extract_demo_trajectories.py -- Pull real per-probe trajectories from
the Fathom atlas v0.3 captures and bundle them as a fixture file
that styxx uses when the CLI needs a "demo" trajectory.

This exists because synthetic random trajectories are not well-behaved
inputs to a classifier trained on real atlas distributions — the
classifier gives meaningless predictions on pure noise. Using actual
atlas trajectories means `styxx ask --demo-kind hallucination` shows
the classifier behaving as it does on real data.

One representative trajectory is picked per category from the
gemma-2-2b-it capture (the strongest single-model classifier in the
atlas). Deterministic — runs produce identical fixtures.
"""

import argparse
import json
import os
from pathlib import Path


CATEGORIES = [
    "retrieval", "reasoning", "refusal",
    "creative", "adversarial", "hallucination",
]


def main():
    ap = argparse.ArgumentParser(
        description=(
            "One-time fixture generator for styxx demo trajectories. "
            "The shipped artifact lives at styxx/centroids/demo_trajectories.json "
            "and is the actual file consumed at runtime by `styxx ask --demo-kind`. "
            "You only need to run this script if you are rebuilding the fixture "
            "from a fresh atlas capture."
        ),
    )
    ap.add_argument(
        "--source",
        default=os.environ.get(
            "FATHOM_ATLAS_CAPTURE",
            "atlas/captures/google__gemma-2-2b-it__v0.1.0.json",
        ),
        help=(
            "Path to a Fathom atlas v0.3 per-model capture JSON. "
            "Override with the FATHOM_ATLAS_CAPTURE env var."
        ),
    )
    ap.add_argument(
        "--out",
        default="styxx/centroids/demo_trajectories.json",
        help="Where to write the fixture (relative to the package root).",
    )
    args = ap.parse_args()

    with open(args.source, "r", encoding="utf-8") as f:
        capture = json.load(f)

    out = {
        "source_model": capture.get("model", "unknown"),
        "source_atlas_version": capture.get("atlas_version", "v0.3"),
        "schema_version": "0.1.0",
        "note": (
            "These trajectories are real atlas v0.3 probe captures "
            "from google/gemma-2-2b-it. They are shipped with styxx "
            "so CLI demos produce honest classifier output on real "
            "data, not on synthetic noise. Each trajectory is a "
            "complete 30-token generation of the stated category."
        ),
        "trajectories": {},
    }

    for cat in CATEGORIES:
        probes = capture.get("probes", {}).get(cat, {})
        # Take the first probe of each category with a full trajectory
        for probe_id, probe in probes.items():
            trajs = probe.get("trajectories", {})
            ent = trajs.get("entropy", [])
            lp = trajs.get("logprob", [])
            t2 = trajs.get("top2_margin", [])
            if len(ent) >= 25 and len(lp) == len(ent) and len(t2) == len(ent):
                out["trajectories"][cat] = {
                    "probe_id": probe_id,
                    "text_preview": probe.get("text", "")[:80],
                    "generated_preview": probe.get("generated", "")[:200],
                    "n_tokens": len(ent),
                    "entropy": ent,
                    "logprob": lp,
                    "top2_margin": t2,
                }
                break
        else:
            print(f"[warn] no qualifying probe found for category {cat}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(out['trajectories'])} demo trajectories to {out_path}")
    for cat, data in out["trajectories"].items():
        print(f"  {cat:<14}  {data['n_tokens']:>3d} tokens  "
              f"'{data['text_preview'][:40]}...'")


if __name__ == "__main__":
    main()
