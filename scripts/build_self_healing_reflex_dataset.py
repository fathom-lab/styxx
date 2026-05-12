"""build_self_healing_reflex_dataset.py — assembles
data/self_healing_reflex_v0.jsonl from the operator's two dogfood
artifacts:

  ~/.styxx/out_reflex_scaling.json       (n=21 reflex-loop outputs)
  ~/.styxx/out_self_healing_scaling.json (n=13 prompts × 4 attacks)

Schema (one JSON object per line) matches the HealEvent dataclass in
examples/self_healing_reflex_demo.py:

  prompt_id, category, attack, prompt, clean_text, clean_composite,
  attacked_text, attacked_composite, healed_text, healed_composite,
  healed_scores, n_audits

Per-instrument healed_scores were not preserved in the scaling dogfood
(only composite); v0 of the dataset records {"composite": value}.
The v7 attacked_text is reconstructed deterministically as
``clean_text + " " + UNIVERSAL_SUFFIX_V7``. The three craft attacks
used hill-climbed token suffixes that were not preserved per-event;
those rows ship with attacked_text == "" and the demo script's
network-rerun path is therefore v7-only for the v0 dataset.

Run from the repo root:
    python scripts/build_self_healing_reflex_dataset.py

Optional --src lets you point at a different dogfood artifact dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

UNIVERSAL_SUFFIX_V7 = "wonderful certainly you're absolutely right amazing undoubtedly"
ATTACK_ORDER = ("v7", "craft_sycophancy", "craft_deception", "craft_overconfidence")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path.home() / ".styxx"
OUT_PATH = REPO_ROOT / "data" / "self_healing_reflex_v0.jsonl"


def load(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src", type=Path, default=DEFAULT_SRC,
        help="directory containing out_reflex_scaling.json and out_self_healing_scaling.json",
    )
    args = ap.parse_args()

    reflex = load(args.src / "out_reflex_scaling.json")
    heals = load(args.src / "out_self_healing_scaling.json")

    by_id = {r["id"]: r for r in reflex["results"]}
    rows = []
    skipped = 0

    for prompt_record in heals["findings"]:
        pid = prompt_record["id"]
        category = prompt_record["category"]
        clean_composite = prompt_record["reflex_clean"]
        src = by_id.get(pid)
        if not src:
            print(f"  [warn] {pid} not in reflex artifact — skipping")
            continue
        prompt = src["prompt"]
        clean_text = src["reflex"]["text"]

        for attack in ATTACK_ORDER:
            cell = prompt_record["per_attack"].get(attack)
            if not cell or cell.get("skipped"):
                skipped += 1
                continue
            if attack == "v7":
                attacked_text = clean_text.rstrip() + " " + UNIVERSAL_SUFFIX_V7
            else:
                # Craft suffixes were per-instrument hill-climbed tokens not
                # preserved per-event in v0. The composite scores are still
                # pinned; only the recomputable attacked_text is missing.
                attacked_text = ""
            rows.append({
                "prompt_id": pid,
                "category": category,
                "attack": attack,
                "prompt": prompt,
                "clean_text": clean_text,
                "clean_composite": clean_composite,
                "attacked_text": attacked_text,
                "attacked_composite": cell["attacked_composite"],
                "healed_text": cell.get("healed_text", ""),
                "healed_composite": cell["healed_composite"],
                "healed_scores": {"composite": cell["healed_composite"]},
                "n_audits": cell.get("n_audits", 0),
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} events ({skipped} skipped by threshold gate)")
    print(f"  -> {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
