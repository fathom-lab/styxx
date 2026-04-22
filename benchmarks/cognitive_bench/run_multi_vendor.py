# -*- coding: utf-8 -*-
"""
benchmarks/cognitive_bench/run_multi_vendor.py

CognitiveBench v0 — the first public cross-vendor cognitive audit.

Runs the same fake-entity fabrication battery against every major
production LLM we have access to:

  - Anthropic: Claude Haiku 4.5 (via API)
  - Meta:      Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct (local)
  - Alibaba:   Qwen2.5-1.5B-Instruct (local)
  - Microsoft: Phi-3.5-mini-instruct (local)
  - OpenAI:    GPT-4o Mini (via API, if key present)
  - Google:    (Gemma / Gemini, if API available)

For each model:
  - Claude / OpenAI / Gemini: via their respective HTTP APIs
  - Local HF: via `transformers` with bf16 on GPU

Scoring:
  - Ground truth: all 50 prompts target entities/papers/drugs that
    DO NOT EXIST. Any confident-concrete response is a fabrication.
  - `fabricated = not is_decline(response)` where `is_decline` is
    the same epistemic-marker detector used in the Claude battery.

Outputs:
  - per-model JSON in results/<model_slug>.json
  - combined leaderboard markdown in results/cognitivebench_v0.md

This is the ship that giant labs cannot ship themselves. Every
published competitor comparison is filtered through their PR teams.
A neutral third-party audit is a structural gap in the industry,
and CognitiveBench v0 stakes the position.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    env_path = Path.home() / ".clawdbot" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def _load_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    env_path = Path.home() / ".clawdbot" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


# Import the battery + grading from the Claude run
sys.path.insert(0, str(ROOT / "benchmarks" / "claude_vs_us"))
from run_battery import FAKE_PROMPTS, is_decline  # noqa: E402


def run_anthropic(model_id: str, prompts: List[Dict]) -> List[Dict]:
    import anthropic
    key = _load_anthropic_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY unavailable")
    client = anthropic.Anthropic(api_key=key)

    results = []
    for i, row in enumerate(prompts):
        try:
            msg = client.messages.create(
                model=model_id,
                max_tokens=256,
                messages=[{"role": "user", "content": row["prompt"]}],
            )
            response = msg.content[0].text
        except Exception as e:
            print(f"  {i+1}/{len(prompts)} {row['id']:>10s}  "
                  f"API ERROR: {e}")
            continue
        declined = is_decline(response)
        results.append({
            "id": row["id"], "kind": row["kind"],
            "prompt": row["prompt"][:100],
            "response_excerpt": response[:300],
            "declined": declined,
            "fabricated": not declined,
        })
        print(f"  {i+1}/{len(prompts)} {row['id']:>10s}  "
              f"declined={declined}")
    return results


def run_openai(model_id: str, prompts: List[Dict]) -> List[Dict]:
    from openai import OpenAI
    key = _load_openai_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY unavailable")
    client = OpenAI(api_key=key)

    results = []
    for i, row in enumerate(prompts):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                max_tokens=256,
                messages=[{"role": "user", "content": row["prompt"]}],
            )
            response = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  {i+1}/{len(prompts)} {row['id']:>10s}  "
                  f"API ERROR: {e}")
            continue
        declined = is_decline(response)
        results.append({
            "id": row["id"], "kind": row["kind"],
            "prompt": row["prompt"][:100],
            "response_excerpt": response[:300],
            "declined": declined,
            "fabricated": not declined,
        })
        print(f"  {i+1}/{len(prompts)} {row['id']:>10s}  "
              f"declined={declined}")
    return results


def run_hf_local(model_id: str, prompts: List[Dict]) -> List[Dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)

    results = []
    for i, row in enumerate(prompts):
        input_ids = tok.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        prefill = input_ids.shape[1]
        with torch.no_grad():
            out = mdl.generate(
                input_ids, max_new_tokens=256,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
        response = tok.decode(out[0, prefill:].tolist(),
                              skip_special_tokens=True)
        declined = is_decline(response)
        results.append({
            "id": row["id"], "kind": row["kind"],
            "prompt": row["prompt"][:100],
            "response_excerpt": response[:300],
            "declined": declined,
            "fabricated": not declined,
        })
        print(f"  {i+1}/{len(prompts)} {row['id']:>10s}  "
              f"declined={declined}")

    del mdl
    torch.cuda.empty_cache()
    return results


VENDOR_MODELS = [
    # Local / open — full GPU load
    {"vendor": "Meta",      "slug": "llama-3.2-1b-instruct",
     "model_id": "meta-llama/Llama-3.2-1B-Instruct", "backend": "hf"},
    {"vendor": "Meta",      "slug": "llama-3.2-3b-instruct",
     "model_id": "meta-llama/Llama-3.2-3B-Instruct", "backend": "hf"},
    {"vendor": "Alibaba",   "slug": "qwen-2.5-1.5b-instruct",
     "model_id": "Qwen/Qwen2.5-1.5B-Instruct", "backend": "hf"},
    {"vendor": "Microsoft", "slug": "phi-3.5-mini-instruct",
     "model_id": "microsoft/Phi-3.5-mini-instruct", "backend": "hf"},
    # Closed APIs — already-run for Anthropic, keep entry for full grid
    {"vendor": "Anthropic", "slug": "claude-haiku-4-5",
     "model_id": "claude-haiku-4-5", "backend": "anthropic"},
    {"vendor": "OpenAI",    "slug": "gpt-4o-mini",
     "model_id": "gpt-4o-mini", "backend": "openai"},
]


def summarize(results: List[Dict]) -> Dict:
    n = len(results)
    if n == 0:
        return {"n": 0, "fab_rate": None, "decline_rate": None}
    n_fab = sum(1 for r in results if r["fabricated"])
    return {
        "n": n,
        "fab_rate": n_fab / n,
        "n_fabricated": n_fab,
        "decline_rate": 1 - n_fab / n,
        "n_declined": n - n_fab,
    }


def render_leaderboard(all_results: Dict[str, Dict]) -> str:
    """Render the public CognitiveBench v0 markdown."""
    lines = []
    lines.append("# CognitiveBench v0")
    lines.append("")
    lines.append("**First public cross-vendor cognitive audit of "
                 "production LLMs.**")
    lines.append("")
    lines.append(f"Battery: 50 fake-entity prompts "
                 "(papers / people / drugs / historical events / "
                 "technical features that do not exist). "
                 "Ground truth: any confident-concrete response is "
                 "a fabrication.")
    lines.append("")
    lines.append("Scoring: epistemic-decline detector over response "
                 "text (same heuristic for every model).")
    lines.append("")
    lines.append("## Fabrication-rate leaderboard (lower is better)")
    lines.append("")
    lines.append("| Rank | Vendor | Model | n | Fabrication rate | "
                 "Decline rate |")
    lines.append("|---|---|---|---|---|---|")

    rows = []
    for slug, rec in all_results.items():
        summary = rec["summary"]
        if summary["n"] == 0:
            continue
        rows.append((
            summary["fab_rate"], rec["vendor"],
            rec["model_id"], summary,
        ))
    rows.sort()   # ascending: lowest fab rate first

    for rank, (fab_rate, vendor, model_id, summary) in enumerate(rows, 1):
        lines.append(
            f"| {rank} | {vendor} | `{model_id}` | {summary['n']} | "
            f"**{fab_rate:.0%}** | {summary['decline_rate']:.0%} |"
        )
    lines.append("")

    lines.append("## Per-category breakdown")
    lines.append("")
    lines.append("| Model | papers | people | drugs | history | tech |")
    lines.append("|---|---|---|---|---|---|")
    for slug, rec in all_results.items():
        if not rec.get("results"):
            continue
        by_kind: Dict[str, List] = {}
        for r in rec["results"]:
            by_kind.setdefault(r["kind"], []).append(r)
        cells = []
        for kind in ("fake_paper", "fake_person", "fake_drug",
                     "fake_history", "fake_tech"):
            subset = by_kind.get(kind, [])
            if not subset:
                cells.append("-")
                continue
            fab = sum(1 for r in subset if r["fabricated"]) / len(subset)
            cells.append(f"{fab:.0%} ({sum(1 for r in subset if r['fabricated'])}/{len(subset)})")
        lines.append(f"| `{rec['model_id']}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- **Ground truth** is that all 50 prompts reference "
                 "non-existent entities. A confident concrete "
                 "response is always a fabrication by definition.")
    lines.append("- The decline detector is a keyword-level heuristic "
                 "(`i don't have`, `cannot verify`, `no record of`, "
                 "etc.). Same rules for every model.")
    lines.append("- This measurement is a *baseline for the industry*, "
                 "not a final evaluation. Future versions will add: "
                 "sycophancy-under-pressure, jailbreak resistance, "
                 "cross-session consistency, multilingual fabrication.")
    lines.append("- Measurements are run by Styxx Lab, independent of "
                 "all vendors tested. Code + prompts + raw data are "
                 "open-source in this repo. Any vendor can re-run "
                 "and publish counter-measurements.")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d')}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(
        ROOT / "benchmarks" / "cognitive_bench" / "results"))
    ap.add_argument("--slugs", nargs="+", default=None,
                    help="Run only these vendor slugs")
    ap.add_argument("--reuse", action="store_true",
                    help="Reuse per-model JSONs if present")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict] = {}

    targets = VENDOR_MODELS
    if args.slugs:
        targets = [m for m in VENDOR_MODELS if m["slug"] in args.slugs]

    for m in targets:
        slug = m["slug"]
        fp = out_dir / f"{slug}.json"
        if args.reuse and fp.exists():
            rec = json.loads(fp.read_text(encoding="utf-8"))
            print(f"reusing cached {slug}")
        else:
            print(f"\n=== running {slug} ({m['backend']}) ===")
            t0 = time.time()
            try:
                if m["backend"] == "anthropic":
                    results = run_anthropic(m["model_id"], FAKE_PROMPTS)
                elif m["backend"] == "openai":
                    results = run_openai(m["model_id"], FAKE_PROMPTS)
                elif m["backend"] == "hf":
                    results = run_hf_local(m["model_id"], FAKE_PROMPTS)
                else:
                    print(f"unknown backend {m['backend']}, skipping")
                    continue
            except Exception as e:
                print(f"FAILED {slug}: {e}")
                continue
            rec = {
                "vendor": m["vendor"],
                "slug": slug,
                "model_id": m["model_id"],
                "backend": m["backend"],
                "runtime_s": round(time.time() - t0, 1),
                "results": results,
            }
            rec["summary"] = summarize(results)
            fp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            print(f"wrote {fp}")

        all_results[slug] = rec
        s = rec["summary"]
        if s["n"]:
            print(f"  {slug}: {s['n_fabricated']}/{s['n']} fabricated "
                  f"({s['fab_rate']:.0%})")

    lb_path = out_dir / "cognitivebench_v0.md"
    lb_path.write_text(render_leaderboard(all_results), encoding="utf-8")
    print(f"\nwrote {lb_path}")


if __name__ == "__main__":
    main()
