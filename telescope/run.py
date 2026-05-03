"""
telescope/run.py — daily cognometric measurement layer.

Runs the curated prompt corpus through every available LLM (anthropic /
openai / openrouter — auto-detects which vendors have keys), scores each
response with styxx, and writes the day's ledger.

This is the data engine behind https://fathom.darkflobi.com/scoreboard.

Usage:
    python run.py
    python run.py --models claude-opus-4-7,gpt-5
    python run.py --dry-run   # validate config, no API calls

Output:
    data/runs/telescope__<ts>.json   # per-run ledger (append-only history)
    data/latest.json                 # most recent run (overwritten)
"""
import argparse, json, os, sys, time
from pathlib import Path

# ── env loading ──────────────────────────────────────────────────────────────
# load .env from this dir if present; otherwise rely on process env vars.
HERE = Path(__file__).parent
LOCAL_ENV = HERE / ".env"
if LOCAL_ENV.exists():
    for line in LOCAL_ENV.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

DATA_DIR = HERE / "data"
RUNS_DIR = DATA_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── model registry ───────────────────────────────────────────────────────────
# one model per tier per vendor; small enough to run in <5 min, large enough
# to be representative.
MODELS = [
    {"id": "claude-opus-4-7",  "vendor": "anthropic", "tier": "frontier", "$out_per_mtok": 75.0},
    {"id": "claude-sonnet-4-6","vendor": "anthropic", "tier": "balanced", "$out_per_mtok": 15.0},
    {"id": "claude-haiku-4-5", "vendor": "anthropic", "tier": "fast",     "$out_per_mtok": 4.0},
    {"id": "gpt-5",            "vendor": "openai",    "tier": "frontier", "$out_per_mtok": 60.0},
    {"id": "gpt-5-mini",       "vendor": "openai",    "tier": "fast",     "$out_per_mtok": 3.0},
    {"id": "deepseek/deepseek-coder-v3", "vendor": "openrouter", "tier": "open",  "$out_per_mtok": 0.28},
]

VENDOR_KEY = {
    "anthropic":  "ANTHROPIC_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

INSTRUMENTS = ["sycophancy", "deception", "overconfidence", "refusal"]


def load_prompts() -> list:
    p = HERE / "prompts.json"
    if not p.exists():
        sys.exit(f"missing {p} — populate it first.")
    return json.loads(p.read_text(encoding="utf-8"))


def have_key(vendor: str) -> bool:
    return bool(os.environ.get(VENDOR_KEY[vendor], "").strip())


def call_model(model: dict, system: str, prompt: str, max_tokens: int = 600) -> str:
    """vendor-specific call. returns response text."""
    vendor = model["vendor"]
    mid = model["id"]
    is_reasoning = mid.startswith("gpt-5") or mid.startswith("o1") or mid.startswith("o3")
    if vendor == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
        resp = client.messages.create(
            model=mid, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text"))
    elif vendor == "openai":
        from openai import OpenAI
        client = OpenAI()
        kwargs = dict(
            model=mid,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        if is_reasoning:
            # reasoning models burn tokens on internal reasoning before output;
            # 600 is too tight — reasoning consumes it all and content comes back empty.
            kwargs["max_completion_tokens"] = max(max_tokens, 4000)
            kwargs["reasoning_effort"] = "minimal"
        else:
            kwargs["max_completion_tokens"] = max_tokens
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
    elif vendor == "openrouter":
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        resp = client.chat.completions.create(
            model=mid, max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""
    else:
        raise ValueError(f"unknown vendor: {vendor}")


def score_response(prompt: str, response: str) -> dict:
    """call styxx, return per-instrument floats. errors → {error}."""
    try:
        from styxx.attack import score_all
        scores = score_all(prompt=prompt, response=response)
        return {k: float(v) for k, v in scores.items()}
    except Exception as e:
        return {"error": str(e)}


def run_one_model(model: dict, prompts: list, dry: bool = False) -> dict:
    """run all prompts through one model. returns model record."""
    print(f"\n=== {model['id']} ({model['vendor']}, {model['tier']}) ===")
    if not have_key(model["vendor"]):
        print(f"  skip — no {VENDOR_KEY[model['vendor']]} in env")
        return {"model": model["id"], "vendor": model["vendor"], "skipped": "no_key"}
    if dry:
        print(f"  [dry-run] would query {len(prompts)} prompts")
        return {"model": model["id"], "vendor": model["vendor"], "dry_run": True}

    rows = []
    totals = {k: 0.0 for k in INSTRUMENTS}
    n = 0
    t0 = time.time()
    for i, p in enumerate(prompts, 1):
        print(f"  [{i:2d}/{len(prompts)}] {p['id']}", end="", flush=True)
        try:
            text = call_model(model, p["system"], p["prompt"])
        except Exception as e:
            print(f"  ! api: {e}")
            rows.append({"id": p["id"], "category": p["category"], "error": str(e)})
            continue
        if not text.strip():
            print(f"  ! empty response — skipping (likely token-budget exhaustion)")
            rows.append({"id": p["id"], "category": p["category"],
                         "prompt": p["prompt"], "response": "",
                         "error": "empty_response"})
            continue
        scores = score_response(p["prompt"], text)
        if "error" not in scores:
            for k in INSTRUMENTS:
                if k in scores:
                    totals[k] += scores[k]
            n += 1
        rows.append({
            "id": p["id"], "category": p["category"],
            "prompt": p["prompt"], "response": text, "scores": scores,
        })
        score_line = " · ".join(f"{k[:4]}={scores.get(k, 0):.2f}" for k in INSTRUMENTS if k in scores)
        print(f"  {score_line}")

    elapsed = time.time() - t0
    avgs = {k: round(totals[k] / n, 4) if n else 0.0 for k in INSTRUMENTS}
    print(f"\n  averages: " + " · ".join(f"{k}={v:.4f}" for k, v in avgs.items()))
    print(f"  elapsed:  {elapsed:.1f}s · scored: {n}/{len(prompts)}")

    return {
        "model": model["id"],
        "vendor": model["vendor"],
        "tier": model["tier"],
        "n_scored": n,
        "n_prompts": len(prompts),
        "averages": avgs,
        "elapsed_sec": round(elapsed, 1),
        "rows": rows,
    }


def build_ranking(model_records: list) -> list:
    """rank models by composite honesty score (lower = more honest)."""
    ranked = []
    for r in model_records:
        if r.get("skipped") or r.get("dry_run") or r.get("n_scored", 0) == 0:
            continue
        avgs = r["averages"]
        # composite: equal-weighted mean of sycoph + decep + overconf
        # (refusal NOT included — high refusal isn't dishonesty)
        composite = (avgs["sycophancy"] + avgs["deception"] + avgs["overconfidence"]) / 3
        ranked.append({
            "model": r["model"],
            "vendor": r["vendor"],
            "tier": r["tier"],
            "composite_dishonesty": round(composite, 4),
            "sycophancy": avgs["sycophancy"],
            "deception": avgs["deception"],
            "overconfidence": avgs["overconfidence"],
            "refusal": avgs["refusal"],
        })
    ranked.sort(key=lambda x: x["composite_dishonesty"])
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated model ids; default = all available")
    ap.add_argument("--dry-run", action="store_true", help="validate config, no API calls")
    args = ap.parse_args()

    selected = MODELS
    if args.models:
        ids = set(s.strip() for s in args.models.split(","))
        selected = [m for m in MODELS if m["id"] in ids]
    available = [m for m in selected if have_key(m["vendor"]) or args.dry_run]

    print(f"[telescope] models in registry: {len(MODELS)}")
    print(f"[telescope] selected:           {len(selected)}")
    print(f"[telescope] vendor keys present: " + ", ".join(
        v for v in VENDOR_KEY if have_key(v)
    ) or "(none)")
    if not available and not args.dry_run:
        sys.exit("[telescope] no usable models — add an API key to env or telescope/.env")

    prompts = load_prompts()
    print(f"[telescope] prompts: {len(prompts)}")

    records = [run_one_model(m, prompts, dry=args.dry_run) for m in selected]

    if args.dry_run:
        print(f"\n[telescope] dry-run complete — no files written")
        return

    ranked = build_ranking(records)

    out = {
        "ts": time.strftime("%Y%m%d_%H%M%S"),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec_version": "telescope-v1",
        "styxx_version": _styxx_version(),
        "n_prompts": len(prompts),
        "n_models_run": len([r for r in records if r.get("n_scored", 0) > 0]),
        "ranking": ranked,
        "model_records": records,
    }

    out_path = RUNS_DIR / f"telescope__{out['ts']}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    latest = DATA_DIR / "latest.json"
    latest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[telescope] saved: {out_path}")
    print(f"[telescope] latest: {latest}")
    print(f"\n=== ranking (lower = more honest) ===")
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r['model']:30s}  composite {r['composite_dishonesty']:.4f}")


def _styxx_version() -> str:
    try:
        import styxx
        return getattr(styxx, "__version__", "unknown")
    except ImportError:
        return "missing"


if __name__ == "__main__":
    main()
