"""Gemini competence-cliff runner — adds a FRONTIER, different-lineage vendor to the cross-vendor
consensus core (kill #1). Mirrors the proven judge+gate flow of run_xvendor_matched; only generation
is swapped to the Gemini REST API. Produces crossfamily_gate_gemini.json, which drops straight into
consensus_failure_core.py / analyze_provider_matrix.py as a 5th provider.

STATUS: UNTESTED — written without a key (none in secrets/ at build time). Verify on first real run.
Reads key from secrets/gemini-key.txt or env GEMINI_API_KEY. Needs the NLI model on GPU (light).

FREE-TIER NOTE: gemini free tier is rate-limited (~15 rpm, ~1500 req/day for flash). 790 items × N
samples via candidateCount=1 loop = 7900 calls = over the daily cap. Mitigations baked in: (a) prefer
candidateCount=N in ONE call per item (≈790 calls); (b) --n-items to cap; (c) 429 backoff. Apparatus
match: SYS_MSG prompt, temperature 1.0, max 32 tokens (as gpt-4o-mini / the open families).

Usage:  python run_gemini_cliff.py --smoke           # ~8 items
        python run_gemini_cliff.py --model gemini-2.0-flash --n-items 0   # full
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

import requests

import run_truthfulqa_benchmark as B
import run_pregeneration_gate as G
from run_local_cliff import NLIJudge, _judge_item, NLI_MODEL
from styxx.audit import (
    _derive_verdict, _DEFAULT_HONEST, _DEFAULT_LOW_STABILITY, _DEFAULT_CONTRADICTION,
)

HERE = Path(__file__).resolve().parent
SECRET = Path(r"C:\Users\heyzo\clawd\secrets\gemini-key.txt")
N_SAMPLES, TEMPERATURE, MAX_TOKENS = 10, 1.0, 32


def _key() -> str | None:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"].strip()
    if SECRET.exists():
        return SECRET.read_text(encoding="utf-8").strip()
    return None


def _gen(key: str, model: str, question: str, n: int) -> list[str]:
    """N stateless samples from Gemini. Tries candidateCount=n in one call; falls back to a loop."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {
        "system_instruction": {"parts": [{"text": B.SYS_MSG}]},
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "generationConfig": {"temperature": TEMPERATURE, "maxOutputTokens": MAX_TOKENS, "candidateCount": n},
    }

    def _post(b):
        for attempt in range(8):
            r = requests.post(url, json=b, timeout=60)
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1)); continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()

    try:
        data = _post(body)
        cands = data.get("candidates", [])
        out = []
        for c in cands:
            parts = c.get("content", {}).get("parts", [])
            out.append("".join(p.get("text", "") for p in parts).strip())
        if len(out) >= n:
            return out[:n]
        # candidateCount>1 unsupported on this model — loop single calls
    except requests.HTTPError:
        pass
    single = dict(body); single["generationConfig"] = dict(body["generationConfig"]); single["generationConfig"]["candidateCount"] = 1
    out = []
    for _ in range(n):
        data = _post(single)
        c = data.get("candidates", [{}])[0]
        parts = c.get("content", {}).get("parts", [])
        out.append("".join(p.get("text", "") for p in parts).strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.0-flash")
    ap.add_argument("--n-items", type=int, default=0, help="0 = full 790")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    key = _key()
    if not key:
        print("NO KEY — add a free Google AI Studio key to secrets/gemini-key.txt (or env GEMINI_API_KEY).")
        print("Free key: https://aistudio.google.com/apikey (no credit card).")
        return 2

    items = B.load_dataset()
    sha = B.hash_answer_key(items)
    if sha != B.EXPECTED_HASH:
        print(f"FATAL: answer-key hash mismatch {sha}"); return 3
    n = 8 if a.smoke else (a.n_items or len(items))
    items = items[:n]
    print(f"Gemini {a.model}: {len(items)} items × {N_SAMPLES} samples (T={TEMPERATURE}, max={MAX_TOKENS})", flush=True)

    nli = NLIJudge(NLI_MODEL, device="cuda")
    results = []
    for k, (q, best, worst, cat) in enumerate(items):
        samples = _gen(key, a.model, q, N_SAMPLES)
        if not samples:
            continue
        jt, jf = _judge_item(nli, samples, best, worst)
        g_t, st_t, c_t = B.grounded_from_batch(jt, len(samples))
        g_f, st_f, c_f = B.grounded_from_batch(jf, len(samples))
        v_t = _derive_verdict(grounded=g_t, stability=st_t, concordance_stateless=c_t, injection_suspected=False,
                              honest=_DEFAULT_HONEST, low_stability=_DEFAULT_LOW_STABILITY, contradiction=_DEFAULT_CONTRADICTION)
        v_f = _derive_verdict(grounded=g_f, stability=st_f, concordance_stateless=c_f, injection_suspected=False,
                              honest=_DEFAULT_HONEST, low_stability=_DEFAULT_LOW_STABILITY, contradiction=_DEFAULT_CONTRADICTION)
        results.append({"idx": k, "question": q, "best": best, "worst": worst, "category": cat, "samples": samples,
                        "g_true": g_t, "stability_true": st_t, "concordance_true": c_t,
                        "n_clusters_true": jt["n_clusters"], "matches_true": jt["matches"], "verdict_true": v_t,
                        "g_false": g_f, "stability_false": st_f, "concordance_false": c_f,
                        "n_clusters_false": jf["n_clusters"], "matches_false": jf["matches"], "verdict_false": v_f})
        if k % 50 == 0:
            print(f"  {k}/{len(items)}", flush=True)

    bench = HERE / "crossfamily_benchmark_gemini.json"
    bench.write_text(json.dumps({"model": a.model, "judge": f"NLI({NLI_MODEL})", "n_items": len(results),
                                 "answer_key_sha256": B.EXPECTED_HASH, "items": results}, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    G.BENCHMARK_RECEIPT = bench
    G.RECEIPT = HERE / "crossfamily_gate_gemini.json"
    G.main()
    print(f"\nDONE: crossfamily_gate_gemini.json — add 'Google-frontier(Gemini)' to consensus_failure_core / matrix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
