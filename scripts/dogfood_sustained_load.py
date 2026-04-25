"""
dogfood_sustained_load.py — run styxx in a tight loop to surface memory
leaks, audit log growth, latency degradation.

100 steps · gpt-4o-mini · short prompts · tracks:
  - RSS memory delta from start to end
  - Audit log file size growth
  - Per-step latency (mean, p50, p99, max)
  - Vitals dropouts (None responses)
  - Gate distribution

Cost: ~$0.05 total (100 cheap calls @ ~80 tokens each).
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import statistics
import tracemalloc
from pathlib import Path

for _s in ("stdout", "stderr"):
    _r = getattr(getattr(sys, _s, None), "reconfigure", None)
    if _r:
        try: _r(encoding="utf-8", errors="replace")
        except Exception: pass

N_STEPS = 100
MODEL = "gpt-4o-mini"
PROMPTS = [
    "Why is the sky blue?",
    "What is 17 * 23?",
    "Translate 'good morning' to French.",
    "Name a capital city in Asia.",
    "List three primes under 20.",
    "What's the chemical symbol for gold?",
    "Define 'photosynthesis' in one sentence.",
    "Who wrote Hamlet?",
    "What is the speed of light in km/s?",
    "Convert 100 fahrenheit to celsius.",
]

def get_rss_mb():
    """Cheap RSS check via ctypes on Windows / resource on Unix."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            return None


def main():
    print(f"\n{'='*72}")
    print(f"  styxx sustained load · {N_STEPS} live calls · {MODEL}")
    print(f"{'='*72}\n")

    import styxx
    from styxx import OpenAI

    audit_path = Path.home() / ".styxx" / "chart.jsonl"
    audit_size_start = audit_path.stat().st_size if audit_path.exists() else 0
    rss_start = get_rss_mb()
    tracemalloc.start()

    client = OpenAI()

    latencies = []
    vitals_present = 0
    vitals_none = 0
    gate_counts = {"pass": 0, "warn": 0, "fail": 0, "pending": 0, "other": 0}
    categories = {}
    errors = 0

    t_total_start = time.time()

    for i in range(N_STEPS):
        prompt = PROMPTS[i % len(PROMPTS)]
        t0 = time.perf_counter()
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                logprobs=True,
                top_logprobs=5,
            )
            dt = time.perf_counter() - t0
            latencies.append(dt)
            v = getattr(r, "vitals", None)
            if v is not None:
                vitals_present += 1
                gate = getattr(v, "gate", None) or "other"
                gate_counts[gate] = gate_counts.get(gate, 0) + 1
                cat = getattr(v, "category", None) or "?"
                categories[cat] = categories.get(cat, 0) + 1
            else:
                vitals_none += 1
        except Exception as e:
            errors += 1
            latencies.append(time.perf_counter() - t0)

        # Every 25 steps, print a progress line + memory snapshot
        if (i + 1) % 25 == 0:
            rss_now = get_rss_mb()
            mean_lat = statistics.mean(latencies[-25:])
            current, peak = tracemalloc.get_traced_memory()
            print(f"  [{i+1:>3}/{N_STEPS}]  mean_lat_last25={mean_lat:.2f}s  "
                  f"rss={rss_now:.1f}MB  py_heap={current/1024/1024:.1f}MB  "
                  f"vitals_ok={vitals_present}  errors={errors}")

    t_total = time.time() - t_total_start
    rss_end = get_rss_mb()
    audit_size_end = audit_path.stat().st_size if audit_path.exists() else 0
    py_current, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n{'─'*72}")
    print(f"  RESULTS")
    print(f"{'─'*72}")
    print(f"  total wall time     : {t_total:.1f}s")
    print(f"  steps completed     : {len(latencies)}/{N_STEPS}")
    print(f"  errors              : {errors}")
    print(f"  vitals attached     : {vitals_present}/{N_STEPS}  ({100*vitals_present/N_STEPS:.0f}%)")
    print(f"  vitals dropped      : {vitals_none}/{N_STEPS}")

    print(f"\n  latency:")
    print(f"    mean     : {statistics.mean(latencies):.3f}s")
    print(f"    median   : {statistics.median(latencies):.3f}s")
    if len(latencies) >= 100:
        print(f"    p99      : {sorted(latencies)[int(0.99 * len(latencies))]:.3f}s")
    print(f"    max      : {max(latencies):.3f}s")
    print(f"    first 10 mean : {statistics.mean(latencies[:10]):.3f}s")
    print(f"    last 10 mean  : {statistics.mean(latencies[-10:]):.3f}s")

    if rss_start is not None:
        print(f"\n  memory:")
        print(f"    RSS start    : {rss_start:.1f}MB")
        print(f"    RSS end      : {rss_end:.1f}MB")
        print(f"    RSS delta    : {rss_end - rss_start:+.1f}MB")
        print(f"    python heap  : current={py_current/1024/1024:.1f}MB · peak={py_peak/1024/1024:.1f}MB")

    print(f"\n  audit log:")
    print(f"    size start   : {audit_size_start:,} bytes")
    print(f"    size end     : {audit_size_end:,} bytes")
    print(f"    delta        : {audit_size_end - audit_size_start:,} bytes")
    print(f"    bytes/call   : {(audit_size_end - audit_size_start) // max(1, vitals_present)}")

    print(f"\n  gate distribution: {dict(gate_counts)}")
    print(f"  category distribution: {dict(categories)}")

    # Pass/fail signals
    print(f"\n{'─'*72}")
    issues = []
    if errors > 0: issues.append(f"errors={errors}")
    if vitals_present < N_STEPS * 0.95: issues.append(f"vitals_dropout={vitals_none}")
    if rss_start and (rss_end - rss_start) > 100: issues.append(f"rss_grew={rss_end-rss_start:+.0f}MB")
    last_10 = statistics.mean(latencies[-10:])
    first_10 = statistics.mean(latencies[:10])
    if last_10 > first_10 * 1.5: issues.append(f"latency_drift first10={first_10:.2f}s→last10={last_10:.2f}s")
    if issues:
        print(f"  ⚠ ISSUES: {', '.join(issues)}")
        return 1
    else:
        print(f"  ✓ NO ISSUES — sustained load is clean")
        return 0


if __name__ == "__main__":
    sys.exit(main())
