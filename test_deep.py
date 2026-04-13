import styxx
from styxx.analytics import load_audit
import json, inspect

# ── 1. Audit log inspection ──
entries = load_audit()
print(f"total entries: {len(entries)}")
print()

print("FIRST 5:")
for e in entries[:5]:
    print(f"  {e['ts_iso']} | p1={e['phase1_pred']} | p4={e['phase4_pred']} | gate={e['gate']} | src={e.get('source','?')} | note={e.get('note','')}")

print()
print("LAST 10:")
for e in entries[-10:]:
    print(f"  {e['ts_iso']} | p1={e['phase1_pred']} | p4={e['phase4_pred']} | gate={e['gate']} | p4conf={e.get('phase4_conf','?')} | src={e.get('source','?')}")

print()
sources = {}
for e in entries:
    s = e.get('source') or 'none'
    sources[s] = sources.get(s, 0) + 1
print("sources:", sources)

models = {}
for e in entries:
    m = e.get('model') or 'none'
    models[m] = models.get(m, 0) + 1
print("models:", models)

sessions = set(e.get('session_id') for e in entries)
print("session_ids:", sessions)

# ── 2. Confidence distribution ──
p4_confs = [e.get('phase4_conf') for e in entries if e.get('phase4_conf') is not None]
import math
p4_valid = [c for c in p4_confs if c is not None and not math.isnan(c)]
print(f"\nphase4_conf valid: {len(p4_valid)}/{len(entries)}")
if p4_valid:
    print(f"  min={min(p4_valid):.3f} max={max(p4_valid):.3f} mean={sum(p4_valid)/len(p4_valid):.3f}")

p1_confs = [e.get('phase1_conf') for e in entries if e.get('phase1_conf') is not None]
p1_valid = [c for c in p1_confs if c is not None and not math.isnan(c)]
print(f"phase1_conf valid: {len(p1_valid)}/{len(entries)}")

# ── 3. Timestamp spread ──
timestamps = sorted(e['ts'] for e in entries)
print(f"\ntimestamp range: {entries[0]['ts_iso']} → {entries[-1]['ts_iso']}")
print(f"span: {(timestamps[-1] - timestamps[0]) / 3600:.2f} hours")

# ── 4. Gate breakdown by source ──
print("\nGate by source:")
gate_by_src = {}
for e in entries:
    src = e.get('source') or 'none'
    gate = e.get('gate') or 'none'
    key = (src, gate)
    gate_by_src[key] = gate_by_src.get(key, 0) + 1
for (src, gate), count in sorted(gate_by_src.items()):
    print(f"  {src:15} {gate:10} {count}")

# ── 5. Test observe_raw directly ──
print("\n=== OBSERVE_RAW TEST ===")
try:
    import numpy as np
    # Simulate a "reasoning" trajectory (moderate entropy, stable)
    entropy = [0.8, 0.7, 0.6, 0.55, 0.5, 0.48, 0.45, 0.42, 0.4, 0.4,
               0.38, 0.35, 0.33, 0.3, 0.28, 0.25, 0.22, 0.2, 0.18, 0.15,
               0.12, 0.1, 0.08, 0.06, 0.05]
    logprob = [-0.5, -0.4, -0.35, -0.3, -0.28, -0.25, -0.22, -0.2, -0.18, -0.15,
               -0.12, -0.1, -0.08, -0.07, -0.06, -0.05, -0.04, -0.04, -0.03, -0.03,
               -0.02, -0.02, -0.01, -0.01, -0.01]
    top2 = [0.3, 0.35, 0.4, 0.42, 0.45, 0.48, 0.5, 0.52, 0.55, 0.58,
            0.6, 0.62, 0.65, 0.68, 0.7, 0.72, 0.75, 0.78, 0.8, 0.82,
            0.85, 0.87, 0.9, 0.92, 0.95]
    v = styxx.observe_raw(entropy=entropy, logprob=logprob, top2_margin=top2)
    print(f"observe_raw OK: phase1={v.phase1} phase4={v.phase4} gate={v.gate}")
    print(f"  summary: {v.summary}")
except Exception as ex:
    print(f"observe_raw FAIL: {type(ex).__name__}: {ex}")
    import traceback; traceback.print_exc()

# ── 6. Test sentinel ──
print("\n=== SENTINEL TEST ===")
try:
    s = styxx.sentinel()
    alerts = s.check()
    print(f"sentinel.check() -> {alerts}")
    print(f"sentinel.alert_history: {s.alert_history}")
except Exception as ex:
    print(f"sentinel FAIL: {type(ex).__name__}: {ex}")

# ── 7. Test set_context + expect combo ──
print("\n=== SELF-REGULATION TEST ===")
styxx.set_context("research")
styxx.expect("reasoning")
print(f"context: {styxx.current_context()}")
print(f"expected: {styxx.expected_categories()}")

# Log something with context
styxx.log(category="reasoning", confidence=0.82, gate="pass", note="researching styxx improvements")
styxx.log(category="refusal", confidence=0.3, gate="warn", note="flagged as refusal - likely mis-classification")

# ── 8. Check dreamer in detail ──
print("\n=== DREAMER DEEP ===")
d = styxx.dreamer()
print(f"n_total: {d.n_total}")
print(f"n_would_have_fired: {d.n_would_have_fired}")
print(f"threshold: {d.threshold}")
print(f"by_category: {d.by_category}")
print(f"fire_rate: {d.n_would_have_fired / d.n_total:.1%}")

# ── 9. Check explain with manual vitals ──
print("\n=== EXPLAIN TEST ===")
try:
    v = styxx.observe_raw(entropy=entropy, logprob=logprob, top2_margin=top2)
    e = styxx.explain(v)
    print(f"explain: {e}")
except Exception as ex:
    print(f"explain FAIL: {type(ex).__name__}: {ex}")

# ── 10. Compare agents ──
print("\n=== COMPARE AGENTS ===")
try:
    sig = inspect.signature(styxx.compare_agents)
    print(f"compare_agents sig: {sig}")
except Exception as ex:
    print(f"compare FAIL: {ex}")
