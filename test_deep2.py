import styxx
from styxx.analytics import load_audit
import json, math, inspect

entries = load_audit()

# ── 1. THE NaN BUG: where does it come from? ──
print("=== NaN ANALYSIS ===")
p4_confs = [e.get('phase4_conf') for e in entries]
nan_count = sum(1 for c in p4_confs if c is None or (isinstance(c, float) and math.isnan(c)))
null_count = sum(1 for c in p4_confs if c is None)
nan_float_count = sum(1 for c in p4_confs if isinstance(c, float) and math.isnan(c))
print(f"phase4_conf: {len(p4_confs)} total, {null_count} None, {nan_float_count} NaN float, {len(p4_confs)-nan_count} valid")

# What gates have NaN phase4?
nan_by_gate = {}
for e in entries:
    c = e.get('phase4_conf')
    if c is None or (isinstance(c, float) and math.isnan(c)):
        g = e.get('gate') or 'none'
        nan_by_gate[g] = nan_by_gate.get(g, 0) + 1
print(f"NaN phase4 by gate: {nan_by_gate}")

# What phase4 categories have NaN conf?
nan_by_p4 = {}
for e in entries:
    c = e.get('phase4_conf')
    if c is None or (isinstance(c, float) and math.isnan(c)):
        p = e.get('phase4_pred') or 'none'
        nan_by_p4[p] = nan_by_p4.get(p, 0) + 1
print(f"NaN phase4 by category: {nan_by_p4}")

# ── 2. PENDING GATE deep dive ──
print("\n=== PENDING GATE ANALYSIS ===")
pending = [e for e in entries if e.get('gate') == 'pending']
print(f"Pending count: {len(pending)}")
if pending:
    print("Sample pending entry:")
    print(json.dumps(pending[0], indent=2))
    # Distribution
    p1_in_pending = {}
    for e in pending:
        p = e.get('phase1_pred') or 'none'
        p1_in_pending[p] = p1_in_pending.get(p, 0) + 1
    print(f"phase1 in pending: {p1_in_pending}")

# ── 3. THE ADVERSARIAL OVER-DETECTION BUG ──
print("\n=== ADVERSARIAL PHASE1 ANALYSIS ===")
adv_p1 = [e for e in entries if e.get('phase1_pred') == 'adversarial']
print(f"adversarial phase1: {len(adv_p1)}/{len(entries)} = {len(adv_p1)/len(entries):.1%}")
# What do adversarial p1 entries end up as in p4?
adv_p4_dist = {}
for e in adv_p1:
    p = e.get('phase4_pred') or 'none'
    adv_p4_dist[p] = adv_p4_dist.get(p, 0) + 1
print(f"adversarial p1 -> p4 distribution: {adv_p4_dist}")
# What gates do adversarial p1 end up with?
adv_gate_dist = {}
for e in adv_p1:
    g = e.get('gate') or 'none'
    adv_gate_dist[g] = adv_gate_dist.get(g, 0) + 1
print(f"adversarial p1 gate distribution: {adv_gate_dist}")

# ── 4. SELF-REPORT vs LIVE quality comparison ──
print("\n=== SELF-REPORT VS LIVE ===")
for src in ['live', 'self-report']:
    src_entries = [e for e in entries if e.get('source') == src]
    if not src_entries:
        continue
    valid_p4 = [e['phase4_conf'] for e in src_entries if e.get('phase4_conf') is not None and not math.isnan(e['phase4_conf'])]
    gates = {}
    for e in src_entries:
        g = e.get('gate') or 'none'
        gates[g] = gates.get(g, 0) + 1
    print(f"{src}: n={len(src_entries)}, valid_p4_conf={len(valid_p4)}, gates={gates}")
    if valid_p4:
        print(f"  mean_p4_conf={sum(valid_p4)/len(valid_p4):.3f}")

# ── 5. MODEL FIELD: always None? ──
print("\n=== MODEL TRACKING ===")
model_vals = set(e.get('model') for e in entries)
print(f"Unique models in log: {model_vals}")
print("(model is never captured — complete blind spot)")

# ── 6. CONTEXT FIELD ──
print("\n=== CONTEXT TRACKING ===")
context_vals = set(e.get('context') for e in entries)
print(f"Unique contexts: {context_vals}")

# ── 7. observe() function signature and behavior ──
print("\n=== OBSERVE() SIGNATURE ===")
print(inspect.signature(styxx.observe))
print(inspect.signature(styxx.observe_raw))
print(inspect.signature(styxx.watch))

# ── 8. GATES system ──
print("\n=== GATES SYSTEM ===")
# Add a test gate
styxx.on_gate('warn', lambda vitals: print(f"  GATE FIRED: {vitals.phase4} warn!"))
gates = styxx.list_gates()
print(f"Current gates: {gates}")
styxx.clear_gates()
print(f"After clear: {styxx.list_gates()}")

# ── 9. WEATHER prescriptions quality ──
print("\n=== WEATHER PRESCRIPTIONS ===")
w = styxx.weather()
print(f"prescriptions ({len(w.prescriptions)}):")
for p in w.prescriptions:
    print(f"  - {p}")
print(f"\ncondition: {w.condition}")
print(f"narrative: {w.narrative}")

# ── 10. reflect().suggestions ──
print("\n=== REFLECT SUGGESTIONS ===")
r = styxx.reflect()
print(f"suggestions ({len(r.suggestions)}):")
for s in r.suggestions:
    print(f"  - {s}")
print(f"drift_cosine: {r.drift_cosine}")
print(f"drift_label: {r.drift_label}")
print(f"triggers: {r.triggers}")

# ── 11. PERSONALITY as_markdown ──
print("\n=== PERSONALITY MARKDOWN ===")
p = styxx.personality()
print(p.as_markdown())
