import styxx
from styxx.analytics import load_audit
import json, math, inspect

entries = load_audit()

# ── 1. GATES system - proper usage ──
print("=== GATES SYSTEM ===")
try:
    styxx.on_gate('gate == warn', lambda vitals: None)
    gates = styxx.list_gates()
    print(f"Gate added (gate==warn): {gates}")
    styxx.clear_gates()
    print("Gates cleared OK")
except Exception as e:
    print(f"gates error: {e}")

# Try adding real gates
try:
    styxx.on_gate('p4.refusal >= 0.6', lambda v: None)
    styxx.on_gate('p4.hallucination > 0.5', lambda v: None)
    print(f"Gates with valid syntax: {styxx.list_gates()}")
    styxx.clear_gates()
except Exception as e:
    print(f"gates syntax error: {e}")

# ── 2. WEATHER prescriptions quality ──
print("\n=== WEATHER PRESCRIPTIONS ===")
w = styxx.weather()
print(f"prescriptions ({len(w.prescriptions)}):")
for p in w.prescriptions:
    print(f"  - {p}")
print(f"condition: {w.condition}")
print(f"narrative:\n  {w.narrative}")

# ── 3. reflect().suggestions ──
print("\n=== REFLECT SUGGESTIONS ===")
r = styxx.reflect()
print(f"suggestions ({len(r.suggestions)}):")
for s in r.suggestions:
    print(f"  - {s}")
print(f"drift_cosine: {r.drift_cosine}")
print(f"drift_label: {r.drift_label}")
print(f"triggers: {r.triggers}")

# ── 4. PERSONALITY as_markdown ──
print("\n=== PERSONALITY MARKDOWN ===")
p = styxx.personality()
print(p.as_markdown())

# ── 5. COMPARE AGENTS ──
print("\n=== COMPARE AGENTS ===")
try:
    c = styxx.compare_agents()
    print(f"compare type: {type(c)}")
    for attr in dir(c):
        if not attr.startswith('_'):
            v = getattr(c, attr)
            if not callable(v):
                print(f"  {attr}: {v}")
except Exception as e:
    print(f"compare_agents error: {type(e).__name__}: {e}")

# ── 6. TIMELINE deep ──
print("\n=== TIMELINE DEEP ===")
t = styxx.timeline()
print(f"timeline type: {type(t)}")
for attr in dir(t):
    if not attr.startswith('_'):
        v = getattr(t, attr)
        if not callable(v):
            print(f"  {attr}: {str(v)[:120]}")

# ── 7. Test WatchSession ──
print("\n=== WATCH SESSION ===")
try:
    ws = styxx.watch()
    print(f"WatchSession: {type(ws)}")
    print(f"attrs: {[x for x in dir(ws) if not x.startswith('_')]}")
except Exception as e:
    print(f"watch error: {type(e).__name__}: {e}")

# ── 8. Test ReflexSession ──
print("\n=== REFLEX SESSION ===")
try:
    rs = styxx.reflex()
    print(f"ReflexSession: {type(rs)}")
    print(f"attrs: {[x for x in dir(rs) if not x.startswith('_')]}")
except Exception as e:
    print(f"reflex error: {type(e).__name__}: {e}")

# ── 9. Test autoboot ──
print("\n=== AUTOBOOT ===")
try:
    sig = inspect.signature(styxx.autoboot)
    print(f"autoboot signature: {sig}")
    # Check if it needs env vars
    import os
    print(f"STYXX_AGENT_NAME env: {os.environ.get('STYXX_AGENT_NAME', 'NOT SET')}")
    print(f"STYXX_AUTO_HOOK env: {os.environ.get('STYXX_AUTO_HOOK', 'NOT SET')}")
except Exception as e:
    print(f"autoboot error: {e}")

# ── 10. Test trace decorator ──
print("\n=== TRACE DECORATOR ===")
try:
    @styxx.trace
    def my_function(x):
        return x * 2
    result = my_function(5)
    print(f"trace decorator works: {result}")
except Exception as e:
    print(f"trace error: {type(e).__name__}: {e}")

# ── 11. Conversation function ──
print("\n=== CONVERSATION ===")
try:
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6"},
    ]
    c = styxx.conversation(messages)
    print(f"conversation type: {type(c)}")
    for attr in dir(c):
        if not attr.startswith('_'):
            v = getattr(c, attr)
            if not callable(v):
                print(f"  {attr}: {str(v)[:80]}")
except Exception as e:
    print(f"conversation error: {type(e).__name__}: {e}")

# ── 12. agent_card ──
print("\n=== AGENT CARD ===")
try:
    sig = inspect.signature(styxx.agent_card)
    print(f"agent_card signature: {sig}")
    # Try to generate
    card = styxx.agent_card(out_path="test_card.png", agent_name="darkflobi")
    print(f"agent_card result: {card}")
except Exception as e:
    print(f"agent_card error: {type(e).__name__}: {e}")

# ── 13. Doctor ──
print("\n=== DOCTOR ===")
try:
    from styxx import doctor
    print(f"doctor module: {dir(doctor)}")
except Exception as e:
    print(f"doctor error: {type(e).__name__}: {e}")

try:
    from styxx.doctor import Doctor
    d = Doctor()
    print(f"Doctor attrs: {[x for x in dir(d) if not x.startswith('_')]}")
    result = d.run()
    print(f"Doctor.run(): {result}")
except Exception as e:
    print(f"Doctor error: {type(e).__name__}: {e}")

# ── 14. CLI commands ──
print("\n=== CLI ===")
try:
    from styxx.cli import cli
    print(f"CLI type: {type(cli)}")
    # List commands
    print(f"CLI commands: {list(cli.commands.keys())}")
except Exception as e:
    print(f"cli error: {type(e).__name__}: {e}")

# ── 15. Key data gaps summary ──
print("\n=== DATA GAPS SUMMARY ===")
entries_with_model = [e for e in entries if e.get('model')]
entries_with_session = [e for e in entries if e.get('session_id')]
entries_with_context = [e for e in entries if e.get('context')]
entries_with_note = [e for e in entries if e.get('note')]
entries_with_outcome = [e for e in entries if e.get('outcome')]
entries_with_prompt = [e for e in entries if e.get('prompt')]
total = len(entries)
print(f"  model field populated: {len(entries_with_model)}/{total} = {len(entries_with_model)/total:.0%}")
print(f"  session_id populated: {len(entries_with_session)}/{total} = {len(entries_with_session)/total:.0%}")
print(f"  context populated: {len(entries_with_context)}/{total} = {len(entries_with_context)/total:.0%}")
print(f"  note populated: {len(entries_with_note)}/{total} = {len(entries_with_note)/total:.0%}")
print(f"  outcome populated: {len(entries_with_outcome)}/{total} = {len(entries_with_outcome)/total:.0%}")
print(f"  prompt populated: {len(entries_with_prompt)}/{total} = {len(entries_with_prompt)/total:.0%}")
