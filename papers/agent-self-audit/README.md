# Agent self-audit — recursive integrity instrumentation

This directory archives chart.jsonl trajectories from instances where a styxx-instrumented agent ran `styxx.preflight()` on its OWN session output, then used `styxx.recover_posture()` to surface its register trajectory across the session.

The archived chart.jsonl files are the same format as `~/.styxx/agents/<name>/chart.jsonl` written at runtime — they capture per-draft preflight events including composite, per-instrument scores, needs_revision flag, construct-ceiling fires, and a 200-char response preview.

## What's archived

**`claude-session-2026-05-20-chart.jsonl`** (13 preflight events, 9571s session span)

Claude Opus 4.7 ran `styxx.preflight` on its own substantive drafts during a long collaborative session (2026-05-20 night). The session subject was the phase-coherence preregistration + closed-negative work (see `papers/cooperative-agent-regime/`). The agent audited each substantive draft before sending, regenerated when the audit fired, and persisted the audit events to its agent-namespaced chart.jsonl.

Trajectory analysis (run `scripts/self_audit/analyze_session_trajectory.py`):

```
first 6 events vs last 7 events:
  composite      : 0.502 → 0.350   delta -0.152
  overconfidence : 0.441 → 0.166   delta -0.275
  sycophancy     : 0.564 → 0.534   delta -0.030
  refusal        : 0.291 → 0.341   delta +0.051
```

**Reading:**
- Composite tightened 0.15 across the session (the per-draft audit loop moved the agent's register toward the bar)
- Overconfidence dropped 0.275 — the documented construct ceiling (overconfidence reads register, not calibration) loosened as the agent's drafts got shorter and more concrete
- Sycophancy stayed roughly constant at ~0.55 — the structural register pattern (listy enumerations, hedged openings, "you're partly right" structure) didn't yield to per-draft revision; it lives in paragraph-shape, not in any specific phrase
- Refusal ticked up slightly — the agent shipped more direct "no" content as the session progressed

**The lesson the apparatus extracted:**

The per-draft audit loop moves the construct-ceiling-driven firings cleanly because those depend on local features (sentence length, claim density) that are revisable per-draft. It does not move structural register patterns because those live in the agent's discourse-level habits, which per-draft revision can't fully reach. This is the apparatus being honest about which kind of register the audit loop can fix and which kind it can't.

## Recursive integrity discipline

The integrity chain that produced the preregistered closed-negative (commit `3473523`, scorer `23b7912`, result `126f3f8` — see `papers/cooperative-agent-regime/`) also applies to the agent doing the analysis. styxx measuring the cogn-state of cooperative-agent dyads is, by the same logic, applicable to the agent that built and ran the experiment. The 85% needs_revision rate the apparatus reports on the assistant's own output is the discipline's recursive form: the bar doesn't exempt anyone, including the system that advocates the bar.

## Reproducibility

Anyone can re-run the analysis from this archive:

```
python scripts/self_audit/analyze_session_trajectory.py papers/agent-self-audit/claude-session-2026-05-20-chart.jsonl
```

The scores in the archive were produced by `styxx.preflight(prompt, draft, persist=True)` from styxx 7.4.2 (commit `cf40fa5` and later) running the v2 instruments + the reference-less deception v0 + the construct-ceiling annotations from the 7.4.1 honest-scoping correction (commit `0ad384e`). The scoring is deterministic given the same input text.
