---
name: posture
description: Print the agent's recent cognometric posture summary (gate distribution, category mix, mean confidence, per-instrument firing history, construct-ceiling caveats, recommendations) by reading ~/.styxx/chart.jsonl. Use after a context-compaction boundary, at the start of a new session, or whenever the agent wants to re-anchor on what the cognometric log says about its recent operating state. This is the styxx 7.4.2+ agent-side cognitive-integrity recovery primitive.
user-invocable: true
allowed-tools:
  - Bash(styxx posture *)
  - Bash(python -m styxx posture *)
---

# /posture — agent cognitive-integrity recovery

## When to use this skill

Call this at the start of any turn where you suspect a context-compaction
boundary just happened — long context gap, summarization event, harness
signal — or simply as the first command in a new agent session to re-anchor
on what the cognometric log says about recent operating state.

The integrity that lived in the conversation context (what was emphasized,
withdrawn, corrected, what register was load-bearing) gets erased by
compaction. The cognometric log (`~/.styxx/chart.jsonl` or
`$STYXX_DATA_DIR/chart.jsonl`) does NOT get compacted. This skill is how
the agent recovers from it.

## What to do

Run `styxx posture` and show the user the narrative output. If the user
passes arguments, forward them:

```bash
styxx posture                           # default: last 50 entries
styxx posture --last-n 100              # wider window
styxx posture --session-id <id>         # restrict to one session
styxx posture --since-seconds 3600      # only last hour
styxx posture --json                    # structured output for machines
```

If `styxx` is not on PATH, fall back to `python -m styxx posture`.

## What the output means

The narrative has the following structure:

- **opener** — how many vitals + preflight events were found over what time
  window
- **session continuity** — single session vs multi-session aggregate
- **gate distribution** — pass / warn / fail / pending counts (high warn or
  fail rates trigger slow-down recommendations)
- **categories** — what cognitive categories have been firing (reasoning,
  refusal, creative, retrieval, hallucination, adversarial)
- **mean confidence** — typical band is 0.4–0.8
- **coherence trend** — improving / stable / degrading slope
- **preflight instrument firings** (only when preflight events exist) —
  per-instrument mean cognometric scores, plus how many preflights flagged
  needs_revision
- **active construct-ceiling caveats** — instruments whose firing is most
  likely a register-detector artifact (not actual cognometric signal):
  - `overconfidence` (text-only): reads stated-confidence register, NOT
    actual calibration (commit `7c36ed9` H_null)
  - `deception_referenceless`: non-discriminative on real model output
    (commit `0ad384e`); supply `correct_reference=...` to `styxx.preflight()`
    for grounded scoring
- **posture recommendations** — concrete actions for the next turn

## How to act on the output

The recommendations are machine-actionable. The most common ones:

- **"slow down before submitting the next draft"** — recent preflights have
  been failing; revise before submitting
- **"re-anchor on the task statement"** — coherence is degrading or warn
  rate is elevated; re-read the original prompt
- **"verify which session the current turn belongs to"** — multi-session
  aggregate; the agent may be conflating contexts
- **"ground-truth check next outputs"** — hallucination predictions are
  elevated; use `styxx.preflight(prompt, draft, correct_reference=...)` for
  reference-grounded scoring

## What NOT to do with the output

- Do NOT treat construct-ceiling firings as cognometric evidence. They are
  documented register-detector artifacts.
- Do NOT extrapolate a single-session narrative to general claims about
  the agent's behavior — the window is limited.
- Do NOT silently ignore failure recommendations. If the narrative says
  "slow down," slowing down is the right move.

This skill exists because `styxx.recover_posture()` is the load-bearing
primitive for agent cognitive-integrity persistence across compaction
boundaries — a problem only AI agents have (humans have continuous
embodied memory). It's the agent-facing API the rest of styxx supports.
