# ROADMAP — the R line: from a coil in a room to a measured agent–environment coupling

Fathom Lab · 2026-08-05. The plan of record for the arc's real question. Kill tests (W1, M1)
run beside this line; they do not gate it. The word discipline from the R1 prereg binds every
document in this line: a positive result is a *measured coupling*, never "the agent feels the
room."

## The ladder (each rung licenses the next; none is skipped)

| rung | question | status | gate to next rung |
|---|---|---|---|
| **R0** — instrument exam | can the pipeline detect planted coupling, absorb a pure-clock confound, and stay silent on nothing — on synthetic worlds with known truth? | prereg frozen, running | `INSTRUMENT_VALID__r1_licensed` |
| **R1** — first measurement | is the room's spectral state legible to darkflobi's internal state, beyond circadian? | prereg frozen; apparatus committed (`run_r1.py`); blocked on R0 + hardware | `ROOM_IS_LEGIBLE` → R2; `CIRCADIAN_ONLY` / `NOT_LEGIBLE` → S1 |
| **R2** — replication | same verdict on a different window, different day? | designed at R1-close (window params only; machinery identical) | replicated → any public claim |
| **S1** — stimulus-driven | deliberate acoustic events at *randomized* times: does the agent's state track events the clock cannot predict? | successor design; breaks the circadian confound by construction | the strongest form of the claim |

## Apparatus state (2026-08-05)

- **Analysis pipeline**: `run_r1.py` committed before any room data exists — 24-dim agent
  vector, 60 s grid, hour-matched null, machinery reused verbatim from the disjoint-worlds
  arc. Smoke green. Frozen apparatus decisions documented in its module docstring.
- **Instrument exam**: `run_r0.py` running (three worlds × three seeds + power surface).
  The power surface — minimum planted coupling detectable at n = 240 — travels with every
  future R1 interpretation, so an R1 null reads "no coupling above the measured floor,"
  never "no coupling."
- **Room side**: `coil-sense` selftest ALL GREEN with no hardware; `room_cortex --record`
  now persists the raw 12-dim vector per emit in exactly the format the R1 loader reads
  (round-trip verified). Missing only the ~$110 Tier-0 hardware (operator).
- **Agent side**: live and needs nothing. `~/.styxx/chart.jsonl` logged its latest record
  today; 1908 records carry the full 21-dim `features_v2`. Measured cadence: ~29–292
  records/day → **200 paired bins ≈ 5–7 days of joint observation**, not hours. The window
  is days; plan recording accordingly.

## Design invariants (set here so no later document can relax them silently)

1. **The hour-matched null is the licensing null.** Both streams have a daily rhythm; beating
   only the free shuffle proves the existence of a clock.
2. **G0 coverage before reading anything** (≥200 paired bins) — the b35-b lesson.
3. **No perception vocabulary.** Feels / senses / experiences do not appear in findings.
   The claim ceiling is: recoverable shared geometric structure beyond time-of-day.
4. **A positive replicates (R2) before it is said in public.** Same rule as W1's.
5. **The instrument is validated before it touches reality (R0), and its measured power floor
   is quoted in every null.**

## What runs when the hardware arrives (the first 10 days, concretely)

1. Day 0: assemble; `selftest.py` against live daemon; `room_cortex --record room_record.jsonl`
   starts. Nothing is scored.
2. Days 0–2: baseline learning (the cortex needs its 2 h minimum; we take 48 h). Recorder
   accumulates. darkflobi is told nothing about windows or hypotheses beyond what is public
   in this repo (it can read the prereg — the *timestamps* of the scored window are what stay
   unannounced).
3. Days 2–9: the joint observation window accrues toward G0's 200 paired bins.
4. Day 9+: `python run_r1.py --room room_record.jsonl` — one command, one scored run, verdict
   from the frozen table, certify + seal + cycle log regardless of branch.

## Owed, by whom

- **R0 verdict** — machine, today.
- **Tier-0 hardware** (~$110: audio interface, magnet wire, lead) — operator.
- **Nothing else.** This line has no Gate 0 (no claim is being endorsed — the prereg predicts
  "open"), no external dependencies, and both software sides are done and committed.
