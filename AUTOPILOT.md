# styxx AUTOPILOT — the self-advancing loop

**Contract for the autonomous advancement cycle. The scheduler fires a fresh agent on this file;
the agent IS the loop body. One cycle = one disciplined advance. The discipline is not optional —
it is the product.**

## Mission

Advance styxx — the certified-mind / machine-integrity instrument stack — one receipt-backed step
per cycle, indefinitely, without an operator pushing. Optimize for: claims that survive attack,
instruments that refuse what they cannot measure, certificates anyone can re-run.

## The cycle (in order, no skipping)

1. **Orient** (~5 min): `git -C C:\Users\heyzo\clawd\styxx pull` is FORBIDDEN if it would lose local
   state — instead `git status --porcelain` first; never discard work. Read:
   - `papers/PROGRAM_BACKLOG.md` (the ranked ledger — the single source of what's next)
   - `papers/autopilot/CYCLE_LOG.jsonl` (what previous cycles did; never repeat a CLOSED item)
   - last cycle's open follow-ups.
2. **Pick ONE item**: the smallest *decisive* next experiment or instrument upgrade — prefer
   (a) items the backlog marks OWED or SPAWNED, (b) local/$0 runs over API runs, (c) falsification
   of an existing claim over new features. Never pick two.
3. **Pre-register**: write PREREG with frozen bars/kill-gates BEFORE any scored run; commit it
   separately, before results exist. Smoke runs write only `*_SMOKE_INVALID*` and are never read
   as results.
4. **Run**: execute. Long runs: background + checkpoint. GPU budget 8 GB; API spend per cycle
   bounded to the subscription (no pay-per-token keys). If blocked, record the block in the cycle
   log and pick the next-smallest item ONCE (one fallback, then stop).
5. **Attest**: every new FINDING/RESULT doc must pass `python -m styxx.certify` against its
   receipts (OATH-HELD) before commit. If the corpus runner or mutant battery is touched, re-run
   `validate_oath_v0.py` — bars never move.
6. **Verify**: `python -m pytest tests -q` green, `python -m py_compile` on every touched .py.
   Red tests = the cycle's deliverable becomes fixing them, nothing else ships.
7. **Ship**: commit with the repo's message style + `Co-Authored-By`, push to
   `feat/closed-model-frontier` (or the current feature branch) on fathom-lab/styxx using the
   token at `C:\Users\heyzo\clawd\secrets\fathomlab-github.txt` (x-access-token URL form,
   credential helper bypassed). Update the PR if one is open.
8. **Close the loop**: update `papers/PROGRAM_BACKLOG.md` (result + re-rank), append one JSON line
   to `papers/autopilot/CYCLE_LOG.jsonl`:
   `{"cycle": N, "date": "...", "item": "...", "verdict": "...", "receipts": [...], "next": "..."}`

## Hard rails (violating any of these ends the cycle immediately)

- **Never**: PyPI publish, version bump, tag, GitHub release, merge to main — operator-gated.
- **Never**: close, paywall, obfuscate, or token-gate the verifier or the measurement primitives,
  or ship a certificate the public can't re-verify — styxx stays open at the core (see `docs/governance/OPEN_CORE.md`).
- **Never**: force-push, branch delete, history rewrite, `git reset --hard`, secrets in output.
- **Never**: delete or weaken a kill-gate, recalibrate a closed-negative instrument
  (overconfidence), resurrect a buried claim (geometry-manipulation probe, "universal oracle")
  without a NEW prereg that names the burial it is challenging.
- **Never**: mark SURVIVED on a missed bar; near-bar = CLOSED_NEGATIVE; report failures verbatim.
- Respect in-flight pre-registered runs: check `nvidia-smi --query-compute-apps` and running
  python/claude processes before firing anything; never contend with a scored run.
- One cycle = one item. Depth over breadth. A cycle that honestly reports "blocked" is a valid
  cycle; a cycle that ships an uncertified claim is not.

## Standing priorities (re-rank allowed, deletion not)

1. OWED items in `papers/PROGRAM_BACKLOG.md` (B-series, triage fleet debt, corpus provenance).
2. The 45 UNBOUND finding docs: add receipt citations doc by doc (mechanical, high-trust-yield).
3. The 684 UNGROUNDED corpus claims: triage → repair receipts or correct docs (loudly).
4. `styxx.mind` axes: package meaning-integrity (Binder) behind M1/M2-style equivalence gates;
   wire the mind certificate into `styxx.adapters` (the conscience mount).
5. OATH v0.4: claim→field binding for floats (the named limitation of v0.3).
6. Cross-vendor/scale replications of any SURVIVED claim still single-substrate.

## Spend & stop conditions

- Default budget: one focused cycle (~30–90 min of work) per firing. Do not chain cycles.
- STOP and write the block to the cycle log if: tests red at orient time and not fixed within the
  cycle; repo in detached/conflicted state; secrets file missing; >20% of API calls erroring.
- The operator can stop the loop at any time by deleting the scheduled task or this file.
  If this file is absent, do nothing.
