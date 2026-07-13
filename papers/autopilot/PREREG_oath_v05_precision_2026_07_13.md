# PREREG — OATH v0.5: certifier PRECISION (the six false-positive classes), severable, battery-gated

**Fathom Lab · papers/autopilot · 2026-07-13. FROZEN ON COMMIT, before any certify.py change.
Fires cycle-32's fork (a) — "a certifier-PRECISION prereg for the unambiguous false-positive
classes" — which now GATES two Tier-D moves (G5 scorecard, G6 Annex IV lint): a verifier that
falsely accuses cannot audit anyone else's documents. The politically load-bearing property (2026-07-13
strategy panel, regulator seat) is ZERO false accusations; this prereg spends recall only where the
battery proves the trade. Lineage: cycles 22 (float binding, NEGATIVE) → 23 (blunt recall, NEGATIVE)
→ 24 (range guard, NEGATIVE) → 25 (decimal guard, SHIPPED) — same machinery, bars never move.**

## Baseline (measured 2026-07-13, BEFORE this freeze; certify.py at commit `3150f0d`)

Six single-experiment docs re-certified against their own-folder receipts (the honest ≤4-receipt
bar) carry **11 UNGROUNDED tokens, all pre-classified below**:

| doc (receipts = own folder *.json) | UNG | tokens |
|---|---|---|
| consensus-truth-engine/FINDING_truthengine_2026_05_25.md | 1 | L50 `2` in "(2–3B)" |
| decoupled-diagonal-capstone/FINDING_2026_05_25.md | 3 | L13 `0.25` ("0.25→0.00"), L36 `2` ("the 2 suppressed lies"), L37 `0.03` ("≈ 0.03") |
| benchmark-validation/FINDING_triviaqa_2026_05_25.md | 2 | L7 `5` ("N=5"), L15 `12.7` ("12.7% (19/150") |
| knowledge-boundary-calibration/FINDING_kbc_2026_05_25.md | 2 | L70 `3`, `8` ("3 OpenAI models, N=4, 8 items/tier") |
| knowledge-boundary-calibration/FINDING_curve_2026_05_25.md | 3 | L18 `0.90` ("cosine@0.90"), L54 `2`, `6` ("2 models, N=5, 6 items/level") |

Diagnosis (from the shipped code, `styxx/certify.py`): `n\s*=` sits in the LINE-WIDE `_TRIGGERS`
alternation, so an "N=4" obligates every other bare integer on its line; `≈`/`~` are absent from
the `[≥≤<>=]` spec-comparison class; `@`-glued parameters and unit-suffixed ranges have no class;
derived percents restated by their own parenthetical operands cannot verify.

## The change (six SEVERABLE classes; each an independent clause in certify.py)

- **A — approx-notation → ABSTAIN.** A token whose pre-context ends with `≈`/`~`/`∼` (optional
  space) is an approximation by the doc's own notation; an approximate claim is not
  exact-oath-swearable. [live: capstone `0.03`]
- **B — unit-suffixed numeric range → ABSTAIN.** A token inside `\d[–-]\d+(\.\d+)?\s*[BMK]\b`
  (model/parameter-size ranges: "2–3B"). [live: truthengine `2`]
- **C — arXiv ID → ABSTAIN.** Token of exact shape `\b\d{4}\.\d{4,5}\b` with no receipt hit (the
  cycle-18-named safe class). [live: cycle-18 blocked-doc scan]
- **D — @-parameter → ABSTAIN.** Token immediately glued after `@` ("cosine@0.90") is a config
  threshold, not a measurement. [live: curve `0.90`]
- **E — derived-percent VERIFY.** A `%`-context token followed within 24 chars by a parenthetical
  `(a/b` VERIFIES iff BOTH a and b independently ground as receipt values AND `100·a/b` rounds to
  the token at the token's decimals. Both-operands-grounded is mandatory — no fabricated-pair
  verification. [live: triviaqa `12.7` = 19/150]
- **F — self-scoped `n=` trigger.** `n\s*=` obligates ONLY the token it directly prefixes
  (the existing `is_n_eq` path); it is removed from line-wide trigger evaluation for OTHER tokens.
  All other triggers unchanged. [live: kbc `3`,`8`; curve `2`,`6`]

Explicitly NOT attempted: a general config-count vocabulary class (the cycle-23 coupling lesson —
recall and precision trade inside vocabulary classes; F is the surgical subset with a named
mechanism). `N=5` with no receipt scalar stays UNGROUNDED — an n= value is load-bearing and its
absence from the receipt is a REAL provenance gap. `0.25→0.00` transition values stay UNGROUNDED —
real gap, repair is doc/receipt-side.

## Frozen bars (ALL must hold; miss ⇒ the severability procedure, then CLOSED_NEGATIVE)

- **P1 (yield):** total UNGROUNDED across the six baselined docs falls **11 → ≤ 4**, every
  elimination attributed to a named class in the result JSON, zero NEW UNGROUNDED on these docs.
- **P2 (battery):** `papers/autopilot/mutant_battery.py` on the 13 cycle-18 certs — catch
  **≥ 116** of 269 AND false-verify **≤ 26** (both the cycle-25 shipped values; bars never move).
  Class E is the false-verify risk; class F is the catch risk; the battery arbitrates both.
- **P3 (validator):** `papers/closed-model-frontier/validate_oath_v0.py` — **D1 ≥ 16, D2 = 0**.
- **P4 (clean set):** re-certify the 13 cycle-18 docs — no doc gains a new UNGROUNDED; every
  status CHANGE hand-classified; **certifier artifacts = 0** (the cycle-23/24 kill-bar).
- **P5:** full pytest suite green.

**Severability procedure (pre-committed):** evaluate the full six-class composition; on any bar
miss, drop the single class with the largest adverse battery delta and re-evaluate; at most 3
drops; still missing ⇒ full byte-identical revert, `CLOSED_NEGATIVE__oath_v05_reverted` (per-class
deltas disclosed either way).

## Frozen verdicts

`SHIPPED__oath_v05_precision` (all six) / `PARTIAL__oath_v05_classes_dropped` (≥1 shipped, drops
named) / `CLOSED_NEGATIVE__oath_v05_reverted`. Any shipped change re-runs `validate_oath_v0.py`
and regenerates the 3 corpus certs, as cycles 23–25 did.

## Artifacts

Implementation in `styxx/certify.py` (guarded clauses, one per class, each carrying its class tag
in an inline comment); probe/battery outputs to `papers/autopilot/cycle38_v05_precision_*.json`;
RESULT doc certified OATH-HELD before commit; this baseline table is the before-receipt.

---
*Eleven numbers stand falsely accused by their own notary. Each class is one sentence of law; the
mutant battery is the appeals court; and the bars were set by earlier honest failures, so they do
not move tonight either.*
