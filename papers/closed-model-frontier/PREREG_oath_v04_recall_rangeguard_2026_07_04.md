# PREREG — OATH v0.4: RANGE-GUARDED trigger-recall (autopilot cycle 24)

**Fathom Lab · papers/closed-model-frontier · 2026-07-04. FROZEN ON COMMIT, before any change to
`styxx/certify.py`. Bars below never move; a missed bar ⇒ CLOSED_NEGATIVE and the code change is
reverted (nothing ships).**

## The burial this names (a NEW prereg, not an amendment)

Cycle 23 (`PREREG_oath_v04_trigger_recall_2026_07_04.md`, `RESULT_…`, commit `83c3fa0`) widened
`_TRIGGERS` with the correlation/similarity register and **CLOSED_NEGATIVE on bar G3**: it recovered
+70 abstain-degrades (battery caught 58 → 128) but introduced **6 certifier ARTIFACTS** — a register
word obligating a non-measurement number. This prereg challenges that negative with a specific,
frozen fix and re-gates on the same G3 = 0 artifacts. It resurrects nothing else.

**The load-bearing observation from the cycle-23 hand-check** (`cycle23_g3_handcheck_result.json`):
**all six artifact numbers lie outside a correlation's range [−1, 1]** — `20` (an API cap under
"entropy"), `1` and `2` (stage ordinals under "drift"), `60`, `60`, `36` (ceiling percentages /
item counts under "ceiling"/"agreement"). Every *legitimate* correlation the recall recovers
(`RSA 0.264`, `reliability 0.735`, `Spearman 0.98`, the raw noise-ceiling `[0.394, 0.557]`) lies
*inside* [−1, 1]. A value-range guard separates the two classes exactly.

## The change (exact, frozen)

Two edits to `styxx/certify.py`, nothing else:

1. Add a second compiled pattern holding the cycle-23 correlation/similarity register verbatim:
   ```
   _TRIGGERS_CORR = rsa | rdm | spearman | pearson | correlations? | rho | consistency |
                    reliability | ceiling | agreement | convergence | drift | entropy |
                    similarity | variance   (word-boundaried, case-insensitive)
   ```
2. In `certify_doc`, after the existing `bound = bool(_TRIGGERS.search(bctx))`, add a
   **range-guarded** contribution — the correlation register obligates a number **only when that
   number is a plausible correlation**:
   ```python
   if not bound and _TRIGGERS_CORR.search(bctx) and -1.0 <= num["value"] <= 1.0:
       bound = True
   ```

Everything else is UNTOUCHED: the base `_TRIGGERS`, receipt flattening, float/count `_match`, the
count claim→field binding, the slash-pair rule, RANGE-SANITY (`unit_kw`/`sign_kw`), spec-constant
and quoted-historical rules, and the VERIFIED > UNGROUNDED > ABSTAIN precedence. The guard can only
add `bound=True` for in-range numbers; an out-of-range number in this register falls through to its
prior v0.3 fate (ABSTAIN), so the six cycle-23 artifacts (all out-of-range) revert to ABSTAIN.

**Provable property (checked empirically by the bars):** every cycle-23 artifact value ∉ [−1, 1],
so the guard removes all six by construction; the recall the guard keeps is exactly the in-range
correlation mutations (the battery mutates VERIFIED tokens, which are raw correlations in [0, 1]),
so catch should be preserved. Both claims are put to frozen bars, not assumed.

## Frozen bars (validation gates, run in this order)

| bar | gate | source |
|-----|------|--------|
| **G1** | `validate_oath_v0.py` (UNMODIFIED, seed 1) D1: caught ≥ 16 of 20 | inherited, PREREG_oath_v0 (bars never move) |
| **G2** | D2 = 0 false UNGROUNDED on the 3-doc corpus AND battery FALSE-VERIFY ≤ 26 of 269 | inherited D2 + cycle-23 |
| **G3** | **clean UNGROUNDED certifier ARTIFACTS = 0** across the 13 cycle-18 docs (every clean UNGROUNDED hand-verifies as a REAL doc↔receipt gap; the artifact class cycle 23 failed) | the decisive bar — frozen here |
| **G4** | committed battery (seed 1, 13 certs): total caught ≥ 116 of 269 (≥ 2× the v0.3 baseline of 58; recall must survive the range guard) | inherited from cycle 23 |
| **G5** | `python -m pytest tests -q` green; `py_compile` on every touched `.py` | repo standing rule |

Any missed bar ⇒ **CLOSED_NEGATIVE**: revert `styxx/certify.py`, publish the negative, done.
Near-bar = CLOSED_NEGATIVE. No amendment after results exist. **If ALL bars pass, the range-guarded
recall extension SHIPS** — `certify.py` is committed with the change and `validate_oath_v0.py` is
re-run so the 3 corpus certificates regenerate under the new verifier SHA (this is the first
certifier upgrade to ship since v0.3; the 13 cycle-18 certs stay pinned to their recorded SHA and
are a separate re-cert decision).

## Artifacts

- `papers/autopilot/cycle24_rangeguard_probe.py` — self-contained: runs the battery + the G3
  artifact classification under the shipped verifier (post-change) → `cycle24_rangeguard_battery_result.json`
  + `cycle24_rangeguard_g3_result.json`.
- `papers/closed-model-frontier/oath_v0_validation.json` + regenerated corpus certs — G1/G2-D2.
- A short RESULT note, itself certified (`python -m styxx.certify`, OATH-HELD) before commit.

Out of scope: claim→CELL float binding (cycle-22 sibling); quantity-specific ranges for the
unbounded register words (entropy/variance can exceed 1 — this cycle simply declines to bind them
out-of-range, a recall limitation honestly accepted, not a new range model); re-shipping the 13
cycle-18 certificate files; any change to `validate_oath_v0.py` or `mutant_battery.py`.

---

*Frozen on commit. The bar structure outranks the upgrade.*
