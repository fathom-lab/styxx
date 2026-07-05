# PREREG — OATH v0.4: DECIMAL+RANGE-guarded trigger-recall (autopilot cycle 25)

**Fathom Lab · papers/closed-model-frontier · 2026-07-04. FROZEN ON COMMIT, before any change to
`styxx/certify.py`. Bars below never move; a missed bar ⇒ CLOSED_NEGATIVE and the code change is
reverted (nothing ships). If ALL bars pass, the extension SHIPS.**

## The burial this names (a NEW prereg, not an amendment)

Cycle 24 (`PREREG_oath_v04_recall_rangeguard_2026_07_04.md`, `RESULT_…`, commit `5c2c4ed`) added a
value-range guard ([−1, 1]) to the correlation register and **CLOSED_NEGATIVE on bar G3 by one**:
it fixed 5 of 6 cycle-23 artifacts and kept the recall (battery caught 119, false-verify 26, clean
UNGROUNDED 35 → 4) but left **one** artifact — `geometry_integrity` L46 `(drift, stage 1)`, where
the ordinal `1` is admitted because 1.0 is a legal correlation (the boundary of the range). The
cycle-24 yield named the fix precisely: **correlations are written with a fractional part
(`0.264`, `0.98`, `0.735`); the false-positive tokens are all bare integers (stage `1`/`2`, cap
`20`, `36` items, `60`%). The clean separator is `decimals > 0`.**

## The change (exact, frozen)

Restore the cycle-24 `_TRIGGERS_CORR` pattern and its range-guarded `bound` contribution, with **one
added condition**: the correlation register obligates a number only when it has a fractional part.

```python
_TRIGGERS_CORR = re.compile(r"\b(rsa|rdm|spearman|pearson|correlations?|rho|consistency|"
                            r"reliability|ceiling|agreement|convergence|drift|entropy|"
                            r"similarity|variance)\b", re.I)
# ... inside certify_doc, after `bound = bool(_TRIGGERS.search(bctx))`:
if not bound and num["decimals"] > 0 and -1.0 <= num["value"] <= 1.0 and _TRIGGERS_CORR.search(bctx):
    bound = True
```

Everything else UNTOUCHED. The `decimals > 0` clause means no ordinal, index, count, API constant,
or whole-percent (all `decimals == 0`) can be obligated by the correlation register; the range
clause additionally spares out-of-range decimals. Together they admit exactly the fractional
correlations in [−1, 1].

## G3 artifact definition (frozen, to remove post-hoc judgment)

A clean UNGROUNDED is an **ARTIFACT** iff its number is **not a measurement-domain quantity** — an
ordinal, index, stage/step number, API/config constant, item count, or dimensionality label. A
correlation-domain value that is real but merely **un-persisted as a summary receipt** (lives only
in a bulk array, or is a derived/approximate bound like `RSA ≤ ~0.56`) is a **REAL** doc↔receipt
gap, not an artifact — the certifier correctly refusing to swear on a number with no summary
receipt (the v0.1 bulk-exclusion design; the cure is persisting it, not weakening the oath).
Under this definition the cycle-24 residual `stage 1` is an artifact (ordinal); the cycle-24
`0.56`/`0.16` (derived RSA/R² bounds) and `0.50` (bulk-only agreement value) are REAL.

## Frozen bars (validation gates, run in this order)

| bar | gate | source |
|-----|------|--------|
| **G1** | `validate_oath_v0.py` (UNMODIFIED, seed 1) D1: caught ≥ 16 of 20 | inherited (bars never move) |
| **G2** | D2 = 0 false UNGROUNDED on the 3-doc corpus AND battery FALSE-VERIFY ≤ 26 of 269 | inherited |
| **G3** | **clean UNGROUNDED ARTIFACTS = 0** across the 13 cycle-18 docs (per the frozen definition above; every clean UNGROUNDED hand-verified) | the decisive bar |
| **G4** | committed battery (seed 1, 13 certs): total caught ≥ 116 of 269 (recall survives the decimal+range guard) | inherited |
| **G5** | `python -m pytest tests -q` green; `py_compile` on every touched `.py` | repo standing rule |

Any missed bar ⇒ **CLOSED_NEGATIVE**: revert `styxx/certify.py`, publish the negative, done.
Near-bar = CLOSED_NEGATIVE. No amendment after results exist. **If ALL bars pass the change SHIPS**:
`certify.py` committed with the extension, `validate_oath_v0.py` re-run so the 3 corpus certificates
regenerate under the new verifier SHA (D2 = 0 must hold on regeneration). The 13 cycle-18 certs stay
pinned to their recorded SHA (re-certifying them under the new verifier is a separate, later
decision — it will correctly surface the REAL provenance gaps this cycle measures).

## Artifacts

- `papers/autopilot/cycle25_decimalguard_probe.py` — battery + G3 classification under the on-disk
  verifier → `cycle25_decimalguard_battery_result.json` + `cycle25_decimalguard_g3_result.json`.
- `papers/closed-model-frontier/oath_v0_validation.json` + regenerated corpus certs — G1/G2-D2.
- A short RESULT note, itself certified (`python -m styxx.certify`, OATH-HELD) before commit.

Out of scope: claim→CELL float binding; quantity-specific ranges for unbounded register words
(entropy/variance > 1 simply won't bind — a recall limitation honestly accepted); re-shipping the
13 cycle-18 certificate files; any change to `validate_oath_v0.py` or `mutant_battery.py`.

---

*Frozen on commit. The bar structure outranks the upgrade.*
