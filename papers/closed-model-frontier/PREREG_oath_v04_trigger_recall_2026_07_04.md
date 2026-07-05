# PREREG — OATH v0.4: trigger-vocabulary RECALL extension (autopilot cycle 23)

**Fathom Lab · papers/closed-model-frontier · 2026-07-04. FROZEN ON COMMIT, before any change to
`styxx/certify.py`. Bars below never move; a missed bar ⇒ CLOSED_NEGATIVE and the code change is
reverted (nothing ships).**

This is the **recall-axis sibling** of the reverted v0.4 float-binding attempt (cycle 22,
`PREREG_oath_v04_float_binding_2026_07_03.md`, CLOSED_NEGATIVE). That prereg named this extension
"out of scope"; it is now its own preregistered cycle. It **resurrects no buried claim** — it
challenges no negative; it attacks a disclosed *recall* gap that has never been attempted.

## Evidence this answers (the named, owed gap)

- Cycle-19 diligence (`papers/autopilot/DILIGENCE_cycle18_mutant_battery_2026_07_03.md`) and the
  committed battery (`papers/autopilot/cycle22_v03_baseline_battery_result.json`, seed 1, 269
  mutants over the 13 cycle-18 certificates): **abstain-degrade 182 / 269 (0.677)** — the single
  largest failure bucket. Mechanism (verbatim from the diligence note): *"these older docs state
  results in a register the UNGROUNDED trigger vocabulary does not cover (ρ/RSA/alignment/
  convergence/drift phrasing rather than AUROC/margin/FPR). An unbound corrupted number cannot fire
  UNGROUNDED — it falls back to ABSTAIN and the document verdict stays HELD."*
- Recon (2026-07-04, not persisted as a result) reproduced the baseline **269/58/26/182/3** exactly
  and word-counted the abstain contexts: the missed register is the **correlation / similarity
  family** — `RSA`, `RDM`, `Spearman`, `correlation`, `ρ/rho`, `consistency`, `reliability`,
  `ceiling` (noise-ceiling / % of ceiling), `agreement`, `convergence`, `drift`, `entropy`,
  `variance`, `similarity`. The current `_TRIGGERS` is AUROC/margin/FPR-centric and misses all of
  these; the affected docs (ai-human-alignment, representational-integrity, rhythm-rescue,
  council, sycophancy) speak this register.

## The change (exact, frozen)

**One edit only:** extend the `_TRIGGERS` alternation in `styxx/certify.py` with the
correlation/similarity register, so a corrupted number sitting in that register **binds** and can
fire UNGROUNDED instead of silently falling to ABSTAIN. The added alternatives (word-boundaried,
case-insensitive, folded into the existing group):

```
rsa | rdm | spearman | pearson | correlations? | rho | consistency |
reliability | ceiling | agreement | convergence | drift | entropy |
similarity | variance
```

Nothing else changes: the receipt-flattening surface, the float/count `_match` logic, the
count claim→field binding, the slash-pair rule, the range-sanity rule, the spec-constant and
quoted-historical rules, and the VERIFIED > UNGROUNDED > ABSTAIN precedence in `certify_doc` are
all UNTOUCHED. The extension only widens which lines are `bound`.

**The load-bearing tension (why a negative can fire):** these docs carry dense per-model
correlation tables. Widening `bound` moves a previously-ABSTAIN mutant to **either** `caught`
(no receipt coincidence-matches the corrupted value ⇒ UNGROUNDED) **or** `false-verify` (the
corrupted value coincidence-matches a *neighboring* leaf ⇒ still VERIFIED — the disclosed v0 float
gap cycle 22 could not close). Recall and binding are therefore **coupled**: if the dense tables
convert abstain→false-verify faster than abstain→caught, this extension makes tamper-evidence
*worse*, not better. G2 and G4 are set precisely to let that outcome fire CLOSED_NEGATIVE.

## Frozen bars (validation gates, run in this order)

| bar | gate | source |
|-----|------|--------|
| **G1** | `validate_oath_v0.py` (UNMODIFIED, seed 1) D1: caught ≥ 16 of 20 seeded mutations | inherited, PREREG_oath_v0 (bars never move) |
| **G2** | same run + the committed battery: **zero** false UNGROUNDED on the clean 3-doc corpus (D2), AND battery **FALSE-VERIFY ≤ 26 of 269** (no false-verify regression vs baseline) | inherited D2 + new, frozen here |
| **G3** | every 13 cycle-18 doc + 3 corpus doc re-certified under the extension: any clean UNGROUNDED that appears **hand-verifies as a REAL doc↔receipt discrepancy** (a reported catch), else the increase is a certifier ARTIFACT ⇒ kill. Baseline clean-UNGROUNDED = 0. | new, frozen here (the D2 rule, applied to the 13 docs) |
| **G4** | committed battery (seed 1, 13 certs): total **caught ≥ 116 of 269** (≥ 2× the v0.3 baseline of 58; equivalently recover ≥ 58 of the 182 abstain-degrades) | new, frozen here — the decisive recall bar |
| **G5** | `python -m pytest tests -q` green from repo root; `py_compile` on every touched `.py` | repo standing rule |

Any missed bar ⇒ **CLOSED_NEGATIVE**: revert `styxx/certify.py`, publish the negative with the
receipts, done. Near-bar = CLOSED_NEGATIVE. No amendment after results exist.

**Reported, not gated:** the exact per-doc `abstain→caught` vs `abstain→false-verify` split (the
diagnostic that makes any negative legible); the battery catch/false-verify/abstain rates; the
VERIFIED-count changes on the clean docs (the recall extension should not move VERIFIED at all —
it only re-classes former ABSTAINs).

## Artifacts

- `papers/autopilot/mutant_battery.py` — the committed battery (seed 1), re-run under the extension
  → `papers/autopilot/cycle23_recall_battery_result.json` (G2 false-verify + G4 catch receipt).
- `papers/autopilot/cycle23_recall_battery_result.json` and a per-doc split dump.
- `papers/closed-model-frontier/oath_v0_validation.json` + the 3 regenerated corpus certificates —
  G1/G2-D2 receipt (the validator regenerates them by design; the extension may not regress them).
- A short RESULT note, itself certified (`python -m styxx.certify`, OATH-HELD) before commit.

Out of scope (named so they cannot creep in): claim→CELL float binding (the cycle-22 sibling, a
separate future prereg); re-binding / re-shipping the 13 cycle-18 certificate files (they stay
pinned to their recorded verifier SHA); any change to `validate_oath_v0.py` or `mutant_battery.py`
(bars and the instrument never move).

---

*Frozen on commit. The bar structure outranks the upgrade.*
