# PREREG — OATH v0.4: float claim→field binding (autopilot cycle 22)

**Fathom Lab · papers/closed-model-frontier · 2026-07-03. FROZEN ON COMMIT, before any change to
`styxx/certify.py`. Bars below never move; a missed bar ⇒ CLOSED_NEGATIVE and the code change is
reverted (nothing ships).**

## Evidence this answers (the burial it does NOT challenge — nothing here is resurrected)

- `styxx/certify.py` v0.3 discloses: *"Floats keep value-only matching (v0.4 owes them full
  claim→field binding)."* Standing priority #5.
- Cycle-19 mutant battery (`papers/autopilot/cycle18_mutant_battery_result.json`, seed 1, 269
  mutants over the 13 cycle-18 certificates): **26 FALSE-VERIFY (0.097)** — a corrupted float
  re-verifies against a *neighboring* receipt leaf (dense `per_model` tables), i.e. **the oath
  swears to the corrupted number**. This is the worst failure class a certifier can have.

## The change (exact, frozen)

In `certify_doc`, floats (`decimals > 0`) that value-match one or more receipt leaves get a
claim→field binding filter before VERIFIED is granted:

1. **Binding surface (leaf side):** the leaf's **last two path segments only** (field name +
   immediate parent; list indices `[i]` are dropped as segments). The full path is deliberately NOT
   used — generic containers (`per_model.*`) would bind against any table whose header says
   "model", which is precisely the cycle-19 false-verify channel.
2. **Binding test:** tokenize both sides into alphanumeric runs (claim side: the claim's
   `binding_context` — its line, plus table header if in a table). A leaf **binds** iff some leaf
   token and some claim token (both length ≥ 2, case-insensitive) satisfy: exact equality, OR
   equal 4-prefixes (both length ≥ 3), OR one contains the other (contained side length ≥ 3).
3. **Semantics (the asymmetry with counts is deliberate):**
   - some value-matching leaf binds ⇒ **VERIFIED**, receipt_ref = the first *binding* hit.
   - value-match exists but NO matching leaf binds ⇒ **ABSTAIN**, marked loudly in the ledger as
     `float-value-match-unbound(v0.4)` — the certificate refuses to swear on a number it cannot
     tie to a field. It does NOT become UNGROUNDED: a value-match that fails binding is "not
     checkable at v0.4 binding strength", not a proven contradiction — floats are demoted, never
     accused. (Counts keep their validated v0.3 full-path rule + UNGROUNDED fall-through,
     untouched.)
   - no value-match at all ⇒ unchanged v0 logic (UNGROUNDED if trigger-bound, else ABSTAIN).
4. Spec-constant, quoted-historical, range-sanity, count-binding, slash-pair, and trigger rules are
   all UNCHANGED.

**Provable safety properties (stated now, checked empirically by the bars):** demotion can only
move VERIFIED→ABSTAIN; it can never create UNGROUNDED, so it cannot fail a clean document. Mutants
previously *caught* (no value-match anywhere) are untouched, so D1 catch cannot decrease via this
change; mutants previously FALSE-VERIFIED either lose their coincidence leaf (→ABSTAIN or caught)
or keep a *binding* coincidence leaf (the residue bar B3 measures).

## Frozen bars (validation gates, run in this order)

| bar | gate | source |
|-----|------|--------|
| **B1** | `validate_oath_v0.py` (UNMODIFIED, seed 1) D1: caught ≥ 16 of 20 seeded mutations | inherited, PREREG_oath_v0 (bars never move) |
| **B2** | same run, D2: **zero** false UNGROUNDED on the clean 3-doc corpus | inherited, PREREG_oath_v0 |
| **B3** | cycle-18 battery replicate (committed script, same method+seed 1, 13 certs): **FALSE-VERIFY ≤ 13 of 269** (strict halving of 26) | new, frozen here |
| **B4** | all 13 cycle-18 docs re-certified in memory under v0.4: **UNGROUNDED = 0 on every doc** (no manufactured false alarms on shipped certs) | new, frozen here |
| **B5** | `python -m pytest tests -q` green from repo root | repo standing rule |

Any missed bar ⇒ **CLOSED_NEGATIVE**: revert `styxx/certify.py`, publish the negative with the
receipts, done. Near-bar = CLOSED_NEGATIVE. No amendment after results exist.

**Reported, not gated:** the VERIFIED→ABSTAIN demotion count on the 3-doc validation corpus and on
the 13 cycle-18 docs (the honesty cost of binding — disclosed, not hidden); the battery
catch/abstain-degrade movement (catch is the trigger-recall extension's job, a SEPARATE owed item,
explicitly out of scope here).

## Artifacts

- `papers/autopilot/mutant_battery.py` — the battery as a committed, re-runnable script (cycle 19
  ran it in memory; that reproducibility gap closes here). Reads each certificate's receipt
  filenames, resolves them next to the doc (SHA-verified), one mutant per VERIFIED token, seed 1.
- `papers/autopilot/cycle22_v04_battery_result.json` — B3/B4 receipt.
- `papers/closed-model-frontier/oath_v0_validation_result.json` + the 3 regenerated corpus
  certificates — B1/B2 receipt (the validator regenerates them by design).
- A short RESULT note, itself certified (`python -m styxx.certify`) before commit.

Out of scope (named so they cannot creep in): trigger-vocabulary recall extension (ρ/RSA register);
re-binding/re-shipping the 13 cycle-18 certificate files (they stay pinned to their recorded
verifier SHA; regeneration is a later, separate decision); any change to `validate_oath_v0.py`.

---

*Frozen on commit. The bar structure outranks the upgrade.*
