# DILIGENCE — mutant battery on the cycle-18 certificates (2026-07-03)

**What this is.** A due-diligence audit (operator-directed) of the certificates shipped in
autopilot cycle 18 (commit `7c2066d`). Not a scored claim run and **not pre-registered**; no
frozen bar. Receipt: `cycle18_mutant_battery_result.json` (seed 1, deterministic, re-runnable).

**Question.** All 13 docs' certificates are OATH-HELD. Do they have *teeth* — if a verified
number in one of those docs were corrupted, would its certificate notice?

**Method.** For every VERIFIED token in each of the 13 certificates, mutate one significant digit
(the `mutate_token` scheme of `validate_oath_v0.py`, seed 1), re-run `certify_doc` in memory on a
temp copy against the cert's exact receipt set, and classify the mutant's fate. Nothing on disk was
modified. 13 docs, 269 mutants total.

## Result: TAMPER-EVIDENCE-WEAK

| fate of the corrupted number | count | rate |
|---|---|---|
| caught as UNGROUNDED — verdict flips to OATH-FAILED | 58 | 0.216 |
| **FALSE-VERIFY — the oath swears to the corrupted number** | 26 | 0.097 |
| abstain-degrade — oath stops swearing; verdict silently stays HELD | 182 | 0.677 |
| dropped by extraction | 3 | — |

- Overall catch: 58 caught of 269 mutants; verdict flips 58 (a doc fails exactly when a mutant is
  caught). For reference, the D1 validation bar was 0.80 catch — an **analogue only**: that bar was
  frozen for the closed-model-frontier corpus and its receipt sets, not for these docs.
- **FALSE-VERIFY mechanism (26 mutants, rate 0.097):** the corrupted value coincidence-matches a
  *different* receipt leaf — e.g. a mutated per-model correlation re-verifies against another
  model's entry in the same `per_model` table. This is precisely the **disclosed v0 float
  limitation** ("v0.4 owes floats full claim→field binding", `styxx/certify.py`); these 13 docs
  carry dense per-model receipt tables, so the coincidence surface is large even with 1–3 receipts.
- **Abstain-degrade mechanism (182 mutants, rate 0.677):** these older docs state results in a
  register the UNGROUNDED trigger vocabulary does not cover (ρ/RSA/alignment/convergence/drift
  phrasing rather than AUROC/margin/FPR). An unbound corrupted number cannot fire UNGROUNDED — it
  falls back to ABSTAIN and the document verdict stays HELD.

## What stands and what does not

**Stands:** the 13 certificates as *ledgers*. Every VERIFIED number genuinely matches its receipt
at the recorded SHAs; the V/A counts are accurate; anyone can re-run them. Nothing in cycle 18 is
retracted.

**Does not stand:** reading OATH-HELD on these 13 docs as *tamper-evidence*. On this corpus the
certificate detects a single corrupted digit at 0.216, not the 0.80 the validated corpus supports.
HELD here means "nothing contradicted, everything sworn was checked" — it is **much weaker** than
HELD on a trigger-rich, field-bound document.

## Owed (spawned to the backlog)

1. **OATH v0.4 float claim→field binding** (standing priority #5) — now with evidence: 26 of 269
   single-digit corruptions falsely re-verify through neighboring leaves.
2. **Trigger-vocabulary recall extension** for the older register (ρ/RSA/alignment/convergence/
   drift) — a certifier-recall prereg, gated by re-running `validate_oath_v0.py` (bars never move).
3. Config-token triage of the 139 blocked docs from the cycle-18 sweep (steering α, arXiv IDs;
   count per CYCLE_LOG cycle 18, outside this battery's receipt) remains owed.
