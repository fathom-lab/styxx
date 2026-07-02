# FINDING — binding census v1 (partial): the residue replicates

**Fathom Lab · 2026-07-02 · prereg: `PREREG_census_v1_2026_07_02.md` (frozen before any new receipts opened,
`4186f2d`; ratified). Status: 2 arms complete, remaining arms BLOCKED on gated receipt access. Publishes
regardless — and partial is what the data allows tonight.**

Framing rule (frozen): gaps are industry properties; deltas with receipts; correction invited; selection
findings reported in both directions.

## Per-arm fidelity (frozen grades; full tables in `census_v1_results.json` + `rebind_result.json`)

| arm | in-scope | MATCH | ABSENT | orphan (VALUE-MISMATCH) | verdict |
|---|---|---|---|---|---|
| Llama-3.2-**1B**-Instruct (rung 3) | 22 | **18 (82%)** — all Δ=0.0 | 3 | 1 — BFCL 25.7 matches nothing | PARTIAL |
| Llama-3.2-**3B**-Instruct | 22 | **17 (77%)** — all Δ=0.0 | 3 | 2 — BFCL 67.0, Nexus 34.3 match nothing (manually verified per the hazard rule) | PARTIAL |
| Llama-3.1-8B (cross-cited), 3.1-70B/405B, 3.3-70B | — | — | — | — | **BLOCKED**: receipts datasets are gated and this account is not in the authorized list — third-party verification stops at a license wall (itself a binding-infrastructure finding) |

## The census's first real discovery: the residue REPLICATES

The failures are not scattered — they are the **same rows on both arms**:

- **BFCL V2 is an orphan twice** (25.7 and 67.0, matching no receipt aggregate on either arm) — whatever
  produced the card's BFCL numbers is not what produced the receipts' BFCL numbers;
- **MATH config-conflicts twice** (card says 0-shot CoT; both receipt datasets carry only a 4-shot record);
- **IFEval is unrecomputable twice** (the receipts carry 2 of the 4 composite components);
- **TLDR9+ is absent twice** (no receipts anywhere in the family's datasets).

Replication across independently-generated receipt datasets means the residue is **structural — a property
of the card-production pipeline, not eval noise**. Exactly the class of inconsistency that stays invisible
until someone re-binds.

## Selection (the never-before-measured gap)

Both arms: the card publishes **22 of 1,246 vendor-measured metric rows (1.8%)**. Flattering-selection check
(receipt pairs offering ≥2 aggregates >0.15 apart): **1 flattering, 0 unflattering, 4 neutral** — the single
flattering instance is Open-rewrite, where the card's chosen aggregate (`micro_avg`/`average` 41.6) sits above
the unpublished `macro_avg` (40.0). n is far too small to call a pattern; the MGSM spot-check (macro-avg 24.5
published over English 42.0) cuts the honest way. Both directions reported, as frozen.

## What unblocks the rest of the census

The 3.1/3.3 receipt datasets require per-repo license acceptance on HF (operator action, ~1 min each:
`Llama-3.1-8B-Instruct-evals`, `-70B-`, `-405B-`, `Llama-3.3-70B-Instruct-evals`). Once granted, the committed
`census_match.py` runs the remaining arms unchanged — including the cross-citation arm (does the 3.2 card's
quote of Llama-3.1-8B match *that* model's receipts?), and the generational-fidelity question the prereg
registered (M7 seed).

## Reproduce
```
python papers/oath-economy/census_match.py     # arms enumerated at execution; gated arms recorded as blocked
```

*Two arms in: the hyperlink-gap holds at ~80% fidelity, the residue is structural, and the wall we hit —
gated receipts — is itself the strongest argument that binding must be an artifact, not a favor.*
