# DESIGN — the epistemics_summary block (styxx-oath/epistemics-summary/v1)

Fathom Lab · 2026-08-30 · **The frozen design.** Produced by a two-proposal, two-attack,
one-synthesis workflow before implementation; implemented verbatim in `styxx/certify.py` the same
day. The attack pass also caught a real drift in the shipped annotation — range-sanity clobbered
`obligation_source` unconditionally while the contract said first-writer — fixed before this
landed.

---

All five inputs read in full (`RESULT_unobligated_oath_2026_08_28.md`, `INVARIANT_epistemics_annotation_2026_08_28.md`, `RECON_v13_not_frozen_the_ladder_2026_08_28.md`, `styxx/certify.py` lines 640–968, and the committed certificate for the RESULT itself). Cross-checked every attack finding against the code before ruling: the range-sanity overwrite is real (lines 851–854 clobber `_ob_src` unconditionally), the derived arm's operands really are path-blind first-hit `_match(..., False)` calls (lines 863–864), `path_checked` really is a pure alias for `decimals == 0` set only on the value-match branch (lines 949–950), and the slash-pair fallback (lines 802–806) really does land tokens in `path_checked=true` without a stem binding. All confirmed. The schema below is designed around those facts, not around either proposal's flattering of them.

# THE SCHEMA TO FREEZE — `epistemics_summary`, styxx-oath/epistemics-summary/v1

## What was stolen, what was killed

**From proposal 2:** the distinct top-level name `epistemics_summary` (kills the name/shape collision with per-entry `epistemics` — attack A.4), the in-band `schema` version string (kills the absence-vs-stripped ambiguity — attack A.4), the full `by_branch` histogram (kills proposal 1's pooled `silencer` count and the silencer/silent two-character transposition trap), the 4-cell value-match matrix (kills proposal 1's BLOCKER — the hidden obligated-path-unchecked cell, 1909 tokens corpus-wide, the actual mutation-exposure surface), `verified.total` restated for self-checking, and the never-mutate-v1 versioning rule.

**From proposal 1:** counts only, no rates, no floats, no paths, no timestamps; integers with every key always present including zeros; consumers own their denominators; the loud assertion at issuance with no `.get()` defaults and no "unknown" bucket; insertion between `counts` and `verdict`.

**Killed, per confirmed attacks:** proposal 1's single `unobligated_path_unchecked` cell (A.1 BLOCKER); proposal 2's "path binding holds by construction" for derived (B.1 BLOCKER — replaced by a derived obligated/unobligated split and honest prose); the `*_path_checked` cell names (B.3 — replaced by mechanism names `integer_filter_ran`/`integer_filter_na`); the first-clause gloss on `obligation_source` (A.2b, B.4 — replaced by last-writer language matching the code, with the INVARIANT doc's drift noted as pre-existing and out of scope); the histogram-with-no-total (B.4 — `obligated_total` added with a sum invariant); the vacuous-pass jq gate (B.2 — replaced by explicit four-way routing); "correctly excluded" for derived (A.3 — replaced by "path_checked is undefined for derived").

## The JSON block, exactly as it would appear

This is the block `RESULT_unobligated_oath_2026_08_28.certificate.json` would carry if reissued — real numbers from its committed ledger (10 VERIFIED all value-match, 7 ABSTAIN all silent, 3 obligated tokens all via `vocabulary`), inserted between `"counts"` and `"verdict"`:

```json
{
  "oath": "styxx OATH v0 (numeric-claim certificate)",
  "counts": { "VERIFIED": 10, "ABSTAIN": 7, "UNGROUNDED": 0 },
  "epistemics_summary": {
    "schema": "styxx-oath/epistemics-summary/v1",
    "note": "attestation composition folded from this certificate's own ledger; counts which door each token came through; says nothing about whether any token is a true claim or a good one",
    "by_branch": {
      "row-ordinal-label": 0,
      "formula-constant": 0,
      "spec-or-historical": 0,
      "notation": 0,
      "derived": 0,
      "unbound-field": 0,
      "value-match": 10,
      "ulp-neighbour": 0,
      "obligated-accusation": 0,
      "silent": 7
    },
    "verified": {
      "total": 10,
      "derived": { "obligated": 0, "unobligated": 0 },
      "value_match": {
        "obligated_integer_filter_ran": 2,
        "obligated_integer_filter_na": 1,
        "unobligated_integer_filter_ran": 4,
        "unobligated_integer_filter_na": 3
      }
    },
    "obligated_total": 3,
    "obligation_sources": {
      "vocabulary": 3,
      "n-glued": 0,
      "range-correlation": 0,
      "precision": 0,
      "range-sanity": 0
    }
  },
  "verdict": "OATH-HELD"
}
```

(All other existing keys — `prereg`, `document`, `document_sha256`, `receipts_sha256`, `verifier_sha256`, `ungrounded`, `abstained`, `ledger` — unchanged in name, position, and meaning.)

## Field semantics, with the one sentence each field carries

Every value is a non-negative integer produced by a pure fold over `ledger[*].epistemics`; every key is always present, zeros included, in exactly the order shown; the block is a deterministic function of the certificate's own ledger and of nothing else.

**`schema`** — *"Exact version string of this block's shape; gates key on this string, never on key presence, and any shape change — new ladder branch, new obligation clause, new field — is a new string, never a mutation of v1."*

**`note`** — *"Constant scope disclaimer, identical in every v1 certificate: this block counts verifier behavior, not claim quality."*

**`by_branch`** — *"Count of ledger entries per ladder arm, all ten labels this verifier version can emit, in execution order (two pre-ladder demotions, then the ladder)."* Invariants: `sum(by_branch) == counts.VERIFIED + counts.ABSTAIN + counts.UNGROUNDED`; `by_branch["obligated-accusation"] == counts.UNGROUNDED`; the ABSTAIN split (structural silencers vs the final-else `silent`, where the verifier had nothing to say) is read directly off the branch names — no pooled `silencer` field exists to misread.

**`verified.total`** — *"Restatement of counts.VERIFIED inside the block, so the partition below is self-checking without reaching into `counts`."* Invariant: `total == counts.VERIFIED == derived.obligated + derived.unobligated + sum(value_match)`.

**`verified.derived`** — *"VERIFIED via the v0.5 derived-percent arm, split by whether the token was obligated at ladder time; excluded from the value_match matrix because `path_checked` is undefined for this branch — its operand matches are path-blind first-hit value matches constrained by the arithmetic identity, a different mechanism, not a stronger one."* (Derivation: `branch == "derived"`, split on `epistemics.obligated`. This makes the RESULT's 4-way census row "unobligated, path filter ran or derived" reproducible from committed summaries alone.)

**`verified.value_match`** — *"Exact 4-cell partition of value-match VERIFIED tokens by (obligated at ladder time) × (the v0.3 integer count-binding filter executed)."* Per-cell derivation: `branch == "value-match"`, `epistemics.obligated`, and `epistemics.path_checked` — which at this verifier version is exactly `decimals == 0`. The cell names describe the mechanism, not a property: `integer_filter_ran` means the filter executed, in either strict-stem or slash-pair count-like-fallback mode — the ledger does not distinguish the two, so neither does this block; `integer_filter_na` means the token is a float and receives no status-level binding at this verifier version (v0.8 float field binding is `CLOSED_NEGATIVE`; the v0.6.2 stem preference reorders float hits but binds nothing). `unobligated_integer_filter_na` is the instrument's weakest attestation: a volunteered value coincidence nothing required the verifier to examine and no binding filter touched. `obligated_integer_filter_na` is the cell proposal 1 hid: obligated-looking oaths with the same zero binding — corpus-wide the larger absolute population (1909 vs 2023 is comparable, and 76.6% of obligated verifications), and the `gpu_memory_fraction`-class exposure surface. A gate computing "VERIFIED tokens with no binding filter" sums the two `_na` cells; it never parses the ledger.

**`obligated_total`** — *"Count of ledger entries, of any status, whose `obligated` flag was true at ladder time — the population the histogram below partitions."* Invariant: `sum(obligation_sources) == obligated_total`. This total appears nowhere else in the certificate; without it, the committed example invites exactly the wrong cross-check (3 coincidentally equals the obligated-verified count).

**`obligation_sources`** — *"Histogram of the recorded `obligation_source` over all obligated entries of any status — VERIFIED, ABSTAIN, and UNGROUNDED alike — reporting the clause in force at ladder time: the last-writing clause, because range-sanity supersedes any earlier source (certify.py lines 851–854), which this summary reports as recorded and does not re-derive."* All five clause names this verifier version can emit, in clause evaluation order. The UNGROUNDED-only view proposal 1 wanted is recoverable as `by_branch["obligated-accusation"]` cross-checked against `counts.UNGROUNDED`; a per-source UNGROUNDED histogram is deliberately omitted because the source field's last-writer semantics make "what obligated the accusation" a question the ledger cannot currently answer in first-clause terms. (The INVARIANT doc's "first clause that set it" language is documented drift from the code — a pre-existing ledger defect, flagged here, to be resolved in its own invariant-gated cycle; the summary must not describe semantics the ledger does not have.)

## What it must never be read as

1. **No claim-share, no quality, no error rate.** `unobligated_*` counts oaths that were volunteered, not oaths that landed on non-claims; the "roughly one in five" figure is 2026-08-27 blind-panel judgement over foreign text, not derivable from any ledger, and is constitutionally excluded. The certificate counts doors; it does not grade what walked through them.
2. **`obligated` does not mean correct; `unobligated` does not mean wrong.** The RESULT is explicit: most volunteered oaths are true claims whose lines carry no trigger vocabulary. No `weak`/`suspect`/`quality`/`confidence` vocabulary appears anywhere in the schema.
3. **`integer_filter_ran` does not mean context→path binding held.** Slash-pair integers can pass in count-like-fallback mode with no stem binding at all; the cell attests only that the filter executed. Recording strict-vs-fallback mode is a future ledger annotation behind its own invariant, and no summary field claims it before the ledger records it.
4. **No rates, ever.** A rate field asserts a denominator choice and a 0/0 convention; both are policy and live in the gate. The corpus numbers 0.5811/0.3399 are measurements over 192 documents and would be volatile and unfalsifiable inside a single certificate.
5. **No verdict input, no second verdict.** `verdict` remains `OATH-HELD iff counts.UNGROUNDED == 0`, computed exactly as before, never reading this block; the block contains no `grade`, no `epistemic_verdict`, nothing verdict-shaped.
6. **Absence of the key means pre-summary certificate, never zeros** — and the `schema` string, not key presence, is what a gate checks, so a stripped or forked block is mechanically distinguishable from a legacy one.
7. **`derived` is not a path-checked or stronger arm** — its separation records a different mechanism (arithmetic identity over path-blind operand matches), and `ulp-neighbour` entries are their own arm, not value matches.

## The consumption pattern (fixed: no vacuous pass, explicit routing)

```
jq -r '
  if .epistemics_summary == null                       then "legacy"
  elif .epistemics_summary.schema
       != "styxx-oath/epistemics-summary/v1"           then "unknown-schema"
  elif .verdict != "OATH-HELD"                         then "fail:verdict"
  elif (.epistemics_summary.verified as $v |
        ($v.total != .counts.VERIFIED)
        or ($v.total != ($v.derived|add) + ($v.value_match|add))
        or ((.epistemics_summary.by_branch|add)
            != (.counts|add))
        or ((.epistemics_summary.obligation_sources|add)
            != .epistemics_summary.obligated_total))   then "fail:partition"
  elif .epistemics_summary.verified.total == 0         then "empty-attestation"
  elif .epistemics_summary.verified.value_match.unobligated_integer_filter_na
       > 0                                             then "fail:composition"
  else "pass" end' cert.json
```

The consumer gates on the string. `legacy` routes to the legacy policy or is refused — explicitly, its own outcome. `empty-attestation` (OATH-HELD with zero verified tokens, which the ladder produces routinely) is a named outcome and never `pass` — the vacuous-green lesson applied at the gate, to the denominator and not just to schema absence. The composition threshold (`> 0` here; any `<= K` variant) is the gate's policy, not the certificate's claim. Unknown keys inside the block or a failed sum are `fail:partition` / `unknown-schema`, never treated as zero.

## The invariant statement for the change (to be frozen before certify.py is touched)

**The change may add exactly one top-level key, `epistemics_summary`, inserted between `counts` and `verdict` in newly issued certificates, and may move nothing.** Specifically:

1. **Zero tokens change.** Per-token `status` and `receipt_ref` vectors, `counts`, `verdict`, `ungrounded`, `abstained`, and `ledger` are identical under an A/B of the changed verifier against the pre-change verifier **at the same commit** (the 117-false-violation lesson: never against stored multi-version certificates), over all 192 committed documents, with the nonzero-comparisons denominator guard (the vacuous-pass lesson: "HOLDS over zero certificates" is a failure), loading via `from styxx.certify import certify_doc` (the name-shadow lesson). Violation is `INVALID__REGRESSION`: revert, nothing learned, the document stays as the record.
2. **Pure derivation.** The block is computed by one pure function `_epistemics_summary(ledger) -> dict`, a fold over `ledger[*].epistemics` and nothing else — no panel data, no corpus data, no judgement. An independent recompute script ships beside the change so any reader can verify the block from the certificate's own ledger.
3. **Loud failure, no holes.** Issuance asserts every ledger entry carries `epistemics` and every `value-match` entry carries `path_checked`; no `.get()` defaults, no "unknown" bucket — a hole is a defect at issuance, not a category. (The two historical bypass `ledger.append` sites, v0.11 row-ordinal and v0.12 formula-constant, already annotate; the assertion is the guard against the next bypass.)
4. **History untouched.** The 192 stored certificates are never regenerated; only newly issued certificates carry the block; `verifier_sha256` changing is the designed drift signal doing its job; consumers treat key absence as pre-summary, never as zeros.
5. **v1 is immutable.** Any future shape change — a new ladder branch, a new obligation clause, a mode split of `integer_filter_ran`, a first-setter fix to `obligation_source` — is `styxx-oath/epistemics-summary/v2`, never an in-place edit, so a pinned gate can never silently read a reshaped block.
6. **The corpus audit learns the sum invariants** (non-gating at first) for any certificate carrying the key: `sum(by_branch) == sum(counts)`; `by_branch["obligated-accusation"] == counts.UNGROUNDED`; `verified.total == counts.VERIFIED == sum(derived) + sum(value_match)`; `sum(obligation_sources) == obligated_total`.
7. **The owed contract amendment** (RESULT "What is owed" item 2) names `epistemics_summary.verified` as where the obligated/unobligated split now lives, cites the schema string, and states that the two `*_integer_filter_na` cells are the counts `OATH_CONTRACT.md` previously could not see. This block discharges owed item 1.

Files: `C:\Users\heyzo\clawd\styxx\styxx\certify.py` (implementation site: after the `counts` fold, line ~953), `C:\Users\heyzo\clawd\styxx\papers\closed-model-frontier\RESULT_unobligated_oath_2026_08_28.md`, `INVARIANT_epistemics_annotation_2026_08_28.md`, `RECON_v13_not_frozen_the_ladder_2026_08_28.md`, `RESULT_unobligated_oath_2026_08_28.certificate.json`.
