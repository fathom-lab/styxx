# PREREG — OATH v0.8: status-level claim→field binding for FLOAT claims (the standing v0.4 debt)

Fathom Lab · 2026-08-23 · frozen BEFORE any change to `styxx/certify.py`. Bars below never move; a
missed bar ⇒ the clause does not ship and the negative is published. No optional stopping.

Provenance: this is the debt `styxx/certify.py` names in its own source, at the v0.3 count-binding
site — *"Floats keep value-only matching (v0.4 owes them full claim→field binding)"* — and that
`PREREG_oath_v07_precision_obligation_2026_08_22.md` explicitly deferred, listing it first in its
out-of-scope block as *"the standing v0.4 debt, and the instrument that would actually attack the
false-verify channel."* v0.7 shipped and closed 176 of 2696 silent claims; it does not touch the
false-attestation channel at all. This prereg attacks that channel. It resurrects no buried claim:
`PREREG_oath_v04_float_binding_2026_07_03.md` closed CLOSED_NEGATIVE on a *different* instrument
(value-binding by float identity); the lead it handed forward was claim→CELL binding, which is
what is tested here.

## The defect this exists to attack (measured, pre-fix, at the shipped 7.45.0 verifier)

`oath_v07_silentpass_census.py` mutates one significant digit of every claim the verifier certifies
VERIFIED and re-certifies. Over 136 documents with fully-resolvable receipts, 3951 claims:

| mutant lands | n | reading |
|---|---|---|
| UNGROUNDED | 1255 | the instrument works |
| ABSTAIN | 2005 | silent pass — the document keeps OATH-HELD |
| **VERIFIED** | **604** | **affirmative false attestation — worse than silence** |
| NOT_EXTRACTED | 87 | extraction gap |

The 604 are the target: a mutated number that coincidentally matches an UNRELATED receipt leaf and
is sworn to. No obligation rule can reach them — obligation decides whether a claim *must* match,
and these already do match. Only binding the claim to the *right field* can.

**The reachable ceiling is 330, not 604.** 274 of the 604 are bare integers, which already pass
through the v0.3 count-binding stem filter and false-verify anyway. Float field binding cannot
reach them. That number is stated here, before the fix, so no result can be read as more than it is.

## The change (exact, frozen)

A per-token clause at the status ladder, **DEMOTE-ONLY**.

```python
V08_FLOAT_FIELD_BINDING = True    # primary, severable
V08_FIELD_BIND_MAX_DECIMALS = 3   # applies to floats at 1..3 fractional digits
V08_FIELD_BIND_PREV_LINES = 1     # binding window: the claim's line plus the previous line
```

For a float claim with `1 <= decimals <= V08_FIELD_BIND_MAX_DECIMALS` and a non-empty `hits`:

1. build the stem set of the binding context **widened by `V08_FIELD_BIND_PREV_LINES` preceding
   lines** (the v0.3/v0.6.2 stem function, unchanged);
2. if any matching leaf's PATH shares a stem with that context → nothing happens (bound, VERIFIED);
3. else apply the **NAMEABLE** test: if NO receipt leaf anywhere in the cited set has a path sharing
   a stem with the context, binding was impossible in principle → nothing happens (VERIFIED);
4. else record `field_unbound_ref` and the ladder yields **ABSTAIN**, reason
   `unbound-field:<receipt>:<path>`.

Ladder placement, frozen: after `derived_ref`, **before** `hits`. `is_spec` / `is_hist` /
`is_notation` still win, unchanged.

**The clause can never produce UNGROUNDED and can never remove one.** It intercepts only claims
that would otherwise be VERIFIED, and sends them to ABSTAIN. Therefore no certificate can flip
HELD→FAILED, and the false-accusation channel is closed *by construction*, not by a bar. This is
stated as invariant **I1** below and asserted in the suite — **it is deliberately NOT a gate,
because a leg that cannot fail must not gate.**

**Everything else UNTOUCHED.** `_TRIGGERS`, `_TRIGGERS_CORR`, `_NUM`, `_SHAISH`, `_DATEISH`,
`_VERSIONISH`, `_MD_STRUCTURE`, `_BULK_PATHS`, `extract_numbers`, `receipt_values`, `_match`,
`_ulp_neighbour`, every v0.5 class, the v0.3 count-binding filter, the v0.6.2 attribution
preference and all v0.7 clauses are byte-identical. The `decimals <= 3` ceiling means this clause
and v0.7's `decimals >= 7` obligation cannot interact.

### Why this predicate, and why an ACCUSING design is already dead

Five design families were swept pre-fix, over both populations at once — the 330 float
false-attestations and the 3037 clean float claims that currently VERIFY. Artifacts:
`oath_v08_fieldbind_precfix_census.py`, `_sweep.py`, `_ctxsweep.py`, `_nameable.py`. Cost is honest
verifications demoted; kill is false attestations removed.

| family | best point | kill | cost | would ACCUSE | cost/kill |
|---|---|---|---|---|---|
| naked stem filter (v0.6.2 test promoted) | line window, all decimals | 165 | 575 | 203 | 3.48 |
| KEEP-widenings (spec containers, generic leaves, receipt name) | W5, dec≤2 | 98 | 116 | 40 | 1.18 |
| context window | prev1, dec≤3 | 116 | 123 | 42 | 1.06 |
| **window + NAMEABLE (shipped candidate)** | **prev1, dec≤3** | **113** | **115** | **40** | **1.02** |
| all-hits-array-indexed | — | 99 | 764 | — | 7.72 |

Two things are settled by that table and are not re-litigated later:

**An accusing design is dead.** Every point of every family would produce dozens of new UNGROUNDED
tokens on honest documents (40 at the best point). The v0.7 ship rule — raising catches while
adding a single false accusation does not ship — kills it outright. Hence demote-only.

**No widening beats parity.** The instrument buys roughly one honest demotion per false attestation
removed, across five independent families. The NAMEABLE gate is included because it improves the
ratio at every single point measured and is the only widening that is principled rather than
lexical: it refuses to demote where the receipt set contains nothing the sentence could bind to.

### The disclosed prior — this clause may well MISS its own bar

An informal 20-row read of the demotion dossier (`oath_v08_fieldbind_adjudicate.py` →
`oath_v08_fieldbind_dossier.json`, 115 rows at the shipped operating point) suggested that **roughly
two thirds to three quarters of clean demotions destroy a GENUINE binding**, not a coincidence.
Scientific prose refers to a measurement narratively — "whole-stack r=16: 0.616–0.626" — while the
receipt field is `points[2].naive_relock_auroc`. Path-stem overlap has no purchase on that, and the
honest population is not lexically separable from the tampered one.

This prior is recorded BEFORE the frozen adjudication runs, so that a miss reads as a
pre-registered negative rather than a surprise, and so that a *pass* cannot be attributed to a bar
set generously. G4 is the gate that decides it, and its bar is set at what a shippable instrument
must clear, not at what this one is expected to score. Independently useful and also recorded
pre-fix: **0 of the 115 demoted claims have their own value present in any leaf the sentence names.**

## Frozen adjudication definition (removes post-hoc judgment)

Each sampled demotion is adjudicated on the FIRST matching leaf — the one the certificate would
have cited — by reading the sentence against the leaf path, into exactly one class:

- **GENUINE-BINDING-DESTROYED** — the leaf IS the claim's home: it records the quantity the
  sentence is about, at the claim's value. The demotion is a true coverage loss. *Counts against
  the clause.*
- **COINCIDENCE-CORRECTED** — the leaf records a different quantity; the claim's own quantity is
  absent from the cited receipts or present at a different value. The claim was never earned.
- **SPEC-CORRECTED** — the claim is a bar / floor / threshold / experiment parameter whose receipt
  is the prereg, not a measurement. It should already ABSTAIN under the v0.1 SPEC-CONSTANT rule and
  escapes only because the operator sits in a JSON `"op"` field (the 157/157 unrescued class
  disclosed in the v0.7 prereg). The demotion is correct.

**Ties and genuine uncertainty resolve to GENUINE-BINDING-DESTROYED** — the conservative direction,
counting against the clause.

## Battery + gates (harness `run_oath_v08_battery.py`, seed 1)

Sampling frame, stated with every condition: documents under `papers/**` carrying a
`*.certificate.json`, excluding `anc/` staging copies, whose recorded receipts ALL resolve next to
the document with matching SHA. Every arm runs flag OFF and flag ON on the identical sample with
the identical mutation seed. Mutants live in temp files; the corpus pass is in-memory; the only
files written are the battery's own result JSON and the census JSON. The harness copies the
sign-aware `substitute()` from `run_oath_v07_battery.py` — `line.replace(tok, mut)` silently no-ops
on U+2212-signed tokens, because extraction normalizes the typographic minus to ASCII while the
document holds U+2212, and a harness miss then scores as a verifier miss.
`run_oath_v061_battery.py` still carries that bug and **is not modified** — the instrument never
moves.

### I1 — structural invariant (ASSERTED IN THE SUITE, NOT GATED)

With the clause ON, across all resolvable documents: zero new UNGROUNDED tokens, zero lost
UNGROUNDED tokens, zero certificates flipped HELD→FAILED. This holds by ladder construction. It is
asserted as a test and reported, and it is **not** counted as a gate, because it cannot fail.

### Gates

- **G1 (POSITIVE CONTROL — VOID condition, not a bar):** the decisive census runs in both arms. **If
  the ON arm's false-VERIFIED count does not fall strictly below the OFF arm's, the run is VOID**,
  whatever else it reads: a battery reporting the same number in both arms is not measuring the
  clause.
- **G2 (BENEFIT — decisive, gated):** on a census at a **fresh mutation seed (seed 2)**, the
  false-VERIFIED count with the clause ON is **at least 60 lower** than with it OFF. The fresh seed
  is required because the operating point was chosen against seed 1; the seed-1 census is also
  reported, for comparability with the 604 baseline, but **does not gate**.
- **G3 (COST RATIO — regression guard, gated):** at the same fresh seed, clean-corpus
  VERIFIED→ABSTAIN demotions ≤ **1.5 ×** the number of false attestations removed. Guards a drifting
  or gameable implementation: a clause that abstains generously would otherwise score G2 for free.
- **G4 (ADJUDICATION — decisive, gated):** 40 clean demotions sampled uniformly at **seed 11** from
  the full demotion set, hand-adjudicated under the frozen definition above. **GENUINE-BINDING-
  DESTROYED ≤ 12 of 40.** The clause withdraws attestations; at least 7 in 10 withdrawals must be
  right, or it destroys more provable coverage than it repairs. This is the bar the disclosed prior
  expects to miss.
- **G5 (SEVERABILITY — gated):** with `V08_FLOAT_FIELD_BINDING` OFF, the ledger is status-identical
  to `oath_v08_baseline_ledger.json`, regenerated at the current verifier before any edit. Any
  difference → the clause is not severable and does not ship.
- **G6 (SUITE — gated):** `python -m pytest tests -q` green; `python -m py_compile` on every touched
  `.py`. No verifier ships with a red suite — CI masked the entire test step for weeks in August and
  that will not be re-bought.

## Outcome table (pre-committed)

- **All gated bars pass → v0.8 SHIPS.** The RESULT publishes the corpus VERIFIED/ABSTAIN delta, the
  full adjudication table, the post-change silent-pass residual (both seeds), and the
  `unbound-field` roster. CHANGELOG carries the residual. The v0.4 debt closes POSITIVE.
- **G2 misses → `V08_INSUFFICIENT_CATCH`.**
- **G3 misses → `V08_COST_EXCEEDED`.**
- **G4 misses → `V08_COVERAGE_DESTRUCTIVE`.** This is the expected outcome under the disclosed
  prior. On this token the clause **ships DISABLED** — `V08_FLOAT_FIELD_BINDING = False`, left in
  tree behind its flag with the measured cost recorded in the source comment, exactly as
  `V05_APPROX_NOTATION` was retained after the cycle-38 severability drop. G5 proves the disabled
  clause is inert. The v0.4 debt closes **CLOSED_NEGATIVE** with a measured structural reason, and
  is no longer carried as owed work.
- **G1 VOID → `V08_BATTERY_VOID`;** no verdict recorded, the harness is the defect.
- **G5 fails → `V08_NOT_SEVERABLE`;** revert `styxx/certify.py` entirely.
- **G6 fails → `V08_SUITE_RED`;** fix the suite or revert.
- No second attempt inside this cycle. A miss hands the lead to the `is_spec` JSON-idiom repair
  (157/157 unrescued, and the SPEC-CORRECTED column of G4 measures how much of this surface it
  owns), and after it to the 274 integer false-attestations that shipped count-binding does not stop.

## Artifacts

- Pre-fix, committed beside this prereg, all generated at the shipped verifier before any edit:
  `oath_v08_fieldbind_precfix_census.py` / `.json`, `_sweep.py` / `.json`, `_ctxsweep.py` / `.json`,
  `_nameable.py` / `.json`, `_adjudicate.py` / `oath_v08_fieldbind_dossier.json`.
- `make_oath_v08_baseline.py` → `oath_v08_baseline_ledger.json` (G5 reference, current verifier).
- `run_oath_v08_battery.py` → `oath_v08_battery_result.json`.
- A short RESULT note, itself certified (`python -m styxx.certify`, OATH-HELD) before commit.

Out of scope (named so they cannot creep in): the 274 integer false-attestations; `is_spec`
JSON-idiom recall; fence and inline-code awareness in `extract_numbers`; any semantic or embedding
similarity between prose and field names (this repo binds lexically or not at all); any change to
`validate_oath_v0.py`, `run_oath_v061_battery.py`, `run_oath_v07_battery.py` or `mutant_battery.py`
— bars and instruments never move.

---

*Frozen on commit. The bar structure outranks the upgrade.*
