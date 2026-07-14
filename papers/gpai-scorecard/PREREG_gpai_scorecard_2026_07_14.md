# PREREG — G5: the GPAI scorecard, population-framed. Receipt-binding rate across the EU Code-of-Practice signatories' flagship model documentation

**Fathom Lab · papers/gpai-scorecard · 2026-07-14. FROZEN ON COMMIT, before any extraction or
scoring pass runs on any document. Fires the backlog's G5 (Tier D, code-unblocked by OATH v0.5's
zero-false-accusation result, `5cc2e07`). Extends oath-economy rung 1 (`ae51f0f`) from three cards
to the signatory population, on the SAME frozen binding ladder. Publishes regardless of verdict —
a healthy genre is exactly as publishable as a deficient one.**

## Question

Model documentation publishes numeric evaluation claims. What fraction are BOUND — linked, on the
document or one direct hop, to enough to re-run them? Rung 1 measured three cards (0 of 292 claims
bound). This rung measures the population that has publicly committed to transparency: the EU GPAI
Code-of-Practice signatories, three weeks before the AI Act's enforcement powers begin
(2026-08-02). The deliverable is a PRE-ENFORCEMENT BASELINE of the genre, with a pre-committed
re-measurement window (2027-Q1) so the longitudinal question — does enforcement move binding? —
becomes measurable.

## Framing rule (frozen, inherited from rung 1 and the rung-3 hazard list — binding on every artifact)

The finding measures the BINDING GAP — the distance between public numeric claims and re-runnable
receipts — as a property of the documentation GENRE, never of any provider's honesty and never as
an AI Act compliance assessment. Mandatory sentences in the deliverable: "receipts may exist
unlinked — this instrument measures binding, not existence" (the rung-1 Meta correction);
"no provider is alleged non-compliant; Art. 101(1)(b) concerns responses to Art. 91 requests, not
public model cards." Zero occurrences of "violation", "non-compliance", or per-provider accusation
language in the headline or abstract; per-provider detail lives in the appendix; corrections
invited (REPLICATIONS.md route). External publication of the deliverable is operator-gated and not
part of this prereg.

## Population (frozen)

- **Authority:** the European Commission's official signatory list at
  https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai (page-update 2026-04-23;
  24 signatories = 23 full + xAI Safety-and-Security-chapter only). Conflicting third-party counts
  are rejected in favor of the Commission page.
- **Strata (every signatory lands in exactly one; assignment disclosed in the manifest):**
  - `SCOREABLE` — a locatable public flagship model/system card with at least 10 numeric eval claims.
  - `LOW-QUANT` — doc exists, fewer than 10 numeric eval claims; reported, excluded from the rate
    (the low claim count is itself data).
  - `NO-PUBLIC-FLAGSHIP-DOC` — no locatable flagship model documentation; reported as a stratum,
    never framed as a failure (the provider may have no GPAI model on market).
  - `PARTIAL-SIGNATORY` — xAI: scored, flagged; the Transparency chapter does not bind them.
    Headline reported with and without xAI (sensitivity).
- **Secondary contrast arm (never headline):** Meta (non-signatory), via the rung-1 receipts
  (`papers/oath-economy/`, 201 claims, 0 BOUND) — the signatory/non-signatory symmetry is what
  keeps the framing genre-level.
- **Doc rule:** one doc per provider — the provider-published model/system card (or technical
  report where that is the provider's primary eval document) for its most capable GPAI model as of
  the freeze date. Format preference when the same content exists in multiple forms: HF raw
  README.md, then provider PDF, then provider HTML. The exact URL per provider is frozen in
  `CARD_MANIFEST.json` (URL, fetch timestamp, sha256, bytes, format, converter+version) committed
  with the fetch; **no swaps after freeze** — a provider shipping a newer card mid-run is noted,
  not substituted. Document texts are NOT committed (license/copyright); sha256 pins re-verification.
- **Contamination disclosure:** a scouting pass on 2026-07-14 (this session) fetched several
  candidate URLs to locate the documents. The freeze is "prereg committed before any EXTRACTION or
  SCORING pass" — not before any human/agent read — exactly as rung 1 disclosed.

## Instrument (frozen)

- **Claim = numeric evaluation claim**: a (benchmark/eval, metric-or-variant, numeric score) tuple
  published in the doc (tables or prose). Config constants, model sizes, dates, and non-eval
  numerics are out of scope (v0, same scope as rung 1).
- **Extraction:** two INDEPENDENT schema-forced LLM-agent passes per doc, different prompt
  framings, blind to each other; each claim carries an evidence quote + location. Per-doc claim-set
  agreement (Jaccard on normalized tuples) at or above 0.90 accepts the union; below 0.90 sends
  the doc to a third adjudication pass; disagreements resolve toward dropping the claim (never
  toward inventing one). `styxx.extract_claims` is NOT the census instrument (rung 1: 0/201 on
  this genre by design — closed template set).
- **Binding ladder (rung 1, verbatim):** a claim is **BOUND** iff the doc (or a page it DIRECTLY
  links) provides enough to re-run it: named harness + version + config (shots/prompt/metric
  variant) + either code or raw outputs. Partial grades: **CONFIG-NO-CODE** (config stated, no
  harness/code/outputs linked), **NAMED-ONLY** (benchmark named, no config), **UNBOUND**.
- **Certifier role (frozen):** `styxx.certify` (v0.5, `5cc2e07`) is the VERIFICATION layer — run
  doc-vs-receipt wherever a claim links a machine-readable receipt — and the dogfood layer (the
  deliverable itself must certify OATH-HELD). It is NOT the census instrument: its measured
  out-of-register abstain-degrade is 0.677 (cycle-18 battery), so trigger-derived population
  numbers would be instrument artifacts. Any certifier-derived number ships only inside the
  method-validation section, next to its error rates.

## Headline metric (frozen)

Per SCOREABLE doc: **BOUND rate** = BOUND claims / numeric eval claims.
**Headline = the MEDIAN per-doc BOUND rate across SCOREABLE providers** (macro: one provider, one
vote), with IQR. Secondary (reported, never headline): pooled claim-weighted BOUND rate; the full
per-provider ladder distribution (appendix table); a claims-published vs claims-bound two-axis
plot (the denominator-gaming disclosure); sensitivity with/without xAI.

## Guards and VOIDs (frozen; any VOID files an instrument finding first, rung-1 discipline)

| id | condition | frozen verdict string |
|---|---|---|
| V1 | the extraction instrument reads fewer than 10 claims on more than half of the eval-bearing docs | `VOID_G5__instrument_mismatch` |
| V2 | more than a quarter of SCOREABLE docs still below the 0.90 Jaccard agreement after adjudication | `VOID_G5__extraction_unreliable` |
| V3 | a certifier-artifact false UNGROUNDED is found in hand-verification of any third-party verdict, and persists after one fix cycle + full rerun | `VOID_G5__certifier_artifact` |
| V4 | PDF transcription fidelity below 19/20 on the per-doc spot-check (20 random numbers, extracted vs visual) excludes that doc; more than 2 docs excluded this way | `VOID_G5__pdf_pipeline` |
| V5 | SCOREABLE count below 10 | `VOID_G5__population_too_small` |

## Frozen verdict (order: VOIDs, then the fork; bars on 4-decimal rounded values; both branches publishable)

- `GENRE_DEFICIT__median_below_0p50` iff the median per-doc BOUND rate is below the threshold of 0.50.
- `GENRE_HEALTHIER__median_at_or_above_0p50` otherwise — this branch would CORRECT rung 1's
  three-card extrapolation, and is written as exactly as reachable.

## Publishable iff ALL of (frozen)

1. No VOID fired.
2. The deliverable's method-validation section discloses the verifier's measured error rates
   verbatim: D1 tamper-catch 16/20 on the validated corpus; D2 0 false flags; v0.5 battery caught
   117/269 with false-verify 26/269; old-register tamper-catch 0.216; abstain-degrade 0.677; plus
   the DILIGENCE caveat quote on what OATH-HELD means on trigger-poor documents.
3. Every UNGROUNDED published against a third party carries a manual sanity-pass annotation (two
   independent hand-checks; disagreement resolves to ABSTAIN, never UNGROUNDED).
4. The deliverable itself certifies OATH-HELD with certifier artifacts 0 (`--out` always passed).
5. Framing lint passes (mandatory sentences present; forbidden vocabulary absent from
   headline/abstract).
6. The `contradicted=` CLI label is annotated as mapping to UNGROUNDED (missing-or-conflicting),
   never a proven conflict.

## Inference bounds (pre-committed)

One doc per provider (flagship only — not the provider's documentation corpus); numeric eval
claims only; the ladder grades linkage, not truth (a BOUND claim can still be wrong — fidelity is
rung 3's construct, out of scope here); LLM-agent extraction is nondeterministic (mitigated, not
eliminated, by the two-pass gate); English-language docs; the as-of date is a snapshot of a moving
genre; per-doc N varies by an order of magnitude (hence median + per-doc N always shown). The
threshold of 0.50 for the fork is a design choice frozen here, movable before this commit, never
after.

## What this prereg does NOT do

- It does not modify `styxx/certify.py` or any shipped verdict string.
- It does not assess AI Act compliance, does not rank providers in the headline, does not
  publish externally (operator-gated), and does not touch the in-flight B7/B2-coupling GPU work
  (this experiment is CPU-only).
- It does not commit third-party document texts.

## Artifacts

`papers/gpai-scorecard/`: this prereg; `card_fetch.py` (fetch + manifest freeze; PDF/HTML→text
with converter versions recorded); `CARD_MANIFEST.json` (the frozen population); extraction
receipts per doc (two passes + adjudication where fired); `scorecard_result.json` (aggregation);
`RESULT_gpai_scorecard_*.md` certified OATH-HELD before commit. Smoke/dev outputs quarantined in
`*_SMOKE_INVALID*`.

---
*Rung 1 bound one vendor's claims and found the receipts existed, unlinked. This rung asks whether
that is the genre — measured on the population that signed up to be measured, three weeks before
measurement acquires teeth. The healthy-genre branch is written as reachable because an instrument
that can only find deficits is not an instrument.*
