# PREREG — B52: the read re-run on a battery whose extraction term is known

Fathom Lab · 2026-09-01 · frozen before any cell of this experiment runs. No number below is a
measurement; every number below is a threshold or an arithmetic derivation of one.

Every published read in this arc is scored over the 462-concept battery at `run_g0clear.py`
lines 31-67. That battery is a hand-authored string literal whose only filter is an
order-preserving dedup removing three words, and no pool of candidates was ever recorded. The
**extraction term** — the share of candidate concepts that survived into the scored set — is
therefore not computable for those reads, and `ADDENDUM_battery_and_holdout_limits_2026_09_01.md`
records that as UNCHECKABLE rather than as a defect in the reads themselves. B52 does not revisit
those reads. It runs the same machinery on a battery whose extraction term **is** computable, and
reports the term beside the read.

The successor battery already exists as a committed artifact: `concept_pool.json`, produced by
`build_concept_pool.py`. It records a pool of 547,373 candidate types drawn by a stated rule from
a pinned corpus, then eight named filters each carrying its rule, its survivor count, its removal
count and a sha256 of what it left. B52 scores the 462-word battery that receipt names, and no
other.

## What is fixed before any cell runs

**Battery.** The `battery.words` list of the committed `concept_pool.json`, identified by
`battery.sha256`. The runner reads the list from the receipt; it does not rebuild it. Rebuilding
is permitted only to *verify* the sha256, never to produce a different list.

**Pool rule and filter chain.** Frozen in `build_concept_pool.py` as committed. Its constants —
`FREQ_FLOOR = 500`, `MIN_LEN, MAX_LEN = 3, 14`, `PROPER_CAP_SHARE = 0.5`, `POOL_SEED = 52`,
`BATTERY_N = 462`, `TARGET_MODELS`, `_SUFFIXES` — are frozen by this document. Changing any of
them after this prereg is committed VOIDS the run; a changed chain needs a new prereg and a new
receipt, not an edit.

**Extraction.** `run_g0clear.extract_multi` unchanged: 12 `CONCEPT_TEMPLATES`, last-token hidden
state, differenced against the same sentence rendered with the neutral word `object`. Source
`meta-llama/Llama-3.2-3B-Instruct`. Targets `meta-llama/Llama-3.2-1B-Instruct`,
`google/gemma-2-2b-it`, `Qwen/Qwen2.5-1.5B-Instruct` — the three b31v2 and b34v3 scored, with
b31v2's committed per-model layer rule.

**Split.** 462 concepts partitioned at **seed 5201** into 323 anchors / 69 selection / 70
held-out — the same proportions and the same integer sizes as `split_concepts`, so chance is
1/70 = 0.0143 and the read is numerically comparable to the published ones. Seed 5201 has not
been used anywhere in this arc; the seeds already spent are 0, 31, 343 and 8080. **The seed is
not shopped.** The held-out set's overlap with previously-scored held-out sets was deliberately
not computed before this document was frozen; the runner computes it and G1 judges it. If G1
fails, the run is VOID and a successor prereg draws one new seed, publishing the discarded seed
so that any shopping is visible in the record.

**Arms, per target.**

- **PAIRED** — the b31v2 M1 two-layer MLP fit on the 323 true anchor pairs, top-1 identification
  of each held-out concept among the 70.
- **LABEL-FREE** — the b34v3 method unchanged: shuffle the target's anchor rows under the split
  seed, discover the correspondence with `TransferMap.fit`, fit ONE MLP on the discovered
  pseudo-pairs, read the 70 held-outs.
- **NULL** — for each arm and each target, the pairing-shuffled twin: same architecture, same
  training budget, correspondence destroyed. Six null cells.

**Hyperparameters, both ways.** The published reads take `layer` and `k` from
`g0clear_result_llama3b.json` (`layer 11`, `k 150`), selected on the *old* battery's SEL_dirs.
B52 runs the gated arms under those same committed values, so the reproduction gate compares like
with like. It **also** re-selects `(layer, k)` on B52's own 69-concept selection split and reports
the held-out read under the re-selected pair as a **diagnostic**, not a gate. Reporting both
prices the hyperparameter carry-over that the addendum names, without letting an ungated number
into a verdict.

## Asserted invariants — recorded, deliberately not gated

`styxx.protocol` refuses a gate no outcome row depends on, on the principle that a leg which
cannot fail must not gate. The following are facts of the frozen artifacts, so gating them would
be decoration; they are required fields of `b52_result.json` and their absence VOIDS the run.

- `extraction_term.pool` = 547373, `extraction_term.eligible` = 5199, `extraction_term.battery`
  = 462, `extraction_term.survival_ratio_battery_over_pool` = 0.00084403,
  `extraction_term.sampling_fraction_battery_over_eligible` = 0.088863 — copied verbatim from
  `concept_pool.json`, which is the artifact that makes them true.
- The per-filter ladder of `concept_pool.json` reproduced into the result, so a reader of the
  result alone can see what each stage cost.
- The same quantities for the 2026-08 battery, recorded as `UNCHECKABLE` with the reason.

## Gates (frozen; scored by `styxx.protocol` against `b52_result.json`)

```gates
{"gates": {"G0_battery_integrity": {"metric": "battery_sha256_matches_pool_receipt", "op": ">=", "value": 1,
             "power_basis": "an identity check, not a statistical bar. It can only fail if the scored word list is not the one concept_pool.json records, which is the single failure that would make every other number in this run unpriced again. Achievable by construction at zero compute; falsifiable if a runner substitutes a list.",
             "metric_means": "1 if sha256 of the newline-joined scored battery equals concept_pool.json battery.sha256, else 0"},
           "G1_heldout_freshness": {"metric": "n_fin_previously_scored", "op": "<=", "value": 7,
             "power_basis": "10 percent of a 70-item held-out set. The bar exists because b34v3 asserted membership disjointness in a frozen prereg and verification falsified it at 13 of 70 (b34v3_fresh_split_addendum.json); this program does not assert freshness again, it measures it. Computable before any forward pass, so failure costs no compute and produces a VOID rather than a tempting re-draw.",
             "metric_means": "count of the 70 held-out concepts that also appear in the seed-0 FIN-70 or the seed-343 FIN-70 of the 2026-08 battery"},
           "G2_machinery": {"metric": "targets.llama_1b.read_paired_top1", "op": ">=", "value": 0.53,
             "power_basis": "b31v2's own G0 bar verbatim (PREREG_b31v2_content_transport_2026_08_01.md: the committed 0.586 minus 0.05 slack), reused so the same-family sanity check is not re-derived on this run's convenience. b31v2_result.json measured 0.8000 against it on the old battery, so the bar is known reachable by this machinery.",
             "metric_means": "same-family paired-MLP top-1 over the 70 held-out concepts"},
           "G3_decisive_paired": {"metric": "targets.gemma_2b.read_paired_top1", "op": ">=", "value": 0.143,
             "power_basis": "b31v2's G1 bar verbatim: 10x chance at 1/70, binomial p < 1e-6 at n=70. Reachability on THIS battery is exactly what is unknown and is the reason the gate is here.",
             "metric_means": "cross-family paired-MLP top-1 for gemma over the 70 held-out concepts"},
           "G4_decisive_labelfree": {"metric": "targets.gemma_2b.read_labelfree_top1", "op": ">=", "value": 0.143,
             "power_basis": "b34v3's G1_bar verbatim (10x chance). Reused unchanged so the label-free arm is judged against the bar its own finding was judged against, not a new one.",
             "metric_means": "cross-family label-free top-1 for gemma over the 70 held-out concepts"},
           "G5_null_bounded": {"metric": "max_null_top1", "op": "<=", "value": 0.0286, "agg": "max", "over": "nulls",
             "power_basis": "2x chance, the null bar of both PREREG_b31v2 and PREREG_b34v3. b34v3 passed it at exactly the boundary and disclosed the knife edge; the bar is kept rather than loosened so a repeat of that is visible as a repeat. A 70-item null at 1/70 expects one hit, so two hits fail and that strictness is deliberate.",
             "metric_means": "the largest of the six pairing-shuffled null reads (two arms x three targets)"},
           "G6_reproduces_old_read": {"metric": "abs_delta_paired_gemma_vs_b31v2", "op": "<=", "value": 0.15,
             "power_basis": "derived, not chosen. b31v2_result.json records gemma paired top-1 = 0.7857 at n=70; the binomial standard error there is sqrt(0.7857*0.2143/70) = 0.0491, and two independent 70-item batteries differ with standard error sqrt(2)*0.0491 = 0.0694. The bar 0.15 is 2.16 of those, so ordinary sampling variation between two batteries cannot fail this gate while a battery effect larger than about a fifth of the read can. This gate is allowed to fail and its failure is a finding, not a miss.",
             "metric_means": "absolute difference between this run's gemma paired top-1 and the 0.7857 recorded in b31v2_result.json"}},
 "outcomes": [{"when": {"G0_battery_integrity": false}, "verdict": "INVALID__battery_is_not_the_recorded_one"},
              {"when": {"G0_battery_integrity": true, "G1_heldout_freshness": false}, "verdict": "VOID__heldout_set_not_fresh"},
              {"when": {"G0_battery_integrity": true, "G1_heldout_freshness": true, "G2_machinery": false}, "verdict": "INVALID__machinery_broken"},
              {"when": {"G0_battery_integrity": true, "G1_heldout_freshness": true, "G2_machinery": true, "G5_null_bounded": false}, "verdict": "INVALID__null_fired"},
              {"when": {"G0_battery_integrity": true, "G1_heldout_freshness": true, "G2_machinery": true, "G5_null_bounded": true, "G3_decisive_paired": false}, "verdict": "POOLED_READ_BELOW_BAR__hand_authored_battery_was_load_bearing"},
              {"when": {"G0_battery_integrity": true, "G1_heldout_freshness": true, "G2_machinery": true, "G5_null_bounded": true, "G3_decisive_paired": true, "G6_reproduces_old_read": false}, "verdict": "POOLED_READ_CLEARS_BAR__BATTERIES_NOT_COMPARABLE"},
              {"when": {"G0_battery_integrity": true, "G1_heldout_freshness": true, "G2_machinery": true, "G5_null_bounded": true, "G3_decisive_paired": true, "G6_reproduces_old_read": true, "G4_decisive_labelfree": false}, "verdict": "PAIRED_READ_REPRODUCES_POOLED__labelfree_below_bar"},
              {"when": {"G0_battery_integrity": true, "G1_heldout_freshness": true, "G2_machinery": true, "G5_null_bounded": true, "G3_decisive_paired": true, "G6_reproduces_old_read": true, "G4_decisive_labelfree": true}, "verdict": "READ_REPRODUCES_ON_POOLED_BATTERY__extraction_term_priced"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## What each outcome licenses, written before it is known which one fires

- **`READ_REPRODUCES_ON_POOLED_BATTERY__extraction_term_priced`** — both reads clear their
  original bars on a battery that is a mechanically-derived sample of a recorded pool, and the
  paired read lands within 0.15 of 0.7857. The reads then carry an extraction term, which we know of no other
  read in this arc to carry: E = 0.00084403 of the recorded pool, 0.088863 of the eligible set. That is
  the whole claim. It does not retroactively supply an E for the 2026-08 reads, and no document
  may report it as if it did.
- **`PAIRED_READ_REPRODUCES_POOLED__labelfree_below_bar`** — the paired ceiling survives the
  battery change and the label-free protocol does not. Licensed reading: label-free discovery is
  more battery-sensitive than paired transport. This is a partial negative and gets reported with
  the same prominence as a pass.
- **`POOLED_READ_CLEARS_BAR__BATTERIES_NOT_COMPARABLE`** — the read is real at 10x chance but is
  far from the published level. **The two batteries are then not interchangeable and neither
  number generalises to the other.** The pooled battery is broader in word class than the
  hand-authored one (no offline part-of-speech tagger exists in this environment, so no filter
  restricts it to concrete singular nouns) and that is the leading hypothesis a follow-up must
  test. Stated now so it cannot be produced later as an explanation that was available all along.
- **`POOLED_READ_BELOW_BAR__hand_authored_battery_was_load_bearing`** — the decisive cross-family
  read does not clear 10x chance on a pooled battery while the same machinery clears the
  same-family bar and the nulls stay down. This is a **full result and the most informative
  outcome this design can produce**: it would mean the published reads are bounded to the
  properties of one hand-authored word list. It gets a FINDING of its own, and this program does
  not re-run the cell with a different pool rule in the hope of a better number — the pool rule is
  frozen above and a second pool needs a second prereg saying why.
- **`INVALID__*` / `VOID__*`** — no read is reported from a run that fires one of these, not even
  as context.

## VOID conditions, in addition to the gated ones

The run is VOID — no numbers reported, not even as diagnostics — if any of the following holds.

1. Any frozen constant of `build_concept_pool.py` differs from the committed version at run time,
   or `concept_pool.json` does not reproduce byte-identically from it.
2. The split seed changes after any read has been computed, or any read is computed before G1 is
   evaluated.
3. A target's extraction fails and its cell is dropped. The gated targets are fixed at three; a
   two-target run is a different experiment.
4. Any concept is added to, removed from, or substituted into the battery by hand at any point.
5. The MLP architecture, training budget, or readout metric differs from b31v2's committed
   apparatus. This run is about the battery; changing the machinery in the same firing makes the
   comparison uninterpretable.
6. `b52_result.json` omits any asserted invariant listed above.

## Rounding convention

Every reported read and every null is rounded to four decimal places, and `max_null_top1` is the
maximum of the *rounded* per-cell values. `styxx.protocol`'s composition check compares the quoted
maximum against a recomputation over the declared `nulls` population at 1e-12, so a runner that
rounds one and not the other refuses loudly before any verdict. That refusal is the intended
behaviour and must not be worked around by relaxing the tolerance.

## The failure this document promises to publish

If the pooled battery does not reproduce the read, that is written up as a FINDING with the same
prominence as a pass, on the same day, with the numbers unhedged. The addendum accompanying this
prereg already records that the extraction term for the existing reads is UNCHECKABLE; a B52 that
comes back negative would sharpen that from "unpriced" to "priced and battery-dependent", and the
arc is more useful having said so than having not run it.

## Discipline

One GPU-bearing extraction pass over 462 concepts on four models, then CPU-only fits. Smoke uses a
40/10 split and writes `_smoke`-suffixed files, INVALID-only by type. Result `b52_result.json`,
scored by `styxx.protocol.Experiment(this_prereg).score(result)` so the verdict is computed rather
than chosen. `run_g0clear.py` and the 462-word `_BANK` are not touched by this experiment, and no
frozen artifact of the 2026-08 arc is edited by it.
