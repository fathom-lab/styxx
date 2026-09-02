# DECLARATION — h: who hands the verifier its target

Fathom Lab · 2026-09-01 · **A declaration and a census, not a result.** No preregistration
covers this document and it carries no headline finding. It declares a mapping — from every
`obligation_source` the verifier can emit to the party that handed the verifier its target —
and it counts the corpus under that mapping in the two populations that must never be pooled.
Mapping: `h_mapping.json`. Receipt: `h_mapping_census_result.json`. Gate: `tests/test_h_mapping.py`.
Sworn: every count below is bound to that receipt at commit `ed084cef8811` by `sworn/0.1`; the sidecar
and verdict receipt sit beside this file, and `tests/test_sworn_dogfood.py` re-derives them.

## Why a mapping has to be declared

`AUDIT_the_whole_program_2026_09_01.md` names M2, **handed-target**: an instrument measured
where the target was supplied collapses when it must find the target itself — found eleven
times, named zero. The audit's own verdict on the prose-reading hypothesis reduces to it:
*instruments that must locate their own target in open-ended prose have failed every time this
lab measured them against readers who did not write that prose.*

For `styxx.certify` the question *who handed the verifier this token?* already has a recorded
answer. Since 2026-08-28 every obligated ledger entry carries `epistemics.obligation_source`,
the clause that made the verifier look. What it does not carry is what that clause **means** for
handedness, and the source names (`vocabulary`, `n-glued`, `range-correlation`, `precision`,
`range-sanity`) are clause names, not a classification. A reader who wants h has to decide, for
each clause, whether the target came from the document's own words, from the token's own shape,
from the receipt, or from somewhere outside all three — and if two readers decide differently,
they publish two different h from one ledger. The mapping is therefore **declared once, in one
file, and enforced**: `tests/test_h_mapping.py` fails the moment the verifier can emit a source
the mapping does not name. A new obligation clause cannot silently create a new stratum of the
corpus that no h accounts for.

**UNVERIFIED — the brief's list.** The commissioning brief for this cycle is recorded as naming
*six* obligation sources, and the observation that the five values the ledger carries are not
that list is what opened this work. The brief itself was not reachable from the session that
wrote this document (it is not in the tree). The six declared below are therefore the five the
verifier can emit plus the one the lane has defined and refused to ship — `structural-precision`,
OBLIGATE-1 — and not a transcription of the brief. If the brief's six differ, `h_mapping.json`
is the single place to reconcile them and the gate test will hold the reconciliation honest.

## The mapping, as declared

| source | the clause, in `styxx/certify.py` | handed by | emittable |
|---|---|---|---|
| `vocabulary` | `_TRIGGERS` matches the token's binding context | object_text | yes |
| `n-glued` | `n=` glued directly before the token (self-scoped since v0.5) | object_text | yes |
| `range-correlation` | `_TRIGGERS_CORR` on the context AND decimals > 0 AND value in [−1, 1] | object_text | yes |
| `precision` | the token prints ≥ 7 fractional digits (`V07_PRECISION_OBLIGATION`) | object_form | yes |
| `range-sanity` | bounded-quantity vocabulary before the token AND the value leaves its range; recorded only when no earlier clause obligated the token | object_text | yes |
| `structural-precision` | OBLIGATE-1: ≥ 2 fractional digits AND outside a code span | object_form | **no** — defined in `PREREG_obligate1_2026_08_31.md`, did not ship |

**The classes.** `object_text`: the target was selected by *words of the document under
judgement*, on or around the token's line — the object hands the verifier its target.
`object_form`: the target was selected by the *shape of the token itself* (its printed
precision), not by any surrounding vocabulary — still the object, not its words. `receipt`: the
target was selected from the evidence side. `external`: the target was declared by a party other
than the object's text and the receipt — a preregistration, a panel, or the author binding a
span at write time, which is what `sworn/0.1` is. **No obligation source of this verifier is in
either of the last two classes.** Every target the OATH obligation predicate has ever checked was
handed to it by the object it was judging; the only question the mapping can answer inside this
instrument is *by the object's words or by the object's form*.

`range-correlation` is a conjunction — a correlation word (necessary) and a numeric guard — and
is classed by its necessary conjunct: without the word, nothing fires. `range-sanity` is
emittable and **zero in both populations** on this date; every out-of-range token so far was
already obligated by vocabulary, and under first-writer recording the earlier source stands.

## The two populations, counted separately

`h_mapping_census.py` folds the mapping over the corpus twice and writes both folds under
separate keys. Verifier at the time: `styxx/certify.py` sha256 `a588a722…`. The corpus on disk
was written by <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_PRINTED/distinct_verifier_builds_on_disk" k="numeric">**15 distinct verifier builds**</sworn>.

**PRINTED — what the committed certificates say.** <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_PRINTED/certificates_on_disk" k="numeric">There are 208 certificates on disk.</sworn> <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_PRINTED/certificates_with_epistemics_summary_v1" k="numeric">Only 16 of them carry the epistemics-summary block</sworn>, and <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_PRINTED/certificates_with_per_token_epistemics" k="numeric">only 18 carry per-token epistemics.</sworn>
<sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_PRINTED/obligated_tokens" k="numeric">Those ledgers hold 233 obligated tokens</sworn>: <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_PRINTED/token_obligation_sources/vocabulary" k="numeric">`vocabulary` 215</sworn>, <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_PRINTED/token_obligation_sources/range-correlation" k="numeric">`range-correlation` 8</sworn>,
<sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_PRINTED/token_obligation_sources/n-glued" k="numeric">`n-glued` 8</sworn>, <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_PRINTED/token_obligation_sources/precision" k="numeric">`precision` 2</sworn>. Handed by <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_PRINTED/handedness_share_of_obligated/object_text/share" k="numeric">object_text 0.9914</sworn> and by
<sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_PRINTED/handedness_share_of_obligated/object_form/share" k="numeric">object_form 0.0086</sworn>. This is the *only* population a reader of the
committed certificates can see, and it is under a tenth of the corpus.

**LIVE — every certifiable document re-certified at the current verifier.** <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_LIVE/documents_recertified" k="numeric">207 documents were re-certified</sworn>
(<sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_LIVE/documents_skipped_unresolvable" k="numeric">1 skipped as unresolvable</sworn>), yielding <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_LIVE/ledger_tokens" k="numeric">8583 ledger tokens</sworn>, <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_LIVE/verified_tokens" k="numeric">6300 VERIFIED</sworn> and
<sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_LIVE/obligated_tokens" k="numeric">3148 obligated</sworn>. Of the obligated: <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_LIVE/obligation_sources/vocabulary" k="numeric">`vocabulary` 2612</sworn>, <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_LIVE/obligation_sources/precision" k="numeric">`precision` 242</sworn>,
<sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_LIVE/obligation_sources/range-correlation" k="numeric">`range-correlation` 210</sworn>, <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_LIVE/obligation_sources/n-glued" k="numeric">`n-glued` 84</sworn>. Handed by <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_LIVE/handedness_share_of_obligated/object_text/share" k="numeric">object_text 0.9231</sworn> and by
<sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_LIVE/handedness_share_of_obligated/object_form/share" k="numeric">object_form 0.0769</sworn>. <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_LIVE/by_source_and_status/n-glued|ABSTAIN" k="numeric">Every one of the 84 `n-glued` tokens is ABSTAIN</sworn> — the register
obligates and never binds in this corpus.

The two populations disagree on object_form by a factor of nine (0.0086 vs 0.0769), and neither
is wrong: the printed 18 are the documents that happened to be re-issued after 2026-08-28, and
they are not a sample of anything. A rate computed in one and quoted against the other is the
handed-target error one level up.

## The two denominators, named

"Bound" has meant two things in this lane. In `certify.py`, `bound` is **OBLIGATED** — an
obligation clause fired. In the RESULT prose, "the number binds" is **VERIFIED** — a value
matched. On the same live run, <sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_LIVE/source_share_of_obligated/vocabulary" k="numeric">`vocabulary` is 0.8297 of obligated tokens</sworn> and
<sworn r="path:papers/closed-model-frontier/h_mapping_census_result.json#/population_LIVE/source_share_of_verified/vocabulary" k="numeric">0.3517 of VERIFIED tokens</sworn>. The figure carried into this cycle as *vocabulary is 35.4% of bound tokens* is
the second of these — vocabulary-obligated VERIFIED tokens over all VERIFIED tokens — and it is
the handedness figure the v1 epistemics summary already prints per certificate as the obligated
half of `verified.value_match`. The receipt keys both shares by their denominator
(`source_share_of_obligated`, `source_share_of_verified`) so neither can be quoted as the other.

## Instrument-level h is a different population from token-level h

Token-level h is a share over tokens inside one instrument. Instrument-level h is a label over
instruments: one row per instrument whatever its token count, saying where that instrument's
targets come from and whether it must find them. `h_mapping.json` carries that table under
`instrument_level`: the OATH obligation predicate (object_text and object_form; does not find
its own target; measured held-out at a false-accusation rate of 0.2596), the diffgate path-claim
accuser (object_text; finds its own target; 0.16 held-out against a 0.95 floor; deleted), STRUCT-1
(object_text; finds its own target; 0.4211 on n=38), and `sworn/0.1` (external; the author binds
the target; unmeasured by construction).

A hypothesis about instruments — that handed-target instruments collapse held-out and
found-target ones do not — is tested on that table and on nothing else. Drawing its test from
the token histogram measures the wrong population: the one the certificates print rather than
the one the hypothesis is about. The two are kept in separate keys of one receipt for that
reason, and no key in the receipt combines them.

## Addendum, 2026-09-02 — object_text has a grain

`RESULT_handedness_v3_header_handed_2026_09_02.md`, preregistered, on the external blind panel: <sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/cells/header/genuine_share" k="numeric">accusations handed by a table header were genuine at 0.9515</sworn> and <sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/cells/line/genuine_share" k="numeric">accusations handed by a trigger word in the line were genuine at 0.6391</sworn>. Both are `object_text` under this mapping, which stays as declared; but the class splits by whether the handing was a structural label the author chose or a word that co-occurred, and the next version of this mapping owes that split a name.

## What this does not say

No token is graded. `object_text` does not mean wrong and `object_form` does not mean right; the
classes say who supplied the target, not whether the target was a claim or whether the oath
held. No rate here is a measurement of the instrument's quality. The mapping is a declaration
and can be argued with in its one file; the census is a count anyone can re-run.

---

*Every target this verifier has ever checked was handed to it by the document it was judging.
That was already true; now it is written down, in one place, with a test that notices when it
stops being the whole story.*
