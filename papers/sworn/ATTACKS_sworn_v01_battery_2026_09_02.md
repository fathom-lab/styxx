# ATTACKS — the sworn output battery: twelve ways in, four rules out — 2026-09-02

Fathom Lab · 2026-09-02 · **An adversarial pass, not a result.** Nothing here was preregistered
and no bar was frozen. It is the pass the lab's standing rule requires before an instrument is
announced — *no instrument is announced before an adversarial pass* — run against `styxx.sworn`
as committed at `320b303`, in memory, on a `MemoryTree`, touching no file. Every row of the table
is pinned by `tests/test_sworn_attacks.py` at the commit this document names, and every number in
this document is sworn to the receipt that holds it. The repairs the battery licensed are
`SPEC_sworn_output_v02_2026_09_02.md`; <sworn r="prereg:d3f4f852acf3cb8acf7c3aa769cc28b0f2d1574a282b6ae3aa3aee6cf867f4b5" k="quote">that spec's largest rule reads `the coverage estimate is withdrawn; counts are printed instead`</sworn>.

## The battery

Twelve attacks were tried. Four are repaired by v0.2 (their rows say what the verifier now
returns). Six are not repaired and cannot be without the verifier reading meaning — which is the
claim detector that failed eleven times — so their rows say what the verifier now *prints*: a
limitation stated beside the verdict is a different object from one left for a reader to find.
Two survived v0.1 unchanged.

| # | attack | at `320b303` (v0.1, probed) | under v0.2 | pinned by |
|---|---|---|---|---|
| A1 | rider clause — a HELD number painting an unchecked qualifier in the same sentence | HELD | HELD, by design; `certifies` says the document is not thereby correct | `test_a1_the_rider_clause_holds_on_the_number_and_says_nothing_about_the_qualifier` |
| A2 | trivial swearing — bind a count, leave the finding in narrative | HELD, coverage 1.0 | HELD; the floor prints the padding | `test_a2_trivial_swearing_holds_and_the_floor_makes_the_padding_visible` |
| A3 | stative dodge — load-bearing prose in shapes the claim detector misses | HELD, coverage 1.0, zero unsworn claims | HELD; the floor counts every sentence, so the dodge buys nothing | `test_a3_the_stative_dodge_no_longer_shrinks_the_floor` |
| A4 | a tag hidden in an HTML comment | HELD, coverage 1.0 | **MALFORMED `hidden_commitment`**, document SWORN-FAILED | `test_a4_a_tag_hidden_in_an_html_comment_is_malformed_not_held` |
| A5 | a `path:` receipt the author itself committed | HELD, nothing said | HELD; provenance printed: *committed object; authorship unchecked* | `test_a5_a_committed_receipt_holds_and_prints_that_its_authorship_was_not_checked` |
| A6 | a `prereg:` digest proves content, not precedence | HELD | HELD; precedence is the ledger's and `styxx.protocol`'s job, and honest-signal's prior art | `test_a6_a_prereg_digest_proves_content_never_precedence` |
| A7 | a manifest the agent minted, with an honest-looking `kind_of_source` | HELD, harness unnamed in the verdict | HELD; harness and rung printed, rung `undeclared`, never L2 | `test_a7_a_manifest_the_agent_minted_holds_and_the_rung_it_declares_is_printed` |
| A8 | receipt shopping — a larger receipt pool | cannot move a sworn span (v0.1 test) | unchanged | `test_receipt_shopping_moves_oath_but_cannot_move_sworn` |
| A9 | `16%` against a leaf holding `0.16` | FAILED `value_mismatch` | unchanged | `test_a9_percent_and_fraction_do_not_coincide_survives_from_v01` |
| A10 | a one-byte `quote` needle over a whole receipt | HELD | **MALFORMED `short_needle`**; pointer leaves, line anchors and `absent` exempt | `test_a10_a_one_byte_needle_over_a_whole_receipt_is_malformed` |
| A11 | post-hoc tagging | undetectable from bytes | undetectable; `certifies` says the write time was not checked | `test_a11_post_hoc_tagging_is_undetectable_and_the_receipt_says_so` |
| A12 | the right value at the wrong leaf | FAILED when the values differ; HELD when they coincide | unchanged; the pairing-validity cell of the measurement prices it | `test_a12_a_coincident_value_at_the_wrong_leaf_holds` |

<sworn r="path:tests/test_sworn_attacks.py" k="quote">The comment attack is pinned as `test_a4_a_tag_hidden_in_an_html_comment_is_malformed_not_held`.</sworn>
<sworn r="path:tests/test_sworn_attacks.py" k="quote">The needle attack is pinned as `test_a10_a_one_byte_needle_over_a_whole_receipt_is_malformed`.</sworn>
<sworn r="path:tests/test_sworn_attacks.py" k="quote">The unrepaired attacks live under `class TestNotRepairedAndSaidSo`.</sworn>

## The finding that was not an attack

The largest change in v0.2 was not paid to an attacker. While probing A2 and A3 the coverage line
printed 1.0 with zero unsworn claims beside documents whose narrative carried obvious load-bearing
sentences, and the reason was in the denominator instrument. `styxx.claimdetect` (STRUCT-1) is a
diff-claim detector for agent pull-request prose — action verb, concrete object, not stative, no
other actor — and its own docstring says result-shaped sentences are never claims. Every sworn
document in this tree is a result-shaped document. The number beside every verdict was a rule
written for one idiom fired in another: M1 and M2 inside the format built to escape them.

The census (`coverage_census_v01.py`, read through git plumbing at the pre-change commit, never
the working tree) receipts it. <sworn r="path:papers/sworn/coverage_census_v01_result.json#/documents" k="numeric">The census read 12 committed sworn documents.</sworn>
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/v01_estimate_min" k="numeric">The lowest estimate printed beside any of them was 0.6667.</sworn>
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/v01_estimate_max" k="numeric">The highest was 1.0.</sworn>
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/v01_unsworn_claims_total" k="numeric">Across all of them the diff-claim detector counted 20 narrative sentences as claims</sworn>
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/narrative_sentences_total" k="numeric">out of 932 narrative sentences</sworn>
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/sworn_total" k="numeric">beside 167 sworn spans.</sworn>
The sharpest single case is the grain synthesis:
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/rows/0/v01_coverage_estimate" k="numeric">it printed a coverage estimate of 0.9412</sworn>
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/rows/0/v01_unsworn_claims_estimate" k="numeric">with 1 narrative sentence counted as a claim</sworn>
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/rows/0/narrative_sentences" k="numeric">against 98 narrative sentences</sworn>,
and the one it counted is a fragment about a mapping column. Under the v0.2 floor, which treats
every narrative sentence as load-bearing and so cannot flatter,
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/rows/0/sentence_share_floor" k="numeric">the same document reads 0.1404</sworn>;
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/floor_min" k="numeric">across the twelve the floor runs from 0.0156</sworn>
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/floor_max" k="numeric">to 0.2185.</sworn>
Neither number is bound recall. The floor understates by construction; the old estimate overstated
by construction; only a blind panel reading the untagged text can say which sentences mattered,
and that panel has not run.

## The run this document swears to

<sworn r="r1" k="numeric">The harness ran the sworn suites and 358 tests passed.</sworn>
<sworn r="r4" k="numeric">0 failed.</sworn>
<sworn r="r2" k="absent">pytest's output carries no `failed`.</sworn>
<sworn r="r3" k="quote">ruff printed `All checks passed!`</sworn>
<sworn r="r5#/rung" k="quote">The manifest declares rung `L1`</sworn> — a committed script on the same
machine as the author, which is the weak rung and is printed as such on every span above that
rests on it.
<sworn r="path:styxx/sworn.py" k="hash">At the commit this document names the verifier's bytes hash to f75aff8a47656ce74fbc34b6a566e17bb9e23c56da45eed3d28ca55503ca22f1.</sworn>

## What this battery does not say

That the format is safe: six attacks stand, are named, and are priced only by a measurement that
has not run. That the four rules are the right rules: sixteen bytes and three hundred code points
are decisions, carried in `DECISIONS`, arguable. That anyone outside this lab has attacked the
format: the attacker and the builder share a session, which is the specimen-chosen-to-pass shape
the audit named, and the honest reading is that this battery is what the builder could think of.
That the coverage finding is a measurement of load-bearingness: it is a census of what two
instruments printed.

---

*A format that was only ever written by its builder has not been attacked. This one now has
been, by its builder, which is the weakest attacker there is — and it still paid four rules and
lost the number it printed beside every verdict. The next attacker should not be us.*
