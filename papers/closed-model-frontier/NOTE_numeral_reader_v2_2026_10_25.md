# NOTE — certify's numeral reader v2: a made-up `$99.9k` is now a ledger row; what would move the committed tree ships off, the operator's decision

Fathom Lab · 2026-10-25 · An instrument change, measured before it ships. No committed certificate, capsule,
receipt, conformance vector, sworn sidecar or charon line is edited or regenerated. Every count inside a sworn
span below is sworn to `papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json`, which
`papers/closed-model-frontier/oath_v15_numeral_reader_census.py` writes by re-deriving every committed certify
output under each flag configuration (run it again to regenerate it; it writes nothing else). The design, its red
team and the rule as built (`RULE_final.md`) are in the operator's product kit, outside the tree, under
`clawd/wt/_product_2026_10_24/certify_suffix/`.

## Why

The product red team of 2026-10-24 (blocker B1) found that certify read no numeral at all in `$67.4k`, `$1.2B`,
`3.4M` or `2.4K`: every alternative of `_NUM` ends with `(?![\w.])`, so a digit run glued to a letter is not a
token. A post reading `BTC is at $99.9k` against a receipt holding `67412.50` got 0 ledger rows and OATH-HELD, and a
made-up `$99.9k` passed unmarked. The operator chose to hold outreach until certify reads suffixed numbers.

## What the reader reads now, as shipped

The numeral reader v2 is a second path in `styxx/certify.py`, behind the master flag `V15_READER_V2`; `_NUM` is not
edited. With the master off, every function reads as it did before v2, byte for byte on the recorded families of
`conformance/numerals/`. As shipped:

- **A magnitude after a currency symbol** (`V15_MAGNITUDE`): `$67.4k`, `US$1.2B`, `€5m`, `$5mm`, and
  (`V15_MAGNITUDE_WORDS`) `$1.2 billion`, read as mantissa times multiplier. Suffixes `k K M B T bn mn`, and `m b t
  mm`, after the symbol only. A leaf binds it when it equals the value, or when, in the suffixed unit, it lies within
  half a unit of the last shown mantissa digit and within 5% of the mantissa: `$1B` binds 0.95B to 1.05B, not 0.5B
  to 1.5B; `$67k` binds 66,500 to 67,500; `$67.4k` binds 67,350 to 67,450; `$67.41k` binds 67,405 to 67,415.
  Percent scaling never applies. A whole-number magnitude keeps the count-binding filter integers keep; a leaf whose
  path names a time or an id (`openTime`, `lastId`, `updated_at`) is never a hit and is recorded as excluded. It is
  obligated exactly as an unsuffixed numeral is: `the mean rate is $99.9k` with no matching leaf is UNGROUNDED and
  OATH-FAILED; `BTC is at $99.9k on Binance spot` is one ABSTAIN row, unbound; `$67.4k` against `67412.50` is
  VERIFIED (`tests/test_certify_numerals_v2_default.py::test_the_product_need_is_met_by_the_shipped_default`).
- **Signs** (`V15_SIGNED_CURRENCY`): `-$1,234.56`, `$-5k`, `($1.2k)` and `- $1.2k` read negative; a markdown bullet
  is not a sign.
- **Separators** (`V15_THIN_GROUPS`): thin and narrow no-break spaces between three-digit groups join one number.
- **Counted, never valued** (`V15_DECIMAL_COMMA`, `V15_GLUED_NONASCII`): a decimal comma or dotted thousands
  (`84,17 €`, `1.234,5`, `1.234.567`), and a digit run glued to a non-ASCII letter or a superscript (`$99.9к`,
  `3.4М`), go to the uncovered band with a reason. A Greek letter after a digit (`2π`, `-3.4σ`) is notation and stays
  silent, as before.
- **On with the master and the magnitude rule, under no flag of their own:** a hyphen word after a numeral does not
  make it formula notation (`3-ish` reads 3; before v2 it read nothing); a currency-led numeral is never formula
  notation or a compound identifier (`sub-$67k`, `$99.9k-ish`); a numeral the reader reads followed by an ellipsis
  (`rate 0.0666…`) is ABSTAIN as a truncation the writer marked (rule C2; before v2 that token could be
  UNGROUNDED); and the lower half of a suffixed range (`60-70k`, `2–3B`) is ABSTAIN as notation, never bound to an
  unrelated leaf (the class-B suffix set widened from `[BMK]` to the suffix table; before v2 `60` in `60-70k` could
  bind an RSI of 60). Each `only_<flag>` configuration of the census runs with the master on, so these moved no
  committed certificate either.
- **Still not read and not counted:** a magnitude no currency symbol leads (`7B`, `124M`, `3.4M`, `2.4K`, `10k`),
  numbers glued to other letters (`10x`, `24h`, `TP1`, `7B-Instruct`), scientific notation, number words, and spans
  the sha/date/version scrub blanks. A made-up `ETH 2.4K` still passes unmarked. A numeral before a sentence period
  is still the band's `trailing-period`.

## What did not move

Under the shipped flags, of the
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/certificates" k="numeric">240 committed certificates whose documents and receipts resolve</sworn>,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/default/certificates_moved" k="numeric">0 re-derive differently</sworn>
in any compared field (verdict, counts, ledger, the band, epistemics) from the reader before v2:
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/default/rows_added" k="numeric">0 rows added</sworn>,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/default/rows_removed" k="numeric">0 rows removed</sworn>.
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/default/capsules_ok" k="numeric">12 committed capsules verify</sworn>
of
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/default/capsules" k="numeric">13</sworn>,
the same ones as with the master off (the one that does not is the HANDOFF diffgate record,
`NOTE_handoff_capsule_reproduces_under_its_minting_build_2026_10_11.md`), and
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/default/capsules_stop_verifying" k="numeric">0 capsules stop verifying</sworn>.
Of the
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/default/charon_lines" k="numeric">257 charon lines</sworn>,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/default/charon_moved_verifier" k="numeric">257 read MOVED_VERIFIER</sworn>
as they do with the master off, and
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/default/charon_skew" k="numeric">0 read SKEW</sworn>,
so the stranger's ferry-log step passes. The `REPLICATIONS.md` audit line re-derives byte for byte:
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/configurations/default/audit_line" k="quote">`corpus papers: 208 certificates | HELD 200  FAILED 8  unresolved 0  verdict-drift 1  receipt-drift 0  incomplete 1  receipt-changed 1`</sworn>.
Each rule that ships on, run alone, moves no committed certificate (`summary.configs.only_<flag>`). Nothing committed
was re-issued: `verifier_sha256` moves, as on every certify change, and a certificate issued from here on carries a
`numeral_reader` block naming the rules that read it and an `uncovered_policy` of schema `v2` written from those
rules.

## The operator's open decision: what each flag that ships off would move

The lead's scope for this build: every flag that moves any committed certificate, capsule or charon line ships off,
so the tree reads exactly as today; the magnitude, form-noise and separator flags ship on. The two halves disagreed
on two flags, and the build followed the rule (section "How the default was chosen"). These are the measured costs,
each against the reader with the master off. A move here is a re-derivation reading differently; nothing committed
changes until someone turns a flag on.

**`V15_MAGNITUDE_BARE`** (a magnitude no currency symbol leads: `7B`, `124M`, `3.4M`, `2.4K`, `10k`, `1.2 billion`;
never obligated, since an obligated bare magnitude made 62 false accusations on model sizes in the design's
counterfactual):
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/only_V15_MAGNITUDE_BARE/certificates_moved" k="numeric">69 certificates re-derive differently</sworn>,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/only_V15_MAGNITUDE_BARE/rows_added" k="numeric">334 rows added</sworn>
(<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/only_V15_MAGNITUDE_BARE/rows_added_verified" k="numeric">2 VERIFIED</sworn>,
both `124M` against `per_model.gpt2.params`;
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/only_V15_MAGNITUDE_BARE/rows_added_abstain" k="numeric">332 ABSTAIN</sworn>,
most of them `B` model sizes),
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/only_V15_MAGNITUDE_BARE/class_flips" k="numeric">0 verdict classes moved</sworn>,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/only_V15_MAGNITUDE_BARE/capsules_stop_verifying" k="numeric">0 capsules stop verifying</sworn>,
and
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/only_V15_MAGNITUDE_BARE/charon_skew" k="numeric">58 charon lines read SKEW</sworn>.
On, `ETH 2.4K` and `24h volume 3.4M` are read and marked; the product needs it for bots that post bare magnitudes.

**`V15_FORM_NOISE`** (clock times, thread markers, word-led version strings and calendar dates become named
exclusions instead of numerals):
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/only_V15_FORM_NOISE/certificates_moved" k="numeric">3 certificates re-derive differently</sworn>,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/only_V15_FORM_NOISE/rows_removed" k="numeric">11 ABSTAIN rows removed</sworn>
(time and date parts),
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/only_V15_FORM_NOISE/class_flips" k="numeric">0 verdict classes moved</sworn>,
and
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/only_V15_FORM_NOISE/charon_skew" k="numeric">3 charon lines read SKEW</sworn>.
On, `14:00` stops being two numerals and `1/` stops counting.

**Both together**, the lead's explicit list (the earlier build session's default):
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/explicit_list/certificates_moved" k="numeric">71 certificates re-derive differently</sworn>,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/explicit_list/class_flips" k="numeric">0 verdict classes moved</sworn>,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/explicit_list/charon_skew" k="numeric">59 charon lines read SKEW</sworn>.
A SKEW line fails the stranger's ferry-log step (`styxx/stranger.py`, which passes only SAME_LINE and MOVED_VERIFIER)
and `tests/test_stranger.py::test_the_ferry_log_passes_against_its_own_head`, which
`tests/test_charon_instrument.py` names as the lab's invariant over the committed log. Turning either flag on is
therefore also a decision about the charon log: amend charon's reading under a dated amendment, as
`AMENDMENT_charon_same_verifier_build_flag_2026_10_25.md` did for one sworn field, or accept a ferry log that fails.

**`V15_SENTENCE_FINAL`** (a numeral before a sentence period is read: `$84.17.` reads 84.17, `all 23,247.` reads
23,247 whole), with the shipped flags otherwise:
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/sentence_final/certificates_moved" k="numeric">157 certificates re-derive differently</sworn>,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/sentence_final/rows_added" k="numeric">373 rows added</sworn>
(<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/sentence_final/rows_added_ungrounded" k="numeric">14 of them UNGROUNDED</sworn>),
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/sentence_final/class_flips" k="numeric">11 committed OATH-HELD certificates re-derive OATH-FAILED</sworn>,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/sentence_final/capsules_stop_verifying" k="numeric">6 committed capsules stop verifying</sworn>,
and
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/sentence_final/charon_skew" k="numeric">139 charon lines read SKEW</sworn>.
With every rule but the magnitude pair on (the design's `v2_all`),
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/all_rules/charon_skew" k="numeric">152 charon lines read SKEW</sworn>.
The design measured 12 class flips with a prototype that had no ellipsis rule; with rule C2 built, the twelfth
(`papers/anchored-validity/FINDING_hardening_part2a_2026_07_21.md` line 38, `(rate 0.0666...)`) is ABSTAIN as a
truncation the writer marked. The eleven, each through a token the period had hidden and the ladder's standing
rules accuse (the list with every accused row is `configurations.sentence_final.class_flips_list` in the census
result; the nearest receipt leaves of each are in the design's `proto/FLIP_DETAIL_OUTPUT.txt` and, re-derived at
`33981121`, `base_remeasure/FLIP_DETAIL_OUTPUT.txt` in the kit):

| document | accused |
|---|---|
| `agent-conscience/FINDING_combined_signal_2026_07_26.md` | line 17 `0.0039506719834589` |
| `agent-conscience/FINDING_framelocality_dogfood_2026_07_29.md` | line 3 `93` |
| `agent-conscience/FINDING_kp_replication_2026_07_28.md` | line 7 `0` |
| `agent-conscience/FINDING_self_verification_2026_07_25.md` | line 12 `0.012253193960511` |
| `anchored-validity/FINDING_r8_sync_corrected_2026_07_20.md` | line 38 `-0.3634153106917754` |
| `closed-model-frontier/RESULT_agent_claim_extractor_baseline_2026_08_30.md` | line 35 `294` |
| `closed-model-frontier/RESULT_external1_the_gate_fails_in_the_wild_2026_08_31.md` | line 39 `0.95` |
| `closed-model-frontier/RESULT_oath_v07_SHIPPED_2026_08_22.md` | line 51 `2` |
| `closed-model-frontier/RESULT_obligate2_split_verdict_2026_08_31.md` | line 37 `624` |
| `closed-model-frontier/RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md` | lines 25 and 50 `0.95` |
| `mind-instrument/FINDING_anatomy_v0_2026_06_10.md` | line 4 `0` |

Some accuse values the receipts hold under another path or rounding (the design's reading: `294` is
`E4.sampled_valid`, `-0.3634153106917754` is within 0.07% of `pi_unclipped`); those are the ladder's debts, not the
reader's. On, the flag also turns the eleven into newly visible drifts that `tests/test_certificate_reproduces.py`
would have to list, or that a re-certification cycle with its own preregistration would answer
(`tests/test_certify_numerals_census.py` holds that none of them is listed while the flag is off).

**`V15_MAGNITUDE_RANGES`** (`60-70k` read as two magnitudes) is measured and off: with the shipped flags it moves
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/ranges/certificates_moved" k="numeric">0 certificates</sworn>,
since a pair it reads must be currency-led; beside the bare arm it moves
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json#/summary/configs/explicit_list_ranges/certificates_moved" k="numeric">75 certificates</sworn>.

## How the default was chosen

The lead's scope had a rule ("every flag that moves ANY committed certificate, capsule or charon line ... ships
DEFAULT OFF behind the master flag, so the tree reads exactly as today") and a list ("the magnitude, form-noise and
separator flags ship ON"). The list rested on a measurement at verdict grain: those rules move 0 committed verdicts,
which is true. At the grain the rule is written in, the bare arm and the form rule move ledgers and charon lines.
The build followed the rule and kept the product need, which a currency-led magnitude alone carries: the magnitude
rule ships on for currency-led magnitudes, its bare arm became its own flag, and both flags that move committed
outputs ship off. `V15_MAGNITUDE_BARE = True` and `V15_FORM_NOISE = True` restore the list exactly
(`tests/test_certify_numerals_v2_default.py::test_two_flags_give_the_leads_explicit_list_back`). The flags are module
constants, as `V08_*` to `V14_*` are; a caller that wants another reading sets them in its own process, and every
certificate it issues names the rules it read under.

## Conformance

`conformance/numerals/vectors_v2.json` pins the reader under the earlier session's default and is not edited;
`conformance/numerals/vectors_v2_default.json` pins the shipped default. Each holds 412 reader texts and 145 binding
cases and the two families recorded from the reader before v2 at `879f6be9`, which the generator refuses to write
unless the master-off reader reproduces exactly; it did. Each set names the flags it was written under, and
`tests/test_certify_numerals_v2_vectors.py` replays every vector, skipping none.

## What this is not

Not a change to the obligation ladder, to `is_spec`, to the count-binding filter or to the trigger vocabulary: what
they do with the tokens the reader can now see is theirs. Not a recall or precision figure: the texts are the probes
run before the reader was built, and a text absent from the vectors is not covered. Not a claim about posts in the
wild. The darkflobi gate runs certify from a snapshot pinned at its own commit and does not move with this change.

## Re-derive

    py -3.12 -B papers/closed-model-frontier/oath_v15_numeral_reader_census.py
    py -3.12 -B -m pytest tests/test_certify_numerals_v2.py tests/test_certify_numerals_v2_default.py \
        tests/test_certify_numerals_v2_vectors.py tests/test_certify_numerals_census.py tests/test_stranger.py
    py -3.12 -B -m styxx.sworn check papers/closed-model-frontier/NOTE_numeral_reader_v2_2026_10_25.sworn-receipt.json \
        papers/closed-model-frontier/NOTE_numeral_reader_v2_2026_10_25.sworn.json --repo .
