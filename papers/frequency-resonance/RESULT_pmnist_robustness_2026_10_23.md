# RESULT — the permuted-MNIST headline under held-out permutations, new seeds and a 3x budget: `BUDGET_CONFOUND__gap_does_not_survive_3x_steps`

**Verdict:
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/verdict" k="quote">`BUDGET_CONFOUND__gap_does_not_survive_3x_steps`</sworn>.**
Scored by `styxx.protocol` over the whole receipt, under the gates block whose sha256 the receipt records as
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/gates_sha256" k="quote">`578924f8e776573cc82db8fbbd8bb9cf84106b81171a5ba6d5783e422c96e854`</sworn>,
the digest the CORRECTION states; the protocol records the PREREG's commit as
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/prereg_commit" k="quote">`3598e59d3fe708ebf2dd20992f253d02d753c88b`</sworn>.
This document takes the verdict from the receipt and decides nothing itself.

**The five held-out permutations were typed by the designer's tag and order nothing after the seal.** The receipt
records the derivation
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/perm_derivation" k="quote">`sha256-of-tag fallback`</sworn>
over the tag
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/fallback_tag" k="quote">`PREREG_pmnist_robustness_2026_10_11`</sworn>,
the labelled fallback of PREREG §3; no beacon drew them.

Fathom Lab · 2026-10-23 · Preregistration: `PREREG_pmnist_robustness_2026_10_11.md`, sealed as SEALS row 7,
<sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md" k="hash">its blob hashes to `2068e6733c1d5d2c4305cddc89e6e5f2ddd3671406e8bcf56332251d05db648e`</sworn>;
read with `CORRECTION_prereg_pmnist_robustness_2026_10_12.md`, sealed as row 7b,
<sworn r="path:papers/frequency-resonance/CORRECTION_prereg_pmnist_robustness_2026_10_12.md" k="hash">its blob hashes to `e591516f49f84595aa3510b66cdab516dc250b903043899a0a564fdda013f0c1`</sworn>.
The receipt names the same two digests
(<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/prereg_blob_sha256" k="quote">`2068e6733c1d5d2c4305cddc89e6e5f2ddd3671406e8bcf56332251d05db648e`</sworn>
and
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/correction_blob_sha256" k="quote">`e591516f49f84595aa3510b66cdab516dc250b903043899a0a564fdda013f0c1`</sworn>).
On 2026-10-12 the operator chose the CORRECTION-file route (CORRECTION §6): the PREREG carries row 7's bytes unedited,
the CORRECTION is sealed on its own as row 7b, and row 8 is not to be anchored. Runner:
`run_pmnist_robustness.py --aggregate`. Receipt: `pmnist_robustness_result.json`, written at
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/wall_clock" k="quote">`2026-10-23T07:29:52-0400`</sworn>,
with <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/schedule/present" k="numeric">65</sworn>
of
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/schedule/expected_leaves" k="numeric">65</sworn>
scheduled leaves present; the leaves and the leaf log (`pmnist_robustness_runs/`) are committed with it (CORRECTION
§3). Every number below that comes from the receipt is sworn to it at the commit this document's sidecar names;
numbers marked operator-attested come from the run's account, and the bars are quoted from the sealed PREREG.

## The gates as the receipt scored them

| gate | metric | receipt | without exclusions | bar (sealed PREREG, section 5) | outcome | read by this row |
|---|---|---|---|---|---|---|
| `G_P_plumbing` | `plumbing_ok` | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/plumbing_ok" k="numeric">1.0</sworn> | — | <sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L234" k="quote">`"G_P_plumbing": {"metric": "result.plumbing_ok", "op": "==", "value": 1.0`</sworn> | held | yes |
| `G_D_stable` | `degenerate_fraction` | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/degenerate_fraction" k="numeric">0.0</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/unexcluded/degenerate_fraction" k="numeric">0.0</sworn> | <sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L236" k="quote">`"G_D_stable": {"metric": "result.degenerate_fraction", "op": "<=", "value": 0.10`</sworn> | held | yes |
| `G_F_floor` | `free_perm_mean` | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/free_perm_mean" k="numeric">0.9274</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/unexcluded/free_perm_mean" k="numeric">0.9274</sworn> | <sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L238" k="quote">`"G_F_floor": {"metric": "result.free_perm_mean", "op": ">=", "value": 0.85`</sworn> | held | yes |
| `G_C_anchor` | `anchor_gap_newseed` | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/anchor_gap_newseed" k="numeric">0.3182</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/unexcluded/anchor_gap_newseed" k="numeric">0.3182</sworn> | <sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L240" k="quote">`"G_C_anchor": {"metric": "result.anchor_gap_newseed", "op": ">=", "value": 0.15`</sworn> | held | yes |
| `G_G_exists` | `gap_perm_ci_lo` | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/gap_perm_ci_lo" k="numeric">0.2066</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/unexcluded/gap_perm_ci_lo" k="numeric">0.2066</sworn> | <sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L242" k="quote">`"G_G_exists": {"metric": "result.gap_perm_ci_lo", "op": ">", "value": 0.0`</sworn> | held | yes |
| `G_G_mean` | `gap_perm_mean` | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/gap_perm_mean" k="numeric">0.2676</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/unexcluded/gap_perm_mean" k="numeric">0.2676</sworn> | <sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L244" k="quote">`"G_G_mean": {"metric": "result.gap_perm_mean", "op": ">=", "value": 0.15`</sworn> | held | yes |
| `G_G_ci` | `gap_perm_ci_lo` | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/gap_perm_ci_lo" k="numeric">0.2066</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/unexcluded/gap_perm_ci_lo" k="numeric">0.2066</sworn> | <sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L246" k="quote">`"G_G_ci": {"metric": "result.gap_perm_ci_lo", "op": ">=", "value": 0.05`</sworn> | held | yes |
| `G_B_long` | `gap_long` | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/gap_long" k="numeric">0.1097</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/unexcluded/gap_long" k="numeric">0.1097</sworn> | <sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L248" k="quote">`"G_B_long": {"metric": "result.gap_long", "op": ">=", "value": 0.15`</sworn> | did not hold | yes |
| `G_R_diversity` | `free_minus_real2_perm_mean` | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/free_minus_real2_perm_mean" k="numeric">0.275</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/unexcluded/free_minus_real2_perm_mean" k="numeric">0.275</sworn> | <sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L250" k="quote">`"G_R_diversity": {"metric": "result.free_minus_real2_perm_mean", "op": "<=", "value": 0.03`</sworn> | did not hold | no |
| `G_R_rotation` | `real2_minus_clamped_perm_mean` | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/real2_minus_clamped_perm_mean" k="numeric">−0.0074</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/unexcluded/real2_minus_clamped_perm_mean" k="numeric">−0.0074</sworn> | <sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L252" k="quote">`"G_R_rotation": {"metric": "result.real2_minus_clamped_perm_mean", "op": "<=", "value": 0.03`</sworn> | held | no |

The outcome column is the receipt's own record (`gates`), and the bar column quotes the sealed gates block; every
gated metric is written rounded once to four decimals before scoring (CORRECTION §2), and the column without
exclusions is the same aggregate with degenerate runs at their own values (§6).

## Permutation 1234: the new-seed anchor gap and the receipt's re-run seeds

On permutation
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/anchor_perm" k="numeric">1234</sworn>,
FREE minus CLAMPED over the seeds the 2026-07-23 receipt never ran
(<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/new_seeds/0" k="numeric">2</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/new_seeds/1" k="numeric">3</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/new_seeds/2" k="numeric">4</sworn>) is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/anchor_gap_newseed" k="numeric">0.3182</sworn>,
the gated `G_C_anchor`. The receipt's own seeds
(<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/rerun_seeds/0" k="numeric">0</sworn>
and
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/rerun_seeds/1" k="numeric">1</sworn>)
re-ran to seed means FREE
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/free_01" k="numeric">0.9195</sworn>
and CLAMPED
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/clamped_01" k="numeric">0.6073</sworn>,
against the 2026-07-23 receipt's
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/receipt/free" k="numeric">0.9195</sworn>
and
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/receipt/clamped" k="numeric">0.6073</sworn>:
the larger deviation is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/anchor_max_abs_dev" k="numeric">0.0</sworn>,
reported and not gated (§6). Over all five seeds the gap is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/anchor_gap_5seed" k="numeric">0.3158</sworn>
(reported).

| arm | seed <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/rerun_seeds/0" k="numeric">0</sworn> | seed <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/rerun_seeds/1" k="numeric">1</sworn> | seed <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/new_seeds/0" k="numeric">2</sworn> | seed <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/new_seeds/1" k="numeric">3</sworn> | seed <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/new_seeds/2" k="numeric">4</sworn> |
|---|---|---|---|---|---|
| FREE | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/free/0" k="numeric">0.924</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/free/1" k="numeric">0.915</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/free/2" k="numeric">0.9289</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/free/3" k="numeric">0.9322</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/free/4" k="numeric">0.9259</sworn> |
| CLAMPED | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/clamped/0" k="numeric">0.5529</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/clamped/1" k="numeric">0.6617</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/clamped/2" k="numeric">0.6637</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/clamped/3" k="numeric">0.5434</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/clamped/4" k="numeric">0.6253</sworn> |
| REAL2 | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/real2/0" k="numeric">0.6619</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/real2/1" k="numeric">0.5962</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/real2/2" k="numeric">0.5854</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/real2/3" k="numeric">0.595</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/real2/4" k="numeric">0.6159</sworn> |

## The five held-out permutations

Each held-out permutation ran the three arms at seeds
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/derived_seeds/0" k="numeric">0</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/derived_seeds/1" k="numeric">1</sworn>
and
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/derived_seeds/2" k="numeric">2</sworn>.
Per permutation, with degenerate runs excluded from the means (§6):

| permutation | n FREE / CLAMPED / REAL2 | FREE | CLAMPED | REAL2 | gap | FREE − REAL2 | REAL2 − CLAMPED | recovery |
|---|---|---|---|---|---|---|---|---|
| <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/held_out_perms/0" k="numeric">3584514041</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3584514041/n/free" k="numeric">3</sworn> / <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3584514041/n/clamped" k="numeric">3</sworn> / <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3584514041/n/real2" k="numeric">3</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3584514041/mean/free" k="numeric">0.927</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3584514041/mean/clamped" k="numeric">0.6823</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3584514041/mean/real2" k="numeric">0.6533</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3584514041/gap" k="numeric">0.2447</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3584514041/free_minus_real2" k="numeric">0.2737</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3584514041/real2_minus_clamped" k="numeric">−0.0289</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3584514041/recovery" k="numeric">−0.1182</sworn> |
| <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/held_out_perms/1" k="numeric">27380109</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/27380109/n/free" k="numeric">3</sworn> / <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/27380109/n/clamped" k="numeric">3</sworn> / <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/27380109/n/real2" k="numeric">3</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/27380109/mean/free" k="numeric">0.9282</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/27380109/mean/clamped" k="numeric">0.6643</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/27380109/mean/real2" k="numeric">0.647</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/27380109/gap" k="numeric">0.2638</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/27380109/free_minus_real2" k="numeric">0.2812</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/27380109/real2_minus_clamped" k="numeric">−0.0174</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/27380109/recovery" k="numeric">−0.0658</sworn> |
| <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/held_out_perms/2" k="numeric">2994719670</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2994719670/n/free" k="numeric">3</sworn> / <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2994719670/n/clamped" k="numeric">3</sworn> / <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2994719670/n/real2" k="numeric">3</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2994719670/mean/free" k="numeric">0.9306</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2994719670/mean/clamped" k="numeric">0.6582</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2994719670/mean/real2" k="numeric">0.6716</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2994719670/gap" k="numeric">0.2724</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2994719670/free_minus_real2" k="numeric">0.259</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2994719670/real2_minus_clamped" k="numeric">0.0133</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2994719670/recovery" k="numeric">0.049</sworn> |
| <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/held_out_perms/3" k="numeric">3317084913</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3317084913/n/free" k="numeric">3</sworn> / <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3317084913/n/clamped" k="numeric">3</sworn> / <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3317084913/n/real2" k="numeric">3</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3317084913/mean/free" k="numeric">0.925</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3317084913/mean/clamped" k="numeric">0.5798</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3317084913/mean/real2" k="numeric">0.5718</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3317084913/gap" k="numeric">0.3453</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3317084913/free_minus_real2" k="numeric">0.3532</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3317084913/real2_minus_clamped" k="numeric">−0.0079</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/3317084913/recovery" k="numeric">−0.023</sworn> |
| <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/held_out_perms/4" k="numeric">2856893334</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2856893334/n/free" k="numeric">3</sworn> / <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2856893334/n/clamped" k="numeric">3</sworn> / <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2856893334/n/real2" k="numeric">3</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2856893334/mean/free" k="numeric">0.9248</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2856893334/mean/clamped" k="numeric">0.7128</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2856893334/mean/real2" k="numeric">0.7167</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2856893334/gap" k="numeric">0.212</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2856893334/free_minus_real2" k="numeric">0.2081</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2856893334/real2_minus_clamped" k="numeric">0.004</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/per_permutation/2856893334/recovery" k="numeric">0.0187</sworn> |

The permutation-mean gap is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/gap_perm_mean" k="numeric">0.2676</sworn>,
its sd across the five permutations
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/gap_perm_sd" k="numeric">0.0492</sworn>
and its smallest permutation
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/gap_perm_min" k="numeric">0.212</sworn>;
the 95% permutation interval (t with four degrees of freedom, the permutation as the unit) is
[<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/gap_perm_ci_lo" k="numeric">0.2066</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/gap_perm_ci_hi" k="numeric">0.3287</sworn>].
Pooled over runs, run as the unit with the normal approximation (n
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/pooled_n/0" k="numeric">15</sworn>
FREE,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/pooled_n/1" k="numeric">15</sworn>
CLAMPED), the gap is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/pooled_gap" k="numeric">0.2676</sworn>
with interval
[<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/pooled_ci/0" k="numeric">0.2415</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/pooled_ci/1" k="numeric">0.2938</sworn>],
the tighter interval M6 asks to print beside the permutation one. The hierarchical bootstrap
(<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/bootstrap/n" k="numeric">2000</sworn>
resamples, generator seed
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/bootstrap/seed" k="numeric">20261011</sworn>)
puts the permutation-mean gap in
[<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/bootstrap/ci_lo" k="numeric">0.2321</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/bootstrap/ci_hi" k="numeric">0.311</sworn>]
and attributes
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/bootstrap/between_perm_var_share" k="numeric">0.8901</sworn>
of the permutation scatter to the permutation (variance of the five gaps
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/bootstrap/gap_perm_var" k="numeric">0.00242</sworn>,
of which seed noise accounts for
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/bootstrap/seed_noise_var_of_gap_p" k="numeric">0.000266</sworn>
by the method of moments); seed noise alone scatters five three-seed gaps, which is why that share and not the
interval says whether the permutation is the source (§4, M6).

## Three times the budget, on permutation 1234

At
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/steps_long" k="numeric">12,000</sworn>
steps under a cosine schedule re-stretched to that budget, FREE (seeds
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/long_seeds/free/0" k="numeric">0</sworn>
and
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/long_seeds/free/1" k="numeric">1</sworn>)
reaches
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/long/free_long" k="numeric">0.9541</sworn>
and CLAMPED (seeds
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/long_seeds/clamped/0" k="numeric">0</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/long_seeds/clamped/1" k="numeric">1</sworn>
and
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/long_seeds/clamped/2" k="numeric">2</sworn>)
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/long/clamped_long" k="numeric">0.8444</sworn>:
gap_long is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/gap_long" k="numeric">0.1097</sworn>.
Between the two schedules, CLAMPED gains
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/long/clamped_long_gain" k="numeric">0.2183</sworn>
and FREE
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/long/free_long_gain" k="numeric">0.0346</sworn>
over the same seeds at
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/steps_main" k="numeric">4,000</sworn>
steps; that compares two schedules, not one curve. Within the long curves, accuracy at step 8000 minus accuracy at
step 4000 is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/long/clamped_long_midgain" k="numeric">0.129</sworn>
for CLAMPED and
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/long/free_long_midgain" k="numeric">0.0384</sworn>
for FREE — the within-schedule quantity the "plateaus" sentence of the 2026-07-23 RESULT lacked — and over the last
quarter of each
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/steps_main" k="numeric">4000</sworn>-step
CLAMPED run, accuracy at step 4000 minus accuracy at step 3000 is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/long/clamped_last_quarter_gain" k="numeric">0.0367</sworn>,
read with the caveat that a cosine schedule drives the learning rate to zero at the budget (§4). None of these gains
is a gate.

| arm | seed | final accuracy |
|---|---|---|
| FREE | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/long_seeds/free/0" k="numeric">0</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/runs/8/final_acc" k="numeric">0.9544</sworn> |
| FREE | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/long_seeds/free/1" k="numeric">1</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/runs/10/final_acc" k="numeric">0.9537</sworn> |
| CLAMPED | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/long_seeds/clamped/0" k="numeric">0</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/runs/0/final_acc" k="numeric">0.8289</sworn> |
| CLAMPED | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/long_seeds/clamped/1" k="numeric">1</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/runs/2/final_acc" k="numeric">0.8537</sworn> |
| CLAMPED | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/long_seeds/clamped/2" k="numeric">2</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/runs/4/final_acc" k="numeric">0.8505</sworn> |

## The untied control: FREE, REAL2 and CLAMPED on held-out permutation average

On held-out permutation average FREE − REAL2 is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/free_minus_real2_perm_mean" k="numeric">0.275</sworn>
(`G_R_diversity`) and REAL2 − CLAMPED is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/real2_minus_clamped_perm_mean" k="numeric">−0.0074</sworn>
(`G_R_rotation`); over the same pools they sum to the permutation-mean gap. The recovery fraction, defined where a
permutation's gap is at least 0.05 (here on
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/recovery_defined_on/0" k="quote">`3584514041`</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/recovery_defined_on/1" k="quote">`27380109`</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/recovery_defined_on/2" k="quote">`2994719670`</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/recovery_defined_on/3" k="quote">`3317084913`</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/recovery_defined_on/4" k="quote">`2856893334`</sworn>),
is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/recovery_perm_mean" k="numeric">−0.0279</sworn>,
interval
[<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/recovery_perm_ci/0" k="numeric">−0.1105</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/recovery_perm_ci/1" k="numeric">0.0547</sworn>],
class
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/recovery_class" k="quote">`rotation`</sworn>.
At fifteen runs per arm the gates block puts `G_R_diversity`'s 0.03 bar at
<sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L251" k="quote">`about 1.5 SE`</sworn>
and `G_R_rotation`'s at
<sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L253" k="quote">`about 1.1 SE`</sworn>,
so the suffix of a ROBUST or ATTENUATED_VARIABLE verdict is a label and this interval is the finding (§7, M3). On
permutation 1234's new seeds the same three read FREE − REAL2
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/anchor_free_minus_real2_newseed" k="numeric">0.3302</sworn>,
REAL2 − CLAMPED
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/anchor_real2_minus_clamped_newseed" k="numeric">−0.012</sworn>
and recovery
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/anchor_recovery_newseed" k="numeric">−0.0378</sworn>
— the attribution on the sealed untied run's own permutation with seeds it never ran. This row does not read the
attribution gates; the numbers are reported, not pronounced.

## Degenerate runs, by arm, and the aggregate without exclusions

<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/degeneracy/main_count" k="numeric">0</sworn>
of the sixty 4000-step runs and
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/degeneracy/long_count" k="numeric">0</sworn>
of the five 12,000-step runs are degenerate: a final accuracy at or below 0.15 on the 10,000-image test set, or a
non-finite loss on any logged line (§6). FREE: none. CLAMPED: none. REAL2: none. FREE runs counted as 0.0 in the floor
rather than dropped: none. A collapsed decay-arm run is excluded from every mean, the conservative direction for the
gap; the gates table above prints every gated metric again with no exclusion.

## Within-model reliance (reported, not evidence)

The within-model θ→0 reliance of the FREE runs (`clamp_theta` on each block of each trained FREE model) has mean
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/reliance_mean" k="numeric">0.8273</sworn>
over
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/reported/reliance_n" k="numeric">22</sworn>
runs. It is recorded because it is free and is not evidence for the between-model gap: a destructive perturbation with
no matched-norm control (`CRITIC.md` M3; PREREG §3).

## Plumbing and the leaf log

plumbing_ok reads
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/plumbing_ok" k="numeric">1.0</sworn>.
The checks of §9.4 that did not hold, as the receipt's `plumbing` block names them: none.

The leaf log holds
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/leaf_chain/entries" k="numeric">65</sworn>
entries. Leaves re-run after a log line (rewrites, visible and not fatal): none. Leaves whose bytes are not the log's
latest entry: none. Leaves the log does not name: none. A lab that deletes a leaf and re-runs it leaves a visible
second entry; a lab that rebuilds the chain is not caught by this (§9.5), and only an outside replication checks the
numbers (`ATTACK.md` §6).

No field of the receipt or of a leaf was written as null for being non-finite, and no run logged a non-finite loss.

## The derivation of the permutations, and what the seal orders

Derivation
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/perm_derivation" k="quote">`sha256-of-tag fallback`</sworn>:
the five held-out permutation seeds are the top 32 bits of sha256(`<tag>:perm:k`) over the tag
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/fallback_tag" k="quote">`PREREG_pmnist_robustness_2026_10_11`</sworn>:
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/held_out_perms/0" k="numeric">3584514041</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/held_out_perms/1" k="numeric">27380109</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/held_out_perms/2" k="numeric">2994719670</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/held_out_perms/3" k="numeric">3317084913</sworn>,
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/held_out_perms/4" k="numeric">2856893334</sworn>.
The fallback is auditable but was typed by the designer's tag (§13 decision 3). A seal bounds this file's existence
from above and orders nothing else, and under the fallback nothing in this run is ordered after the seal: neither the
forty-five held-out-permutation trainings nor the twenty anchor-permutation runs.

## The sealed untied verdict, beside this one

The sealed untied receipt (`pmnist_untied_result.json`, run 2026-10-11, `PREREG_pmnist_untied_2026_09_02.md`) has not
been unsealed for this RESULT, and its verdict is not printed here. Once it is, it goes beside this one with the note
that its anchor postdates its run (run end 18:54:26 on 2026-10-11, PREREG §10).

## What happened in the run (operator-attested)

The numbers and dates in this section come from the operator-facts file the lead filled in, not from the receipt: the
smoke receipts are gitignored by the lab's `papers/**/*_smoke.json` rule (CORRECTION §3), and the crash log, the
relaunch log and the operator's record are kept outside the commit, each named below by its path under the lab's
worktree folder and by the sha256 of its bytes. Each smoke is named by the sha256 of its receipt's bytes.

**The smokes.** Two smokes ran before night 1: 18.15 minutes per 4000-step run (smoke receipt sha256
`541f0bf08cec733e76c9ddf4baecb9122ffe8bf30fa0b15e9fd1bbfbf3a86790`), the smoke run at 18:31 on 2026-10-12 while agent
workflows and a model download loaded the box; kept as smoke_attempt1_2026-10-12T1835.json; 18.22 minutes per
4000-step run (smoke receipt sha256 `fcc51b31cd8fdeaed768a16c6f2ed489bb68408ea3c63c2a4fcf2f1b3c8f56d8`), the smoke
re-run at 18:37 the same evening; the receipt in place when every night was launched, so the one the nights read.
Every one projects more than the 18.0-minute refusal threshold of §8, so a night refuses to start without
`--accept-schedule`; above 17.78 minutes night 3's twenty-seven runs also exceed the 8-hour window (CORRECTION §7).
The design does not change with the schedule (§8).

**The schedule, accepted.** The operator started nights 1, 2 and 3 with `--accept-schedule`: both smokes projected
more than the 18.0-minute refusal threshold of PREREG section 8 (18.15 and 18.22 minutes per 4000-step run), and the
operator chose in chat on 2026-10-12 to start night 1 with --accept-schedule; nights 2 and 3 were started with the
same flag under the same decision, the smoke receipt being unchanged. The operator's record of the run is
`_frequency_2026_10_11/next/robustness/smoke_attempts/OPERATOR_ACCEPT_SCHEDULE_2026_10_12.md` (sha256
`39120f911ba182bf37fcd94e28bf2f931546a291f35e0696b4c9b3ee3fccc0ad`, 2,063 bytes), kept in the lab's run folder for
this PREREG.

**The nights.** Night 1 ran on 2026-10-12 as scheduled: 18 leaves, none degenerate, the measured runs taking about
14.5 minutes each against the smoke's 18.2. Night 2 ran on 2026-10-22, not on 2026-10-13 as scheduled: the lab's
weekly usage limit stopped its launcher on the scheduled night; started 2026-10-22 at 19:44 under the box lease, with
the labelled sha256-of-tag fallback permutations. Night 3 ran on 2026-10-23, not on 2026-10-14 as scheduled: the lab's
weekly usage limit stopped its launcher on the scheduled night; started 2026-10-23 at 01:20 through the night launcher
under the box lease, straight after night 2, with the same fallback permutations and the same accepted schedule. These
dates are the operator's record, checked at write time against the `wall_clock` the runner wrote into each night's
leaves (the receipt's `runs`) and never read from a file's modification time.

**Night 1's crash and its resumption.** Night 1 stopped in its run 4 of 18 in the runner's order
(`p1234_free_s3_4000_h64.json`) with a CUDA out-of-memory error, caused by another process's use of the host's commit
charge. the same command re-run the same evening at 19:59, once the other process had exited and free commit was back
above 14 GB; the runner skipped the three leaves on disk and wrote the remaining fifteen. The leaves already written
persisted and the runner skipped them on the restart (§8: a crash loses at most one run). The crashed attempt's log is
kept as `_frequency_2026_10_11/next/robustness/smoke_attempts/night1_attempt1_oom.log` (sha256
`8e89243a2ecf7740cb54a525a92d10e019cc396b3ba09b3f310c05e60f03e545`, 3,939 bytes), kept in the lab's run folder for
this PREREG, as the operator's record names it (the same bytes as `robustness_night1.log` beside the runner, which the
`*.log` rule ignores), and the relaunch's log as `robustness-run/papers/frequency-resonance/robustness_night1b.log`
(sha256 `5f4dbe0edf593aca790ceeed33ab264f6235c4bb7e8a1b5af8680a68731876e3`, 8,696 bytes), kept beside the runner in
the run worktree at 62f8ff44, where the `*.log` rule ignores it; neither is committed. The receipt's leaf log lists no
re-run for it: the runner appends a leaf's log line only after the run has trained (CORRECTION §3), so a run that
crashed while training left no line. Its leaf in the receipt reads final accuracy
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/runs/13/final_acc" k="numeric">0.9322</sworn>.

**The permutations of nights 2 and 3: the labelled fallback.** The operator declared `--fallback-perms`: rows 7 and 7b
were never anchored; the operator's decision of 2026-10-12 named the labelled sha256-of-tag derivation for that case.
The receipt's derivation is sworn in the opening lines and agrees with this account.

**Compute.** The operator's account records no overrun of the approved compute past 1.5×, at which §8 has the operator
stop the run and the RESULT say so.

**Repairs by hand.** None: no torn leaf log was repaired by hand (CORRECTION §3).

**Row 8.** No memo carrying row 8's digest (`18aa265c…`) is recorded; row 8 draws nothing and seals nothing this run
reads (CORRECTION §5).

**Note.** night 2 ran under the box lease (clawd/wt/_box/lease.py hold), installed 2026-10-22 after night 1's crash;
the lease refused other heavy work on the box while the run held the GPU.

## The receipt's seeds 0 and 1: the PREREG's three sentences and the CORRECTION's section 1

Three sentences of the PREREG are wrong as written and stay as they are: the opening's
<sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L18" k="quote">`those two enter no gated metric (§4)`</sworn>;
§3's
<sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L140" k="quote">`(reported) and **no gated metric**`</sworn>;
and, inside the frozen gates block, `G_C_anchor`'s power basis,
<sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L241" k="quote">`seeds 0 and 1 are the receipt's own and enter no gate`</sworn>.
The CORRECTION, sealed beside it as row 7b, corrects all three in prose
(<sworn r="path:papers/frequency-resonance/CORRECTION_prereg_pmnist_robustness_2026_10_12.md#L33" k="quote">`By value the six enter exactly one gated metric`</sworn>),
and asks this RESULT to repeat its section 1 beside them. It is repeated here, verbatim:

> Three sentences of the PREREG are wrong as written, and stay as they are: the opening's "those two enter no gated
> metric (§4)"; §3's "they feed `anchor_max_abs_dev` (reported) and **no gated metric**"; and the last clause of
> `G_C_anchor`'s power basis, "seeds 0 and 1 are the receipt's own and enter no gate", which sits inside the frozen
> gates block. This section corrects all three in prose. No byte of the block changes, and a power basis decides no
> row.
>
> **By value.** The six re-run trainings (FREE, CLAMPED and REAL2 at seeds 0 and 1 on permutation 1234) are 4000-step
> runs, and `degenerate_fraction` (§4) is degenerate 4000-step runs over 60: the sixty 4000-step runs, the six among
> them, and not the five 12,000-step runs, which §4 leaves out of it. The runner counts them (its pool for that metric
> is every 4000-step leaf at H = 64). By value the six enter exactly one gated metric, `degenerate_fraction`
> (`G_D_stable`), and no other: not the floor and not `anchor_gap_newseed` (both read anchor seeds 2, 3, 4), not any
> permutation-averaged gate (held-out permutations only), not `gap_long` (its own 12,000-step runs). This is kept,
> not changed in code: `G_D_stable` is a validity gate over the sixty 4000-step runs, and dropping six of them would
> let a collapse of the re-run go uncounted. The bar stays at 0.10.
>
> **As leaves.** "And no other" holds for the six runs' values only. Like every leaf, the six leaves count toward
> `G_P_plumbing`: `plumbing_ok` reads 0 if any of them is absent, reads back module values other than the
> schedule's, names other imported-runner blobs, another data digest, another PREREG digest or another digest of this
> file (section 5), lacks `preflight_ok`, carries a curve of the wrong length or spacing, has a final accuracy unequal
> to its last curve point, or is not the leaf log's latest entry for its name (§9.4, §9.5).
>
> **What a reader of the sealed receipt knows.** The committed `run_pmnist_untied.py` trains FREE, CLAMPED and REAL2
> again at seeds 0 and 1 on permutation 1234 and writes each re-run's final accuracy (`test_acc` per arm and seed): the
> sealed receipt holds six re-run accuracies, not the "two" the opening says. Four of the six configurations, FREE and
> CLAMPED at seeds 0 and 1, have public values in `pmnist_ablation_result.json` (`test_acc`: FREE 0.924 and 0.915,
> CLAMPED 0.5529 and 0.6617); the sealed receipt's re-runs agree with those only up to device nondeterminism, which
> that receipt's `anchor_max_abs_dev` measures on the two arms' seed means (§7 lists
> `INVALID__plumbing_anchors_drifted`, a deviation above 0.03, among the verdicts it can carry). REAL2 at seeds 0 and 1
> has no public value. Whoever reads the sealed receipt before this run therefore knows six of this design's sixty-five
> final accuracies, each up to device nondeterminism, and could foresee at most 6 of the 60 runs that
> `degenerate_fraction` counts.
>
> **Withdrawn.** This branch's CHANGELOG entry as written at `4e3d8be7` and reworded at `e8b91392`, and those two
> commits' messages, said that four of the six accuracies in the sealed receipt "are already public" and that its
> reader learns two new numbers. That is withdrawn: the four public numbers are the 2026-07-23 run's, the sealed
> receipt's are re-runs, and how closely they agree is a fact about a receipt no one writing this has read.
>
> The RESULT repeats this section beside the PREREG's three sentences, citing this correction.

In this receipt the six re-run trainings read as below; by value they enter `degenerate_fraction` only, and of the six
the degenerate ones are: none.

| arm | seed <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/rerun_seeds/0" k="numeric">0</sworn> | seed <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/rerun_seeds/1" k="numeric">1</sworn> |
|---|---|---|
| FREE | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/free/0" k="numeric">0.924</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/free/1" k="numeric">0.915</sworn> |
| CLAMPED | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/clamped/0" k="numeric">0.5529</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/clamped/1" k="numeric">0.6617</sworn> |
| REAL2 | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/real2/0" k="numeric">0.6619</sworn> | <sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/anchor/per_seed/real2/1" k="numeric">0.5962</sworn> |

## The sentence this verdict licenses

The PREREG's §11 gives the sentence for this row
(<sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L513" k="quote">`- **BUDGET_CONFOUND:**`</sworn>).
With this receipt's numbers, and nothing stronger:

**On permutation
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/anchor_perm" k="numeric">1234</sworn>,
at
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/steps_long" k="numeric">12,000</sworn>
steps under a cosine schedule re-stretched to that budget, the gap is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/gap_long" k="numeric">0.1097</sworn>
(two FREE seeds against three CLAMPED seeds); at
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/steps_main" k="numeric">4,000</sworn>
steps on three new seeds it is
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/result/anchor_gap_newseed" k="numeric">0.3182</sworn>.
The
<sworn r="path:papers/frequency-resonance/pmnist_robustness_result.json#/config/steps_main" k="numeric">4000</sworn>-step
gap does not survive the longer budget on this permutation.**

Not licensed (§11):
<sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L515" k="quote">`Not: "the +0.3122 was a training-budget`</sworn>
<sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L516" k="quote">`reading" — one permutation and five runs do not license the general sentence (M8).`</sworn>

For every row:
<sworn r="path:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md#L526" k="quote">`No outcome bears on deployed models, on attention`</sworn>
… on importance, on priority, or on what the lab added that others did not (§11).

## What this does not establish

- Nothing about what the task needs. Correctness is not coverage: a gap inside this architecture says nothing about
  what permuted MNIST requires, and at matched parameters the transformer's mean capacity in
  `necessity_attention_result.json` stands at
  <sworn r="path:papers/frequency-resonance/necessity_attention_result.json#/arms/transformer/mean_kcap" k="numeric">15.33</sworn>.
- Nothing beyond a three-block complex-diagonal linear recurrence (LRU-style; `CLRU` in `run_pmnist_ablation.py`): the
  architecture is named by its code, and the prior-art reads `ATTACK.md` §5 names (the LRU and S4D real-versus-complex
  rows) are owed before any public sentence; no priority claim is made.
- Nothing about deployed models, attention, importance or priority (§11, §14).
- Nothing from the within-model reliance: no matched-norm control was run (a separate design,
  `PLAN_next_experiment.md` §1.3).
- Nothing ordered after the seal: the held-out permutations were typed by the designer's tag.

## Owed with this RESULT, and not done by the kit that wrote it

- The banner for the 2026-09-02 CHANGELOG sentence ("The arc's headline survives its own confound; the permuted-MNIST
  re-run is the owed next preregistration."), whatever the verdict (§10), because that sentence was never supported by
  a receipt on this benchmark. Its text is in the kit's output, `CHANGELOG_banner_2026_09_02.txt`.
- Forward pointers to `RESULT_pmnist_robustness_2026_10_23.md` in `RESULT_pmnist_ablation_2026_07_23.md` and in
  `PREREG_pmnist_untied_2026_09_02.md`, in the same commit (§10). The untied PREREG is sealed (SEALS row 2,
  <sworn r="path:papers/frequency-resonance/PREREG_pmnist_untied_2026_09_02.md" k="hash">its blob hashes to `507ef9aceb297ba06511d9e96c79ab9e1d6eca867d2ab73390e884c455c4a600`</sworn>)
  and `CERT_pmnist_ablation.json` names the 2026-07-23 RESULT, so an edit of either moves bytes a seal or a
  certificate names; the pointer text is in `FORWARD_POINTERS.txt`, and how it lands is the operator's call.
- The headline's rows in `papers/INDEX.md` (the frequency-resonance row and the `+0.3122` receipt row) point to this
  RESULT; their status changes accordingly (§11).

## Sworn

This document swears 229 spans. At write time `make_result.py` verified them in process against the bytes it read (the
receipt, the PREREG, the CORRECTION, the 2026-07-23 receipt, the untied PREREG and the attention receipt): 229 HELD, 0
FAILED, 0 UNRESOLVED, 0 MALFORMED; sentence_share 0.3567 (advisory, a floor that counts every narrative sentence as
load-bearing; `styxx.sworn`). The sidecar committed beside this file is the receipt for these counts.
