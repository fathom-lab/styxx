# NOTE — the checksum RESULT re-run on a second machine: every verdict reproduces, no magnitude does

**status: a replication record with the honest outcome, sworn to both machines' bytes. It corrects
nothing in `RESULT_checksum_smollm_quant_2026_09_13.md`, which is not edited and whose receipt is
history; it states what that RESULT's sentence "rebuild both with `run_smollm_quant.py`" turns out
to mean, and it is the reason `BOUNTY.md` now has a tolerance rule.**

## what was done

`python papers/checksum/run_smollm_quant.py`, the recipe the RESULT names, was run on the lab's
second machine — os <sworn r="path:papers/checksum/replication_alienware_env.json#/os" k="quote">`Windows-11-10.0.26200-SP0`</sworn>, torch <sworn r="path:papers/checksum/replication_alienware_env.json#/torch" k="quote">`2.5.1+cu121`</sworn>, quantized engine <sworn r="path:papers/checksum/replication_alienware_env.json#/torch_quantized_engine" k="quote">`x86`</sworn>, on cpu —
in an isolated checkout of the series. The outputs were kept beside the sworn bytes as
`replication_alienware_*` and never written over them; `replication_alienware_compare.json` lays
the two runs side by side, leaf by leaf, and this NOTE swears to those three files and to the
committed certs.

## what reproduced

- The null pair, exactly: <sworn r="path:papers/checksum/replication_alienware_smollm_quant_certs.json#/reloaded/distance/mean_abs_nats" k="numeric">A vs A′: mean |Δ log-prob| = 0 nats per token</sworn>, <sworn r="path:papers/checksum/replication_alienware_smollm_quant_certs.json#/reloaded/distance/verdict" k="quote">verdict `SAME`</sworn>.
- Every verdict: <sworn r="path:papers/checksum/replication_alienware_compare.json#/certs/verdicts/int8/replication" k="quote">the quantized arm reads `DRIFT`</sworn> and <sworn r="path:papers/checksum/replication_alienware_compare.json#/certs/verdicts/random/replication" k="quote">the random arm reads `DRIFT`</sworn>, as committed.
- The full-precision sanity count: <sworn r="path:papers/checksum/replication_alienware_smollm_quant_certs.json#/sanity/first_token_top1_hits/A" k="numeric">the full-precision model gets the first continuation token right on 30 canaries</sworn>, as committed; the random arm on <sworn r="path:papers/checksum/replication_alienware_smollm_quant_certs.json#/sanity/first_token_top1_hits/random" k="numeric">0</sworn>.
- The measured floor: <sworn r="path:papers/checksum/replication_alienware_smollm_quant_certs.json#/sanity/null_floor_nats" k="numeric">0</sworn> here and <sworn r="path:papers/checksum/smollm_quant_certs.json#/sanity/null_floor_nats" k="numeric">0</sworn> there — within a machine, a reload is bit-identical.

## what did not

Of the leaves the two certs files share (<sworn r="path:papers/checksum/replication_alienware_compare.json#/certs/leaves_in_common" k="numeric">72</sworn>), <sworn r="path:papers/checksum/replication_alienware_compare.json#/certs/leaves_moved_excluding_timestamps" k="numeric">22</sworn> carry a different value, timestamps excluded. Every one of them is a magnitude the RESULT swears to:

- The quantized arm's distance: committed <sworn r="path:papers/checksum/smollm_quant_certs.json#/int8/distance/mean_abs_nats" k="numeric">1.51 nats per token</sworn>, second machine <sworn r="path:papers/checksum/replication_alienware_smollm_quant_certs.json#/int8/distance/mean_abs_nats" k="numeric">1.41 nats per token</sworn>; its interval committed <sworn r="path:papers/checksum/smollm_quant_certs.json#/int8/distance/ci_mean_abs/0" k="numeric">from 1.17</sworn> <sworn r="path:papers/checksum/smollm_quant_certs.json#/int8/distance/ci_mean_abs/1" k="numeric">to 1.90</sworn>, second machine <sworn r="path:papers/checksum/replication_alienware_smollm_quant_certs.json#/int8/distance/ci_mean_abs/0" k="numeric">from 1.09</sworn> <sworn r="path:papers/checksum/replication_alienware_smollm_quant_certs.json#/int8/distance/ci_mean_abs/1" k="numeric">to 1.75</sworn>; its belief-geometry agreement committed <sworn r="path:papers/checksum/smollm_quant_certs.json#/int8/distance/rdm_r" k="numeric">0.83</sworn>, second machine <sworn r="path:papers/checksum/replication_alienware_smollm_quant_certs.json#/int8/distance/rdm_r" k="numeric">0.83</sworn>; its sanity count committed <sworn r="path:papers/checksum/smollm_quant_certs.json#/sanity/first_token_top1_hits/int8" k="numeric">16</sworn>, second machine <sworn r="path:papers/checksum/replication_alienware_smollm_quant_certs.json#/sanity/first_token_top1_hits/int8" k="numeric">17</sworn>.
- The random arm's distance: committed <sworn r="path:papers/checksum/smollm_quant_certs.json#/random/distance/mean_abs_nats" k="numeric">9.50 nats per token</sworn>, second machine <sworn r="path:papers/checksum/replication_alienware_smollm_quant_certs.json#/random/distance/mean_abs_nats" k="numeric">9.72 nats per token</sworn>; agreement committed <sworn r="path:papers/checksum/smollm_quant_certs.json#/random/distance/rdm_r" k="numeric">0.07</sworn>, second machine <sworn r="path:papers/checksum/replication_alienware_smollm_quant_certs.json#/random/distance/rdm_r" k="numeric">0.09</sworn>.

## where the difference comes from, arm by arm

The fingerprints say which step moved. The full-precision arm itself moved, barely: across the
two machines the per-item mean log-prob differs by at most <sworn r="path:papers/checksum/replication_alienware_compare.json#/arms/A/max_abs_diff_mean_lp" k="numeric">0.000017 nats per token</sworn> —
float32 BLAS order, under the instrument's resolution, which is why the null pair still reads SAME
and why "the floor is exactly zero" is a within-machine sentence: across machines the floor is
that number, not zero. The quantized arm moved by up to <sworn r="path:papers/checksum/replication_alienware_compare.json#/arms/Q/max_abs_diff_mean_lp" k="numeric">1.85 nats on one item</sworn>:
`torch.ao.quantization.quantize_dynamic` runs the per-tensor int8 kernels the machine's quantized
engine provides, and the recipe pins no engine. The random arm moved by up to <sworn r="path:papers/checksum/replication_alienware_compare.json#/arms/R/max_abs_diff_mean_lp" k="numeric">3.23 nats on one item</sworn>:
`from_config` under `torch.manual_seed(343)` draws its weights in an order the torch and
transformers versions decide, and the recipe records neither. The recipe at the head of the series
also writes a key the committed certs do not have (<sworn r="path:papers/checksum/replication_alienware_compare.json#/certs/keys_only_in_replication/0" k="quote">`coefficients_sha256`</sworn>):
the sworn bytes were produced by an earlier revision of the recipe than the one the RESULT points
a stranger at.

One reading rule the RESULT left unsaid: its "nats per token" is the mean over the items of each
item's per-token mean, every item weighted equally. The token-weighted pooled mean is lower —
committed <sworn r="path:papers/checksum/replication_alienware_compare.json#/aggregation/token_weighted_pooled_mean_abs/committed/int8" k="numeric">1.32</sworn>, second machine <sworn r="path:papers/checksum/replication_alienware_compare.json#/aggregation/token_weighted_pooled_mean_abs/replication/int8" k="numeric">1.25</sworn> —
and a reader who takes the unit as the aggregation gets a different number for no reason.

## what this settles

By the letter of `BOUNTY.md` as it stood this morning ("a number in a sworn RESULT that does not
reproduce from its own recipe on a second machine"), the lab owed itself a bounty by noon. The
number was never wrong: it is what that machine computed, the receipt binds it, and the verifier
holds it. What was wrong was the promise that the recipe reproduces it anywhere. So: a sworn
magnitude is covered only against a tolerance and an environment the RESULT, or a correction
beside it, states (`BOUNTY.md`, Tolerance); the checksum's cert now records the measured and the
effective floor, the seed and the written hashes of both fingerprints; the deploy-scale runner
records the versions, the engine and the device; and the sentence a RESULT owes its reader is not
"rebuild both" but "rebuild both, and expect the verdicts, not the digits".

Receipt history, stated because the law says it must be: the RESULT's receipt was written at
870a034a over certs from a 15:10 run, rewritten at c0bba384 over certs regenerated in place at
15:17, and rewritten again at 9da2da0c on this machine because c1751053, the commit it named,
existed only on the machine that built the series. From here a re-swear is a new dated receipt
file, and a regenerated certs file is a new file name; `tests/test_receipts_name_reachable_commits.py`
makes a ghost commit a CI failure.
