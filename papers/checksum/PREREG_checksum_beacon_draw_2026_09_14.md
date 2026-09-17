# PREREG — checksum on canaries nobody could study for: the 48 drawn from the committed pool by the block hash of the slot that sealed this document, Qwen2.5-1.5B, GPU

**frozen 2026-09-14, before any run. seal: the sha256 of this file's git blob, anchored from the
creator wallet with a memo built by `python -m styxx.clock memo sealed-prereg <digest>`. the
beacon is the block hash of the slot in which the EARLIEST confirmed memo carrying that digest
landed, as `python -m styxx.clock verify` prints it on a line that reads ANCHORED; the run does
not start before that line exists. the RESULT reports every number below whatever it says.**

## question

The 2026-09-13 PREREG froze 48 hand-written canaries that anyone could read in the repository
before the run (`DUE_DILIGENCE_2026_09_13.md` §3 named that first). This PREREG asks the same
questions of a canary set that did not exist when anyone could have prepared for it: (a) does the
checksum read the null pair as SAME, a real 4-bit method as DRIFT in a predicted band, int8 as a
gentler DRIFT, and random weights as far — on 48 items chosen by a value the network produced
after the seal was out of the lab's hands; (b) does the drawn set read what the hand-written set
read on the same weights; and (c) can a stranger check from the bytes alone that the run came
after the seal.

## design, fixed

- model: `Qwen/Qwen2.5-1.5B`, as in the 2026-09-13 PREREG, so that (b) is a comparison on the
  same weights and the same machine.
- pool: `styxx.beacon.POOL` at the commit that adds this file — the 48 hand-written items plus
  template items with one-token answers (sums, products, differences, capitals, days, months,
  opposites, counting), <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/canaries/draw/pool_size" k="numeric">778</sworn> items, pool sha256
  <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/canaries/draw/pool_sha256" k="quote">`9e450999977a274fe63f1f7358378a7b42b710c54572dbc0e72bd2b45ab9906f`</sworn>.
  A draw whose record names any other pool hash is not this experiment.
- draw: `styxx.beacon.draw(beacon, 48)` — `seed_i = sha256(pool_sha256 || beacon || i)`, index
  `seed_i mod 778`, duplicates skipped, until 48 items; pure sha256, no PRNG.
  `python -m styxx.beacon <beacon> 48` prints the ids and the canary hash a stranger reproduces.
- beacon: the block hash of the slot of the earliest confirmed memo carrying this file's blob
  digest from the creator wallet, decoded from base58 to 64 hex by `styxx.clock.blockhash_to_beacon`.
  A later memo carrying the same digest does not move the beacon (`SEALS_2026_09_13.md`, the
  earliest-memo rule; `clock.verify` reads `EARLIER_MEMO_EXISTS` for a record that is not the
  earliest). The signer chooses when to submit; the signer does not choose a slot's hash.
- the record travels: every fingerprint carries the draw record (`styxx.beacon/draw/v0`: pool
  hash, pool size, beacon, n, canary hash); every cert digests it (`styxx.checksum/compare/v2`);
  `checksum.check_draw_record` re-runs the beacon against the pool and refuses a record whose
  beacon does not produce exactly the items it names.
- arms, probe, distance: exactly the 2026-09-13 design — **A** bf16, **A′** and **A″** bf16
  reloaded, **Q4** bitsandbytes NF4 (double-quant, bf16 compute), **Q8** bitsandbytes LLM.int8,
  **R** random init seed 343; teacher-forced, greedy; `torch.use_deterministic_algorithms` best
  effort and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; `styxx.checksum.distance(n_boot=2000,
  seed=20260913)`; first-token top-1 hits per arm.
- floors, two of them, both written by the runner: the **worst pairwise** floor among A, A′, A″
  (`sanity.null_floor_nats`) gates K1 and grades Q4, Q8 and R; the **held-out** floor — the
  larger of A–A″ and A′–A″, or the instrument's 1e-4 resolution when that is larger — grades
  A vs A′ (`h1_held_out.cert`, written with `unpreregistered: false` under this PREREG). This is
  the form `CORRECTION_prereg_deploy_quant_H1_2026_09_13.md` rule 5 promised: the graded pair is
  not inside the floor that grades it.
- command: `python papers/checksum/run_deploy_quant.py --prereg beacon_draw --beacon <64 hex>`,
  untagged, on the commit that carries this file sealed. The certs carry `provenance.prereg` =
  this file, `provenance.prereg_blob_sha256` = the sealed digest, `canaries.draw` = the record,
  `is_the_experiment: true`. A run under any other beacon, any other model or a `--tag` is an
  instrument check and never this experiment.

## what the instrument check showed, and why the bands below differ from the hand set's

The recipe ran on `Qwen/Qwen2.5-0.5B` under a test beacon (the sha256 of a sentence, not a block
hash; tagged `_dryrun_qwen0.5b_beacon`; `is_the_experiment` false) at commit `8b2df603`, so that
the bands here are written from a drawn set and not from the hand set. That draw took
<sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/canaries/n" k="numeric">48</sworn> items of which one was hand-written and the rest template
(24 sums, 7 differences, 7 products, 5 opposites, 2 counting, 1 capital, 1 day). Read from
`deploy_quant_certs_dryrun_qwen0.5b_beacon.json`:

- three bf16 loads bit-identical: null floor <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/sanity/null_floor_nats" k="numeric">0.000000</sworn> nats per token;
  A vs A′ under the held-out floor <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/h1_held_out/cert/distance/verdict" k="quote">`SAME`</sworn>.
- NF4: <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/Q4/distance/verdict" k="quote">`DRIFT`</sworn>, mean |Δ log-prob| <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/Q4/distance/mean_abs_nats" k="numeric">0.347</sworn> nats per token,
  interval from <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/Q4/distance/ci_mean_abs/0" k="numeric">0.211</sworn> to <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/Q4/distance/ci_mean_abs/1" k="numeric">0.533</sworn>,
  geometry r <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/Q4/distance/rdm_r" k="numeric">0.969</sworn>, first-token argmax lost on
  <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/sanity/top1_loss_vs_A/Q4" k="numeric">15</sworn> of 48 items. On the hand set, same model, same day
  (`deploy_quant_certs_dryrun_qwen0.5b.json`): mean <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b.json#/Q4/distance/mean_abs_nats" k="numeric">0.427</sworn>, argmax lost on
  <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b.json#/sanity/top1_loss_vs_A/Q4" k="numeric">4</sworn> of 48. Template items have one right token and NF4 flips
  it far more often than it flips the hand items' continuations; the magnitude is of the same
  order.
- int8: <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/Q8/distance/verdict" k="quote">`DRIFT`</sworn>, mean <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/Q8/distance/mean_abs_nats" k="numeric">0.121</sworn>,
  r <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/Q8/distance/rdm_r" k="numeric">0.997</sworn>, argmax lost on <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/sanity/top1_loss_vs_A/Q8" k="numeric">2</sworn> of 48.
- random weights: mean <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/R/distance/mean_abs_nats" k="numeric">10.13</sworn>, r
  <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/R/distance/rdm_r" k="numeric">0.35</sworn> — against
  <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b.json#/R/distance/rdm_r" k="numeric">0.11</sworn> on the hand set: arithmetic items share tokens
  with each other, and a random model's geometry reflects that shared surface. Top-1 hits
  <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/sanity/first_token_top1_hits/R" k="numeric">0</sworn> of 48, against
  <sworn r="path:papers/checksum/deploy_quant_certs_dryrun_qwen0.5b_beacon.json#/sanity/first_token_top1_hits/A" k="numeric">34</sworn> for the model.

So the bands below are wider than the 2026-09-13 PREREG's on argmax loss and on the far control's
r, for a reason the check shows, and they were written for a model three times larger, which the
check did not run: a prediction, not a fit.

## hypotheses and predicted bands (written before data)

- **H1 (floor, held-out form).** The worst-pairwise null floor is ≤ 1e-3 nats/token; zero is
  allowed and expected (the 0.5B loads were bit-identical). A vs A′ reads SAME against the
  held-out floor, read from `h1_held_out.cert.distance.verdict`. There is no "floor > 0" clause.
- **H2 (NF4 on the drawn set).** A vs Q4 reads DRIFT; mean |Δ log-prob| in **[0.05, 0.60]
  nats/token**; geometry r ≥ 0.90; argmax lost on ≤ 24 of 48.
- **H3 (int8).** A vs Q8 reads DRIFT with mean |Δ log-prob| below Q4's; r ≥ 0.98; argmax lost on
  ≤ 8 of 48.
- **H4 (far control).** A vs R: mean |Δ log-prob| > 5 nats/token; r < 0.6; top-1 hits ≤ 4 of 48.
- **H5 (the drawn set against the hand set).** Evaluated only if the 2026-09-13 experiment's certs
  (`deploy_quant_certs.json`, `is_the_experiment: true`, same model, same machine as this run)
  exist at the RESULT's commit: every arm's verdict agrees between the two sets (A′ SAME under the
  held-out reading on both, Q4, Q8 and R DRIFT on both), and the drawn set's NF4 mean lies within
  a factor of two of the hand set's, either way. Otherwise H5 is **PENDING**, the RESULT says so,
  and a later RESULT evaluates it against this text.
- **H6 (portability of this run).** Evaluated when a second machine's certs under this beacon
  exist, by `styxx.portability` with this machine's run as the base arm: verdicts AGREE on every
  arm; Q4's and Q8's mean |Δ log-prob| move by at most
  <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13_v1.json#/arms/int8/numbers/mean_abs_nats/max_abs_diff" k="numeric">0.10</sworn> nats/token and R's by at most
  <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13_v1.json#/arms/random/numbers/mean_abs_nats/max_abs_diff" k="numeric">0.22</sworn> — the two-machine tolerances
  measured on a CPU int8 recipe and a different model (`RESULT_portability_smollm_quant_2026_09_13.md`).
  The hypothesis is that those numbers transfer to GPU bitsandbytes arms; it may fail, and a
  failure is a measured tolerance for this recipe, not a defect. **PENDING** until a second
  machine runs.

## kill gates

- **K1.** Worst-pairwise null floor > 1e-2 nats/token → the serving is not deterministic enough
  for this probe; the run is INCONCLUSIVE and says so; no other hypothesis is evaluated
  (evaluated in code; the fingerprints are still written).
- **K2.** A vs Q4 reads SAME → the drawn set cannot see a deployed quantization; the RESULT is
  titled with the kill.
- **K3.** A vs Q4 > 1.5 nats/token, or argmax lost on > 36 of 48 → the quantization pipeline, not
  the model, is suspect (compare against a published NF4 perplexity delta before believing it).
  The thresholds are wider than the hand set's 1.0 and 12 because the check above lost 15 of 48.
- **K4.** A vs R inside H2's band → the distance cannot tell a random model from a quantized one;
  the metric is broken.
- **K5 (validity).** Any of: `checksum.check_draw_record` refuses the certs' draw record; the
  certs' `canaries.draw.beacon` is not the beacon `python -m styxx.clock verify` prints ANCHORED
  for the earliest memo carrying this file's digest; `provenance.prereg_blob_sha256` is not that
  digest; `is_the_experiment` is false → the run is not this experiment; nothing is evaluated;
  the files stay as an instrument check under whatever tag they carry.
- **K6 (precondition).** `clock verify` reads anything but ANCHORED for this file's seal line
  (EARLIER_MEMO_EXISTS, EARLIEST_UNKNOWN, BEACON_UNAVAILABLE, or any failed check) → the run does
  not start. If it started anyway, K5.

## what a stranger checks, in order

1. `git show <commit>:papers/checksum/PREREG_checksum_beacon_draw_2026_09_14.md | sha256sum`
   equals the digest in the memo; the memo is from the creator wallet; the transaction is the
   earliest carrying it (`python -m styxx.clock verify papers/charon/anchors.jsonl`).
2. `python -m styxx.beacon <that line's beacon> 48` prints the ids and canary hash the certs and
   every fingerprint carry, and `checksum.check_draw_record` accepts the record.
3. The fingerprints' `mean_lp` under that draw could not have been computed before the slot
   existed: nothing in the tree, and nothing the lab could have written, named those 48 items
   before the network chose them. That is the ordering the 2026-09-13 seal did not carry
   (`SEALS_2026_09_13.md`, "the rule").
4. The RESULT's sworn spans bind every number here to `beacon_draw_certs.json` at its commit.

## what is not claimed

A threshold for "meaningful" drift. Anything about API models or sampled serving. Anything
beyond this one model, these two quantizations and this pool. That the template pool is hard —
it is dull by design; fixedness and breadth are the point. That H2's band, written from a 0.5B
check, is tight: it is the lab's prediction for a model it did not check. That a seal orders
anything but this run: the 2026-09-13 run's canaries were frozen by hand, and its seal proves
only that its PREREG existed before its block.

## outputs

`papers/checksum/beacon_draw_certs.json`, `beacon_draw_fingerprints.json`,
`beacon_draw_plates.png`, and a sworn `RESULT_checksum_beacon_draw_<date>.md` binding every
number above to those files at the commit that carries them, with H5 and H6 marked PENDING where
they are.
