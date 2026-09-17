# CORRECTION — PREREG_checksum_beacon_draw_2026_09_14: what the beacon orders, what H1's pair clause can read, and five readings fixed before the run

**status: a withdrawal and a set of reading rules for a frozen document, written before any run and
to be sealed beside it (anchor 6b in `SEALS_2026_09_13.md`). The PREREG is not edited; its seal digest
is unchanged. Everything here follows the lab's own red team of 2026-09-14, whose findings were each
put to two skeptics before they were acted on. Nothing here reads a number from the run, which has
not started.**

## 1. withdrawn: the PREREG's claim that the drawn values could not precede the seal

The PREREG's "what a stranger checks", step 3, says the fingerprints' values under the draw "could
not have been computed before the slot existed". That is false, and the red team's blocker finding
on it was confirmed by both skeptics. A fingerprint scores each item independently of the others
(`styxx/checksum.py`, `fingerprint`), and the pool the beacon draws from is committed and public:
<sworn r="path:papers/checksum/beacon_draw_certs_dryrun_qwen0.5b.json#/canaries/draw/pool_size" k="numeric">778</sworn> items,
pinned by a test. Anyone — the lab included — can compute every item's value on every arm before
any seal and keep the 48 a beacon later names; the result is byte-identical to a run made after the
seal. A model can also be tuned against the whole pool.

What the beacon does order, and all it orders: WHICH 48 items are graded was fixed by a value no one
could choose, after the seal. The PREREG's question (c) — whether a stranger can check from the bytes
that the run came after the seal — is answered no: the bytes show that the selection came after the
seal, not the computation. The title's "canaries nobody could study for" is read as "canaries nobody
could choose". `SEALS_2026_09_13.md` repeated the claim ("that run cannot precede its seal") and is
corrected in the commit that adds this file; `styxx/beacon.py`'s docstring, which said the module
"removes the advance knowledge", was corrected before it. No hypothesis, band or gate depends on the
withdrawn sentence, and none changes.

## 2. H1's pair clause, measured before the run

H1 freezes the pair clause in the held-out form: A vs A′ is graded against the larger of the A–A″ and
A′–A″ distances, "the form CORRECTION_prereg_deploy_quant_H1 rule 5 promised". The red team argued,
and both skeptics agreed, that this escapes only the case where SAME was impossible: `distance()` reads
SAME only when the pair's bootstrap upper bound lies below the floor, and the held-out floor is a point
estimate of a quantity distributed exactly like the pair's own mean. `probe_h1_held_out.py` measures
it with the runner's exact rule, N_BOOT and SEED, on the synthetic construction of
`probe_h1_by_construction.py`, <sworn r="path:papers/checksum/probe_h1_held_out_result.json#/n_trials_per_jitter" k="numeric">60</sworn>
trials per jitter:

- bit-identical loads (jitter 0): SAME in <sworn r="path:papers/checksum/probe_h1_held_out_result.json#/jitters/0/held_out/SAME" k="numeric">60</sworn> trials.
- jitter 1e-4: SAME in <sworn r="path:papers/checksum/probe_h1_held_out_result.json#/jitters/0.0001/held_out/SAME" k="numeric">9</sworn>,
  INCONCLUSIVE in <sworn r="path:papers/checksum/probe_h1_held_out_result.json#/jitters/0.0001/held_out/INCONCLUSIVE" k="numeric">51</sworn>.
- jitter 3e-4: SAME in <sworn r="path:papers/checksum/probe_h1_held_out_result.json#/jitters/0.0003/held_out/SAME" k="numeric">9</sworn>,
  INCONCLUSIVE in <sworn r="path:papers/checksum/probe_h1_held_out_result.json#/jitters/0.0003/held_out/INCONCLUSIVE" k="numeric">50</sworn>,
  DRIFT in <sworn r="path:papers/checksum/probe_h1_held_out_result.json#/jitters/0.0003/held_out/DRIFT" k="numeric">1</sworn>.
- jitter 1e-3: SAME in <sworn r="path:papers/checksum/probe_h1_held_out_result.json#/jitters/0.001/held_out/SAME" k="numeric">12</sworn>,
  INCONCLUSIVE in <sworn r="path:papers/checksum/probe_h1_held_out_result.json#/jitters/0.001/held_out/INCONCLUSIVE" k="numeric">47</sworn>,
  DRIFT in <sworn r="path:papers/checksum/probe_h1_held_out_result.json#/jitters/0.001/held_out/DRIFT" k="numeric">1</sworn>.
- jitter 5e-3: SAME in <sworn r="path:papers/checksum/probe_h1_held_out_result.json#/jitters/0.005/held_out/SAME" k="numeric">11</sworn>,
  INCONCLUSIVE in <sworn r="path:papers/checksum/probe_h1_held_out_result.json#/jitters/0.005/held_out/INCONCLUSIVE" k="numeric">49</sworn>.

On the same draws the worst-pairwise form of the 2026-09-13 PREREG read SAME in
<sworn r="path:papers/checksum/probe_h1_held_out_result.json#/jitters/0.001/worst_pairwise/SAME" k="numeric">12</sworn>
of the trials at 1e-3, and it read DRIFT in no trial at any scale. The held-out form is no better at
reading SAME on identical weights, and it read DRIFT on them, which the worst-pairwise form did not.
Three loads give three
exchangeable pair distances; no rule that grades one pair's interval against the other two's point
estimates reads SAME reliably unless the loads are bit-identical.

The reading rule, fixed before the run:

1. H1's floor clause is evaluated as frozen, from `sanity.null_floor_nats`.
2. If the floor is zero, the pair clause is evaluated as frozen; the probe expects SAME.
3. If the floor is above zero and at most 1e-3 and the pair reads INCONCLUSIVE, the RESULT records
   **H1: HELD on the floor, INCONCLUSIVE on the pair by construction**, citing this correction. That is
   neither HELD nor FAILED for H1 as written, and the RESULT says so in those words.
4. If the floor is above zero and the pair reads DRIFT, H1 is FAILED as written, and the RESULT
   reports beside it that the probe saw this rule read DRIFT on identical weights at two of the four
   nonzero scales.
5. `papers/checksum/score.py` applies rules 1 to 4 and prints the status of rule 3 in those words.
6. The next PREREG on this instrument does not grade the null pair by a bootstrap bound against a point
   floor. It states H1 as a claim about the floor alone (bit-identical loads, or a floor at most a
   stated bound), or grades the pair against a floor with its own interval.

## 3. the seal is the earliest memo the creator wallet signed

The PREREG defines the beacon as the slot of "the earliest confirmed memo carrying that digest from the
creator wallet". That rule stands. `styxx.clock` enforced something weaker until commit `7ef0e7ac`: its
scan counted every memo the wallet's listing showed carrying the digest, and the listing includes
transfers another key sent the wallet, so anyone who read the digest in `SEALS_2026_09_13.md` could
have sent the wallet a memo carrying it first and made the lab's seal read EARLIER_MEMO_EXISTS forever,
which K6 would turn into a run that never starts. Both skeptics confirmed it. K6 is read under the
module as corrected: a candidate counts only if the wallet signed it, it succeeded, and one of its memo
instructions is exactly the seal memo; a candidate that does not count is listed as foreign and never
moves the beacon; `clock verify` prints a beacon only on a line that reads ANCHORED. What the correction
cannot remove: a stranger who floods the wallet with more memos carrying the digest than the scan
resolves makes the seal read EARLIEST_UNKNOWN, K6 holds the run, and the RESULT would say who sent
them. A flood can delay the run; it cannot choose the beacon.

## 4. readings fixed before the run

- **K4.** "A vs R inside H2's band" is read as A vs R's mean |Δ log-prob| inside H2's mean band,
  inclusive. It is the only band in H2 stated in nats.
- **A floor above 1e-3 and at most 1e-2.** H1's floor clause fails, K1 does not fire, and H2 to H6 are
  evaluated. The PREREG does not say this; it follows from the two thresholds, and the RESULT says it.
- **H5's "same machine".** Read as the same `provenance.cuda_device` in both certs, which names a device
  model and not a machine; the RESULT names the machine in prose. It is not checkable from the bytes.
- **PENDING does not expire.** Every RESULT on this PREREG lists each hypothesis still PENDING, with its
  date, and a later RESULT that evaluates one cites this correction.
- **The reading of record** is `papers/checksum/score.py` at the commit that carries this correction
  (scorecard schema `styxx.checksum/scorecard/v2`). Beyond the PREREG's K5 it re-derives K1 from the
  recorded floor, compares `provenance.prereg_blob_sha256` with the sealed digest without being asked,
  requires a draw of exactly 48, and requires every arm's cert to grade the drawn set. The runner now
  refuses the experiment when the PREREG at HEAD is not the sealed text or the commit is not clean. The
  RESULT swears to the scorecard.

## what this does not change

The model, the pool, the draw, the beacon rule, the arms, the probe, H2 to H6 and their bands, and
gates K1 to K6 stand as frozen. The PREREG's seal is unchanged. This file is sealed on its own (6b) and
the run does not start before both 6 and 6b read ANCHORED, as the 2026-09-13 correction was sealed
beside its PREREG.
