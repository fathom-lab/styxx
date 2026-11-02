# BOUNTY: paused

**Paused 2026-11-02. No bounty is offered, and none is paid, while this notice stands.** Rules v0
of 2026-09-13 are withdrawn as an offer of payment; they stay readable with
`git show 310df453:BOUNTY.md`.

## Why

- **No amount was ever set.** Every payout in rules v0 read `[amount]`, the reply window `[days]`,
  and the one bounty already owed `Paid: [tx]`. The file has been on main since #87 merged it
  (2026-09-17 UTC), with its own line "Fill the bracketed amounts before this file is public" unmet.
- **The funding source is gone.** Rules v0 paid from the lab's share of the creator fees of the
  $STYXX mint `93ihpGjLVnhghXciSeFovXwKF762rwcigW12kSQBpump`, which
  [`docs/token/README.md`](docs/token/README.md) lists as discontinued 2026-09-15. The current
  contract has creator tax 0 and no creator claim; its creator fees go to all holders pro-rata, so
  there is no lab share of creator fees for rules v0 to pay from.

A page that offers money with no amount and no source of funds is not an offer, so it is
withdrawn rather than left standing.

## What still counts

None of this carries money. Each is still recorded, credited and answered.

- **Replications.** A re-run of a target in [REPLICATIONS.md](REPLICATIONS.md), matching or not,
  earns the ledger row and the named credit that file states; a verdict flip that survives our
  re-run earns co-credit on the correction (REPLICATIONS.md, Credit).
- **Challenge records.** A record from `python -m styxx.challenge` (see
  [`papers/plates/SAND_CHECK.md`](papers/plates/SAND_CHECK.md)) is a challenge when the verifier
  says it disagrees. Open an issue with the record attached; the lab answers there with EXPLAINED
  (an environment difference, documented in the issue) or DISPUTED (a sworn NOTE the community
  can re-run). A dispute never closes an issue; a re-run does.
- **Tolerance** (kept from rules v0, because `papers/checksum/STRANGER.md` and
  `papers/plates/SAND_CHECK.md` cite it). A sworn RESULT number is challenged only against the
  tolerance and the environment that the RESULT, or a correction committed beside it, states. A
  RESULT that states neither is not covered until a correction does. The lab's own second-machine
  re-run of `RESULT_checksum_smollm_quant_2026_09_13` moved every int8 and random-arm number
  (`papers/checksum/NOTE_replication_alienware_2026_09_13.md`); that is why this rule exists.
- **Not covered.** Anything about price. Anything that requires the lab to trade. Anything about a
  document at a commit it does not name.

## Owed before the pause

#1: the wrong gold answer in `bench/tasks/reasoning.jsonl` (reas-021), reported 2026-09-02 by
stacc and fixed in 25a2617e (gold Wednesday to Tuesday). The repository records no payment, and no
amount was ever set. The pause does not cancel it; how it is settled is the operator's decision.
