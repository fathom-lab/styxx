# BOUNTY — paid in $STYXX to whoever proves us wrong

**Rules v0, 2026-09-13.** Fill the bracketed amounts before this file is public.

- **Funding.** The lab's share of the creator fees of the $STYXX token (mint
  `93ihpGjLVnhghXciSeFovXwKF762rwcigW12kSQBpump`). Half of creator fees go to MIRI regardless.
- **What qualifies.** See `papers/plates/SAND_CHECK.md` §"what earns a bounty". The verifier
  decides whether a record is a challenge; the lab decides only the amount, from the table below.
- **Table.** reproduction failure of a sworn RESULT number: [amount]. challenge record with
  `agree: false, same_build: true` the lab cannot explain: [amount]. a sworn span HELD but false
  about the world: [amount]. a bug in `styxx.sworn`, `styxx.charon`, `styxx.checksum` or
  `styxx.clock` that changes a verdict: [amount]. earliest external replication of an arc (a
  `REPLICATIONS.md` line with a record): [amount].
- **Tolerance.** A sworn RESULT number is covered by the first row only against the tolerance and
  the environment that the RESULT, or a correction committed beside it, states. A RESULT that
  states neither is not covered by that row until a correction does. The lab's own second-machine
  re-run of `RESULT_checksum_smollm_quant_2026_09_13` moved every int8 and random-arm number
  (`papers/checksum/NOTE_replication_alienware_2026_09_13.md`); that is why this line exists.
- **Process.** Open the issue with the record attached. The lab replies within [days] with one of:
  PAID (tx signature in the issue), EXPLAINED (environment difference, documented in the issue),
  or DISPUTED (the lab publishes a sworn NOTE and the community can re-run). A dispute never
  closes an issue; a re-run does.
- **Already owed.** #1 — the wrong gold answer in `bench/tasks/reasoning.jsonl` (reas-021),
  reported 2026-09-02 by stacc. Fixed in 25a2617e (gold Wednesday → Tuesday); no committed receipt,
  certificate, capsule or charon line cites that file, so nothing else moved. Paid: [tx].
- **Not covered.** Anything about price. Anything that requires the lab to trade. Anything about a
  document at a commit it does not name.
