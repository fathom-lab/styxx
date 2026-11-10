# BOUNTY: the rules for whoever proves us wrong

**Rules v0, 2026-09-13. Payment terms withdrawn 2026-11-10.** No payment is offered under these
rules today. The amounts were never set. The rules below still say what counts as a challenge and
what the lab must then do. The number of days in the Process rule was never set.

- **What qualifies.** See `papers/plates/SAND_CHECK.md` §"what earns a bounty". The verifier
  decides whether a record is a challenge; the lab does not decide that. Where
  `papers/plates/SAND_CHECK.md` or `papers/checksum/STRANGER.md` says a challenge is paid, this
  file governs. None is paid today.
- **Table.** reproduction failure of a sworn RESULT number. challenge record with
  `agree: false, same_build: true` the lab cannot explain. a sworn span HELD but false
  about the world. a bug in `styxx.sworn`, `styxx.charon`, `styxx.checksum` or
  `styxx.clock` that changes a verdict. earliest external replication of an arc (a
  `REPLICATIONS.md` line with a record).
- **Tolerance.** A sworn RESULT number is covered by the first row only against the tolerance and
  the environment that the RESULT, or a correction committed beside it, states. A RESULT that
  states neither is not covered by that row until a correction does. The lab's own second-machine
  re-run of `RESULT_checksum_smollm_quant_2026_09_13` moved every int8 and random-arm number
  (`papers/checksum/NOTE_replication_alienware_2026_09_13.md`); that is why this line exists.
- **Process.** Open the issue with the record attached. The lab replies within [days] with one of:
  UPHELD (the lab says so in the issue), EXPLAINED (environment difference, documented in the
  issue), or DISPUTED (the lab publishes a sworn NOTE and the community can re-run). A dispute
  never closes an issue; a re-run does.
- **Already upheld.** #1: the wrong gold answer in `bench/tasks/reasoning.jsonl` (reas-021),
  reported 2026-09-02 by stacc. Fixed in 25a2617e (gold Wednesday → Tuesday); no committed receipt,
  certificate, capsule or charon line cites that file, so nothing else moved. This file listed it
  as owed before 2026-11-10. No amount was set for it and this file records no payment.
- **Not covered.** Anything about a document at a commit it does not name.
