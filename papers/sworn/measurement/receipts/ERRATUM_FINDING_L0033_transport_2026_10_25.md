# Erratum beside `FINDING_L0033_transport_2026_10_24.md` (2026-10-25)

`FINDING_L0033_transport_2026_10_24.md` in this folder is the lead's finding of 2026-10-24, committed byte for byte
(sha256 `2fe2ff030055b3a392e990115c6348846ff62c22b4f43e0eac2a8fb56b25831d`). It is not edited: the tranche A RESULT and
its drafts name it by that sha256, and a receipt is history. This file corrects three statements in it and records
that its proposed route for tranche B was not the one taken.

**1. "(after the runner's two retries)".** The bytes show less. `seat_claude.cli` makes up to three attempts
(`RETRIES = 2`, l.39; `for _ in range(1 + RETRIES)`, l.73), overwrites its reason on every failed attempt (l.80,
l.89) and returns only the last one (l.91). So each of the three seat files records that the last of three attempts
failed with the class "the transport's bytes were not the JSON envelope the transport promises" (l.49); the reasons
of the two earlier attempts were not kept.

**2. "Cause (measured)".** Part of it is receipted and part is the lead's account. Receipted: L-0033 is the one
Panel L item whose text begins with "-" (`tranche_A_packet_facts.json`, this folder), and `prompt_for` passes a
Panel L item's text verbatim as the value of `-p` (`seat_claude.py` l.155-157, l.68-69). The lead's account: that the
claude CLI reads an argument beginning with "-" as an option, so the call returned no JSON envelope. No kept byte
shows that mechanism: the three calls' output was not kept (each ledger row has `raw_sha256` null,
`seat_outputs/claude/ledger.jsonl` lines 34, 94 and 154), and no call was re-made.

**3. "Length is not the cause (L-0033 is 2,199 JSON chars; L-0008, 2,918, passed)".** Those counts are of each
packet item serialized as a JSON object, id included; the texts themselves are 2,130 and 2,853 characters
(`tranche_A_packet_facts.json`). One
comparison shows that length alone does not explain the failure, not that length plays no part.

**Not an error, superseded.** The finding proposed a dated amendment for tranche B's leading-dash route. On
2026-10-25 the operator chose instead that tranche B runs on a fixed transport under a new preregistration, with this
route and the twins' line-feed route both in it (the tranche A RESULT records the decision).

**Found by** the independent review of the RESULT draft on 2026-10-24 (its finding m1, and the draft's own
correction of "Cause (measured)"); this erratum is the form that fix takes for a committed receipt.
