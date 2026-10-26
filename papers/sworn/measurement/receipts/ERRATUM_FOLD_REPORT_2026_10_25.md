# Erratum beside `FOLD_REPORT.md` (2026-10-25)

`FOLD_REPORT.md` in this folder is the lead's report of tranche A's fold, committed byte for byte (sha256
`a19a6a5c605628e39a3a8034aae5bf1e800666d01b90573b6da62c2cd4f85e5e`). It is not edited: the tranche A RESULT and its
drafts name it by that sha256, and a receipt is history. This file corrects one sentence in it.

**The sentence.** l.201-202: "Whether L-0033 is a decoy or a document window is in the sealed key and was not read
or derived here."

**What is wrong.** "is in the sealed key". Each Panel L decoy's text and side are committed in
`papers/sworn/measurement/decoys_L.json` (added in `7a8a0729`, 2026-09-05; locked at blob `75a030fd`, PREREG l.578),
and `build_packets.py` makes each decoy's text its packet item's text (l.204-207). Counted, with no id and no side:
30 of `packet_L.json`'s 59 item texts equal a committed decoy text (`tranche_A_decoy_text_count.json`, this folder).
So whether L-0033 is a decoy, and of which side, is decided by committed bytes (`decoys_L.json` against
`packet_L.json#/items/32/text`), not by the sealed key alone.

**What stands.** "was not read or derived here": the fold did not read or derive it, and neither does the tranche A
RESULT, which leaves it unprinted (the lead's default D4). Everything else in the paragraph stands: L-0033's Claude
family label is `NO-MAJORITY` by the PREREG's rule, and the verdict string cannot move with it, because G_F_L needs
both families and the local family is at 14 of 30 on Panel L.

**Found by** the independent review of the RESULT draft on 2026-10-24 (its finding M1); the draft was fixed the same
day, and this erratum is the form that fix takes for a committed receipt.
