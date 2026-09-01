# PREREG v2 — the extraction ceiling, with a reliability gate that exists

Fathom Lab · 2026-09-01 · Frozen after the decoy set was built and hashed, and **before any seat
is prompted**. Successor to `PREREG_extraction_ceiling_2026_09_01.md` (frozen at `38ab585`,
voided by `ADDENDUM_extraction_ceiling_gate_unsatisfiable_2026_09_01.md`). **Neither predecessor
is edited.** A receipt is history and so is a preregistration.

**The standing commitment:** do not ship an accusing verdict whose precision has not been
measured by a blind panel. Absence of evidence is never a contradiction. Never "first". Never
"nobody". Always "we know of no other."

---

## What changed, and only what changed

v1 froze `G-E1` on an assumption about `v14_packet.json` that turned out to be false: that its
30 sealed decoys split 15 claim / 15 not-a-claim. All 30 are claims. The panel was voided and no
E was reported.

**Everything else in v1 stands unamended and is inherited by reference**: the decomposition
`P = E x A`, the substituted question, the 100 sampled accusations, the exact item-by-item
computation of E and A against the committed key, `G-E2` reconciliation, `G-E3`'s bars, `G-E4`'s
no-regeneration rule, and every clause of *Honest limits*. This document replaces `G-E1` and
adds the disclosures the new decoy set requires. Nothing else.

---

## The decoy set

Built by `extraction_decoys.py`, emitted as `extraction_decoys.json`, hashed into
`extraction_decoys_digest.txt`, and **committed and pushed before this document was frozen**.

| side | n | construction |
|---|---|---|
| CLAIM | 15 | taken **unmodified** from `v14_packet.json`'s `decoy_verified` — real claims the gate verified. Nothing synthesised. |
| NOT-A-CLAIM | 15 | synthesised from real VERIFIED path claims drawn from the **DEVELOPMENT split only**, by three frames in a committed 5/5/5 ratio |

The three frames, fixed before the draw:

  * **negation** — *"To be explicit about scope: this change does not touch `P`. That file is
    unchanged here and stays exactly as it is on main."*
  * **quotation** — *"For reference, the linked issue asks for the following: `"<the original
    claim>"` That work is tracked separately and is not part of this pull request."*
  * **comparative reference** — *"The approach here follows the same pattern `P` already uses,
    which was settled in an earlier pull request. Nothing in this change modifies that file."*

**The DEVELOPMENT restriction is `SPLIT_external_corpus_2026_08_31.md` rule 1**: held-out prose
is not consumed to build the instrument that scores held-out prose. The development population
came to **15,617 PRs / 3,762 verified path claims**, and the PR count reconciles exactly with
`v14_gates.json` `development_bucket.prs = 15617`.

**The perturbation is the same SHAPE the packet already ships.** `decoy_synthetic_contradiction`
was manufactured by a stated string transform of a real item (prefix the path with `zz_`). This
side is manufactured by a stated string transform of a real item too — on the speech act instead
of the path. This decoy side is no more synthetic than the one already sealed.

---

## G-E1, restated — and now two-sided

**G-E1a — overall reliability.** Panel accuracy across all 30 decoys **>= 27/30**, as in v1.

**G-E1b — neither class may be carried by the other.** Accuracy **on each side separately must
exceed chance**: at least **9 of 15** correct on the CLAIM side *and* at least **9 of 15** on the
NOT-A-CLAIM side. A panel whose answer is stuck on one word now fails on the class it agrees
with as well as the class it does not — which is precisely what v1's all-positive decoy set
could not do.

**Either sub-gate failing voids the panel and no E is reported.** Non-negotiable, and a void
panel is published as a void panel. This has already happened twice today; it is not a
formality.

---

## Authorship, disclosed rather than buried

**The NOT-A-CLAIM side is written by us.** The addendum conceded this in advance
("authorship is judgment") and it does not stop being true because the frames are stated. What
the construction buys is not neutrality — it is that the transform is declared before the draw,
mechanical given the source sentence, applied at a fixed ratio, and auditable line by line by
anyone who opens `extraction_decoys.py`.

**A specific failure mode this creates, named now:** if the frames are too easy, `G-E1` is passed
by a panel that would not survive a real ambiguous case, and E inherits that looseness. The
decoys are a *reliability* check, not a difficulty benchmark — the same standing the packet's own
sealed decoys have, which were also unambiguous by construction ("must read as contradicted").
Any result quoting E must carry this paragraph.

**What would settle it properly**, and is not claimed here: NOT-A-CLAIM items found in the wild
and adjudicated by a party with no stake in the instrument. We know of no such corpus and did
not build one today.

---

## What is unchanged and must not be re-read as new

E is still **UNMEASURED** at the time of freezing. No panel has been prompted. No seat has seen
an item. The `tests_pass` census counts remain **CONTAINMENT** figures — where a match sits, not
whether extraction was wrong — and the 1,234 bare-assertion count still may not be divided by
5,514. Nothing in this document licenses a number.
