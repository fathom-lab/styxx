# ADDENDUM — the extraction-ceiling panel was not run, because its own reliability gate cannot be built

Fathom Lab · 2026-09-01 · Written after opening `v14_packet.json` and before any panel was
convened, any seat was prompted, or any value of **E** existed. Amends
`PREREG_extraction_ceiling_2026_09_01.md`, which is frozen and pushed and is **not edited by
this document**. A receipt is history; so is a preregistration.

**No value of E is reported here, and none was computed.** Nothing below is a finding about
extraction. It is a finding about the preregistration.

---

## What was established before the defect appeared

Two things were checked on opening the packet, and both hold.

**G-E2 — reconciliation. PASS.** The packet is 130 items keyed by `v14_key_SEALED.json`:
100 `sampled accusation`, 15 `decoy_verified`, 15 `decoy_synthetic_contradiction`. Scoring the
committed answers in `v14_answers.json` against the accusation subset gives **16 upheld of 100
scored = 0.1600**, which re-derives the published 0.16 exactly. The join named in the
preregistration is sound and the item identifiers line up with no gaps.

**The packet's question is the one the preregistration assumed.** `v14_packet.json`
`instructions` reads: *"For each item answer SUPPORTED or CONTRADICTED: do the PR's changed
files support the claim, or contradict it? You are not told the gate's verdict."* So
`CONTRADICTED` on an accusation item is an upheld accusation, and the decomposition
`P = E x A` is well-posed over this packet.

---

## The defect

**G-E1 as frozen is unsatisfiable, and it is our error, not the packet's.**

The preregistration says the 30 sealed decoys are *"re-purposed as extraction decoys: 15 in
which a claim is unambiguously being made, 15 in which it unambiguously is not (a path inside a
code fence, an unticked template line, a comparative reference)."*

That sentence was written without opening the packet. The packet does not contain such a split.
**All 30 decoys are items in which a claim is unambiguously being made.** The two decoy families
differ in whether the claim is *true*, never in whether it is *a claim*:

| family | n | what it is | is a claim being made? |
|---|---|---|---|
| `decoy_verified` | 15 | real claims the gate verified — e.g. *"Created a root composer.json file that references the PHP SDK in the subdirectory"* | **yes** |
| `decoy_synthetic_contradiction` | 15 | the same real claims with the **path** perturbed to a `zz_`-prefixed name — e.g. *"**Created `zz_full-index.ts`**: New CLI entry point…"* | **yes** |

The perturbation that manufactured the second family operates on the path, not on the speech
act. Both families answer CLAIM to the substituted question. The packet therefore supplies
**30 positives and zero negatives** for the extraction question.

---

## Why the obvious substitute is not a gate

The tempting repair is to keep the 30 decoys, expect CLAIM on all of them, and require >= 27/30.

**That gate is passed by the exact failure it exists to detect.** A panel that answers CLAIM to
every item scores 30/30 on it — and returns `E = 1.00`, the maximally thesis-refuting value,
for a reason that has nothing to do with the prose. A one-sided reliability check over an
all-positive decoy set cannot distinguish a careful panel from a panel with its answer stuck on
the majority class. Shipping E behind it would be shipping an unmeasured number behind a gate
that reads green by construction, which is the defect class this instrument exists to name and
which `RESULT_v14` already paid for once.

G-E1 was written as **non-negotiable**, with the consequence stated in the same sentence: *"the
panel is VOID and no E is reported."* That clause is honoured here rather than reasoned around.

**The panel was not convened. No seat was prompted. E is UNMEASURED, not low, not high.**

---

## Why the decoys cannot be repaired mechanically, either

The first repair we reached for was to draw NOT-A-CLAIM decoys mechanically — matches the
`extraction_census` rules place inside a code fence, a blockquote, or an unticked task box.

**That is circular and must be refused.** Whether a path inside a fence or an unticked template
line constitutes a claim *is the question this preregistration asks a panel to answer*. Building
the reliability gate out of the answer would license whatever the panel then said. It is the
same error this lab corrected earlier today, when calling those census counts "provably not
assertions" was replaced with the observation that they are **containment** figures: they record
where a match sits, not whether the extraction was wrong.

Using the gate's own `_REFERENTIAL` and `_CONTAINMENT` guards to build decoys fails for a
sibling reason — the panel would then be scored on agreement with the instrument under test.

---

## The repair, stated now so that it is frozen before it is used

The existing packet already demonstrates the honest construction: `decoy_synthetic_contradiction`
was **synthesised by a stated perturbation** of real items. The repair applies the same method to
the speech act instead of the path.

A successor preregistration must, before any panel:

1. **Synthesise 15 NOT-A-CLAIM decoys** from real claim sentences drawn from the development
   split only, by a committed frame perturbation with the frame stated in advance — an explicit
   negation frame, a quotation-of-someone-else frame, and a comparative-reference frame
   (*"fixed the same way `sla.py` was"*), in a fixed ratio.
2. **Keep 15 CLAIM decoys** from `decoy_verified`, unmodified.
3. **Commit the decoy file and its SHA-256 digest before the panel is prompted**, exactly as
   `v14_key_digest.txt` was committed before the V14 panel judged.
4. Re-state G-E1 as a **two-sided** bar over that 30, and additionally void the panel if either
   class is answered correctly at below chance, so a stuck answer fails on the class it agrees
   with as well as the class it does not.
5. Draw the NOT-A-CLAIM frames from the **development split only**, so the held-out prose the
   accusations are drawn from is not consumed to build the instrument that scores them
   (`SPLIT_external_corpus_2026_08_31.md`, rule 3).

Item 1 is authorship, and authorship is judgment. It is disclosed as such: the frames are
written by us, and a successor result must say so in the same breath as any E it reports.

---

## The honest summary

A preregistration written on Monday specified a reliability gate that Monday's packet cannot
supply. The defect was found by executing the document rather than by re-reading it, it was
found before any answer existed, and the consequence the document itself committed to — void the
panel, report no E — is what happened.

**What is licensed by this addendum:** that G-E2 reconciles at 0.1600, that the packet's 130
items carry 100 accusations and 30 all-positive decoys, and that
`PREREG_extraction_ceiling_2026_09_01.md` cannot be executed as frozen.

**What is not licensed:** anything at all about the size of E, in either direction, for any
instrument. The extraction term remains exactly as unmeasured as it was this morning, and the
sentence "we know of no other measurement of it, ours or anyone's" still stands — now including
our own.
