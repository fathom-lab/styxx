# RECON — the OATH lane has never had a prior-art survey, and the neighbourhood is occupied

Fathom Lab · 2026-08-26 · **RECON. Licenses no claim.** One reader, one afternoon, four sources
actually opened. This is not a systematic review and the absence of a hit here is weak evidence
of anything.

---

## why this exists

`RECON_landscape_2026_08_21.md` surveys the SILENT-PASS lane — measurement integrity in code —
and concludes that the position is unoccupied. That survey is about a different subject and
nothing below contradicts it.

The OATH lane has no equivalent. It carries the programme's north star, it is twelve
preregistered cycles deep, and searching the repository for the obvious neighbours returns
nothing: `statcheck`, `arXiVeri`, *Proof-Carrying Numbers* and *Agent-Native Research Artifacts*
appear in no document here. An instrument built for twelve cycles without once asking who else
built one is exactly the shape of mistake this lab exists to catch, so it is caught here.

## what is actually out there

**statcheck** (Nuijten et al.; R package and web app, in wide use in psychology since the
mid-2010s) extracts statistical results from a paper's text and recomputes p-values from the
reported test statistic and degrees of freedom, flagging internal inconsistencies. It is the
long-standing precedent for automated numeric checking of published claims, it operates on PDFs
and HTML with no cooperation from the author, and it is deployed at journal scale.

*Difference that matters:* statcheck checks a claim against **itself** — does the p-value follow
arithmetically from the statistic beside it. OATH checks a claim against **an external artifact
the author shipped**. Neither subsumes the other, and statcheck's regime is the one that needs no
contract at all, which is why it works on documents nobody wrote for it.

**arXiVeri** (Shin, Xie & Albanie, 2023) does automatic table verification: given a target table,
find the source table in the cited document and match cells across them, using GPT-4 as the
baseline. It is explicitly framed as a spell-checker for numbers copied between papers. It
produces a benchmark and baselines; it does not emit a certificate or any persisted verifiable
artifact.

**Proof-Carrying Numbers** (arXiv 2509.06902) is the closest work found, and it is close enough
to be uncomfortable. Numeric spans in LLM output are emitted as claim-bound tokens; a verifier
sitting in the **renderer rather than the model** checks each against a declared policy — exact
equality, rounding, aliases, or tolerance with qualifiers — and only passing tokens are marked
verified. It is **fail-closed**: an absent mark means unverified. The stated principle is that
trust is earned only by proof.

That is the same architecture as OATH and the same slogan, published earlier, under a name from
the same family. The matching policy it describes is close to what `_match` in `styxx/certify.py`
implements, and the renderer-not-model separation is the same "the mind must not mark its own
homework" move the north star makes.

**Agent-Native Research Artifacts** (arXiv 2604.24658, 37 authors) proposes replacing the
narrative paper with a structured package: a scientific-logic layer, executable code with
specifications, evidence grounding every claim in raw outputs, and — this one should sting — **an
exploration graph that preserves failed experiments.** Plus an ARA-native review system that
automates objective verification.

Evidence bound to every claim, and the negatives kept as a first-class part of the record, is a
fair description of what this repository has been building. It was published in April 2026 by
thirty-seven people.

## what appears to remain, stated narrowly

Three things were searched for and not found. Each is stated at the strength one afternoon
supports, which is: *not found*, not *does not exist*.

1. **Preregistration compiled into the scorer.** `styxx/protocol.py` refuses to emit a verdict
   unless the preregistration is committed in git history — a prereg on disk is a draft, and
   there is no API to pass a bar at scoring time. Searching turned up `preregr` (machine-readable
   preregistrations in JSON) and a large methods literature on preregistration as practice, but
   no tool that makes it a **runtime precondition**. Of everything here this is the claim that
   survives a hostile expert best, because the expert can run it and watch the refusal fire.
2. **A persisted certificate that can later stop holding.** OATH certificates bind a document
   hash, its receipt hashes, and the verifier's own hash, so a certificate is re-checkable years
   later against a moved verifier. That makes **drift** detectable, and drift is not hypothetical
   — one was found in this corpus today, hidden for months. PCN verifies at render time; ARA
   verifies at review time; neither, as far as this survey goes, keeps an artifact whose failure
   *later* is the signal.
3. **The instrument audited by its own standard, at scale.** 163 logged cycles, a published
   negatives record, and a cycle whose entire content was retracting four of the verifier's own
   accusations under a nine-gate battery. This is a practice rather than a mechanism and it is
   the hardest thing to claim credit for, but no surveyed source does it to itself.

## the honest read

The category is not new, and the framing that the OATH lane has been using — *for the whole of
history no mind has been able to prove its own sincerity* — is a poor fit for a neighbourhood
that contains a decade-old deployed tool and at least two recent papers doing something close.
The gap between the north star's rhetoric and the literature is now measured, and it is wide.

What today's other recon adds is worth more here than any priority claim. `RECON_oath_external_reach_2026_08_26.md`
points the verifier at documents nobody wrote for it and reports that it abstains on almost
everything and that every accusation it makes is false. **PCN and ARA both assume the contract is
kept** — they describe systems where claims are emitted bound to evidence by construction. Nobody
in this survey measured what the same machinery does when pointed at prose that was never written
to carry receipts. That measurement is a contribution those lines of work need, and it is a
negative, which is the kind this lab is supposed to be good at.

## what follows

- The north star document needs a related-work section, and its opening claim needs rewriting to
  survive contact with statcheck.
- Any external publication from the OATH lane must cite PCN and ARA and state the difference
  precisely, or a reviewer will do it for us and less generously.
- This survey is one reader deep. A real one — systematic, with a search protocol frozen in
  advance — is owed before anything from this lane goes outward.

---

*Twelve cycles of instrument work, and the first question anyone would ask got asked in the
thirteenth.*
