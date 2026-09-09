# Prior art for constraint accrual and the first-claim residue

Fathom Lab · 2026-09-09 · **A prior-art check, not a result.** It was commissioned after
`THE_BOUNDARY_2026_09_09.md` emptied its class two and stated a residue, and its job was to find out
whether either idea is new. Method: WebSearch and WebFetch over the literatures named in the brief,
plus one re-run of this lab's own receipt to confirm the numbers quoted below. Not sworn. Written by
the same lab that produced the finding, which is the obvious limit on it.

**Verdict in one line: nothing in the finding is new as an idea.** Both claims are named,
old, and better developed elsewhere — one of them in a literature (metrology) that also has a better
answer than ours. The specific artifact — a predicate that refuses a signed evaluation certificate
because a prior signed certificate in the same log measured the same nuisance factor as non-zero —
is not something this search located, but it is a routine instance of a standard quality-control
move, and it should be published as an instance, not as an idea.

---

## The claims under check

Quoted verbatim from `THE_BOUNDARY_2026_09_09.md`:

1. *"A log is not a pile of certificates. It is a set of mutually constraining claims, and the
   constraint grows with its length. A party who lies must be consistent not with one document but
   with everything they have ever signed, and they must have been consistent before they knew which
   check would be written."*
2. *"The residue is not a list of unreachable fields. It is a single unreachable act: the first
   claim about anything. … Consistency accrues; it cannot be bootstrapped."*
3. The member-1 predicate, stated operationally: *a prior logged measurement of a nuisance factor's
   effect on a subject refuses a later certificate that claims the same factor separates nothing.*

The numbers this document refers to were re-read today by running
`papers/v8/class_two_empty_2026_09_09/member1_demo.py` against
`papers/v8/first_verdict_2026_09_09/log/entries`. Its printed output: the published floor at entry
`00000006` records `1v8=0.03125`, `1v32=0.015625`, `8v32=0.046875` on channel `exact`, the
corresponding values on `seqlp` and `topk`, and `1v1=0` on all three; the honest floor against
itself compares **12 factor-level pairs, none contradicted**; the roster's forgery is refused on
**9 factor-level pairs across 3 channels**; the same forgery on a subject with no logged history
returns **`unconstrained`**. Those are the numbers cited below, and they are this lab's own, not the
literature's.

---

## Idea 1 — "a self-written append-only log constrains its author, and the constraint grows with the log"

### Closest prior work

**(a) Accounting — this is the closest match found, and it is not close, it is the same sentence.**
Jan Barton and Paul J. Simko, *The Balance Sheet as an Earnings Management Constraint*, The
Accounting Review 77 (2002), supplement, 1–27.
[SSRN abstract 320641](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=320641).
What it establishes: because the income statement and the balance sheet articulate, every prior
earnings-increasing accounting choice is also recorded as an increase in net assets, so *the
accumulated record of past choices puts an upper limit on how much a manager can manipulate the next
period's number*. Their empirical proxy is beginning net operating assets scaled by sales, and they
report that the likelihood of reporting a favourable earnings surprise **falls** as that proxy rises.
Follow-on work (Zhang, *Bloated balance sheet, earnings management, and forecast guidance*, Review of
Accounting and Finance 11 (2012), 120–140,
[Emerald](https://www.emerald.com/raf/article-abstract/11/2/120/358570/Bloated-balance-sheet-earnings-management-and))
finds managers switch to real earnings management or forecast guidance once the accrual channel is
constrained — that is, the constraint is measurable enough that the constrained party is observed
routing around it.
**Our sentence with "log" substituted for "balance sheet" is their thesis, published 24 years
earlier, with an estimator and a sample attached.** We have neither.

**(b) Auditing — using previously reported figures to constrain new ones is the standard procedure,
not an observation.** PCAOB [AS 2401, *Consideration of Fraud in a Financial Statement Audit*](https://pcaobus.org/oversight/standards/auditing-standards/details/AS2401)
directs auditors, where improper revenue recognition is a fraud risk, to run substantive analytical
procedures on disaggregated data *comparing the current period against comparable prior periods*.
The whole practice of analytical review is the operational form of claim 1.

**(c) Transparency logs — the constraint-accrues property is the design goal of the field, stated
explicitly.** Melara, Blankstein, Bonneau, Felten and Freedman, *CONIKS: Bringing Key Transparency to
End Users*, USENIX Security 2015 ([PDF](https://www.cs.wm.edu/~smherwig/readings/papers/15-sec-coniks.pdf)):
because a provider publishes signed, chained snapshots, *"any equivocation to two distinct parties
must be maintained forever or else it will be detected."* That is claim 1, for key directories, in
2015. The same property underlies Certificate Transparency's consistency proofs (RFC 6962; Laurie,
Langley, Kasper), formalised by Dowling, Günther, Herath and Stebila, *Secure Logging Schemes and
Certificate Transparency*, [ePrint 2016/452](https://eprint.iacr.org/2016/452.pdf).

**(d) Tamper-evident logging — the mechanism.** Crosby and Wallach, *Efficient Data Structures for
Tamper-Evident Logging*, USENIX Security 2009
([PDF](https://static.usenix.org/event/sec09/tech/full_papers/crosby.pdf)): an untrusted logger is
kept honest by auditors, and the history tree lets it prove that the log *"is consistent with how it
was seen in the past"* in logarithmic size. Earlier: Schneier and Kelsey, *Cryptographic Support for
Secure Logs on Untrusted Machines*, USENIX Security 1998, and *Secure audit logs to support computer
forensics*, ACM TISSEC 2 (1999), 159–176
([ACM](https://dl.acm.org/doi/10.1145/317087.317089)). Note what the field is honest about: Crosby
and Wallach's guarantee is consistency of the log with its own past, and Schneier–Kelsey's guarantee
degrades sharply after compromise (an attacker at time *t* can delete earlier entries).

**(e) Distributed systems — commitment to a history, then replay.** Haeberlen, Kouznetsov and
Druschel, *PeerReview: Practical Accountability for Distributed Systems*, SOSP 2007
([PDF](https://www.sigops.org/s/conferences/sosp/2007/papers/sosp118-haeberlen.pdf)). A node keeps a
tamper-evident record of its observable actions; witnesses replay that record against a reference
implementation and detect deviation. The log constrains the node precisely because it cannot revise
what it already signed.

**(f) Deception detection — and this literature is evidence against us.** Consistency across
statements is the *most used* lay and professional cue to deceit and it is used badly: see the
[consistency heuristic literature](https://www.researchgate.net/publication/303702363_Deception_Detection_examining_the_Consistency_Heuristic)
and Nahari, Vrij and Fisher's verifiability approach (2014), replicated with smaller effects by
Verschuere et al., Applied Cognitive Psychology 35 (2021)
([Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/acp.3769)). The field's finding is that
*checkability* (can a third party verify this detail?) discriminates better than *consistency*, and
that observers who substitute consistency for verifiability get worse. Our own document reaches the
same conclusion about challenges; this literature got there first and measured it.

**(g) Logic — the ceiling on the whole idea.** A consistent set of sentences is satisfiable, not
true; it has *a* model, not necessarily the intended one. The coherentist position that consistency
of a growing web can substitute for grounding is the circular horn of Agrippa's trilemma
([Münchhausen trilemma](https://en.wikipedia.org/wiki/M%C3%BCnchhausen_trilemma), Hans Albert, 1968).
This is the formal reason claim 1 can never be upgraded from "raises the cost of lying" to "detects
lying", and our document's own invariant — *a check on bytes an issuer wrote can only ask whether
that party contradicted itself* — is a restatement of it.

### What is new here, and what is not

**Nothing is new.** The claim is Barton–Simko's thesis in accounting, CONIKS' design goal in
security, the premise of analytical review in auditing, and the circular horn of Agrippa's trilemma
in epistemology. The only thing this lab added is the observation that *its own four-member
impossibility roster fell because each argument examined one certificate instead of the log* — which
is a report of four of our own errors, not a contribution.

---

## Idea 2 — "the residue is the first claim about a subject, which nothing logged can contradict"

The brief's suspicion was that this is trust-on-first-use. It is, and it is also three other named
things, one of which is sharper than ours.

### Closest prior work

**(a) Trust on first use.** [Wikipedia, TOFU](https://en.wikipedia.org/wiki/Trust_on_first_use);
[GnuTLS manual, *Verifying a certificate using trust on first use authentication*](https://www.gnutls.org/manual/html_node/Verifying-a-certificate-using-trust-on-first-use-authentication.html).
What it establishes: when no prior identifier exists for an endpoint, the client either asks a human
or trusts what it is handed; there is no mechanism inside the protocol to verify the first
observation, and every later consistency check inherits whatever the first one accepted. Named,
deployed in SSH `known_hosts` since the 1990s, surveyed in [*Failures of public key infrastructure: a
53 year survey*](https://arxiv.org/pdf/2401.05239).
**Our "control B returns `unconstrained`" is TOFU's first connection.** The one difference worth
stating: TOFU's convention is to *accept* the first observation and bind to it; our predicate
*declines to answer* and reports the claim as unconstrained. That is a reporting choice, not a new
security property, and TOFU implementations that prompt the user make the same choice.

**(b) Weak subjectivity — the closest structural analogue, because it is about a log rather than a
key.** Vitalik Buterin, *Proof of Stake: How I Learned to Love Weak Subjectivity*, Ethereum
Foundation blog, 25 November 2014
([blog.ethereum.org](https://blog.ethereum.org/2014/11/25/proof-stake-learned-love-weak-subjectivity));
see also [ethereum.org, Weak subjectivity](https://ethereum.org/developers/docs/consensus-mechanisms/pos/weak-subjectivity/)
and Poelstra, *On Stake and Consensus* (2015), which named costless simulation and long-range
attacks. What it establishes: a node joining a proof-of-stake chain with no prior state cannot
distinguish the real history from a costlessly simulated one, because *both are internally
consistent signed histories*; the fix is an out-of-band checkpoint, i.e. a byte the chain did not
write. That is our residue and our proposed remedy (a second party's bytes), stated twelve years
earlier for exactly the object we are building — an append-only signed log — and with the added
result that consistency of the candidate history is no help at all.

**(c) Metrological traceability — and this is the one that should sting.** JCGM 200 (VIM),
[2.41 metrological traceability](https://jcgm.bipm.org/vim/en/2.41.html) and
[2.42 metrological traceability chain](https://jcgm.bipm.org/vim/en/2.42.html); NIST's
[Metrological Traceability policy and FAQ](https://www.nist.gov/metrology/metrological-traceability).
What it establishes: a measurement result is credited only via *"a documented unbroken chain of
calibrations, each contributing to the measurement uncertainty"*, terminating at a primary standard
whose value is, by definition, **accepted without reference to other standards of the same
quantity.** Metrology has known for a century that the regress in a measurement chain must be
stopped by an externally agreed realisation, has named the stopping point, has built institutions
(BIPM, NMIs, ISO/IEC 17025 accreditation) whose entire purpose is to be that external reference, and
requires every link to carry an uncertainty. **We rediscovered the need for the anchor and did not
build the anchor.** Our "reproduction count" is a weaker, unaccredited version of "traceable to a
stated reference".

**(d) Bootstrapping trust in platforms.** Parno, *Bootstrapping Trust in a "Trusted" Platform*,
USENIX HotSec 2008 ([PDF](https://www.usenix.org/legacy/event/hotsec08/tech/full_papers/parno/parno.pdf));
Parno, McCune and Perrig, *Bootstrapping Trust in Modern Computers* (Springer, 2011). The cuckoo
attack: a verifier knows it is talking to *a* genuine TPM, not to *the* TPM it means, and the paper
states plainly that no fully satisfying instantiation of the fixes exists. Same shape: the first
binding cannot be established from inside.

**(e) Byzantine agreement's setup assumption.** The authenticated/unauthenticated split — a PKI
"trusted setup" raises tolerable faults from *t < n/3* to *t < n/2* for agreement and *t < n* for
broadcast — is the formal statement that the initial trust cannot be produced by the protocol; see
e.g. [Optimal Communication Complexity of Authenticated Byzantine Agreement](https://eprint.iacr.org/2020/1569.pdf)
and work on [bootstrapping a PKI from scratch](https://link.springer.com/chapter/10.1007/978-3-319-76581-5_16).

**(f) Epistemology.** The general form is the dogmatic horn of the
[Münchhausen trilemma](https://en.wikipedia.org/wiki/M%C3%BCnchhausen_trilemma): justification ends
in regress, circularity, or an ungrounded stopping point. "Consistency accrues; it cannot be
bootstrapped" is that sentence.

### What is new here, and what is not

**Nothing is new. The honest finding is that this lab rediscovered trust-on-first-use in a
measurement setting**, and that at least three mature fields — key management, proof-of-stake
consensus, and metrology — had already named it, characterised it, and (in metrology's case) built
the institutional answer we are gesturing at. The brief asked us to try hard to confirm this rather
than preserve novelty; it is confirmed.

The single thing that survives as *ours* is small and procedural: the choice that the predicate
returns a third verdict, `unconstrained`, rather than passing or failing a first claim, and the
proposal to print the amount of prior logged material beside every verdict. Neither is an idea. The
first is a well-known API design choice for partial information; the second is a reproducibility
metric, and reproducibility metrics are a crowded field (see below).

---

## Idea 3 — the member-1 predicate: using a prior logged measurement of a nuisance factor to refuse a later fabricated one

This was the specific question in the brief. Answer: **the move is standard in three measurement
disciplines and older than any of us. The particular instance is not something this search found.**

### Closest prior work

**(a) "Too good to be true" — the founding case, and it is our predicate's superset.** R. A. Fisher,
*Has Mendel's Work Been Rediscovered?*, Annals of Science 1 (1936), 115–137. Fisher took the
*known error inherent in Mendel's experimental design* — i.e. a prior, independently established
account of the nuisance variation — and showed the reported data hugged expectation far more closely
than that variation permits. See
[the Mendel–Fisher controversy literature](https://arxiv.org/pdf/1104.2975) and Franklin et al.,
*Ending the Mendel–Fisher Controversy* (Pittsburgh, 2008). Our predicate is the degenerate special
case: reported dispersion of exactly zero where the log records positive dispersion for the same
factor levels on the same subject. Fisher's version handles the harder case — plausible but too-small
dispersion — which our own receipt admits it does not reach.

**(b) Proficiency testing — using an established dispersion to score a new participant's result is
the entire method.** ISO 13528, *Statistical methods for use in proficiency testing by
interlaboratory comparison* (editions 2005 and 2015 consulted:
[2005 sample PDF](https://cdn.standards.iteh.ai/samples/35664/7621722d89914ff69ee9f5d83d1c3d82/ISO-13528-2005.pdf),
[2015 sample PDF](https://cdn.standards.iteh.ai/samples/56125/afcddf885d3746e6a839fc5147c7d945/ISO-13528-2015.pdf));
Eurachem, [*Understanding PT performance assessment*](https://www.eurachem.org/images/stories/leaflets/pt/pt_perf_assmnt/PT_Understand_Evaluation_V1_EN.pdf).
The *z* score divides a participant's deviation from the assigned value by σ_pt, the standard
deviation for proficiency assessment — one of whose permitted derivations is *from previous rounds
or from a published reproducibility standard deviation*. Companion standard ISO 5725 fixes
repeatability (*r*) and reproducibility (*R*) limits in advance from interlaboratory studies, against
which later results are checked. That is claim 3, standardised, with an ISO number.

**(c) Laboratory quality control.** Levey–Jennings charts with Westgard rules (1-2s, 1-3s, 2-2s,
R-4s) set control limits from *historical* dispersion and flag a new run that violates them —
including runs whose dispersion is anomalously small. See
[Laboratory quality control](https://en.wikipedia.org/wiki/Laboratory_quality_control) and
[EPA MARLAP Vol. III, ch. 18](https://www.epa.gov/sites/default/files/2015-05/documents/402-b-04-001c-18-final.pdf).

**(d) Trial-integrity statistics.** Carlisle, *Data fabrication and other reasons for non-random
sampling in 5087 randomised, controlled trials*, Anaesthesia 72 (2017), 944–952
([Wiley](https://associationofanaesthetists-publications.onlinelibrary.wiley.com/doi/10.1111/anae.13938)),
flagging baseline tables that are *either* too unbalanced *or* too balanced against expected sampling
variability — with published criticism of the method's independence assumption
([Bordewijk et al. / arXiv:2209.00131](https://arxiv.org/pdf/2209.00131),
[Heathers & Brown reanalysis](https://www.biorxiv.org/content/10.1101/179135.full.pdf)) and automated
over/under-dispersion detection in [F1000Research 11:783](https://f1000research.com/articles/11-783).
Related digit- and moment-based tools (GRIM, SPRITE, Benford) are surveyed in
[*Tools of the data detective*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12121900/).

**(e) The threat model already has a name in forensic science.** Max M. Houck, *Systems fail people:
dry labbing as a distinct category of scientific fraud*, Science & Justice 66 (2026), issue 5,
DOI [10.1016/j.scijus.2026.101491](https://www.sciencedirect.com/science/article/abs/pii/S1355030626000973).
Dry labbing is reporting a result for a procedure never run. Houck's stated point is the one our
document keeps rediscovering: *a falsified result can sometimes be caught by re-examining the
underlying materials; a dry-labbed result has no underlying materials to re-examine at all.* Our
class-two argument is a restatement of that sentence for signed certificates. (Caveat on sourcing: I
found this via a [CASRAI editorial](https://casrai.org/news/dry-labbing-distinct-category-scientific-fraud-2026)
and confirmed the paper exists on ScienceDirect; I have not read the paper itself and cannot vouch
for the editorial's characterisation beyond the abstract-level match.)

**(f) The forensic-statistics doctrine that names our exact fallback.** From
[embassy.science's forensic-statistics summary](https://embassy.science/wiki/Theme:467f5cf6-d41f-42a0-9b19-76556579845d):
*"In the absence of an external source for comparison, criteria for assessing the raw data file were
internal consistency and plausibility."* That sentence is this whole document's subject: internal
consistency is what you use **when you have no external source**, and the field treats it as the
weaker fallback, not the mechanism.

### Prior art on the nuisance factor itself

The batch-size effect our predicate exploits is documented independently of us: Thinking Machines
Lab, [*Defeating Nondeterminism in LLM Inference*](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
(batch size as the dominant nondeterminism source; batch-invariant kernels as the fix), now shipped
in [vLLM's batch-invariance mode](https://docs.vllm.ai/en/latest/features/batch_invariance/) and
[SGLang](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/), with numerical-source analysis
in [arXiv:2506.09501](https://arxiv.org/pdf/2506.09501). **The effect we measure is prior art; only
our measurement of it on this subject is ours.** Note the direction of travel: the ecosystem is
actively removing batch sensitivity, so a predicate that depends on it having a non-zero value has a
shelf life.

### What is new here, and what is not

**The idea is not new: it is proficiency testing's *z*-score against an established σ, and Fisher's
1936 argument, applied to a computation instead of a wet lab.** What this search did not find is any
published instance of the specific composition — *a prior cryptographically signed certificate's
measurement of a named nuisance factor on a hash-identified subject used as the reference dispersion
that refuses a later signed certificate about the same factor and subject*. That composition, and the
receipt showing it refusing on 9 of 9 factor-level pairs while agreeing on 12 and declining on a
fresh subject, is the only defensible novelty claim in the whole finding, and it is a novelty of
*artifact*, not of idea. It should be written up as "PT's method, mechanised over a signed log", with
ISO 13528 and Fisher cited in the first paragraph.

---

## Work that attacks the same gap and is ahead of us

The brief did not ask for this, but a prior-art check that omits it would be dishonest. Our document
says the challenge mechanism *"is the only part of it that introduces a byte the issuer did not
write."* That is false about the design space, and three lines of work show it:

- **Schnabl, Hugenroth, Marino and Beresford, *Attestable Audits: Verifiable AI Safety Benchmarks
  Using Trusted Execution Environments*, arXiv:2506.23706 (30 June 2025)**
  ([abs](https://arxiv.org/abs/2506.23706)). Runs the benchmark evaluation inside a TEE so that
  provider and auditor need not trust each other. This is our exact application domain — verifiable
  model evaluation — with a hardware root of trust doing the job our log cannot.
- **Karvonen, Reuter, Rinberg, Marks, Garriga-Alonso and Warr, *DiFR: Inference Verification Despite
  Nondeterminism*, arXiv:2511.20621 (25 November 2025)** ([abs](https://arxiv.org/abs/2511.20621)).
  Untrusted inference provider; seed-synchronised reference implementation bounds valid outputs;
  activation fingerprints via random orthogonal projections. Directly addresses "did this inference
  actually run", *and* handles the benign-noise problem our floor exists to characterise.
- **Zero-knowledge evaluation.** [*Verifiable evaluations of machine learning models using
  zkSNARKs*, arXiv:2402.02675](https://arxiv.org/pdf/2402.02675) and the survey
  [arXiv:2502.18535](https://arxiv.org/pdf/2502.18535).

All three introduce bytes the issuer did not choose — by hardware attestation, by a reference
implementation, or by a proof system. Our design's reliance on a second party is one option among
several, and the weakest of them in the sense that it requires someone else's willingness.

---

## Claims of ours the literature already covers — withdraw or re-attribute

| our claim | already covered by | action |
|---|---|---|
| "A log is … a set of mutually constraining claims, and the constraint grows with its length." | Barton & Simko 2002; CONIKS 2015; PCAOB AS 2401 | withdraw as a finding; cite Barton & Simko and CONIKS |
| "A party who lies must be consistent with everything they have ever signed." | CONIKS: equivocation *"must be maintained forever or else it will be detected"* | withdraw; quote CONIKS |
| "The residue is a single unreachable act: the first claim about anything." | trust-on-first-use; Buterin, weak subjectivity (2014); VIM 2.41–2.42 primary standard | withdraw as novel; state plainly that this is TOFU in a measurement setting |
| "Consistency accrues; it cannot be bootstrapped." | Münchhausen/Agrippa trilemma, dogmatic horn; Parno's cuckoo attack; authenticated-vs-unauthenticated BA | withdraw as an observation; keep only as a section heading with attribution |
| "A second party's bytes are the only thing that can constrain a first claim." | **false as stated** — TEE attestation (arXiv:2506.23706), reference-implementation replay (DiFR, PeerReview), zkSNARK evaluation all constrain it without a second measuring party | rewrite: *an input the issuer did not choose* is what constrains it; a second party is one such input |
| "The challenge mechanism … is the only part of it that introduces a byte the issuer did not write." | same three lines of work | rewrite as a statement about **our current design**, not about self-written records generally |
| "The reproduction count is … cheap to compute and currently computed nowhere." | ACM badging already separates *Results Reproduced* from *Results Replicated*; a [scoping review of reproducibility metrics](https://royalsocietypublishing.org/rsos/article/12/7/242076/235444/A-scoping-review-on-metrics-to-quantify) finds "a multitude of metrics" | rewrite to "computed nowhere **in this system**" |
| "This is the ordinary distinction between an audit and an attestation." | correct, and the citation exists: attestation engagements require a written assertion from the responsible party ([PCAOB AT 101](https://pcaobus.org/oversight/standards/attestation-standards/details/AT101); [ICAEW, attestation vs direct reporting](https://www.icaew.com/technical/audit-and-assurance/assurance/process/scoping/assurance-decision/attestation-vs-direct-reporting)) | keep, add the citation |
| "The prior on a fifth impossibility claim is poor." | not a claim about the world; consistent with Parno's "no fully satisfying instantiation" and with the Schneier–Kelsey post-compromise result | keep |
| Commitment (plan logged before runs) as one of five things the record establishes | preregistration literature: *"pre-registration … is powerless in the face of fraud"* and does not weed out fabricated data ([Pre-registration: Why and How](https://merit.url.edu/ws/portalfiles/portal/46674012/Pre-registration_Why_How_-_preprint.pdf)) | keep the property, add the limit explicitly beside it |

Two of ours that the literature **strengthens** rather than pre-empts, and should be cited for:

- The `unconstrained` third verdict is defensible, and interlaboratory statistics gives it a
  vocabulary: results whose spread is inconsistent with stated uncertainties are handled by named
  adjustment procedures rather than a pass/fail, and unexplained excess spread has a name — *dark
  uncertainty* ([IJMQE 2019](https://www.metrology-journal.org/articles/ijmqe/full_html/2019/01/ijmqe180025/ijmqe180025.html)).
- The verifiability-approach result — that *checkability* of a detail beats *consistency* of an
  account as a deception cue — is independent psychological support for our conclusion that the
  challenge, not the consistency check, is the load-bearing part. Cite Nahari, Vrij & Fisher (2014)
  and the Verschuere et al. (2021) replication, including its smaller effect sizes.

---

## What this check did not do

- No paywalled paper was read in full. Barton & Simko, Fisher 1936, Carlisle 2017, Houck 2026 and
  ISO 13528 were assessed from abstracts, standards samples, and secondary summaries. Any of those
  characterisations could be wrong in detail.
- ISO 13528 was consulted as the 2005 and 2015 sample PDFs. A later edition may differ on how σ_pt
  may be derived, which is the exact clause the claim-3 comparison rests on.
- No systematic search of the metrology literature for "fabricated result detected by prior
  nuisance-effect measurement" was run beyond proficiency testing and interlaboratory comparison;
  a negative result there is weak evidence, not absence.
- The CASRAI editorial used to locate Houck 2026 is a secondary source; the ScienceDirect record
  confirms the paper, not the editorial's reading of it.
- No search was run in the databases/knowledge-representation literature on integrity constraints,
  where claim 1 very likely has a fourth independent statement.
- Nothing here was reviewed outside this lab, which is the same limit the finding it checks carries.
