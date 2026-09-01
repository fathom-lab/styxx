# PLAN — prior art, the pioneer ledger, and the adjusted next move

Fathom Lab · 2026-08-31 · A strategy document, not a certified result: it makes no numeric
claims about our instruments. Written after a four-lens strategy panel chose "the witnessed
gate," and a four-sweep prior-art review (supply-chain attestation, agent-output
verification, proof-carrying documents, 2024–2026 literature) tested whether that move is
genuinely ours to make. It adjusts the plan. A pioneering claim is a claim, and this lab
does not publish claims without receipts.

## What the prior art actually says

**The gap is real, and four independent sweeps agree on its shape.** No shipping tool
deterministically gates natural-language claims against evidence bytes and fails a build on
contradiction. The landscape splits cleanly:

- **Supply-chain attestation** (in-toto, SLSA v1.2, sigstore/cosign, GitHub artifact
  attestations, reproducible builds) attests *artifact identity and build provenance* —
  hashes, identities, environments. The semantic content of what an author *said* about a
  change is out of scope for all of it, by design.
- **AI code review** (CodeRabbit, Copilot code review, Qodo/PR-Agent, Greptile) is
  LLM-judged and advisory. Qodo's "ticket compliance" is the nearest neighbour and is still
  an LLM rating, benchmarked LLM-as-judge against LLM-generated ground truth. A comment is
  not a gate.
- **Agent-provenance research** (execution-provenance surveys, claim-to-evidence trace
  graphs, LTL trace checking, TEE attestation) produces graphs and bounds for human review,
  with at least one system stating plainly that its own layers are not fully deterministic.
  Provenance of *actions*, never adjudication of *claims*.
- **Self-verifying artifacts**: sigstore bundles verify offline via CLI plus an external
  trusted root; browser ports of sigstore verification exist as hosted network-fetching apps.
  No project ships a single self-contained file that seals its evidence and re-derives its
  own verdict offline in the reader's browser.

**Two corrections the sweeps forced on us.**

1. **We must not build our own execution recorder.** in-toto's *Witness* (Apache-2.0,
   donated to in-toto in 2024, actively released) already records command runs with exit
   codes and output streams and verifies them against signed policy; in-toto publishes a
   **Test Result predicate** (PASSED/WARNED/FAILED with test-name lists) for exactly the
   evidence we wanted to capture. Building a parallel recorder would be weaker security,
   duplicated effort, and a novelty claim we would lose. The correct architecture is the
   opposite: **we adjudicate, we do not record.** Our evidence leg becomes an adapter that
   consumes ordinary machine-readable test output (JUnit XML, pytest) for zero-friction
   adoption and signed in-toto attestations where they exist for stronger evidence.
2. **The name "witness" is taken, hard.** It belongs to an active project in the in-toto
   ecosystem whose entire domain is CI attestation, and to at least one other protocol.
   Our module ships as `styxx/evidence.py`; the third leg is "the evidence leg."

**And one asset we did not know we had.** The MSR 2026 Mining Challenge publishes **AIDev**,
an openly available dataset of agent-authored pull requests spanning five commercial coding
agents and tens of thousands of repositories, and the accompanying study measured
message-code inconsistency across tens of thousands of agentic PRs — reporting that the most
common inconsistency type is a description claiming changes the diff does not contain, and
that inconsistent PRs are accepted far less and merge far slower. The study calls for
automated verification of this in CI and releases no tool.

The corpus we assumed we would have to accumulate over six months of adoption is public
today, and the phenomenon our instrument was built to catch has already been measured, by
someone else, at scale, with human annotation.

## The adjusted move

**Two legs, one launch.**

1. **The evidence leg + the Action** (the product): diffgate gains `--evidence`, consuming
   JUnit XML / pytest output and in-toto attestations, so that a claim like "all tests pass"
   moves from UNCHECKABLE to VERIFIED or CONTRADICTED when evidence the claimant did not
   author is present — and stays UNCHECKABLE, loudly, when it is not. Ships as a GitHub
   Action that gates the PR and attaches a capsule. We adjudicate claims; the CI provider
   and the attestation ecosystem supply the evidence and its trust boundary. That boundary
   is stated in the artifact, not implied away.
2. **The measurement** (the receipt): run the gate across the public agentic-PR corpus and
   publish what it finds — preregistered, four-gated, with precision measured against the
   study's human annotations and every false accusation counted as a failure of ours. This
   is the receipt no self-testing can buy, and it is available now rather than in six months.

The two are one move: the measurement proves the tool on data we did not choose, and the
tool is what makes the measurement possible for anyone at all.

## The pioneer ledger — what may be said, and how

**Defensible as stated:**

- No shipping tool we could find performs deterministic, claim-level gating of an author's
  natural-language summary against the diff it describes, failing CI on contradiction. The
  MSR mining-challenge study names the problem and releases no tool.
- No project we could find ships a single self-contained file that seals its evidence and
  re-verifies itself offline in a reader's browser, with a local command re-deriving the
  verdict. Sigstore bundles verify offline via CLI and an external root; browser
  verification exists as hosted apps.
- Capsules of failed results are first-class here, and the whole certified corpus ships as
  one self-verifying file.

**Only with qualifiers:**

- "Tests pass, made contradictable" — credit the in-toto Test Result predicate as the
  evidence schema and the CI provider as the recorder. We supply the adjudication, not the
  attestation.
- Any "first" — carries "we know of no other," never "there is no other."

**Must not be said:**

- Nothing may be called "proof-carrying data" (that term belongs to a specific cryptographic
  literature). Where "proof-carrying" is used for capsules, it is used with its lineage
  acknowledged and its difference stated: proof-carrying code carries a deductive proof of a
  safety policy checked by a tiny trusted checker; a capsule carries *evidence* and a
  re-derivable empirical verdict. Same asymmetry — the producer does the work, the consumer
  only checks — different epistemic tier. Say so every time.
- Nothing ships under the name "witness."

## What we credit, openly

Necula's proof-carrying code for the asymmetry principle and the small trusted checker;
in-toto, SLSA, and sigstore for the attestation rails we build on rather than beside;
browser-side sigstore verification for proving WebCrypto verification is practical; and the
MSR mining challenge and its dataset for the corpus and the problem statement. Crediting
prior art is not a concession here. An instrument that overstates its own novelty has
already failed its own standard.

---

*The gap is real; the recorder is not ours to build; the corpus already exists; and the
claim to being first survives only with the qualifier attached. That is the honest version,
and the honest version is the one that ships.*
