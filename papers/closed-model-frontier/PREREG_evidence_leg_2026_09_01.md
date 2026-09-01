# PREREG — the evidence leg: "tests pass" becomes a function of bytes, and the accusing branch is deleted

Fathom Lab · 2026-09-01 · Frozen before `styxx/evidence.py` exists. Successor to
`RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md`, which measured a repaired
accusation at **0.16 precision against a 0.95 floor** on prose its authors had never read,
and retired the path-claim class rather than repair it a fourth time. Corpus and split
governed by `SPLIT_external_corpus_2026_08_31.md`; the AIDev shelf is the same one
EXTERNAL-1 used.

**The standing commitment this document is written under:** do not ship an accusing verdict
whose precision has not been measured by a blind panel. Report-only bands are fine and
honest. Absence of evidence is never a contradiction. "UNCHECKABLE" is a first-class verdict
here and is printed loudly, not hidden.

## What changes

Today `tests_pass` is adjudicated by **re-executing the tests on the verifier's machine**.
`styxx/diffgate.py:447-453` shells out with `subprocess.run(run, shell=True, cwd=repo,
timeout=1800)` and sets the verdict from an exit code. After this cycle, `tests_pass` is
adjudicated by **reading attestation bytes** — JUnit XML, or an in-toto Test Result
predicate — under a new module, `styxx/evidence.py`, spec string `styxx-evidence/v0.1`.

The module is a pure function. No subprocess. No network. No clock. No `os.environ`. No
randomness. The same input bytes produce the same verdict forever.

## The defect being repaired, named plainly

**A `--run` verdict cannot be re-derived, and therefore cannot be capsule-sealed.** Its true
inputs are the working tree, the installed dependency set, the interpreter, the network, the
filesystem, the clock and the shell — none of which the capsule binds, and none of which are
bytes it could bind. The same sealed summary and the same sealed diff yield VERIFIED today
and CONTRADICTED next week when a transitive dependency releases or a flaky test flips.
Re-derivation by a stranger in a year is not merely hard; it is undefined, because the thing
that decided the verdict no longer exists.

The capsule module already refuses this by name, in three places:

- `styxx/capsule.py:676` hardcodes `run=None` at the mint. There is no argument that lets a
  `--run` verdict into a capsule.
- `styxx/capsule.py:686-691` is refusal **R4**: a supplied gate carrying a `tests_pass`
  verdict other than `UNCHECKABLE` is rejected with *"environment legs cannot be capsuled in
  v0.2; a --run-resolved tests_pass verdict would require executing an embedded shell string
  to verify."*
- `styxx/capsule.py:459-460` publishes it to the reader as a boundary: *"that tests passed —
  environment legs are refused at mint; tests_pass can only appear here as UNCHECKABLE, by
  construction."*

So the shipped `tests_pass` verdict is the one claim kind in the whole template set that sits
**outside the trust stack's own transport format**. That is the defect. A bytes-only
adjudicator's verdict re-derives from sealed material forever and *is* sealable; `--run` is
unsealable by construction rather than by oversight.

Three further defects in the incumbent are recorded here so the repair cannot be mistaken for
a refactor, all verified against shipped code:

1. **`tests_pass` is exempted from the no-evidence guard.** `diffgate.py:357` reads
   `if no_evidence and kind != "tests_pass":`. Every other claim kind abstains when the diff
   parsed to nothing; this one proceeds to run the subprocess. The CLI can therefore print
   *"UNMEASURED — this gate did not run"* and exit 1 carrying an accusation.
2. **Absence of evidence already produces CONTRADICTED.** Every nonzero exit code is read as
   a lie, including pytest's `rc=5` (*no tests collected*), a misspelled command, and `rc=4`
   (usage error). "No tests ran" and "your command was misspelled" are this module's own
   definition of UNCHECKABLE, and that branch calls them lies.
3. **`--run` on an untrusted pull request is remote code execution with extra steps.**
   `cwd=repo` is the claimant's checkout; `pytest` imports their `conftest.py` at collection,
   `npm test` runs a string they wrote, `make test` runs their Makefile. The adversary also
   controls the exit code in both directions, so the check is simultaneously dangerous to the
   defender and useless against the attacker. The shipped Action does **not** use this path
   (`diffgate_action.py:80` passes no `run`); the exposure is the library and CLI surface.

## The three laws — commitments, not features

These are stated as things we can be held to and can fail, not as properties of an
implementation that does not exist yet.

**Law 1 — PURE FUNCTION OF BYTES.** The verdict is a total function of the bytes read. No
subprocess, no network, no clock, no `os.environ`, no randomness, at or below the adjudicator.
If reading the environment is ever required, it happens in a separately named `capture` step
that is *labeled impure*, writes bytes, and is not the adjudicator. **We commit in advance
that this buys purity and zero trust:** whoever ran `capture` could have written anything
into that file, and the channel is still ASSERTED. Believing the design is fixed because it
got architecturally clean is how a module ships a binding it does not have.

**Law 2 — BINDING GATES THE ACCUSATION.** A test report not cryptographically bound to the
commit under test may never produce CONTRADICTED. An unbound report may be from a different
commit, and accusing on it is precisely the false-accusation mode that retired the path-claim
class. A caller-supplied `--evidence-commit` is recorded as **ASSERTED-NOT-VERIFIED** and
licenses nothing — in the common CI deployment that flag is populated by the same workflow
the claimant wrote, and a binding supplied by the claimant is not a binding.

**Law 3 — ABSENCE IS NOT FAILURE.** Zero tests, no file, unparsable file, a harness that
could not load a module, a shard whose report never arrived: all UNCHECKABLE. Never
CONTRADICTED. This is the law the incumbent breaks today and the one we will be graded on.

## Amendments to the frozen contract, made before implementation

The `styxx-evidence/v0.1` contract was frozen before a red-team pass read it. The pass found
holes that would make a faithful implementation violate the three laws. Recording them here,
before any code, rather than discovering them in a panel later.

**A1 — Row 3's conjunction is struck.** The table read *"binding not verified AND commit was
given → UNCHECKABLE"*. With `commit=None` — the default, and the only thing
`python -m styxx.evidence <paths...>` can pass — the binding check does not run at all and a
FAILED outcome falls straight through to CONTRADICTED with `binding.kind == "none"`. The
module's default invocation would be its most dangerous mode. Row 3 now reads: **binding is
not verified → UNCHECKABLE, unconditionally.**

**A2 — `binding.verified` is split in two.** `intoto-subject` is CRYPTOGRAPHIC about *report
identity* and ABSENT about *commit binding*: a subject digest binds an envelope to a byte
string and is silent on which commit the tests ran against. One boolean cannot span two
objects. The record carries `report_identity_verified` and `commit_binding_verified`
separately, and **only the second may ever gate an accusation.** Never let one word cover
both.

**A3 — Binding is per-source, not per-bundle.** One `binding` dict over a list of sources
lets one attested file's `verified=True` license an accusation sourced from unattested
siblings globbed out of the same directory. Binding attaches to each source, and any
accusing arithmetic runs only over individually-bound sources.

**A4 — `commit_binding_verified` requires a digest equality that is stated as an obligation.**
It may be true only when the attestation's subject digest equals the SHA-256 of the source
bytes actually loaded. A valid signature over a *different* file binds nothing. Comparison is
full-length, lowercase-normalised, exact. **No prefix matching** — an empty or abbreviated
value would prefix-match every commit and turn the whole Law-2 gate into a no-op in one line.

**A5 — `errors` and `failures` do not share a verdict.** A test that ran and failed is
evidence; a harness that could not run a test (a pytest collection failure from a missing
optional dependency, a jest suite that failed to load) is absence of evidence, and Law 3
makes it UNCHECKABLE.

**A6 — EMPTY is defined over *executed* tests.** An all-skipped suite and an all-`DISABLED_`
googletest binary both report nonzero `tests=` with nothing executed. A resolved `passed`
count is added and VERIFIED requires `passed > 0`. A green badge certifying that nothing ran
is the mirror of the retirement risk: a badge that is always the same colour devalues every
other badge on the page.

**A7 — Any unparsed source blocks an accusation.** The frozen row 1 abstained only when
*every* source was unparsed, which permits accusing on a tenth of the evidence. The
asymmetry is deliberate: **a partial read may honestly decline; it may not honestly accuse.**

**A8 — Verdicts are resolved from `<testcase>` child elements, never from attributes, with
explicit precedence error > failure > skipped > pass and exact tag-name equality.** No
substring, prefix, suffix or case-insensitive tag matching anywhere: Surefire's
`<flakyFailure>` marks a test that failed and then *passed*, and `tag.endswith("failure")`
would call a green Maven build red — the exact false-accusation shape this lab retired. Root
count attributes are recorded as-found and never trusted; a root/children mismatch is
**report-only** and is specified behaviour in at least three dialects, not evidence of
tampering.

**A9 — Path-convention binding is REFUSED, not merely unbuilt.** `artifacts/<sha>/junit.xml`
is a string the producer chose, recovered by a regex. This lab has measured twice what
happens when it infers structure from strings a stranger wrote. It is written into the spec
as refused so nobody adds it later as an obvious convenience.

**A10 — The record carries an absence log.** `sources` lists what was read and `unparsed`
lists what failed to parse; nothing lists what was never offered. A missing failing shard
currently leaves no trace, which makes a false VERIFIED silent and makes a sealed
CONTRADICTED unable to diverge under a differently-globbed re-gate. The resolved path list
and the expected-but-absent set become first-class fields, and a change in the source set is
capsule refusal **R5** divergence.

## What ships, and what does not

**What ships is a binding classifier and a census. Not an adjudicator.**

`tests_pass` stays **UNCHECKABLE on every path in this version**, with a specific, citable
reason replacing today's misleading one. The current reason — *"no --run command supplied"* —
implies the fix is to supply a shell string, which is exactly what capsule R4 forbids. The
replacement names the channel that is missing: *this report carries no binding to any commit;
the channels that would have bound it are X, Y, Z; none are present.* **Strictly more
informative at identical verdicts** is a pure gain with zero precision exposure.

**No code path may produce CONTRADICTED for `tests_pass` in this version — not a flag set to
off, no such branch.** `WITHHOLD_PATH_ACCUSATION` exists in this codebase today and is a
flag, and flags get flipped by people who did not read the paper. Absence of a branch is a
stronger guarantee than a flag set to off. Enabling the accusation later must require writing
new code, and therefore re-deriving the traps below.

**The branch is deleted, not withheld, and the distinction is the whole commitment.**
"Withheld" names a capability that exists and is being held back by a decision, and a decision
is revisitable by someone who never read this document. "Deleted" names a capability that is
not in the bytes. This commitment is being honoured in code: there is no accusing branch to
re-enable, no constant to flip, no environment variable to set, nothing that a future
maintainer can turn on without writing the branch themselves — in a diff a reviewer can see,
against the traps recorded here.

Two reasons found since this document was first drafted make deletion the right shape rather
than a cautious one. Neither is a reason to try harder.

1. **The accusing branch's precision is structurally unmeasurable.** Across **1,775,765**
   changed files in the corpus, the channel Law 2 permits to accuse fires **11** times. The
   protocol this lab is bound to needs a hundred held-out items and thirty sealed decoys;
   eleven events cannot furnish one, and no protocol we would accept turns eleven into a
   measured precision. Under the standing commitment — do not ship an accusing verdict whose
   precision has not been measured by a blind panel — a verdict whose precision *cannot* be
   measured is not "pending measurement". It is unearnable on the evidence this corpus
   contains, and the honest word for a branch that can never satisfy its own gate is
   *deleted*. (Eleven is a count of bound events, not an adjudicated precision, and must never
   be reported as one.)
2. **The binding that was supposed to gate the accusation was never cryptographic.** In the
   path as built, a DSSE envelope whose signature was the ASCII text `not-a-signature` reached
   an accusation. That is Law 2 defeated by a string literal. The honest boundary below already
   says we parse and do not verify; this is what "we do not verify" costs the moment anything
   downstream is allowed to accuse. `binding.verified` was, on that path, a claim about the
   shape of a field and not about any key. A gate a fifteen-character string walks through is
   not a gate, and the branch standing behind it had no right to exist.

Both are reasons the accusing verdict for `tests_pass` may never be earnable at all — which is
what the permanence paragraph below already obliges us to design for, now with a count attached
rather than a worry.

The publishable deliverable is a **preregistered, report-only census** of binding-channel
distribution over the AIDev corpus: for each `tests_pass` claim, which channel (if any) binds
its evidence to the commit, and at what strength. It carries no precision claim.

## Gates — thresholds fixed now, before any measurement

- **G-E1 (purity), blocking, mechanically checked.** A static check over
  `styxx/evidence.py` finds **zero** references to `subprocess`, `socket`, `urllib`,
  `requests`, `http`, `os.environ`, `getenv`, `time.time`, `datetime.now`, `random`, or
  `Path.cwd`, and **zero** imports of any module that transitively provides them at
  adjudication. Additionally: two runs over the same byte inputs produce **byte-identical**
  output records. One reference fails the gate. The check ships as a receipt.
- **G-E2 (no accusation without binding).** Over the whole AIDev corpus,
  **ZERO CONTRADICTED verdicts are issued on unbound evidence.** Any single one fails the
  gate. In this version the stronger form holds by construction — zero CONTRADICTED verdicts
  are issued at all — and the test asserts the absence of the branch, not merely the absence
  of the output.
- **G-E3 (absence is not failure).** **Zero CONTRADICTED** on empty, unparsable, absent,
  all-skipped, harness-errored, or partially-read evidence. Asserted over a synthetic
  fixture suite covering every shape named in A5–A8, and over the corpus. One failure fails
  the gate.
- **G-E4 (precision, unearned).** The accusing verdict is **not implemented**, and nothing
  short of a blind panel measuring it at **≥ 0.95** may write it — the same floor, the same
  protocol, the same reliability gate as the path-claim class: a fresh sample, new sealed
  decoys, key digest committed publicly before any item is judged, and **fewer than 27 of 30
  decoys correct voids the measurement with no headline.** Nothing else licenses that code.
  Not an internal review, not a maintainer's confidence, not a clean fixture suite. On the
  count above, that panel cannot presently be assembled at all, so the gate is not a schedule.
- **G-E5 (sealability).** A capsule minted over a `styxx-evidence/v0.1` record re-derives the
  verdict from the sealed bytes alone, and capsule refusals R4 and R5 remain untouched and
  still fire on a `--run`-resolved gate. If the new record cannot be sealed, it has not
  repaired the defect it was built to repair.
- **G-E6 (subset invariant on the incumbent).** No pull request in the corpus gains a
  `tests_pass` accusation relative to the shipped gate run without `--run`. Since the shipped
  gate without `--run` accuses zero times, the bar is zero and the gate is trivially
  satisfiable — which is the point: it is the tripwire that fires the moment someone adds the
  branch.

**Two blind panels, not one, before any future accusing verdict.** Panel A on EXTRACTION —
*does this sentence assert that the tests passed for this change?* — and Panel B on
ADJUDICATION. This lab has only ever measured B, on three instruments, and lost three times.

## What would make us not ship, and the failure we commit to publishing

- **G-E1 fails** (any impurity, or non-identical repeated output): the module does not ship.
  A bytes adjudicator that is not a function of bytes has no advantage over `--run` and
  inherits its unsealability. We publish that the repair did not repair.
- **G-E2 or G-E3 fails** (any CONTRADICTED on unbound, empty, or unparsable evidence): the
  module does not ship, and the failure publishes with the count attached, under its own
  name, at the same speed as the three failures before it. A single false accusation is a
  gate failure here, not a rounding error.
- **G-E5 fails**: the module ships only as a report-only reader with the sealability claim
  struck from every document, and we say we did not get what we came for.
- **The census comes back near-zero-bound.** If almost no `tests_pass` claim in the corpus
  carries any commit-bound evidence, **the correct move is to publish "unbound" as the
  finding and stop** — not to widen the channels until something binds. Widening until
  something binds is how the previous three repair cycles started, and this sentence exists
  so that a future maintainer has to argue with it in writing.

**Design as though report-only is PERMANENT, not a stepping stone.** The bound branch — the
only one Law 2 permits to accuse — fires **11 times across 1,775,765 changed files**, which
makes its precision **structurally unmeasurable**: you cannot assemble a hundred-item blind
panel with thirty sealed decoys from a channel that fires eleven times in a corpus. Under the
standing commitment, that implies the accusing verdict for `tests_pass` may never be earnable
at all. This document says so in those words rather than writing "pending measurement" and
letting it be read as "pending".

## The honest boundary

Stated here so it can be pasted into the tool's own output, not paraphrased away later.

**We do not verify signatures.** We base64-decode a DSSE payload and read it. The in-toto
spec's own words for what we are not doing: *"To obtain predicate information that is
authenticated, consumers MUST parse the Envelope's `payload`, and verify it against its
`signatures`."* We parse. We do not verify. Every run prints `signature: NOT CHECKED` beside
every value derived from that payload. If a caller ever supplies a raw public key, a passing
check upgrades exactly one proposition — *these bytes are the bytes that key signed* — and
prints `key trust: NOT ESTABLISHED BY US` immediately after. It never upgrades a verdict
about test outcomes. **Partial Sigstore verification will not be implemented at all**; half a
chain check reporting "verified" would be the most dangerous line of code in the repo.

**We inherit the CI's trust.** `result: "PASSED"` is an assertion by a producer, not an
observation by us. It can never yield CONFIRMED — laundering an unverified third-party
assertion into our own verdict is the move this lab has spent three failed cycles learning
not to make. The digest binds a *statement* to a commit id; it does not bind a *test
execution* to that commit. Nothing in the format stops a producer emitting PASSED against any
subject digest it likes. Our threat model is **an honest-but-unverified producer and a trusted
filesystem**, and on attacker-supplied repositories — exactly where EXTERNAL-1 found this
lab's instruments failing in the wild — that model does not hold and this output must not be
treated as a security control.

**Binding is necessary and not sufficient, and the insufficiency is invisible.** A report can
be perfectly bound, entirely truthful, and describe a suite that tests nothing: one matrix
leg of twelve, an `continue-on-error` leg, a `-k` subset, a `--maxfail` short-circuit, a
shard of N, the superseded first attempt of a re-run job, or a suite where every test is
skipped. Jest, under its default `reportTestSuiteErrors: false`, **deletes** a suite whose
file failed to compile — producing a fully green report with no trace of the broken file, a
blind spot no rule over these bytes can detect. Completeness is a second unsolved axis, and a
module that solves binding will be read as having answered sufficiency. It has not.

**The extractor is upstream and unmeasured.** `adjudicate_tests_pass` receives an evidence
dict and a commit — never the claim's scope, polarity, or provenance. The template that
decides *when* to consult the evidence is one regex, `diffgate.py:76`, with no capture groups,
no negation guard, no blockquote or code-fence stripping, and none of the mention-versus-use
repairs bought by three failed cycles — all of which are gated on `_PATH_KINDS`
(`diffgate.py:131`), which does not include `tests_pass`. A prior keyword sweep over
`external1_ledger.jsonl` found roughly 5,500 `tests_pass` extractions across the 71,016
eligible pull requests, of which a large share carry an explicit subset scope, a PR-template
checkbox (some of them **unchecked**, i.e. the author declining the claim), a conditional or
future tense, a "locally" qualifier, a self-disclosed exception, a quotation, or a negation.
**Those are shape counts from a sweep, not adjudicated precision, and must never be reported
as precision.** The point they establish is a direction, not a number: fixing the adjudicator
does not fix the extractor, and the extractor is where the problem starts.

## Prior art and credit

**The novelty claim in the plan of record was tested and refuted. We say so here rather than
wait to be told.**

The retired sentence is *"nobody deterministically gates natural language claims against
evidence bytes in CI."* It is false on at least three counts: **Cucumber/Gherkin** has bound
controlled natural language deterministically to execution and failed builds since 2008;
**Doc Detective** ships a GitHub Action that extracts testable steps from prose documentation
and fails CI when the documented claim no longer holds; **Jdoctor/Toradocu** (ISSTA 2018)
translated free-form Javadoc into executable oracles deterministically at 92% precision —
eight years ago, with a measured precision we are currently nowhere near. Separately,
*"nobody gates a build on evidence the claimant did not author"* has been false for roughly
eighteen years: **Gerrit's Verified label** with a `user=non_author` submit requirement exists
precisely so the author cannot supply their own verification, and **cosign
`verify-attestation --policy`** and **in-toto Witness** already evaluate a Test Result
predicate and fail non-zero.

**Do not let "deterministic" carry the novelty.** Danger exposes `github.pr_body`, danger-junit
parses JUnit XML and calls `fail()` — those three together are the whole substrate,
deterministic, running in CI, years old. A twenty-line Dangerfile does the mechanism. Our
contribution is that **we know of no one who pointed it at prose and measured what happened**,
and the measuring is the part that has cost this lab three published failures. **We claim the
measurement, not the machinery.** Any sentence anywhere in this project that leans on
"deterministic" to do novelty work is to be struck on sight; the rule stated below binds this
sentence too.

The conjunction that survived the sweep, with every clause load-bearing:

> We know of no other tool that adjudicates an author's **free-form** claim about a change
> they just made — prose not written to be machine-read — against evidence bytes **produced
> by a party other than the claimant**, deterministically, in CI. Controlled-language gates
> (Cucumber), docs-as-tests (Doc Detective) and NL-to-oracle translation (Jdoctor) each
> occupy part of this; supply-chain policy engines (cosign, Witness) and code-review labels
> (Gerrit) occupy the third-party-evidence half; **we know of none that combine the two, and
> none that adjudicate a retrospective report.**

Strike any clause and a named system already covers what is left, which is the test of whether
every clause is load-bearing. Note also what the conjunction no longer says: an earlier draft
ended it with *"emitting VERIFIED / CONTRADICTED / UNCHECKABLE and failing the build only on
CONTRADICTED."* This version has no CONTRADICTED branch to emit, so that clause would have
been a claim about code that does not exist. It is removed rather than softened.

And the honest centre of the position, which survived the sweep intact:

> The industry has built extensive infrastructure to verify that a claim was **made** —
> checklist enforcers require the box beside "I have run the tests" to be ticked, DCO requires
> the sign-off trailer, Conventional Commits requires the prefix — and **we can find none that
> verify the sentence is true** against a test attestation the claimant did not author.

We do not claim there is none. We claim we looked and did not find one, and the size of that
look is stated next.

**This sweep publishes as a REPORT-ONLY band, and this is its coverage.** One agent. One day.
Roughly **twenty queries**. Abstracts and product documentation, **not claim sets** — no full
texts, no patent families, no systematic protocol, no preregistered query list, no second
reviewer, no inter-rater check, and unverified leads left counted rather than chased. **It is a
pilot, not a systematic review**, and it publishes with those numbers attached exactly as
`RESULT_oath_prior_art_survey_2026_08_26.md` did.

**A prior-art claim is itself an accusing verdict about a field**, and this one is built on
twenty queries — thinner evidence than any accusation this lab has already retired for being
thin. **Absence of a hit in twenty queries is absence of evidence, and absence of evidence is
never a contradiction.** Never "first". Never "nobody". Always "we know of no other." Any hit
that refutes a clause retires that clause the same day, in writing, under its own name — the
way the plan of record's sentence was retired at the top of this section.

### Credit

**Credited openly, because an instrument that overstates its novelty has already failed its
own standard:**

- **in-toto** — the Test Result predicate (`result` / `passedTests` / `warnedTests` /
  `failedTests` / `configuration`, type URI `https://in-toto.io/attestation/test-result/v0.1`)
  as our evidence schema, and the Statement/DSSE envelope we read. We consume it; we did not
  invent it. The URI ends in `/v0.1` and that *is* the stability signal — a future major
  version is a different literal string and our matcher must emit UNCHECKABLE rather than
  guess.
- **in-toto Witness** — the CI execution recorder with signed policy. Better than we would
  build, and the name is theirs. We adjudicate; we do not record.
- **SLSA** — the non-forgeability trust boundary (Build L2/L3) that makes "an attestation the
  claimant did not author" mean anything at all.
- **sigstore / cosign** — `verify-attestation` with Rego and CUE policy as the deployed
  pattern for failing on attestation content.
- **Gerrit and the Android Open Source Project** — the Verified label and `user=non_author`:
  the cleanest statement we found that the author must not supply their own verification. We
  did not establish that it is the earliest, and we do not claim it.
- **Blasi, Goffi, Gorla et al.** — Toradocu/Jdoctor (ISSTA 2018) and C2S (FSE 2020), for
  deterministic natural-language-to-specification translation and for the precision bar.
- **Doc Detective** and the docs-as-tests community — for putting prose-as-testable-claims in
  CI. We know of no earlier deployment of that idea; we did not establish that there is none.
- **The Cucumber/BDD lineage** — two decades of binding natural language to execution.
- **cargo-semver-checks / japicmp / revapi** — the deployed pattern of failing CI when an
  author's declared claim contradicts the evidence of the diff. The closest mainstream
  analogue this sweep found; the only missing leg is that their claim is a structured
  field rather than a sentence.
- **The MSR 2026 Mining Challenge, the AIDev dataset authors, and the PR-MCI team** — for the
  corpus, the taxonomy, the human annotations and the problem statement. Our measurement is
  only possible because they published.
- **CodeFuse-CommitEval** — for a message-code-inconsistency benchmark, the earliest we know
  of, and for ceiling numbers that keep us honest.
- **Qodo (PR-Agent / Qodo Merge)** and **CodeRabbit** — the nearest commercial neighbours on
  compliance and prose-based merge blockers. Cite them before a reviewer does. The relevant
  asymmetry is not that we are deterministic and they are not; it is that we are obliged to
  publish a measured precision and they are not.
- **The EBTE, ECA and Evidence-Carrying Termination authors** — for the untrusted-claim-versus-
  server-held-fact architecture, and specifically for ECA's **opposing bet** that free-form
  text is inadmissible evidence. RESULT_v14's 0.16 is evidence in its favour, and that
  deserves a paragraph in whatever we publish next, not a footnote.
- **LEDGER** and the agent execution-provenance survey — for claim-to-evidence trace structure
  and for stating their own non-determinism plainly.
- **RealDiff** — whose exit code 3 for incomplete evidence is our UNCHECKABLE under another
  name, shipped by someone else, in CI, for pull requests. The abstention design is not unique
  to us either.
- **Necula's proof-carrying code** — the producer-proves / consumer-checks asymmetry (the
  party that ships the artifact bears the cost of the proof; the party that accepts it runs
  only a small trusted checker), with the epistemic-tier difference restated every time:
  proof-carrying code carries a *deductive proof of a safety policy*, checked by a tiny
  trusted checker, and a failed check is a theorem about the artifact. A capsule carries
  *empirical evidence and a re-derivable verdict*, and a failed re-derivation is a
  disagreement about bytes. We inherit the shape of the asymmetry and none of its soundness,
  and this module in particular checks no proof of anything.
- **Nuijten et al. (statcheck)** — the decade-old deployed precedent for deterministic checking
  of claims in prose nobody wrote for a checker.
- **honest-signal** — still owed credit for preregistration-as-runtime-precondition and for the
  vacuity check we still do not have.

Nothing ships under the name "witness". Nothing is called "proof-carrying data".

## Disclosed before the fact

**Reconnaissance preceded this document and is not hidden.** Before freezing this, we counted
file *shapes* in the corpus — how many changed files look like JUnit XML, how many paths look
like attestations. Those counts are small, and they are counts of shapes, not of verdicts:
they informed the decision to ship a classifier rather than an adjudicator, and they are not
the census result. The census result is what the preregistered classifier reports, and it does
not exist yet.

**One prior-art finding the gates should be read against.** The PR-MCI study measured a **1.7%
high-inconsistency base rate** across 23,247 agentic pull requests, its own embedding baseline
scored **F1 0.150**, and CodeFuse-CommitEval's best LLM reached roughly **80% precision on
rule-generated synthetic inconsistencies**. A base rate that low puts a hard ceiling on the
precision any accusing verdict can reach on this corpus. That number, rather than our template
set, may be the real explanation for 0.16 — and it is measurable before another panel is
convened. It belongs in the next preregistration, not in this one's gates.

## What this cannot support

It cannot support any statement about whether an agent lied. It issues no accusation and, in
this version, has no branch that could. It cannot support "the tests pass" as a verified fact
— at best it reports that an unverified attestation bound to this commit *asserts* PASSED. It
cannot be charted against EXTERNAL-1's precision figures: different object, different
population, and any such chart would be dishonest.

What it can support: a truthful, specific reason where today there is a misleading one, a
verdict that a stranger can re-derive from sealed bytes in a decade, and a census of how much
of the agentic-PR ecosystem carries any commit-bound test evidence at all. If that census
reads near zero, the honest finding is that the ecosystem is unbound, and we publish it and
stop.

---

*The strongest binding available — re-running the suite at the commit — is the one this
architecture has already, correctly, refused. This module exists because the one sound channel
is the one we will not take, and the honest consequence is a loud UNCHECKABLE with the missing
channel named. We are shipping a better reason, not a new accusation, and the gate that
matters is the one that keeps it that way.*
