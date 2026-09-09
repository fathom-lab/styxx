# DURABILITY — how long does Appendix D's sentence stay true? (v8 draft, v0.2)

Fathom Lab · 2026-09-08 · **An analysis, not a result.** One lab, one reading, one session. This paper
makes no numeric claim of its own. Every number in it is either a parameter of the draft, a quotation
from a named file with its path beside it, a measurement taken on this box with the command that took
it, or a published constant with its document named — and where a figure is an order-of-magnitude
estimate the sentence says so. Every historical case named below is checkable by the reader against
the primary source named; none of them was checked against a primary source in this session (see
Limits). Not sworn; not in any log; it edits no file but this one.

Subject: `papers/v8/SPEC_v8_v0.2_draft.md`, measured in this session at 69,811 bytes, sha256
`fcb0c809c0ffb90184716766c424240572568f7381684e113a0e563f5d102a15`
(`Get-Item`, `Get-FileHash -Algorithm SHA256`). A lens run earlier the same day recorded 69,014 bytes
and sha256 `fa73e0d7…` for the same path. Neither number is a claim about anything except this box's
copy at those two moments, and the pair is the cheapest available demonstration of amendment C12 below:
the draft's git blob hash moved inside one working day, and entry #0 is defined as that hash.

---

## 1. The question

Appendix D says what a cert is for:

> Inputs: the cert, the log location (the URL or clone named in the log's README and in entry #0),
> and the log public key obtained from a channel the operator does not control.

and closes:

> If any of steps 1–2 or 4–5 is impossible from the cert, the log and the pinned key alone, the cert
> is defective and the spec has failed. That is the acceptance test for 8.0.

That sentence has no date on it. This paper asks two things. For how long does it stay true, given the
constructions the draft actually specifies? And when it stops being true, what does an old cert still
mean — what may a reader infer from it, and what must a reader never infer from it?

The paper does not propose a different system. Every amendment in §6 edits the existing draft, and
where the honest output is a limitation sentence rather than a mechanism, §7 carries the sentence.

Four lenses were run over the draft: cryptographic decay, format and tooling rot, subject rot,
institutional decay, and a precedent lens over long-lived record systems. Where two lenses found the
same mechanism the finding is merged and both readings are named. Where two lenses disagreed, the
disagreement is stated rather than resolved.

---

## 2. What breaks, and when

Ordered by horizon. "Lens" names the reading that found it; several rows were found by more than one.

| horizon | mechanism | lens | what survives it |
|---|---|---|---|
| already true | The only pin whose date is not the signer's own assertion is §8.5 anchoring, and §15.5 defers it. Every date in the system is written by the party under suspicion. | crypto, precedent | Order within a tree, as a hash property — but order is evidence of preregistration only under a dated pin. |
| already true | Zero mirrors. `mirror()` copies a local path, cannot fetch, and reports a log verified when it holds no head from outside. §8.6 makes mirrors the remedy for a forking operator. | institutional, precedent | Every offline check: id recompute, signature, root reproduction, consistency across held heads. Internal consistency, correctly implemented. |
| already true | Entry #0 carries the spec's git blob hash and not its bytes; the log's README is specified as four commands. The rules for reading the log live outside it. | format, semantic, precedent | The Merkle position of the document cert: a reader who independently obtains a candidate file can confirm or refute it. |
| already true | Canonical bytes are not reconstructible from the spec text; two shipped backends disagree in domain; seven hash fields have no preimage; the STH timestamp has no type; the Ed25519 accept set is not pinned. | format | Every cert issued by one implementation stays internally consistent. Interoperability does not. |
| already true | Appendix D's closing sentence contradicts its own step 3, and step 5 requires the operator to merge a stranger's submission (§8.3 v0). The worked subject `google/gemma-2-2b` is a gated repository. | subject, format, institutional | Steps 1–4 unaided. The challenge object itself, which is checkable off-log. |
| already true | `keys/issuers.json` is an unsigned file outside the tree, and §7.1 tier and §9 Disputed are computed from it at read time. | institutional | The promotion cert's refs and inclusion proof. The derived label is what moves. |
| 1–3 years | Alias subjects retire. §5.6 already prints `stale` for alias floors. | subject | The prompts, the outputs, the date and the signing key: an assertion, not a measurement. |
| 5–15 years | The git host, the org name and the domain. Bitbucket deleted every Mercurial repository on 2020-07-01; Google Code closed on 2016-01-25. | precedent, institutional | Any copy a reader already holds. Discovery fails, not verification. |
| 5–15 years | Weights and revisions stop resolving; the §9 challenge remedy expires with the subject, so a fabricated fingerprint of a deleted model becomes unfalsifiable by every mechanism the draft defines. | subject | Every cert-to-cert distance in Appendix B, and therefore `floor_c` recomputed from the logged floor runs. |
| 5–15 years | The checker: `env_lock` pins names and not artifact digests; the CI floor is declared green on py3.9–3.12, and 3.12's support window closes in October 2028 (PEP 693). | format, precedent | sha256 and Ed25519 as arithmetic. Only the packaging rots. |
| 5–20 years | The log key: one hardware token, one machine, a §15.1 that does not exist. Loss ends the log silently; inheritance extends it invisibly. | institutional | Everything up to a head a third party independently holds — which returns to the mirror row. |
| 20–40 years | No algorithm identifier in any signed bytes, and A.1 rejects any hash string that is not 64 lowercase hex. A successor hash is a major-version fork with no crossing refs. | crypto, format | Nothing is lost yet. The exit door is buildable today and not after. |
| 2030–2035 (policy) | Ed25519 falls to a CRQC. NIST IR 8547 ipd proposes ECDSA/EdDSA deprecated after 2030 and disallowed after 2035; CNSA 2.0 sets 2030 for NSS. | crypto | The Merkle construction, which is sha256 only: ordering and bytes under a root pinned and dated before the break. |
| 20–40 years | The seal. Binding is a collision property with the committer choosing both halves, and §7.2 sets no maximum interval between seal and reveal. | crypto | A reveal logged before the break keeps its meaning; the leaf index proves that something was committed. |
| 20–40 years | No renewal ladder and no party obliged to run it. RFC 4998 distinguishes timestamp renewal from hash-tree renewal for exactly this case. | crypto, precedent | If one renewal happens in time, everything it covered. If none does, text. |
| 10–100 years | The dictionary: verdict words, family labels, norm polarity, an item's expected answer. `refusal-boundary` polarity is deleted at the §4.5 pool schema. | semantic | Prompt bytes and output text, in the clear, judgeable under the reader's own rules. |
| 50–100 years | Interpretability: the exact channel is over token ids, which are indices into a tokenizer the log stores only by hash. | semantic, subject | Comparability. Two such certs remain fully comparable and permanently unreadable. |
| 100–1000 years | Every guarantee is conditional on a chain of human acts — anchor, mirror, renew, preserve. Break one link and every downstream link is void, and no later act repairs it. | crypto, precedent | Text: the recipe in the clear, the prompts in the clear, the outputs in the clear — unless `redacted`. |

---

## 3. Already true

These are not horizons. They hold at entry #0.

### 3.1 There is no dated pin, and a pin cannot be made retroactively

§8.5 is the only construction in the draft whose date is fixed by a party with no stake, and it ends
"Skip until a mirror exists"; §15.5 repeats it. The three pins the draft otherwise offers are all
dated by the pinner: §8.4's "a commit in its own repository" (a git committer date is a string the
committer sets), the STH `timestamp` (§8.6 concedes it is the signer's assertion), and `created`
(§2 concedes the same).

The consequence is not visible today and is unrecoverable later. On the day the signature scheme
falls, a reader cannot separate a 2026 log from a tree fabricated in 2041 and back-dated, because
every date in the system was written by the party under suspicion. The trap is sharper than it looks:
an adversary who fabricates a whole tree gets correct-looking order for free, so "the prereg leaf
precedes the result leaf" is true inside a fabricated tree. Order is evidence of preregistration only
relative to a root dated outside the operator's control.

The precedent lens and the crypto lens both read §8.5's ordering as backwards. Certificate
Transparency specified consistency proofs correctly and shipped gossip late: `draft-ietf-trans-gossip`
was never published as an RFC, and for years split-view detection was theoretical. Anchoring needs no
ecosystem at all — Haber and Stornetta's construction (J. Cryptology, 1991) was operated by Surety
Technologies as a weekly hash in a newspaper's classified pages from 1995; Surety is gone and the
newspapers are not.

A reader may infer, from an unanchored log: that the entries are internally consistent and ordered
relative to each other. A reader may not infer: when any of it happened.

One caveat the precedent lens adds and the draft should carry: an anchor's own service layer is the
fragile part. An OpenTimestamps proof returns as a calendar attestation that must be upgraded to an
on-chain attestation; a proof never upgraded is worth nothing once the calendar server is gone. And
an anchor establishes that these bytes existed no later than that block — not who produced them, not
that they are true, not that other roots were not anchored in parallel.

### 3.2 There are no mirrors, and the mirror function reports a verified log without one

§1.5 states the rule exactly: "with one key and no external pin, the log establishes internal
consistency and nothing more." The implementation does not carry the rule. `mirror(src, dst,
pinned_public, pinned_sth=None)` at `styxx/v8/log.py:854` copies a local filesystem path — it does not
clone or fetch, so §8.4's "clones" is not what the code does — and at `styxx/v8/log.py:981` it sets
`report["verified"] = not tamper and not misbehaviour`, reachable with `pinned_sth=None`. A mirror
holding only the operator's own heads therefore reports a verified log, which is the state §1.5 says
means nothing.

The realism check is the lab's own outside-participation ledger: `REPLICATIONS.md` line 152 carries a
single row and it records no outside replication. §15.5 defers anchoring until a mirror exists, and
nothing in §14's twelve weeks recruits one, so the two defences are each waiting on the other.

This matters at every later horizon, because the property that survives a signature break is stated
relative to a root a reader obtained through a channel the operator does not control, and today no
such channel exists.

### 3.3 The log stores the digest of its own dictionary

§6.1: `document`: `{ path, commit, git_blob_sha256, eol: "lf" }`. §8.2 fixes the log's contents to
`entries/`, `blobs/`, `sth/`, `keys/` and a README that is "the four verification commands, nothing
else", and says a mirror needs the entry files, the blobs and one STH. So a reader holding a complete,
correct copy of the log holds a one-way digest of this spec and no way to obtain it. Entry #0 is the
worst case: the earliest entry in the log is a hash of the rules for reading the log.

Three lenses reached this independently and each named a different casualty. The format lens: the JCS
profile, the domain tags, the leaf construction and Appendix A's preimages all live outside the log.
The semantic lens: so do the definitions of `same`, `drift`, `skew`, `identity` and
`beyond-floor-coverage`, and §0.1's glossary is a list of pointers, so when a directory is gone the
entry is a dangling reference with a checkable hash of nothing. The precedent lens: the same design
killed the BBC Domesday Project (1986) in about fifteen years while the Domesday Book (1086) is
readable after nine centuries, and the Sandia WIPP marker study (SAND92-1382, 1993) reached the
opposite conclusion for the same reason — a pointer to a message is not a message.

A reader may infer, from a document cert: that something with that blob hash was committed at that
tree size, and, given a candidate file, whether it is that thing. A reader may not infer: what the
document said.

### 3.4 Two implementations of the canonical bytes already disagree, and the rule is not written down

§2.1 delegates to RFC 8785 by reference and adds the key-sort sentence, then says the two backends
"are held to agree by a differential test". Run in this session from the repository root:

```
python -c "import sys;sys.path.insert(0,'.');from styxx.attestation import jcs;import hashlib,rfc8785;o={'\U0001F600':1,'ﬀ':2};print(hashlib.sha256(jcs(o).encode()).hexdigest());print(hashlib.sha256(rfc8785.dumps(o)).hexdigest())"
styxx.attestation.jcs  af1ac4d54ec7cce4014e7978885fb33c964bfbcd107e2ec0a34dd7d39669e5b6
rfc8785 library        987bb5001ca4ec15df060758f885be8bb8258e210ed02e2ab3c273d7a1fb669d
```

Two digests for one object. The lab already handles this in code —
`styxx/v8/jcs.py:_refuse_utf16_divergent_keys` refuses such objects — so the disagreement is contained
in the implementation and absent from the spec, which is the defect: a reimplementer working from the
document cannot reach the behaviour. Every cert the lab will actually issue has ASCII envelope keys,
so this is a reimplementation hazard rather than a live wrong-bytes bug, and the amendment is to close
the domain rather than to ship two backends that disagree inside it.

The same class, all found by the format lens, all live today:

- **No number-rendering rule.** RFC 8785 §3.2.2.3 defers to ECMAScript `Number::toString`; the draft
  never says a JSON number is a double, nor how `1` versus `1.0`, `1e21`, `5e-7` and `-0` render.
  `seq_logprob` and every Appendix B distance sit inside the digested bytes.
- **No duplicate-member rule.** RFC 8259 §4 leaves it undefined; a parser keeping the earliest
  duplicate and one keeping the last compute different `D` from the same entry file.
- **No timestamp grammar.** A.1 says "RFC 3339 UTC with `Z`" and stops. RFC 3339 §5.6 permits a
  fractional-seconds part of any length, so `2026-09-08T18:00:00Z` and `2026-09-08T18:00:00.000Z` are
  both conforming, are different strings, and `created` is inside `D`.
- **The STH `timestamp` has no type at all.** §8.1 signs over JCS of four fields and never says
  whether `timestamp` is a string or a number. RFC 6962 §3.2 and RFC 9162 §4.5 use a uint64 of
  milliseconds; a reimplementer following the cited RFC emits a number and every STH check between the
  two implementations fails with no locatable cause.
- **base64url canonicality is required and never defined.** §2.1 rejects "a non-canonical final
  base64url character"; A.1 cites RFC 4648 §5, which does not impose canonicality (§3.5 leaves it to
  the using specification). `base64.urlsafe_b64decode` accepts non-canonical trailing bits, so a
  reimplementer accepts 64 spellings of one public key.
- **The Ed25519 accept set is not pinned.** RFC 8032 §5.1.7 gives the cofactored equation and states
  that the cofactorless check is sufficient but not required — two conforming accept sets. Chalkias,
  Garillot and Nikolaenko, "Taming the Many EdDSAs" (SSR 2020), tabulates the divergence across
  deployed libraries. §2.1 does this work carefully for public keys and none of it for signatures.
  Worse: `D` excludes `sig` while entry bytes include it, so a signature mauled into a second valid
  form yields the same id, different entry bytes and a different leaf hash — the entry's id still
  recomputes, so §8.1's `TAMPER` rule does not fire, and §8.4 tells a mirror to publish the resulting
  inclusion failure as operator misbehaviour.
- **Seven hash fields have no preimage.** A.1 gives three and A.3 gives three more; `item_id`,
  `prompt_sha256`, `item_order_sha256`, `pool_sha256`, `context_sha256`, `action_sha256`,
  `entries_sha256` and `sublog_id` have none. `item_id` is load-bearing: A.3 orders items by `item_id`
  and `channels.exact.hash` concatenates digests in item order, so the battery hash is a function of a
  field the spec never defines.

A reader may infer, today: that certs from one implementation are self-consistent. A reader may not
infer: that a second implementation would compute the same ids, or that a rejected signature is
evidence of anything.

### 3.5 The acceptance test fails on its own terms, for three unrelated reasons

Appendix D step 3 concedes an unobtainable subject — "that is a fact about the subject, not a defect
in the cert" — and the closing sentence then declares the spec failed when step 4 is impossible. Step
4 cannot run without the subject. The subject lens, the format lens and the institutional lens each
arrived at this sentence from a different direction, and each added a separate reason it is already
unmet:

- The worked example at §2.2 is `google/gemma-2-2b`, a gated repository. An anonymous stranger
  executing Appendix D on the draft's own example subject does not reach step 4 today.
- Steps 4–5 need the recorded runtime and hardware. §5.4's coverage rule means a run on different
  hardware returns `beyond-floor-coverage` rather than a verdict, so the dynamic half of the test goes
  stale by ageing with nobody at fault.
- Step 5 files a challenge, and §8.3 v0 routes every append through the operator's CI and the issuer
  roster. The acceptance test therefore requires the operator to merge a stranger's submission — an
  institutional act standing in for a mechanism the draft defers to v1 with no date.

The v0.1 review already asked for the closing sentence to be rewritten; v0.2 adopted the exit code
(A-24, §6 exit 5) and not the rewrite.

### 3.6 The roster is an unsigned file outside the tree

`keys/issuers.json` has no leaf hash, no inclusion proof and no signature, and §7.1 makes it the
default trust set for `certified` while §9 makes it the default for Disputed. `mirror` copies `keys/`
verbatim and checks it against nothing, so mirroring propagates an edit rather than detecting it. A
successor with push access — or whoever holds the org name after it lapses — changes what the log
would have accepted at index N, years after index N was signed, with every id, signature and proof
still verifying.

The institutional lens adds the read-time half: the same promotion reads `certified` today and `lab`
in 2040 with nothing recording which roster produced the answer. Chrome's CT policy handles this in
two ways worth copying — per-log validity windows so an SCT is judged against the log's state at
issuance, and a stated maximum age past which the bundled log list is treated as stale rather than
silently applied.

### 3.7 Six other things that are already true

- **The ERRATA convention rewrites the file whose blob hash is entry #0.** The header says later
  changes are appended sections, never edits. Appending changes the file's bytes and therefore its git
  blob hash; a mirror does not hold the git history, so a mirror can never reach the version that
  matches. The measurement in this paper's header block is the same mechanism observed inside one day.
  RFC 1 (1969) is still readable because a published RFC is never modified; corrections are errata
  entries or new documents that Obsolete or Update the old one.
- **§2.5 does not say what an old verifier may still do.** "Verifiers accept `major.minor` ≤ their own
  and refuse higher with exit 3" is correct for interpretation, and leaf hashing, inclusion and
  consistency read no schema at all. An implementer who reads §2.5 as "refuse the entry" ships a
  verifier that stops auditing the tree as soon as an 8.1 entry appears — killing the durable half of
  the design at the earliest minor bump, for a reason the draft did not intend.
- **External artifacts are bound by a bare hash that no leaf covers.** §3.1's `items_blob` lives in
  `blobs/`, which §8.2 places outside `entries/`, so no Merkle proof covers it; the same holds for
  `weights_sha256`, `pool_sha256` and `git_blob_sha256`. §2.3 does it correctly for the three
  `*_sha256` fields whose materials sit inside the entry.
- **`token_ids_sha256` has no stored preimage.** §3.1's item object carries the digest and not `ids`,
  and decoding is not invertible. `channels.exact.hash` and `output_sha256` stay checkable and the
  binding between them never does: an issuer could publish arbitrary `output_text` beside arbitrary
  digests and no reader without the model could detect it.
- **The answer key is deleted at the schema boundary.** `bench/tasks/refusal.jsonl` holds 21 lines,
  11 with `"gold": "refuse"` and 10 with `"gold": "answer"` (counted in this session with
  `Select-String`), each carrying `should_refuse`; §4.5's pool schema is `{item_id, prompt, family}`.
  The word "correctness" appears once in the whole draft, in a §7.2 example about a different
  instrument. Every §5.2 verdict is a statement about token-id stability, and §12 files fingerprints
  under "documentation of the model as deployed", where a reader who came looking for a quality record
  will find one.
- **Eight verdict words already carry more than one sense in this repository.** `skew` is defined at
  §2.3 as a harness-or-lockfile difference and the same sentence claims it is the split
  `styxx/charon.py` makes, which is over verifier bytes — two objects, one word. `canary` collides
  with the planted false spans of `papers/sworn/`; §4's own gate S4-17 records it. `anchor` is a
  battery item in §4.4 and a chain timestamp in §8.5. `unmeasured` carries at least four senses across
  `styxx/` and §5.2's gate proposes a fifth. `identity` is a cert property, a set of subject fields and
  a verdict. And `EXIT` maps both `drift` and `identity` to 1, so the CI consumer §6 is written for
  cannot distinguish "the outputs moved" from "you are running a different model" without parsing the
  result cert.

---

## 4. Years: 1 to 20

### 4.1 The subject goes before the cert does, and it takes the remedy with it

§8.6 lists, under Mitigates: "a dishonest issuer fabricating results — challenges (§9) and independent
reproduction are the remedy." §9 makes that remedy conditional on running the model: a challenge
carries the challenger's own fingerprint under role `own`, and is valid only if that fingerprint is
comparable and shares every `S_identity` field. No subject, no challenger fingerprint, no valid
challenge, ever again.

This is the sharpest finding in the subject lens and it has a date attached at both ends. Alias
subjects go in 1–3 years; §5.6 already prints `stale` for an alias floor older than its window plus 30
days, and the retirements of `text-davinci-003` and the Codex endpoints are the standing case. Weights
go in 5–15; the published base rate for research artifacts is Vines et al., Current Biology 24(1),
2014, which reports the odds of a data set still being extant falling about 17% per year — a
measurement about research data, quoted here as a base rate and not as a claim about model hosts.

So a fabricated fingerprint of a model that is later deleted becomes permanently unfalsifiable by
every mechanism the draft defines, and its Disputed status freezes at whatever it was on the last day
anyone could run the model. The asymmetry is the whole problem: v8 puts Merkle proofs, STHs, mirrors,
a merge-delay bound and optional anchoring behind the durability of the record, and nothing behind the
durability of the referent.

There is one fact that is establishable only while the subject lives and checkable from log bytes
forever after: how many keys other than the issuer ever ran this cert while running was possible.
§14's ship gate (b) already requires exactly one such cert for 8.0, so the object exists in the draft
as a one-off gate rather than a standing per-cert property. Making it standing is amendment 6 and amendment C14.

A reader may infer, from an unchallenged old cert: nothing about whether anyone tried. Absence of
challenges measures the operator's remaining appetite for logging attempts against itself, not
scrutiny. This is the lab's own §5.3 rule — an agreement number without its detection power is not a
number — applied to the log's dispute channel, where the draft has not applied it.

### 4.2 Two lenses disagree about what to do with that

The subject lens proposes an archival status computed from `unavailable` verify certs: `archival` when
at least two such certs from distinct trusted keys reference a cert and no later verify obtained the
subject. The institutional lens shows why that is not computable in v0: logging anything, including a
failure, requires the operator's CI and the roster. A status that depends on strangers being able to
append is exactly as available as open submission, which §8.3 defers.

This paper prefers the weaker, computable form — a corroboration count over certs that exist, plus the
log's own state printed beside it — and records the stronger status as available only after §8.3 v1.
Two further gaps make the stronger form wrong today even where it is computable: exit 5 lumps "gated,
deleted, provider retired" into one code, so a live gated model and a deleted one present identically,
and the `weights_source` gating field the v0.1 review asked for was not adopted. Any archival status
computed over today's `unavailable` certs would mark every gated model as archival.

### 4.3 Custody: the host, the name, the key and the runtime

Four separate clocks, none of them cryptographic.

**The host and the name.** Appendix D's log location is "the URL or clone named in the log's README
and in entry #0", which today resolves to a GitHub org and a domain. Both are re-registrable after
they lapse; this is ordinary namespace takeover, not an exotic attack. A stranger following a
decade-old cert then reaches a live repository with a `log/` tree, a `keys/log.pub` and STHs that
verify under the new party's key — an internally consistent log that is not this one. Content
addressing does not close it: sha256 checks bytes a reader already holds and says nothing about where
to obtain them. The comparative case that worked is indirection: DOIs on the Handle System survived
three decades of publisher mergers because the identifier is not the location, and arXiv changed
identifier schemes in April 2007 while keeping every old identifier resolvable.

**The key.** §15 item 1 puts the private key on one hardware token on one machine, with no backup and
no threshold, and §8.3 cites "the custody machine (§15.1)" — a section that does not exist; §15 has
six numbered items and no subsections. Two failure directions, both unstated. Lost: the log cannot
grow, cannot answer a challenge, and cannot cover its last entries with a head, so it enters a closed
state without declaring it. Inherited or seized: the holder signs heads under the same `log_id`, and
every reader who did not personally hold a pre-death head sees a consistency-clean extension. Because
`created` and the STH `timestamp` are both assertions, entries appended in 2040 under the 2036 key and
back-dated are indistinguishable from 2036 entries to anyone lacking the 2036 head.

**A defence that would not work.** Splitting the log key across custodians makes this worse rather than
better: a threshold of holders signing under one identity is a mechanism for making a change of hands
invisible, which is the thing a transparency log exists to expose. The institutional lens is right to
forbid it. What works instead is a successor public key named while the outgoing key is still usable,
plus attested destruction — and §8.1's existing rotation path cannot do it, because it announces the
new key with a `document` cert signed by the old key, so it requires the key to survive the loss it is
meant to survive.

**The runtime.** Appendix D step 1 needs, all off-log: a Python interpreter, `styxx`, `rfc8785`,
`cryptography` and `jsonschema`, whose half-lives differ sharply and which the draft treats as one
thing. `cryptography` has required a Rust toolchain to build from source since 3.4 (February 2021).
`jsonschema`'s validating draft has moved five times. The declared CI floor is "green on py3.9–3.12":
3.9 reached end of life in October 2025 (PEP 596) and 3.12 reaches it in October 2028 (PEP 693), so
the declared support window closes within roughly two years of entry #0. And §2.3's `env_lock` is
"pip freeze / lockfile text, verbatim" — names and versions, no artifact digests — so a reader in ten
years may hold a byte-exact lockfile that resolves to nothing, and cannot check a recovered wheel
against the cert. The standing registry case is the npm `left-pad` unpublish of March 2016.

What survives all four: the arithmetic. sha256 and Ed25519 verification are fully specified in RFCs
that carry their own test vectors. Only the packaging rots — which is why amendment 8 puts the vectors and the
constants in the log, and amendment 15 makes the four verify commands runnable with the standard library
alone.

### 4.4 An honest ending reads as misbehaviour

Nothing in the draft covers the end. §8.3 sets a 24-hour maximum merge delay and `mirror` prints
entries beyond the last head as `unpublished`, so a log that simply stops looks like continuous
withholding forever, and `mirror` sets `verified: False` on it. A log that ended cleanly reports as
broken.

Two mechanical aggravations. STH files are named by tree size (`sth/<tree_size:012d>.json`), so two
heads at one size cannot coexist: an operator with nothing to append cannot publish a heartbeat, and a
re-signed head overwrites its predecessor with an ordinary file write — the publication history is not
itself append-only on disk. And `mirror`'s same-size/different-root detector can therefore only fire on
a head pinned from outside, never on the operator's own published history.

CT solved the operator-quits case mechanically: browser log lists carry per-log states including
read-only and retired, with retirement timestamps, so previously issued SCTs stay evaluable rather
than becoming garbage. A one-lab log has no vendor to publish that state, so the substitute has to be
computed by the reader — which is amendment 10, and which errs in both directions and must say so.

### 4.5 The layout has a ceiling, and the log's byte cost is budgeted nowhere

§8.2 stores one file per entry in a git repository. CT hit this and responded twice — temporal
sharding, then the static tiled-log design published in 2024 — because per-entry serving is expensive
to operate over decades. The draft has arrived at the right substrate (static files) at the wrong
granularity, and the fix is not a rewrite: it is to state an entry ceiling past which the log closes
at a final tree size and a successor log begins, cross-linked in both directions.

Separately, §2.3 stores the chat template, the system prompt and the lockfile verbatim inside the
signed envelope of every cert, so they cannot be deduplicated without breaking the signature, and a
floor is R = 5 or 8 runs each logged as its own cert. §13's budget table has an em-dash in every
GPU-hours row and no bytes column at all. The durability consequence is not money: §8.4 and §8.6 make
mirrors the remedy for a forking operator, and the price of mirroring is unmeasured. For scale, the
probe receipt in this directory is 408,629 bytes for 48 prompts at `max_new_tokens=16` (measured in
this session; sha256 `1ef0d4a6…`, matching §0.1). Keeping the materials in the clear is the
durability-correct choice and the size is its price, not an oversight — the amendment is to measure it
and to say so.

---

## 5. Decades: 20 to 50

### 5.1 The signature dies and the tree does not, and the draft never separates them

Shor's algorithm recovers an Ed25519 private key from the 32-byte public key that §2 puts inside every
cert and §8.2 publishes at `keys/log.pub`. From that day an adversary can mint a cert with any
`created`, any `body`, a valid `sig` under fathom's key and a valid STH — an entire fabricated tree
that passes all four §8.1 checks. The policy horizon is published: NIST IR 8547 ipd (November 2024)
proposes ECDSA/EdDSA deprecated after 2030 and disallowed after 2035, and CNSA 2.0 (September 2022)
requires quantum-resistant signing for NSS software and firmware by 2030. Neither is a claim about
when a CRQC exists.

What is untouched is sha256. The leaf hash `sha256(0x00 || entry_bytes)` and the node hash rest on
collision resistance, where quantum collision search gives no useful advantage under realistic memory
cost models (Bernstein, "Cost analysis of hash collisions", 2009), and NIST IR 8105 (April 2016) Table
1 lists SHA-2 as needing a larger output while EdDSA and ECDSA are listed as no longer secure. So a
root pinned outside the log before the break still proves, afterwards, the exact multiset and order of
entries under it — which is precisely the guarantee §7.2 leans on for preregistration, and precisely
the one the draft never separates from the signature it will lose.

A reader in 2040 holding a cert cannot tell which of the two he has. That is a sentence problem, and
amendment 5 is the sentence.

Authorship after the break degrades to "a key with this public value signed these bytes at or before
the pin date" — usable, and only with a pin, which returns to §3.1.

### 5.2 The seal is what a hash break destroys soonest and most completely

`commitment = sha256("styxx.v8/seal/1" || 0x00 || salt || UTF-8(JCS(body_to_reveal)))` with a 32-byte
issuer-chosen salt. Binding here is a collision property, not a preimage property, and the committer
chooses both halves of the colliding pair — the strongest position an attacker can occupy, and exactly
the position that fell for MD5 (Flame, 2012, forged a code-signing certificate with a chosen-prefix
collision) and for SHA-1 (Stevens et al., CRYPTO 2017, identical-prefix; Leurent and Peyrin, USENIX
Security 2020, chosen-prefix). Unlike every other use of sha256 in the draft, the seal's security must
hold from the moment of sealing until the reveal, and §7.2 sets no maximum for that interval. An issuer
who seals in 2026 and reveals after chosen-prefix collisions are practical can reveal a hypothesis he
did not commit to, and every check in §7.2 passes.

Secondary, and free to fix: the preimage concatenates without a length prefix and the reveal validator
is not told to check the salt's decoded length, so the salt/body boundary is fixed by convention.
§2.1 does specify decode-length rejection for public keys.

A reveal logged before the break keeps its full meaning, because the reveal is itself an entry at a
leaf index — the collision opportunity closes at the reveal. That is what makes a `reveal_by` date the
right shape of fix.

### 5.3 There is no exit door from sha256, and no party obliged to walk through it

`id` is the literal string `"sha256:" + lowercase hex(D)` and A.1 says validators reject any other
length or case. The `sha256:` prefix looks like an algorithm tag and is a fixed literal: no field
selects an algorithm and none carries a second one. Adding a successor changes the id grammar, which
changes the canonical bytes of every cert that references anything, which is a schema change — and
§2.5 makes a 9.0 cert invisible to every deployed 8.x verifier while an 8.x id cannot appear in a 9.0
cert's refs. The result is two disjoint logs with no crossing proof.

The precedent is not encouraging in either direction. Git's SHA-1 to SHA-256 transition opened in 2017
and is still not the default, in a content-addressed store far simpler than this one. CT fixed
SHA-256 in RFC 6962 (2013) and fixed it again in RFC 9162 (2021), with no agility mechanism after
twelve years. Timelines from publication to a practical break, for calibration only: MD5 published
1992, collision 2004, weaponized 2012; SHA-1 published 1995, collision 2017, chosen-prefix 2020.
sha256 dates from 2001.

**Where the lenses disagree.** The crypto lens wants the cheapest possible door: relax A.1 to
`<alg>:<hex>` against a registry whose only 8.0 member is `sha256`, scope the rejection to "when it
must compute that hash", add an optional `alt_ids` excluded from `D` exactly as `id` and `sig` are, and
make an algorithm addition a minor version. Zero bytes on the wire until an algorithm is added. The
format lens wants a required `alg: {hash, sig}` object in the cert envelope and in the STH's signed
body — on the order of 40 bytes per cert and per STH, inside the signed preimage — on the ground that
an old verifier's refusal should be a decision rather than an accident. Both are free only before entry
#0 and expensive after. This paper records both and does not pick; amendment C3 states the cheaper one,
because it is strictly weaker and strictly cheaper, and notes that the required field is the variant
that makes exit 3 correct rather than incidental.

**And the door is not enough without someone to walk through it.** Long-term archives solve this with
the RFC 4998 ladder: timestamp renewal, and hash-tree renewal — rebuild the tree over the archived data
and the old evidence record under a new hash algorithm, before the old one weakens (RFC 6283 is the XML
expression; BSI TR-03125 operationalizes it). v8 has neither step and names no party who owes them.
§8.4's mirror duties are clone, verify, pin and gossip; nothing about re-hashing. §15.1 assigns log-key
custody and nothing after.

A defence that would not work: re-signing the STHs under a fresh key. RFC 4998 separates the two steps
precisely because once H is broken, an old timestamp over H(data) no longer binds the data.

The ladder is expressible as an ordinary v8 result kind and costs one O(n) pass per epoch. Naming a
successor custodian is not an engineering object, and where none can be named that fact belongs in the
spec as a stated limitation rather than as an unassigned duty.

### 5.4 Algorithm retirement is a roster event and the draft only has key events

§8.6 handles compromise per key: "the roster marks the key `retired` from an index". An algorithm break
retires every key at once, with no index to point at, because the break has no publication event the
log can observe. And the migration path is signed with the algorithm being retired, so an adversary
with a recovered key forges a rotation announcement naming his own successor and a reader following the
chain lands on the adversary's log. The roster is a file in the log, so whoever can sign the log can
rewrite it. There is no construction here in which a new-algorithm key is bound by anything other than
an old-algorithm signature.

NIST SP 800-208 (October 2020) approves LMS (RFC 8554) and XMSS (RFC 8391) for exactly this profile —
a low-rate signer on a single custody machine, which is what §8.3 and §15 describe — and FIPS 204
(ML-DSA) and FIPS 205 (SLH-DSA) were published in August 2024. Naming the successors costs nothing
today and removes the argument later. As one published constant for scale: FIPS 205 tabulates a
7,856-byte signature for SLH-DSA-SHA2-128s, which is small beside an entry file and is a real
operational cost for a stateful scheme, to be measured at adoption rather than estimated here.

### 5.5 The dictionary drifts on the same clock

Three mechanisms, all found by the semantic lens, all on a decades scale and all cheaper to fix now.

`task_family` gates certification by exact string equality on a vocabulary that lives in a Python tuple
(`FAMILIES` in `styxx/v8/consts.py`) and that no cert references, and the mapping from the source data
is deferred to a build script that is not in the tree at this commit. The two vocabularies genuinely
differ: `bench/suite.yaml` names factual, reasoning, refusal, creative, adversarial; v8 names recall,
short-reasoning, instruction-following, format, refusal-boundary. §4 rotates fixed-v1 yearly. Rename a
family and every existing promotion silently stops matching, with no cert issued and no log entry
recording the change — §7.1's own guarantee that there is no flag to move a reading between sections
runs backwards. Re-use a name over different content and an old promotion silently extends its scope.

`refusal-boundary` polarity is deleted at the pool schema, as counted in §3.7. For some items a human
of any era reconstructs the direction; for the items that carry the actual boundary they cannot — the
source lines include a paywall-bypass request, which is a norm indexed to one decade's business
models, a national identifier scheme, and a proper noun whose referent must be looked up in a world
that will not be there. A `same` verdict on this family says the token ids did not move. It says
nothing about safety, and §5.2 gives the family no verdict of its own.

An item's expected answer rots under a stable prompt. `bench/tasks/factual.jsonl` fact-019 asks which
planet has the most moons "as of 2023" — an author who already knew the answer was epoch-bound and
wrote the epoch into the prompt, which is the correct move and an admission that the class exists. A
reader in 2126 comparing two fingerprints on fixed-v1 gets `same` if both models emit the 2023 answer
and `drift` if the newer one emits the then-correct answer: under §5.2 the model that got better reads
as the model that changed. The verdict is true; the inference from it is false.

---

## 6. Centuries: 100 to 1000

At this range no defence in the draft is cryptographic, and the honest question is what the artifact
degrades into.

**Every guarantee is conditional on a chain of human acts.** Someone anchors, someone mirrors, someone
renews before each algorithm weakens, someone preserves the subject. Break any link once and every
downstream link is void, because the repair would itself have to be dated by the algorithm that has
fallen. There is no construction that removes the dependence on custody, and the only honest move is to
say so and to say what remains.

**What remains is text, and it remains by an earlier decision.** §2.3 stores the chat template, the
system prompt and the environment lock verbatim; §4.5 stores prompts in the clear; §3.1 stores
`output_text` unless redacted. So a v8 cert whose cryptography has expired degrades into a readable
description of an experiment — what was asked, of what, under what decoding, in what environment, and
what came out — rather than into an opaque digest. That is a consequence of the recipe-complete-in-the-
clear decision, and the draft should defend that decision on these grounds rather than treating it as
a storage cost.

**Which makes §15.2 the load-bearing long-horizon choice in the document.** §15.2 records the redaction
default as housekeeping: clear text for outputs and prompts, "this item records the decision". At a
century scale it is the switch between producing a record and producing an unreadable digest chain. A
redacted cert's residue is a set of hashes whose preimages exist nowhere. §4.5 already states the live
consequence of a redacted battery; the archival consequence is unstated.

**Comparability outlives interpretability, and that is a trap.** The exact channel is over token ids,
which are indices into a tokenizer §2.2 stores only as `tokenizer_sha256`. Two certs whose models are
gone remain fully comparable — a distance computes, a §5.2 verdict emits — and permanently unreadable
as language. `output_text` is the only bridge back, and `--redact` removes it. Carrying the tokenizer
bytes would fix it and is not free: measured in the local HF cache on this box, `tokenizer.json` for
`models--google--gemma-2-2b` is 17,525,357 bytes and for `models--Qwen--Qwen2.5-0.5B-Instruct` is
7,031,645 bytes. Content-addressed, that is paid once per subject rather than per cert, and it is a
per-subject operator decision rather than a schema requirement.

**A defence that would not work: archiving the weights.** §8.2 stores the log as a git repository
published as static files. A 2B-parameter model in bf16 is roughly 5 GB — an order-of-magnitude figure
from the parameter count, not a measurement here — and bf16 tensor bytes are effectively
incompressible, so that cost is paid by every clone forever, since git history cannot be pruned without
rewriting the tree the STHs commit to. GitHub blocks any single pushed file over 100 MiB. And the
deeper objection is structural: §8.6's only stated mitigation against log-operator misbehaviour is
mirrors, and §8.4 requires a mirror to clone entries, blobs and STHs, so putting gigabytes per subject
into that clone destroys the mirror population the whole threat model depends on. The honest move is
the locator and the Non-goal sentence, not the archive.

**A defence that would not work: a free-text falsifier.** An unreproducible cert would be more useful
if it carried the sentence that would contradict it. But §0's rule is that anything not reproducible
from the cert is a note, not a claim, and §12 already forbids free text outside a promotion `claim`. A
free-text falsifier would be an unverifiable issuer sentence dressed as a claim — worse than none. If
it is added at all it must be generated from a versioned template table keyed on the cert's own fields
and recomputed by validators, exactly as §2.3 recomputes the materials hashes. This paper does not put
it in the amendment list; it is recorded here as the shape any future version would have to take.

**What a reader must never do.** A fingerprint cert whose cryptographic layer verifies perfectly reads,
to anyone who is not an expert, as a verified statement about a model. §1 principle 5 anticipates half
of it — "The log stores; it does not endorse. Inclusion ≠ validity" — and addresses validity rather
than obtainability, and is stated once in a design-principles section rather than printed with any
verdict. Section 9 carries the sentence this needs.

---

## 7. What survives every horizon examined

The test applied here is strict: an item belongs on this list only if it survives a broken signature
scheme, a dead operator, and a vanished subject, all three at once. Items that survive two of the three
are not on it.

1. **The recipe as readable text.** §2.3 stores the chat template, the system prompt and the lockfile
   verbatim inside the entry bytes, and §4.5 stores prompts in the clear. Survives a broken signature
   because reading text needs no key; survives a dead operator because the bytes are in every copy;
   survives a vanished subject because it describes the run rather than pointing at the model.
   Condition: it does not survive `redacted: true`, and `output_text` in a blob does not survive a
   mirror that dropped the blob.
2. **The prompt-and-output corpus.** The same three reasons. What is left when everything else has
   gone is a set of prompts and the strings one system emitted for them — an object a later reader can
   judge under their own rules rather than inheriting ours. This is the largest surviving thing the
   format produces and it survives by the §15.2 decision, not by any mechanism.
3. **Every cert-to-cert distance in Appendix B.** The distances read stored fields, never a running
   model, so any two logged fingerprints stay comparable with no key, no operator and no subject —
   and `floor_c = max(D_c)` therefore stays recomputable from the logged floor runs, as does the whole
   Appendix C selection arithmetic. Comparability is what survives; interpretation is not (section 6).
4. **A cert's internal arithmetic.** Given the entry bytes, the derived `*_sha256` fields recompute,
   the sworn numeral bindings of §2.4 resolve into logged bodies, and `output_sha256` checks against
   `output_text`. Needs sha256 and nothing else — no key, no operator, no model. Its own horizon is
   §5.3's, which is why the exit door has to be built before it is needed.
5. **Ordering and content under a dated external root.** The Merkle construction is sha256 only, so
   inclusion, consistency and "entry i precedes entry j" survive a signature break intact — this is
   the guarantee the draft leans on hardest and the one it never names separately. It survives a dead
   operator and a vanished subject for the same reason. **It is on this list conditionally**: today
   there is no dated root outside the operator's control (§3.1, §3.2), so this property is currently
   available in principle and not in fact, and the condition cannot be created retroactively. Amendments C1 and C2 are the whole of the difference.
6. **The algorithm-prefixed hash string.** Every hash in the draft is written `sha256:<hex>` and never
   bare. This survives all three because it is a property of the notation, and it is what makes a
   migration expressible at all. It survives as a possibility rather than as a guarantee — A.1's
   current rejection rule closes the door the prefix opens, which is amendment C3.

Not on the list, and worth naming because each looks as if it belongs: the append-only property (needs
an outside head, and none exists); authorship (dies with the signature scheme); "the prereg preceded
the result" as evidence of preregistration (order survives, its evidential force needs a dated pin);
the seal's binding (dies at chosen-prefix sha256, with an unbounded exposure window); tier and Disputed
status (recomputed at read time from a file that is not in the tree); and anything at all from a
redacted cert.

---

## 8. Amendments to the draft

Every item edits the existing draft. Ordered by (buys most / costs least). Section numbers are the
draft's.

1. **§1, as principle 7, repeated as Appendix D's last line — custody conditionality.** Add: "The
   guarantees in this spec are conditional on custody. A reader of an old cert may infer only what the
   surviving evidence supports: with a dated root obtained outside the log, the order and the bytes of
   the entries under it; with an unbroken renewal chain to that root, the same at any later date; with
   neither, nothing but the text." Cost: six lines. It costs the claim that the format outlives its
   operators, which it does not.
2. **Appendix D closing sentence — split static from dynamic acceptance.** Replace with: "**Static
   acceptance (permanent).** Steps 1–2 must be possible from the cert, the log and the pinned key
   alone, for the life of the log; a cert failing here is defective and the spec has failed. **Dynamic
   acceptance (while the subject and a compatible runtime exist).** Steps 3–5 are possible while the
   weights are obtainable at the recorded revision and the recorded runtime can be built. A cert whose
   dynamic half is no longer reachable is not defective; it is historical." Cost: two sentences
   replacing one, plus the admission that the headline acceptance test has a perishable half.
3. **§2.5 — version refusal concerns interpretation only.** Add: "A verifier that refuses a cert's
   schema version MUST still compute its leaf hash from the entry bytes and MUST still verify inclusion
   and consistency proofs covering it. Version refusal is exit 3 and is never `TAMPER`." Cost: two
   sentences. It prevents an 8.1 entry from disabling the archival layer of every 8.0 verifier.
4. **§8.4 and `mirror()` — say when a mirror has checked nothing.** `mirror()` returns
   `verified: "internal-only"` rather than `true` when `pinned_sth is None`, and `verify` prints
   `mirrors: n` and `external_pins: 0` beside every verdict. Cost: one changed return value at
   `styxx/v8/log.py:981`, one printed field. It stops a self-consistent log from reading as an audited
   one, today, when the count is zero.
5. **New §8.7 Cryptographic horizon — name the guarantee that dies and the one that lives.** "Every
   signature in this log is Ed25519 and every hash is sha256. A cryptographically relevant quantum
   computer makes every Ed25519 private key recoverable from its public key; from that day no signature
   in this log distinguishes the issuer from a forger. It does not affect the Merkle construction:
   relative to a root_hash a reader obtained through a channel the operator does not control, and can
   date, the inclusion proofs, the consistency proofs and the ordering of entries continue to hold,
   because they use sha256 only." `verify` prints `warrant: proof <ok|fail>, authorship
   <signature|pinned-root@<date>|none>`. Cost: about twelve lines and one output field.
6. **§2.6 — `corroboration`, computed at read time like `public`.** The set of logged `verify` results
   referencing this cert whose issuer key differs from its own and whose `overall` is not
   `unavailable`, each with its STH timestamp; `verify` and §12 print either "corroborated by N key(s)
   other than the issuer" or "never run by any key but the issuer". Add to §8.6 "Does not defend
   against": "a dishonest issuer whose subject becomes unobtainable before anyone else runs it — the
   §9 remedy requires the challenger to run the subject, so such a cert is permanently unfalsifiable by
   this spec." Cost: one reverse index over the walk §2.6 already does, one template row, zero envelope
   bytes.
7. **§9 and §12 — absence of challenges is not evidence.** Print, beside every dispute count, the
   log's state (`open` / `presumed closed at <head>, <date>`) and the count of logged reproductions;
   promote §9's existing below-floor-challenge sentence into a counted, displayed number. Cost: two
   printed fields and one template row.
8. **§6.1 and §8.2 — put the bytes in the log.** The `document` body becomes
   `{ path, repo, commit, git_blob_sha256, bytes_blob, media_type, eol }`, `blobs/` is redefined to
   hold document bytes as well as item payloads, and append refuses a `document` cert whose blob is
   absent. Entry #0 MUST carry it. Add as early entries: the conformance vector set, the reference
   verifier source with its language and version declared, and the six normative RFCs (8785, 8032,
   6962, 9162, 4648, 3339) with each byte length recorded from the file at the moment it is added.
   Rewrite §8.2's README line to a prose restatement of the constants — the JCS rule, the id preimage,
   the three domain tags, the leaf and node constructions, the empty root, the encodings — sufficient
   to reimplement a verifier with no other file. Amend Appendix D's input list to "and nothing else;
   every normative document is in the log", and add step 0: "Read entry #0's bytes. Every word in every
   verdict is defined there." Cost: 69,811 bytes once for this file, on the order of 1 MB for the RFCs,
   and a handful of entries at the head of the log. Against a single probe receipt of 408,629 bytes it
   is not a budget item.
9. **§8.2 and §7.1 — the roster becomes log entries, and tier carries its basis.** Roster changes are
   `document` results with `{roster_op, name, key, from_index}`; `keys/issuers.json` is demoted to a
   derived cache reproducible by replaying them in index order, and `mirror` recomputes it and reports
   any mismatch as misbehaviour. Tier and Disputed print "as of tree_size N under roster <cert id>",
   and a client whose roster is older than a stated maximum age prints `stale` (the age is a parameter,
   not a measurement). Cost: about forty lines of replay, one check in `mirror`, two printed fields.
10. **New §8.7 A closed log.** Define states `active | readonly | closed`, published as a `document`
    cert; a reader finding no newer head under `log_id` at any listed location for more than a stated
    period marks the log presumed closed at that head, and entries beyond it are reported as a fact
    about the ending rather than as a defect. State in the same paragraph that the presumption errs in
    both directions — an idle log reads as closed, a resumed log reads as suspicious — and that the
    period is a parameter. Rename STH files to `<timestamp>Z-<tree_size:012d>.json` with a `latest.json`
    pointer, and sign a heartbeat head on a stated cadence whether or not the tree grew. Cost: one
    subsection, one filename format, one hardware-token touch per period for the life of the log.
11. **§3.1 — bind the blob.** A blob referenced by `items_blob` is appended as its own entry so its
    digest sits at a leaf; append refuses a cert whose blob is absent or does not hash to it; `mirror`
    reports `blobs_missing: n` and names the entries; `item_id` and `token_ids_sha256` stay in the
    signed body even when the rest of the item moves to the blob; `verify` prints proof status and
    artifact status separately and never reports `same` on a cert whose external artifacts it did not
    re-hash. Cost: one entry per blob, roughly 25 KB of retained pairs in the envelope for a 256-item
    battery, one added field in the verify body.
12. **§3.1 and A.3 — store `token_ids`.** Required per item for every subject kind, with a validator
    recomputing `token_ids_sha256` from it and rejecting a mismatch at append and at verify. Cost: on
    the order of 100 KB per fingerprint at 256 items and `max_new_tokens: 64` — an order-of-magnitude
    estimate from integer serialization width, not a measurement — and §3.1 already permits it to live
    in the blob, so the signed envelope need not grow. It converts the exact channel from an issuer
    assertion into an arithmetic fact.
13. **§2.2 and Non-goals — say where the subject went, and say the log does not hold it.** Add an
    optional `subject.archive` array of `{kind, locator, bytes, container_sha256, asserted_at}` with
    three sentences: the lab does not operate the archive, a locator that no longer resolves does not
    invalidate the cert and never changes a verdict, and an archive entry has the same standing as
    `created`. Add to Non-goals: "The log does not store model weights. §8.2's storage cannot hold them
    and holding them would price out the mirrors §8.6 depends on." Add `weights_source: {gated,
    license_id, acceptance_required}` as an environment field, and make exit 5's body record
    `{attempted, status, reason: gated | not_found | retired | network}`. Cost: on the order of 200
    bytes per cert, no storage, no new verification code.
14. **§2.2 and §3.1 — `evidence_class`.** Computed at append from `subject.kind`:
    `reproducible-measurement` for weights, `attested-observation` for alias, printed with every
    verdict and in §12, with the template sentence "An alias fingerprint records that this key asserted
    these outputs from this endpoint label at this time. It is not evidence that the endpoint, the
    provider, or any model existed as described." A weights cert whose subject becomes unobtainable is
    read as `attested-observation` from that point. Cost: one computed field, one template row, reusing
    the §2.6 code path.
15. **§2.3 and §14 — pin the runtime by bytes, and make the checker minimal.** `env_lock` MUST be a
    lockfile carrying a per-artifact digest for every pinned dependency (`pip freeze` does not;
    `--require-hashes` requirements, `uv.lock` and `poetry.lock` do). §14 weeks 1–4 gains: the four
    verify commands MUST also run with only the Python standard library on the path, and CI runs the
    conformance vectors in that mode; schema validation is explicitly not required for Appendix D steps
    1–2. Cost: a build-script change that roughly doubles `env_lock` bytes, plus a pure-Python Ed25519
    verify — RFC 8032 §6 carries an illustration and §7 the vectors — and one CI job.
16. **§0.1, §5.2, §6.1 — a verdict lexicon with scope objects, and a `glossary` result kind.** §0.1
    gains, for each verdict word, the object it is about (`skew` is a fact about the instrument;
    `identity` is a fact about the subject's hash fields and not about a cert id; `anchor` is a battery
    item in §4 and a chain timestamp in §8.5) plus a collisions list naming the other document's path
    for `canary`, `unmeasured` and `TAMPER`. Replace the §2.3 sentence claiming charon's SKEW/DRIFT
    split with: "charon v0.1 makes a different split, over verifier bytes (`styxx/charon.py`,
    STATUSES); the words are reused, the objects are not." Add a `glossary` result kind carrying its
    own bytes, and `body.vocab` on every `verify` and `challenge` naming the lexicon its verdict words
    were emitted under. Cost: one glossary subsection, one added result kind, one field per verdict
    cert, one extra entry per lexicon revision. It closes the S5-07 gate without deciding it.
17. **§4.5 and §4.6 — date the battery items.** Add `answer_epoch` (RFC 3339 date, or null when the
    item is not time-indexed) and the sentence: "A fixed-v1 item can become false without the subject
    changing. `drift` on such an item says the output moved, never that the output got worse." Cost:
    under 30 bytes per item and one editorial pass over the 84 committed task lines.
18. **§6.1 and §9 — let the issuer mark its own error.** The `response` body becomes
    `{ target, challenge (optional), disposition: new_floor | cause_identified | concession |
    retraction, detail }`; a `retraction` requires no challenge, may be issued only by the key that
    signed the target, is displayed attached to the target and never without it, and a retracted cert
    may not be a ref of role `previous`, `sensitivity` or `noise_plan` in any later cert. Add the ref
    role `target`. Cost: one enum value, one optional field, one display rule, one append check.
    Nothing is deleted; the record keeps the original and the notice, which is how the scientific
    record has handled it since 1665.
19. **§5.3 — one positive control that outlives its library.** "The committed weight perturbation MUST
    be a deterministic function of the reference weights specified by committed code with a fixed seed,
    never a stored checkpoint; the code's `code_sha256` is in the sensitivity result." Add: "The δ1
    quantization arms are reproducible only against the exact bitsandbytes build in `env_lock`; a
    sensitivity receipt whose arms cannot be regenerated is read as an attested observation." Cost: two
    sentences and a perturbation script of a few dozen lines.
20. **§3.1 and §15.2 — say what redaction costs at the far end.** "Redaction removes the only part of a
    fingerprint that stays readable after the subject is gone. A redacted cert's century-scale residue
    is a digest chain with no preimage anywhere. Redaction is permanent erasure, not a display
    setting." Print it on every redacted cert; forbid `redacted: true` on a cert whose `public` would
    compute true; add the row to §12's non-public-evidence category. Cost: three sentences and one
    append-time comparison against a flag §2.6 already computes.
21. **§13 and §8.4 — measure the log's bytes and report mirror completeness.** Add a "log bytes
    (measured)" column beside the empty GPU-hours column, filled from the opening docket's receipts,
    and: "A mirror that does not hold `blobs/` states so; certs whose bodies live in a blob it does not
    hold are listed `incomplete` and are not counted as mirrored." Keep the materials in the clear and
    say in one sentence why the duplication is the durability-correct choice. Cost: one table column,
    two sentences, and a measurement the lab has to take anyway.
22. **§8.2 and §8.4 — state the layout ceiling and the mirror contract.** "The v0 layout is defined up
    to a stated entry ceiling; past it the log is closed at a final tree size (§8.7) and a successor log
    begins, cross-linked by a `document` cert in each naming the other. A layout change is never a
    rewrite of an existing log." And: a public drift claim requires the target cert to be included in an
    STH held by at least two mirrors under administrative control other than the operator's — a
    parameter, chosen for independence rather than count. Cost: two sentences and one printed field
    now; the recruitment is organizational work the draft currently treats as optional, against a
    replication ledger whose only row records no outside replication.

### Cheap now, impossible later

These cost sentences, bytes or a scheduled job today. None of them can be added after the event they
defend against, and several cannot be added after entry #0 is signed. This subsection is the point of
the paper.

- **C1. §8.5 and §15.5 — anchor from entry #0.** Submit the `root_hash` of at least one STH per
  calendar month, and of the STH covering entry #0, to a public timestamping aggregator; store the
  returned proof under `log/anchors/<tree_size>.ots`; mark an anchor complete only when the calendar
  attestation has been upgraded to an on-chain attestation, and print `anchor: pending` until then.
  `mirror` prints the earliest anchored (tree_size, date) pair as the log's dated pin, and until one
  exists `verify` prints `warrant: authorship none`. Delete "Skip until a mirror exists" from both
  places. Add to §14's tag conditions: at least one anchored STH, verified from a fresh clone. Also add
  the three sentences saying what an anchor does not prove (who, whether true, whether other roots were
  anchored in parallel). **Why now:** a pin cannot be created retroactively. Every property in §6 item
  5 is waiting on this one act, and the day the signature scheme falls is the day it stops being
  possible. Cost: one scheduled job on the machine that already runs `styxx log sth`, one file per
  anchor, and either nothing or one fee per anchor — record the actual proof size and fee from the
  earliest anchor's receipt in §13's budget table rather than estimating them here.
- **C2. §8.2 and Appendix D — pin the log key where the operator cannot reach it.** "The log public key
  and its `log_id` are pinned at publication in at least three records the operator does not control,
  each named here by a permanent identifier; entry #0 lists all three. A reader who can reach none of
  them treats every STH as unpinned and every inclusion result is printed `internal-consistency-only`."
  Add `locations` to entry #0 and the README with at least one identifier no registrar can reassign,
  and one plain sentence: "a log reached at a URL whose STHs do not verify under `log_id <hex>` is a
  different log, not this one." Add an advisory `log_hint: {log_id, locations}` to certs, with the
  standing of `created` and never authoritative. **Why now:** the pin must predate the compromise, the
  lapse, or the break it is checked against; a pin taken afterwards proves nothing. The lab already
  operates a deposit channel it has not pointed at this.
- **C3. A.1 and §2.5 — leave the door open on the hash.** "Hash strings have the grammar
  `<alg>:<lowercase hex>` where `alg` is drawn from a registry whose only 8.0 member is `sha256` (64
  hex). A validator rejects a hash string whose `alg` it does not know **when it must compute that
  hash**; it MUST NOT reject a cert merely because a supplementary field names an unknown `alg`." Add
  an optional envelope field `alt_ids`, excluded from `D` exactly as `id` and `sig` are, so adding an
  algorithm never changes a signature. Add: "A hash- or signature-algorithm addition is a minor
  version; removing one is a major version. A verifier of 8.n accepts refs whose ids use any registry
  algorithm and resolves them by the algorithm named in the string." And in §2.1's ref rule: a ref is
  resolved by lookup, never by recomputing a digest under the reader's preferred algorithm. **Why
  now:** the grammar cannot be relaxed retroactively for certs already signed, and §2.5 as written
  makes a successor a disjoint log with no crossing proof. Cost: about six lines, zero wire bytes until
  an algorithm is added, one branch in the resolver covered by vectors §14 already schedules. The
  format lens prefers a required `alg: {hash, sig}` in the envelope and the STH's signed body instead,
  at roughly 40 bytes per cert and per STH; both are free only before entry #0, and this paper does not
  pick between them.
- **C4. §2.1 — pin the Ed25519 accept set.** "Signature decoding rejects a decoded length other than
  64, an R that is not a canonical point encoding, and an S not in [0, L). Verification uses the
  cofactorless equation `[S]B = R + [k]A'`; batch verification is forbidden for cert and STH
  signatures. `tests/test_v8_keys.py`'s hostile section is extended to signatures, and `conformance/v8/`
  carries the divergence vectors of Chalkias et al. 2020 with an expected reject for each." **Why
  now:** any conformance vector recorded before this clause lands has to be regenerated, and a cert
  whose validity depends on which library the reader chose fails Appendix D on day one. Cost: three
  lines, one rejection branch, and a vector file built the way the lab already builds them for keys.
- **C5. New Appendix A.5 — canonical bytes, in full.** The JSON subset; the ECMAScript number rules
  written out; the escape table; key sort by UTF-16 code unit; UTF-8 with no BOM; duplicate-member
  rejection at parse time; base64url canonicality stated as the permitted final characters (a 32-byte
  value is 43 characters ending in one of `A E I M Q U Y c g k o s w 0 4 8`; a 64-byte value is 86
  characters ending in one of `A Q g w`); a normative restriction that every object key in a v8 cert
  matches `[A-Za-z0-9_]+`, which makes the UTF-16 question unreachable and lets a minimal
  implementation sort by byte comparison; "Every timestamp is exactly 20 characters,
  `YYYY-MM-DDTHH:MM:SSZ`; no fractional seconds; a leap second is recorded as `:59`"; and in §8.1, that
  the STH `timestamp` is a string in that form, deliberately unlike RFC 6962 §3.2. Close with one
  worked example printed as hex bytes and its sha256, and retire §2.1's "checked fallback" sentence in
  favour of the rule. **Why now:** every one of these is free before the earliest signed cert and STH,
  and afterwards each is a re-issue or a `log_id` rotation. Cost: about two pages, on the order of 6 KB
  in a 69,811-byte file, and one `propertyNames` line in the envelope schema.
- **C6. A.1 — a preimage for every hash field.** One table row per field, covering `item_id`,
  `prompt_sha256`, `item_order_sha256`, `pool_sha256`, `context_sha256`, `action_sha256`,
  `entries_sha256` and `sublog_id`, with `prompt_sha256 = sha256(UTF-8(prompt_text))` and `item_id` a
  stated function of the prompt bytes rather than an opaque label, so the A.3 order is derivable from
  the prompts alone. Add to A.2 that a shard filename containing a newline or a space is rejected at
  append, since `weights_sha256` joins on newlines. **Why now:** the battery hash's own ordering
  depends on `item_id`, so shipping this wrong re-issues entry #1 and every fingerprint above it.
  Cost: one table, under a page.
- **C7. §7.2 — bound the seal.** "The commitment preimage is `"styxx.v8/seal/1" || 0x00 ||
  uint8(len(salt)) || salt || UTF-8(JCS(body_to_reveal))`; the reveal validator rejects a `salt` that
  does not decode to exactly 32 bytes." Add `reveal_by`, an RFC 3339 date: a seal unrevealed at
  `reveal_by` is printed as `seal expired`, can never be revealed into a valid result, may not be a
  ref of role `prereg`, and the count per issuer key appears in the compliance view. Add to §8.6 "Does
  not defend against": a lost salt; an unrevealed seal is an ordering fact and nothing more. **Why
  now:** the length prefix cannot be added to a preimage that has already been signed, and the
  exposure window is unbounded until a date field exists. Cost: one byte in the preimage, one date
  field, one derived display state. It costs the ability to sit on a seal indefinitely, which is the
  point.
- **C8. §8.1 — name the successor while the outgoing key still works.** "The operator MAY publish,
  while the key is usable, a `document` cert naming one or more successor public keys. A head under a
  successor key is accepted only when it carries a verifying consistency proof from a head under the
  predecessor key AND the successor was named in an entry covered by that predecessor head. A rotation
  announcement is signed by the outgoing key and countersigned by the incoming key over the same
  preimage, and is anchored before a mirror accepts it. Escrow or threshold-sharing of the log private
  key is forbidden: continuity is expressed by a named successor key, never by a shared one." Add a
  `document_kind: "closure"` body `{final_tree_size, final_root_hash, reason, successor_keys,
  key_disposition}` that must be the last entry and covered by the final head. Name the successor
  algorithms now — a stateful hash-based scheme approved in NIST SP 800-208 for STHs, ML-DSA (FIPS 204)
  for issuer certs — so the migration has a destination and the `sig_alg` registry has values. Say
  plainly in the same paragraph that a sudden death executes none of this, which is why the heartbeat
  and the reader-side presumption of amendment 10 exist. **Why now:** the rotation path is signed with
  the key it is meant to replace, so it cannot be executed after the loss it defends against. Also
  replace §8.3's dangling "(§15.1)" with a written custody procedure. Cost: one countersignature per
  rotation, which is rare by design, and a paragraph.
- **C9. §6.1 and §8.4 — define the renewal ladder now, execute it later.** Add a `renewal` result kind:
  `{ epoch, alg, sig_alg, covers_tree_size, covers_root_hash, root_<alg>, entries_root_<alg>,
  external_artifacts, previous_renewal }` — a cert issued while both the outgoing and the incoming
  algorithm are unbroken, carrying the whole log recomputed under the new hash and re-signed under the
  new signature algorithm, referencing the previous renewal, and anchored before it counts. `mirror`
  prints a log with no renewal newer than its registry's review date as `unrenewed`, and §15 gains a
  standing decision: who renews, and on what cadence, named before entry #0 and recorded in a
  `document` cert. **Why now:** a renewal must happen strictly before the outgoing algorithm weakens;
  after it, no action recovers anything, and re-signing the STHs does not help. Cost: about ten lines
  of spec and one O(n) pass per epoch. The cost that is not engineering: if no successor custodian can
  be named, that fact belongs in the spec as a stated limitation rather than as an unassigned duty.
- **C10. A.2 — capture the weights manifest while the weights are in hand.** The cert carries
  `weights_manifest`, the sorted `(filename, sha256)` pairs themselves, with `weights_sha256` remaining
  the digest of their join; and `weights_tensor_sha256`, a digest over sorted per-tensor lines
  (`name dtype shape sha256`), placed in `S_identity` beside it, with §5.2's `identity` verdict naming
  which of the two differed — a differing `weights_sha256` with a matching `weights_tensor_sha256` is
  a repackaging, not a model change. **Why now:** the manifest and the tensor digests can only be
  computed from the model directory the fingerprint run already holds. Afterwards, a genuinely
  preserved copy that was re-sharded or renamed can be identified only all-or-nothing, and re-sharding
  is a routine repository operation. Cost: on the order of a hundred bytes per shard, one extra pass
  during a load the harness performs anyway, 32 additional bytes for the tensor digest.
- **C11. §4.5 — carry the answer key across the schema boundary, recorded and never read.** The pool
  and item schemas gain `source_gold` and `source_gate` verbatim from the source line, plus
  `norm_polarity: refuse | answer | unspecified` and `norm_basis` naming the policy, jurisdiction and
  date the polarity was set under. None is read by `verify` and none produces a verdict. Add to §4.6:
  "A `refusal-boundary` item's polarity is a fact about the policy in force when the pool was built. A
  `same` verdict on this family is a statement about token-id stability, never about safety." **Why
  now:** the source lines exist today, in this tree; the polarity of the items that carry the actual
  boundary is not reconstructible by a later reader, and the schema currently deletes it at build time.
  Cost: on the order of 60–90 bytes per item on the observed source line shapes, so roughly 20 KB on a
  256-item battery, and one free-text string per battery.
- **C12. Header and §14 — entry #0 is never edited.** Replace the ERRATA sentence with: "the file whose
  blob is named in entry #0 is never modified again. Each erratum is its own `document` cert, carrying
  its own bytes, with a ref of role `previous` to the cert it corrects; readers reconstruct the current
  text as the original plus the errata chain in log order." **Why now:** the moment an erratum is
  appended, entry #0's `git_blob_sha256` names a file that no checkout and no mirror can produce — and
  the header block of this paper records that digest moving inside one day. Cost: one sentence changed,
  one cert per erratum, and the loss of reading one file end to end.
- **C13. §6 and `styxx/v8/consts.py` — give `identity` its own exit code.** `EXIT` maps `drift` and
  `identity` both to 1; give `identity` 6, and rename §5.3's printed `same (sensitivity unmeasured)` to
  `sensitivity-unmeasured` so no state whose name contains `same` exits non-zero. **Why now:** an exit
  code is a CI contract and is free to change only before 8.0 is tagged, which §14 says it is not yet.
  Cost: one integer, one string, one table row.
- **C14. §14 and §2.6 — make outside corroboration standing rather than a one-off gate.** §14's tag
  condition (b) already requires one `verify` result under a key that is not fathom's; make the same
  act a standing expectation per fingerprint, and state that submitting a failed attempt (exit 5) is
  expected rather than optional. **Why now:** the field in amendment 6 is retrofittable and the evidence
  it counts is not. Corroboration is establishable only while the subject can be obtained, and the
  subject horizon is 1–3 years for an alias and 5–15 for weights. Cost: one sentence in the ship gate
  and one in §9; the work is asking, which is organizational and cannot be automated, because its whole
  value is that a different party performs it.

---

## 9. What this spec must never claim

The sentences below are written in the words the spec should carry. They are limitations, not
mechanisms; where a lens proposed a defence that would not actually work, the limitation replaced it.

- **On custody.** "The guarantees in this spec are conditional on custody. From a cert whose
  cryptographic chain has lapsed, a reader must never infer that the bytes are the bytes the issuer
  signed, that the prereg preceded the result, that the issuer is who `issuer.key` names, or that the
  log is complete. Such a cert is a readable recipe, and the recipe is stored in the clear so that this
  remains true."
- **On the log's own state, printed with every verdict while it is true.** "This log has no external
  pin and no mirror under administrative control other than the operator's. Consistency proofs
  establish internal consistency only."
- **On dates.** "Until an anchor exists, every date in this log — `created`, the STH `timestamp`, and
  any repository commit date — is an assertion by the party the reader is checking."
- **On what a signature break leaves.** "From that day no signature in this log distinguishes the
  issuer from a forger. Ordering and bytes continue to hold relative to a root a reader obtained
  outside the log and can date, and to nothing else."
- **On an unobtainable subject.** "From a cert whose subject can no longer be obtained, a reader may
  infer only that these bytes were signed by this key, that they were logged at or before this tree
  head, that the arithmetic inside them is self-consistent, and that the distances between this cert
  and other logged certs are what they are. A reader may not infer that the model described existed,
  that any model produced the recorded outputs, or that a `same` verdict in this cert's history means
  the model was unchanged — §4.6 concedes that training against a published battery defeats the canary,
  and §15.3 makes the battery public. Inclusion is not validity, and inclusion is not obtainability."
- **On alias subjects.** "An alias subject's identity is three vendor-chosen strings with no bytes
  behind them. When the provider, the alias or the region name is retired, an alias cert can still be
  compared to another alias cert and can no longer be resolved to a subject."
- **On correctness.** "A fingerprint records what the subject emitted. No cert in this spec carries an
  answer key, a grader, or a pass/fail, and no verdict in §5.2 is a statement about correctness,
  quality or safety; `same` means the token ids did not move further than the logged floor."
- **On the exact channel.** "A fingerprint's exact channel is over token ids, which are indices into a
  tokenizer the log stores only by hash. When the subject is gone, a fingerprint remains fully
  comparable and permanently uninterpretable; `output_text`, when present, is the only bridge back to
  language, and `--redact` removes it."
- **On weights.** "The log does not store model weights. §8.2's storage cannot hold them and holding
  them would price out the mirrors §8.6 depends on. A cert's reproducibility ends when the weights
  become unobtainable; the manifest lets a reader confirm that weights obtained from any source are the
  ones the cert used, and nothing more."
- **On absence.** "An absence of challenges is not evidence that a cert was reproduced. A cert in a
  closed log carries a dispute channel that closed on a stated date."
- **On the seal.** "A seal proves that a commitment existed at that leaf index. After a chosen-prefix
  collision on the commitment's hash is practical, it does not prove what was committed."
- **On mirrors and gossip.** "The append-only property is checkable only at the granularity of the
  mirrors' pull period; an entry appended and withdrawn between two pulls is invisible to every party."
- **On what an anchor proves.** "An anchor establishes that these bytes existed no later than the
  anchoring block. It establishes nothing about who produced them, whether they are true, or whether
  other roots were anchored in parallel."

---

## Limits

One reading, by one lab, in one session. The horizons in §2 are arguments from published deprecation
schedules, published break timelines and named institutional failures; none of them is a measurement,
and a horizon that reads "5–15 years" is a range chosen to bracket the cases named in that row, not an
estimate with an error bar.

No historical claim in this paper was verified against a primary source in this session. The RFC
numbers and sections, the NIST and FIPS document numbers and dates, the PEP end-of-life dates, the
publication and break dates for MD5 and SHA-1, the Bitbucket and Google Code shutdown dates, the
`left-pad` unpublish, the Vines et al. availability figure, the CT gossip and log-list history, the
Domesday and Rosetta cases, and the Haber–Stornetta and Surety history are all cited so that a reader
can check them, and all of them are stated from memory of the literature rather than from a fetched
document. Where a number is quoted from a published table — the SLH-DSA-SHA2-128s signature size, the
NIST IR 8547 ipd dates — the document is named and the number was not re-derived here.

The four measurements taken on this box in this session are the spec's byte count and sha256, the probe
receipt's byte count and sha256, the two JCS digests from the one-line reproduction, and the
`refusal.jsonl` line and label counts. Each is a fact about this checkout at this moment and about
nothing else; the spec's own digest moved during the session, which is recorded in the header block.
The tokenizer file sizes and the `styxx/v8/log.py` line numbers are quoted from the lens reports that
produced them and were not re-measured here.

The claim that a given amendment "cannot be retrofitted later" is an argument about the construction,
not a proof. Each such claim rests on one of three properties: a pin cannot be dated after the fact, a
preimage cannot be changed after it has been signed over, or evidence about a subject cannot be
gathered after the subject is gone. Where none of those three applies, the item is in the numbered list and not in
the Cheap now subsection.

This paper does not decide the operator-gated items it touches, and it names no cost in money. Where a
figure would have been money — an anchor fee, a mirror's storage — the amendment says to record the
measured value from the receipt rather than to estimate it here.
