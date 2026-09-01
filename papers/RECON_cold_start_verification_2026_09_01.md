# RECON — what a stranger can verify, on a machine that has never run this code

Fathom Lab · 2026-09-01 · **RECON, not a result.** No preregistration covers this work. Nothing
here was frozen before it was measured, the sample is one environment and the pass is one, and
so **this document carries no headline finding**. It reports what a cold machine did when it was
pointed at the published artifacts and told to behave like a hostile-but-fair third party.

Environment: Windows 11, Python 3.12.10, no GPU, a virtualenv containing nothing but `pip`.
Package installed from PyPI before the repository was cloned, and kept separate from it
throughout — a stranger has the package, not the tree. Repository cloned at `daf5881`.

Every number below is either quoted from an observed run or marked UNVERIFIED. Where this
document says a thing was not found, that is absence of evidence and not a contradiction.

---

## What a stranger can actually do today

`pip install styxx` succeeded in 46 s into an empty virtualenv and installed **styxx 7.47.0**,
matching the version the capsules pin. It pulled seven dependencies — numpy, scikit-learn,
scipy, joblib, narwhals, threadpoolctl, cloudpickle. **No torch, no CUDA, no compilation step,
nothing that requires a GPU.** Eleven console scripts landed. Nothing failed.

From that install alone, and nothing else, a stranger can:

- **Verify all nine published capsules.** All nine reproduce their embedded verdict. Run from
  outside the clone, against the installed package, exit 0 on all nine.
- **Reproduce the corpus audit** in REPLICATIONS.md, on CPU, in 12 s, with all nine disclosed
  exceptions and every count matching.
- **Certify their own document** against their own receipts, mint their own capsule over it, and
  verify that capsule. This works end to end. On a document written for the purpose, the
  instrument correctly accused a number no receipt supported.
- **Do all of it with the network switched off.** See *First-run honesty* below.

That is a real and unusual amount of working machinery, and it is worth saying plainly before
the defects, because the defects are all repairable and this part is the hard part.

What a stranger **cannot** do today, following the published instructions, is run the CPU-only
replication targets. Those are covered under *Defects*.

---

## The capsule claim, adjudicated

**A capsule opened in a browser by someone with no styxx install does verify something, entirely
offline, and it is not the verdict.**

The precise position, stated in the two halves that matter:

**The offline half holds.** The capsules are genuinely self-contained. Grepping all nine finds no
`src`, no `href`, no `fetch`, no `XMLHttpRequest`, no CDN, no external host of any kind. Loading
one in a real browser and reading the network log afterwards returns **no network requests
recorded**. The document, the receipts, the certificate and the verifying code all travel inside
the single file, and the hashing is done by the browser's own WebCrypto. Nothing is fetched and
nothing is phoned home. On this point the artifact is exactly what it says it is.

**The self-verifying half is narrower than the phrase suggests.** What the embedded script
computes is the SHA-256 of the embedded document and of each embedded receipt, and compares each
against a hash recorded in the embedded certificate. That is an **internal consistency check**:
it establishes that the bytes in this file are the bytes this file's own certificate claims. It
does not re-derive the verdict. The verdict is read out of the certificate —
`['verdict', C.verdict]` — and displayed. The certificate carries no signature, no public key and
no anchor outside the file; a search of all nine certificates for any signature-shaped field
returns nothing.

The consequence is the thing worth writing down. Anyone who can edit the file can edit the
document, recompute the hash, write it back into the certificate, and the browser layer will
report **"integrity: all hashes match"** with no warning. We did this and it does. Re-deriving
the verdict — the step that catches such a forgery — requires the styxx package. **So the strong
form of the claim, that a capsule re-derives its own verdict in a browser with no install, is not
what these capsules do.** The weak form — that a capsule carries its own bytes, its own receipts
and its own certificate, and can show a reader offline that they are mutually consistent — is
exactly what they do.

**The capsule itself states this correctly.** Its own footer reads, in part, that the bands are
drawn faithfully from the certificate at layer 1 while "the verdict is reproducible by re-running
the real verifier over the embedded bytes (layer 2)", and that a capsule is "a portable binding,
not a portable oath of origin". That is an accurate description of the mechanism. The artifact
does not overclaim. Prose elsewhere that compresses this to "self-verifying" does, and the
compression is the part that should be retired.

---

## The tamper test, with the exact observed behaviour

No tamper had been run against these capsules before. Three copies of
`RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.capsule.html` were made in a temporary
directory; the committed file was never touched and the tree was confirmed clean afterwards.

**Tamper 1 — one byte, certificate left alone.** Inside the sealed payload, the paper's central
finding `Observed 0.16 against a floor of 0.95` was changed to `Observed 0.96 ...`. Exactly one
byte of 4,792 differs.

- Layer 2 — **detected**, exit 1:
  `CAPSULE FAILS VERIFICATION: - document bytes != certificate.document_sha256`
- Layer 1, in a browser — **detected**. Red banner, `INTEGRITY: FAILED`. The banner reads
  "TAMPERED — embedded bytes do not match this capsule's certificate. Nothing below can be
  trusted."

**Tamper 2 — the same edit, plus the certificate hash recomputed to match.** The file is now
internally consistent.

- Layer 2 — **detected**, exit 1, and detected for the right reason:
  `- verdict not reproduced: live OATH-FAILED vs embedded OATH-HELD`
  `- counts not reproduced: live {'VERIFIED': 6, 'ABSTAIN': 5, 'UNGROUNDED': 1} vs embedded {'VERIFIED': 7, 'ABSTAIN': 5, 'UNGROUNDED': 0}`
  The forged 0.96 is not supported by the sealed receipts, so re-running the instrument turns it
  into an accusation and flips the verdict. This is the mechanism working as designed, and it is
  the strongest single observation in this document.
- Layer 1, in a browser — **not detected.** `integrity: all hashes match`, no banner, verdict
  badge `OATH-HELD`, `accused 0`, and the document body renders the forged
  `Observed 0.96 against a floor of 0.95`. A reader without the package sees a clean capsule
  asserting the opposite of the paper's finding.

**Tamper 3 — the same capsule where WebCrypto is unavailable.** `crypto.subtle` is exposed only
in a secure context. Loaded in a context where it is absent (`isSecureContext: false`), the
embedded script throws at its first hash —
`TypeError: Cannot read properties of undefined (reading 'digest')` — before the tamper flag can
ever be set. Observed page state, for the **tampered** payload:

| element | rendered to the reader |
|---|---|
| verdict badge | `OATH-HELD` |
| integrity badge | `checking integrity…` |
| tamper banner | not shown |
| receipt rows | 0 |
| document body | empty |

**The capsule fails open.** The verdict badge is static HTML present in the file before any
verification runs; the script only recolours it. So wherever the crypto path does not complete,
the page still asserts a verdict, shows no error, and shows no banner — and a tampered capsule is
indistinguishable from a clean one. A verification artifact should fail closed.

---

## Defects, ranked by how badly they block a stranger

**1. The CPU-only replication targets cannot be run from the published instructions.**
`papers/disjoint-worlds/REPLICATE_legibility.md` says `pip install numpy scipy`. With numpy,
scipy and scikit-learn present, both `run_b37.py` (the row advertised as "One command from a
clone, no GPU") and `run_b45.py` (advertised as "CPU-only, ~4 s — the single easiest check in
this repo") die immediately:
`ModuleNotFoundError: No module named 'torch'`. This is the lowest-barrier, most-promoted
replication in the repository, and it is the one that stops first. A stranger with no repo
knowledge stops here.

The cause is worth stating because it is not what it looks like. **None of the six CPU scripts
uses torch at all** — `run_b37`, `run_b40`, `run_b41`, `run_b42`, `run_b45`, `run_b46` contain
zero references to it. They import the constant `CONCEPTS` from `run_g0clear.py`, which imports
torch at module scope for a GPU extraction path they never call. Substituting a stand-in `torch`
module that raises on any real use, `run_b45.py` completed in **3 s**, exit 0, verdict
`SHARED_FRAME_CONFIRMED_GEOMETRICALLY__island_rotated_away` — and reproduced **every scientific
field of the committed receipt exactly**, including `median_clique_affinity_k20` 0.848,
`clique_affinity_minus_null_p95_k20` 0.7914 and `seeds_qwen_below_clique_k20` 5. The only field
that differed was `prereg_commit`, which records the commit of whatever tree it ran in. The
computation is CPU-only and ~4 s exactly as advertised. The import graph is what is broken.

**2. A number that ends a sentence is never examined, and the omission is silent.**
`_NUM` in `styxx/certify.py` ends every alternative with `(?![\w.])`, so a numeric token
immediately followed by a period does not match. It is not abstained on; it does not appear in
the ledger at all. Observed, on documents written for the purpose:

| document text | verdict | exit |
|---|---|---|
| `This document reports a precision of 0.55.` | **OATH-HELD**, 0 tokens | 0 |
| `This document reports a precision of 0.55` | **OATH-FAILED**, 1 accused | 1 |

One period is the difference between an accusation and a clean certificate. `precision` is
measurement vocabulary, so the second form is obligated to ground and is correctly accused; the
first is never seen.

Across the corpus, **90 of 211 certified documents contain at least one sentence-final decimal
absent from their own ledger — 170 tokens in all.** These are not artifacts; several are the
outcome numbers of the gates their documents report, and in more than one case the threshold on
the same line *was* examined while the achieved value was not:

- `FINDING_p1_third_quarantine_2026_08_08.md` [OATH-HELD] — "G2 demanded 0.85 and got 0.75. G3
  demanded a 0.30 margin over the best constant and got 0.25." The demanded values are in the
  ledger. The achieved values, 0.75 and 0.25, are not.
- `FINDING_adjudicated_loop_2026_07_24.md` [OATH-HELD] — "This is the gate cycle 62 failed at
  0.7931." Unexamined.
- `PROGRAM_SYNTHESIS_2026_07_30.md` [OATH-HELD] — "the rescue rate on initially-wrong answers is
  0.08333333333333333." Unexamined.

This does not mean any of those numbers is wrong, and it does not mean any verdict is wrong. It
means the certificate's coverage is narrower than "every numeric claim in the document", and that
the shortfall is invisible rather than declared. This lab has already named the failure class, in
a comment in the same file describing an earlier instance of it: *"certified-by-omission, the
inverse of the oath."* This is another instance, still open.

**3. The capsule fails open where WebCrypto is unavailable.** Detailed above. Ranked here rather
than first because it needs a non-secure context to bite, but it inverts the artifact's purpose
when it does.

**4. Layer 1 cannot detect a coordinated forgery, and the page does not say so where a reader
will look.** Detailed above. The footer is accurate; it is the last paragraph of the page, below
a large green badge the reader has already read.

**5. REPLICATIONS.md never says to install styxx, and pins no version for the CPU target.**
Searching it for install or version instructions returns nothing. The corpus-audit output it
pins character-for-character is a function of the verifier version; the capsules pin
`styxx==7.47.0`, the replication document pins nothing. A stranger arriving in six months with a
later styxx has no way to know whether a divergence is theirs or the tool's.

**6. Step 4 promises CI that does not cover four of the seven targets.** REPLICATIONS.md says CI
runs `python scripts/verify_replication.py <target>`. That script knows exactly four:
`b2-adaptive`, `b2-static`, `parity-control`, `e1`. The three CPU-only targets and the corpus
audit have no entry, so the replications a stranger is most able to perform are the ones the
promised check cannot process.

**7. Running the advertised replication destroys the artifact it is compared against.**
`run_b37.py` and `run_b45.py` write `b37_result.json` and `b45_result.json` into their own
directory — the exact filenames of the committed canonical receipts. In a single clone, the act
of replicating overwrites the baseline. This audit ran them only in a throwaway copy.

**8. The pinned corpus-audit block is stale again, in the way the document confesses it went
stale before.** Running `python -m styxx.corpus_audit papers/` exactly as written reproduces
**every pinned line character-for-character** — all nine exceptions, all counts — and then prints
one line the block does not contain:

```
  epistemics: 6330 verified | obligated 2660 unobligated 3670 (rate 0.5798) | weakest 2116 (0.3343) | 0 pre-v1
```

Against a bar of "character for character", an undocumented extra line is a divergence, and a
replicator has no way to know it is benign.

**9. Two smaller ones in the same document.** The targets table still describes the corpus audit
as having "five disclosed exceptions" while the block below it lists nine and the prose says
nine. And the command exits **1**, which the document does not mention; a replicator scripting it
sees a failure.

**10. On Windows the pinned block cannot match byte-for-byte.** The audit prints CRLF; the block
is stored LF. A literal `diff` reports all ten lines as differing. Only after normalising line
endings does the comparison mean anything. Given that this document already carries a note about
CRLF hashes on Linux, the same care is owed to its own pinned output.

**11. The capsule is absent from the front door.** `README.md` mentions `styxx.certify` once, in
prose, with no command line, and the word "capsule" zero times. `styxx.capsule create --help`
documents `--cert CERT` but never says how to obtain a certificate. The mint flow is documented
only in the two SPEC papers and an integrations skill file. This audit found the sequence by
reading `--help` output and source. `OATH_CONTRACT.md` is the exception and is good: it gives
`pip install styxx` and a working `styxx.oathready` one-liner, and that command produced clear,
actionable output with an honest disclaimer attached.

---

## What is UNCHECKABLE from this environment, and why

- **The `file://` double-click path.** This is how a stranger would actually open a capsule, and
  it is the case this audit most wanted. The browser harness available here rewrote `file://`
  loads into `data:` documents, so no real `file://` origin was ever loaded. Layer 1 was verified
  over `http://127.0.0.1`, which is a secure context. Whether `crypto.subtle` is present for a
  double-clicked capsule is **UNCHECKED** here, and it decides whether defect 3 is an edge case
  or the default experience. It is one afternoon's work with a real browser and should be done
  before any claim about double-clicking is made.
- **Cold-network install time.** pip reported `Using cached` for every wheel; this machine had a
  warm HTTP cache. The 46 s figure is resolution and unpacking, not download. True cold-network
  install time is UNVERIFIED.
- **Every GPU target.** B2-adaptive, B2-static, parity control and E1 need ~8 GB VRAM. There is
  no GPU here. Four of the seven advertised targets are untouched by this audit.
- **Whether the receipts truthfully record reality.** No capsule claims this and neither does
  this document. That chain lives in repository provenance and cannot be checked from a sealed
  file.
- **Any platform but this one.** One Windows machine, one Python, one browser engine.
- **Whether defect 2 changes any published verdict.** Establishing that would mean re-certifying
  the corpus with a repaired extractor and comparing. This audit measured the coverage gap and
  stopped there, deliberately.

---

## This document, held to the contract it audits

`python -m styxx.oathready papers/RECON_cold_start_verification_2026_09_01.md` returns
**OATH-FAILED**: 65 numeric tokens, 0 bound, 22 accused, 43 abstained. It is published failing
rather than reworded until it passes.

The accusations are almost entirely on **quoted** text. The instrument accuses `0.16`, `0.95` and
`0.96` on the lines where this document reports the tamper string
`Observed 0.16 against a floor of 0.95` and its forged counterpart — because the word "floor" on
those lines is measurement vocabulary, so every number on them is obligated to ground in a
receipt. They are not this document's claims; they are the specimen it is reporting.

That is a cold-start reproduction of a finding this lab has already published and already
committed a failing document to: `SYNTHESIS_mention_and_use_2026_08_26.md` is itself OATH-FAILED,
accused on the digits of the formula it quotes as its specimen, because four instruments could
not tell a mention from a use. An outside pass, with no knowledge of that paper, walked into the
same wall by writing down what a tampered file said. The abstention count is the more interesting
half and is not examined here; the tool's own note — that on measured corpora roughly two in five
abstained tokens are claims nobody checked — applies to this document as much as any other.

---

## Our own limits

One agent, one environment, one pass, no preregistration. The defects were found by looking, not
by a protocol that said in advance what would be looked for, so the list is not a census and its
ranking is a judgement rather than a measurement. A second pass would likely find more; a
different platform would likely find different ones. Nothing here was replicated even once, by
anyone, including us.

Two of the findings — the sentence-final extraction gap and the fail-open badge — are the kind
that a preregistered adversarial pass should be pointed at properly, with the hypotheses written
down first. This document is the reconnaissance that says where to point it.

Nothing in this pass was repaired. The defects are left standing so that what the audit found is
legible, and a repair that arrives bundled with its own audit is a repair nobody has checked.

---

*Nine capsules verified. One tamper caught at both layers, one caught only by the installed
package, and one context where a tampered capsule shows a green badge and no warning at all. The
corpus audit reproduced. The easiest advertised replication did not run.*
