# CORPUS STATE — the whole program, sworn at once

Fathom Lab · 2026-08-31 · Receipt: `corpus_census.json` — one leaf per certificate ever
stored in this repository: name, verdict, counts, and the SHA-256 of the stored bytes.
This document is minted into a capsule, so the file you are reading can prove, offline,
in your browser, the state of every result this lab has ever certified.

## The numbers

The census holds 206 stored certificates. Of these, 200 were sworn OATH-HELD at
issuance and 6 were sworn OATH-FAILED — because this lab publishes its failures under
the same seal as its wins.

Two measures of the same corpus, both kept: the numbers above are what the stored
certificates swear; the live audit re-derives every verdict at the current verifier and
tracks one historical verdict that has drifted since issuance. Stored history is never
edited — drift is recorded, not erased.

## What this capsule proves

Every certificate hash in the receipt pins the exact bytes of one certified result.
Change any certificate anywhere in the program's history — soften a failure, inflate a
count, delete an accusation — and its hash no longer matches this census; re-run the
census and it no longer matches this capsule; open this capsule and the tamper banner
says so. The entire research program is now one verifiable object.

## What it does not prove

That the receipts behind each certificate record reality — that chain lives in git
history and preregistration seals, per the capsule's own footer. And this census is a
snapshot: certificates issued after it are simply absent, which the count discloses.

## Repaired 2026-09-01, and what the repair taught

The counts in the section above are larger than the ones this document was first sworn
with, and the repair was forced by CI rather than noticed by a human: the certificate
stopped reproducing, and the guard that asserts no committed certificate silently stops
holding caught it on every supported Python. The superseded figures are in git history
where they belong, not restated here — a narrative retelling of a number is exactly the
kind of unbound claim this verifier is built to refuse, and it refused this paragraph
once already.

The cause is worth recording, because it is a defect in how this lab handles receipts
rather than a defect in this paper. While dogfooding `styxx.undeclared`, a generator was
allowed to rebuild `corpus_census.json` — the very receipt this document cites. The
census was regenerated, the corpus had grown, and the counts moved underneath a document
that had already sworn to them. Nothing was edited dishonestly and nothing was concealed;
the instrument simply noticed first.

**A receipt is history too.** Certificates are already treated as immutable — re-issues
are new commits and drift is tracked, never erased — but the receipts those certificates
cite were being regenerated in place, which silently invalidates every document citing
them. A claim measured against a moving receipt cannot stay true, and the fix is the same
binding law the evidence leg is being built on: a snapshot claim has to be bound to the
digest of the bytes it described, not merely to their filename. This document is repaired
here; the general repair is not yet built, and saying so is cheaper than pretending the
treadmill is a design.

---

*One file. Two hundred and six oaths. Nothing crosses unseen.*
