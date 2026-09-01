# CORPUS STATE — the whole program, sworn at once

Fathom Lab · 2026-08-31 · Receipt: `corpus_census.json` — one leaf per certificate ever
stored in this repository: name, verdict, counts, and the SHA-256 of the stored bytes.
This document is minted into a capsule, so the file you are reading can prove, offline,
in your browser, the state of every result this lab has ever certified.

## The numbers

The census holds 203 stored certificates. Of these, 197 were sworn OATH-HELD at
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

---

*One file. Two hundred and three oaths. Nothing crosses unseen.*
