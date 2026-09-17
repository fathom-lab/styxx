# SURVEY — the neighbours of the sand, priced against twenty fetches

Fathom Lab · 2026-09-13 · **A survey, not a result.** It runs the procedure frozen the same evening in
`papers/plates/PROTOCOL_sand_prior_art_2026_09_13.md` (commit 9f5fd42c), which named its sources,
its clauses and its outcome table before anything was fetched, and it prices exactly one sentence:
the one the lab was asked to say about the sand and refused to say from memory. Every count below is
sworn to `papers/plates/sand_prior_art_survey.json`, which holds the per-source answers with the
URL as fetched, the fetch time, the sha256 of the bytes saved, the verdict per clause and the quoted
sentence behind it; `build_sand_survey.py` beside it is the script that wrote the record from the
surveyor's entries. This document is itself sworn.

## what was read

<sworn r="path:papers/plates/sand_prior_art_survey.json#/counts/sources_in_list" k="numeric">The frozen list named 20 sources and closed itself against additions.</sworn> <sworn r="path:papers/plates/sand_prior_art_survey.json#/counts/read" k="numeric">10 were READ end to end</sworn> (the pages and the one paper the fetch could process whole); <sworn r="path:papers/plates/sand_prior_art_survey.json#/counts/skimmed" k="numeric">9 were SKIMMED</sworn> — the papers read from their bytes for their first pages only, and the Nature article from its abstract because its body is paywalled — and under the procedure a skimmed source may occupy a clause or stay silent but may never retire one; <sworn r="path:papers/plates/sand_prior_art_survey.json#/counts/unfetchable" k="numeric">1 was UNFETCHABLE</sworn> (the archived Proof of Existence page carries only its title). <sworn r="path:papers/plates/sand_prior_art_survey.json#/counts/leads_not_scored" k="numeric">6 leads surfaced while reading and are recorded without being scored</sworn>, because the list was closed.

Two limits, stated: the pages were read through a fetch-and-summarise tool with the whole page as
its input, and the surveyor set every verdict from the quotes it returned and from the paper pages
read directly; and one agent ran one pass with no independent re-fetch. Nine skimmed papers is nine
places where a retiring sentence may sit in a section nobody read.

## the sentence under test

> We know of no lab that binds every published number to bytes at a commit, re-derives every
> verdict from those bytes into a chained log, gives every receipt a face a stranger reads without
> json, fingerprints a model's behavior on hashed canaries against a measured null floor under a
> preregistration sealed on a public chain before the run, and pays a standing bounty against its
> own verifier — at once.

## what the sources did to it

- **C1, binding numbers to bytes**, is not re-priced: it inherits <sworn r="path:papers/plates/sand_prior_art_survey.json#/clauses/C1/status" k="quote">`OCCUPIED`</sworn> from the 2026-09-05 survey of sworn output, with its neighbours named there.
- **C2, the chained log of re-derived verdicts**: <sworn r="path:papers/plates/sand_prior_art_survey.json#/clauses/C2/status" k="quote">`OCCUPIED`</sworn>. Certificate Transparency (RFC 6962) chains certificates in an append-only Merkle log with a signed head that monitors check, and Rekor does the same for signed supply-chain metadata; neither logs a verdict re-derived from receipt bytes. The mechanism is theirs; the object is not.
- **C3, a face for every receipt**: <sworn r="path:papers/plates/sand_prior_art_survey.json#/clauses/C3/status" k="quote">`OCCUPIED`</sworn>. Perrig and Song drew hash visualizations so people could compare key fingerprints by eye in 1999; OpenSSH shipped random art for host keys in 2008; Don Park drew identicons from hashed addresses in 2007. The plate is that idea applied to receipts of scientific claims, and the sentence must say so.
- **C4a, the behaviour fingerprint against a measured floor**: <sworn r="path:papers/plates/sand_prior_art_survey.json#/clauses/C4a/status" k="quote">`OCCUPIED`</sworn>, and crowded. Dutta et al. measure per-item flips and KL divergence between a model and its compressed copy and argue that accuracy hides them — the checksum's measurement, without the hashed set, the in-situ floor or the interval verdict; Thinking Machines measure the same-weights spread of served completions — the null floor's idea, on completions; Chen, Zaharia and Zou count per-prompt mismatches across service versions; Kriegeskorte compares dissimilarity matrices by their upper triangles with a noise floor; Hochlehnert et al. measure seed variance of benchmark scores; REEF and instructional fingerprinting fingerprint models for ownership. Nothing retires the clause; almost everything in it has a neighbour.
- **C4b, the seal on a public chain**: <sworn r="path:papers/plates/sand_prior_art_survey.json#/clauses/C4b/status" k="quote">`RETIRED`</sworn>, by <sworn r="path:papers/plates/sand_prior_art_survey.json#/clauses/C4b/retired_by/0" k="quote">`S05`</sworn>, OpenTimestamps, which anchors the hash of any data to the Bitcoin chain and lets anyone verify it later — a preregistration is data. Haber and Stornetta set the problem in 1991, Registered Reports and OSF registrations freeze protocols before data without a public ledger, and proof-of-learning publishes training proofs to one. The seal is a practice the lab adopted, not one it can say it knows no other of.
- **C5, the standing bounty against the lab's own verifier**: <sworn r="path:papers/plates/sand_prior_art_survey.json#/clauses/C5/status" k="quote">`OCCUPIED`</sworn>. Immunefi runs standing bounties against deployed code, adjudicated by Immunefi; the Preregistration Challenge paid a thousand dollars for publishing preregistered work until 2018; the NeurIPS reproducibility challenge asked the community to reproduce papers for recognition. None pays a stranger for a record in which the payer's own verifier disagrees with the payer.
- **The conjunction**: <sworn r="path:papers/plates/sand_prior_art_survey.json#/conjunction/status" k="quote">`RETIRED`</sworn> as written, because one of its clauses is; no single source does the surviving clauses at once, so the sentence survives as their conjunction — <sworn r="path:papers/plates/sand_prior_art_survey.json#/sentence/status" k="quote">`SURVIVES_WITHOUT_C4b`</sworn>. <sworn r="path:papers/plates/sand_prior_art_survey.json#/counts/clauses_occupied" k="numeric">5 clauses are OCCUPIED</sworn>, <sworn r="path:papers/plates/sand_prior_art_survey.json#/counts/clauses_retired" k="numeric">1 is RETIRED</sworn>, <sworn r="path:papers/plates/sand_prior_art_survey.json#/counts/clauses_free" k="numeric">0 are FREE</sworn>.

## the sentence the lab may now say, and the words it may not

The surviving sentence is in the record at `#/sentence/text`, with every neighbour named inside it.
Its shape: *we know of no lab that does these five things at once — and here is who does each one
alone.* The word "first" is not licensed by this survey under any reading of it, nor "novel", nor
"revolutionary"; the licensed form is "we know of no", and only for the conjunction, and only with
the neighbours in the same breath. The seal is out of the sentence entirely.

What would change this document: a source the list missed that does the five surviving things at
once; or a full read of any skimmed paper that turns out to do C4a or C5 for the lab's own object,
which would retire the sentence outright. The leads are recorded so the next survey starts from them.
