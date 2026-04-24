# Zenodo attribution fix packet — record 19703527

**Problem:** the Cognometry v0 paper (DOI `10.5281/zenodo.19703527`, the
record v4/v5/v6 READMEs all point at) was deposited under the
**Flobi / Fathom Lab** Zenodo account, while the earlier methodology
paper (DOI `10.5281/zenodo.19504993`) is under the
**Alexander Rodabaugh / Fathom Intelligence** Zenodo account. This
splits the publication record across two profiles — a reader
citation-walking from one paper won't see the other in the author's
works list.

**Fix:** do (1) now (5 min), request (2) in parallel (takes ~1 week
for Zenodo support).

---

## (1) Metadata edit — executable today, keeps the DOI

Log in at https://zenodo.org with the account that owns record
`19703527` (the Flobi / Fathom Lab account), then:

1. Open https://zenodo.org/records/19703527
2. Click **Edit** (top-right).
3. Under **Creators**, ensure the list contains (in this order):
   - **Alexander Rodabaugh** — Affiliation: `Fathom Intelligence` —
     ORCID: *(fill in Alexander's ORCID here if available)*
   - **Flobi** — Affiliation: `Fathom Lab`
4. Under **Communities**, ensure the record is linked to any Zenodo
   community the Fathom Intelligence account already uses (so both
   profiles index it).
5. Under **Related identifiers**, add:
   - `10.5281/zenodo.19504993` — relation `isContinuationOf` — type
     `publication` (the v3 methodology paper).
6. Under **References**, cross-reference
   `doi.org/10.5281/zenodo.19504993`.
7. Click **Publish**. Zenodo mints a new version DOI internally but
   the concept-DOI-of-record keeps resolving; citations continue to
   work.

This makes the record appear on Alexander Rodabaugh's Zenodo profile
page via the creator link, even before the full ownership transfer.

---

## (2) Ownership transfer request — opens the background ticket

Send the following email to Zenodo support:

**To:** info@zenodo.org
**Subject:** Record ownership transfer request — zenodo.19703527

**Body:**

```
Hello Zenodo team,

I'd like to request an ownership transfer for record 19703527
("Cognometry v0: 8-Benchmark Cross-Validated Hallucination Detection
in Production LLMs", DOI 10.5281/zenodo.19703527).

Source account:       [email associated with Flobi / Fathom Lab]
Destination account:  [email associated with Alexander Rodabaugh /
                       Fathom Intelligence]

Reason: the record was deposited under a secondary lab account, but
belongs in the Fathom Intelligence publication record alongside our
earlier methodology paper (10.5281/zenodo.19504993, already on the
destination account). We want a unified author profile for the
cognometry research program before an external launch push.

Both accounts are controlled by the same research group. I can
confirm this from either email address on request.

Please let me know what additional verification you need.

Thanks,
[your name]
```

Zenodo support typically responds within 5 business days.

---

## (3) Optional: future-proof by registering a Zenodo Community

Once the transfer completes, consider creating a **"Fathom
Intelligence"** Zenodo Community and adding all Fathom papers to it.
Any future papers deposited to the community are auto-linked
regardless of which lab-member account uploaded them. This prevents
the split from recurring.

Community setup: https://zenodo.org/communities/new

---

## What does NOT change

The DOI `10.5281/zenodo.19703527` is preserved through all of this.
Every existing citation (in the repo, on the site, in awesome-list
PRs, in outreach emails) continues to resolve correctly.
