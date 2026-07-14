# OPERATOR BRIEF — 2026-07-14: the external-counterparty flip

**Three items, all yours, ~30 minutes total. Each converts the diary into a record. The agent
lanes (B7 3B run, B2-coupling chain, G5 scorecard) run without you today.**

## 1. arXiv (~20 min) — the discoverability fix

- Create the account at https://arxiv.org/user/register with the professional identity
  (Alexander Rodabaugh, Fathom Intelligence). First cs submission may require ENDORSEMENT —
  if prompted, the endorsement-request email cites the two published DOIs below.
- Upload the two papers already published + DOI'd on Zenodo (download your own deposit PDFs):
  - **Calibration Poisoning, Not Erasure** — DOI 10.5281/zenodo.21241185 (Fathom v26)
  - **read≠write v0.3 (private-calibration re-lock)** — DOI 10.5281/zenodo.21263158 (Fathom v28)
- Category suggestion: primary cs.LG, cross-list cs.CR + cs.CL. License: CC-BY-4.0 (matches
  Zenodo). In the report-number/DOI field cite the Zenodo DOI so the records link.
- Note: if the PDFs were TeX-generated, arXiv demands the TeX source; pandoc/typst-generated
  PDFs upload as-is.

## 2. OPEN_CORE §4 excision (~2 min) — the Brussels de-poisoning

Review `docs/governance/PROPOSED_open_core_s4_excision_2026_07_14.md`. Say "apply the excision"
and the agent commits the three edits (remove token passage, neutralize one anti-goal line,
quarantine the idea to docs/token/NETWORK_NOTES.md). Nothing about the open-core rails changes.

## 3. main fast-forward (~1 min) — publish the receipts

`paper/read-neq-write` is 60+ commits ahead of `origin/main` (everything since cycle ~29: the
E-series, the parity arc, B2 static+adaptive, ladder v1, OATH v0.5, REPLICATIONS.md, the G5
prereg+manifest).

    cd C:\Users\heyzo\clawd\styxx
    git checkout main
    git merge --ff-only paper/read-neq-write
    git push origin main
    git checkout paper/read-neq-write

(If you prefer the agent does it: say "fast-forward main" — push-to-main is yours to authorize.)

## Today's board (context, no action needed)

- **B7 (3B erasure, the paper's scale gate):** in flight, crash-safe, seed-0 already SURVIVES
  twice per the overnight log; verdict expected late tonight.
- **B2-coupling (the dose-response law):** auto-launches the moment B7's result lands
  (`card_chain_watcher.py` armed; `--dry` verdict validation passed all three branches).
- **G5 scorecard:** prereg frozen `a059ba9` BEFORE extraction; population frozen `2368248`
  (18 docs / 24 signatories); extraction fleet running. Result lands in-repo today.
  External publication of the scorecard = your call after you see the numbers.
- **Zenodo next version:** staged paste-and-publish (`ZENODO_NEXT_VERSION_DRAFT.md`), gated on
  B7 + coupling + your fast-forward.
