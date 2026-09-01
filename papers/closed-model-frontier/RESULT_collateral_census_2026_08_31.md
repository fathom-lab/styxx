# RESULT — the collateral census: the floor is low, and that decides a design

Fathom Lab · 2026-08-31 · Prereg: `PREREG_collateral_census_2026_08_31.md`, frozen before
the classifier existed. Receipts: `collateral_census.json`,
`collateral_census_fidelity.json`, `collateral_census.py`. Corpus: AIDev (HuggingFace
`hao-li/AIDev`, CC-BY-4.0; Zenodo 10.5281/zenodo.16919272).

**No claim was read and nothing is accused.** This measurement extracts nothing from prose
and issues no verdict about any pull request or any agent. It sorts changed files by what
they are.

## The gates

**G-C1 (fidelity) — PASS**, run before any corpus number was computed: every constructed
fixture classified correctly, and the file-status folding that the same-day correction found
wrong in the EXTERNAL-1 harness re-verified against its own cases. **G-C2 (determinism)**,
**G-C3 (exhaustiveness)** — every changed file resolved to exactly one class and the class
counts sum to the file count, asserted in code — and **G-C4 (no accusation)** hold.

## What is in an agent's change

Across the eligible pull requests, the classifier sorted well over a million changed files.
The shares, in the frozen resolution order:

| class | share of files | share of PRs containing it |
|---|---|---|
| lockfile | 0.69% | 9.13% |
| generated | 5.46% | 6.09% |
| snapshot | 1.46% | 1.46% |
| migration | 0.49% | 1.88% |
| whitespace-equivalent | 2.87% | 9.52% |
| **substantive** | **89.04%** | **99.56%** |

**The collateral floor is about one file in nine.** Nearly nine in ten changed files are
none of the recognised incidental categories, and essentially every pull request contains at
least one such file.

## Why this number was worth measuring before building anything

The design this lab was moving toward — an agent declaring its changes structurally, with a
third band for what it changed and did not declare — has one obvious practical objection,
and it was raised as the likely fatal one: real diffs are full of incidental churn nobody
would ever describe, so a "changed but not declared" band would drown in lockfiles,
generated output and formatting noise and be useless.

That objection now has a number attached, and the number does not support it. Even reading
the collateral share generously — the whitespace-equivalent class is a **lower bound**, as
the preregistration states, because reformatting that reorders tokens or rewraps lines lands
in `substantive` — incidental files are a small minority of what agents change. A band
listing undeclared changes would be dominated by ordinary edited source, not by noise.

The reverse reading is equally load-bearing and less comfortable: because collateral is
scarce, **an exemption list would not have saved anyone much**, which removes the main
argument for building one — and an exemption list is exactly the structure that gives a
dishonest change somewhere to hide.

## Per agent, described and not ranked

The per-agent table is in the receipt. It differs more than expected: one agent's changes
are roughly a fifth incidental while another's are about one in twenty, driven mostly by
generated files and snapshots. Read this as a description of what different tools touch —
one agent working in repositories with heavy code generation is not more or less honest than
one that is not. **No ranking is published and none is implied**; nothing here measures
disclosure, because nothing here read a claim.

## Named limitations

`whitespace-equivalent` is a floor on formatting churn, not a measurement of it — the honest
test needs per-language parsing and this one compares content with whitespace removed. The
generated and lockfile lists are closed and quoted in the receipt; anything they miss is
counted as substantive, which again biases the collateral share **downward**. And the
corpus's own patch-truncation boundary is inherited from the dataset.

None of these numbers may be charted against EXTERNAL-1's coverage figure: different
population, different denominator.

---

*Two instruments failed today at the moment prose becomes a claim. This one never reaches
that moment, and it still decides something: the ground under the next design is mostly
real code, so the band worth building is not drowning in noise.*
